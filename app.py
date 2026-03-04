from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import anthropic
from flask_login import LoginManager, current_user, login_required
from models import db, User
from auth import auth_bp
from decorators import analysis_gate, basic_required, pro_required, elite_required
import stripe
import numpy as np
from scipy.stats import norm
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import requests as http_requests
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY', 'jaydawolfx-secret-2026')
_db_url = os.environ.get('DATABASE_URL', 'sqlite:///jaydawolfx.db')
if _db_url.startswith('postgres://'):
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = _db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_DURATION'] = 60 * 60 * 24 * 30
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24 * 30
app.config['SESSION_PERMANENT'] = True

stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PRICES = {
    'basic': os.environ.get('STRIPE_BASIC_PRICE_ID', 'price_REPLACE_BASIC'),
    'pro': os.environ.get('STRIPE_PRO_PRICE_ID', 'price_REPLACE_PRO'),
    'elite': os.environ.get('STRIPE_ELITE_PRICE_ID', 'price_REPLACE_ELITE'),
}
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', '')

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login_page'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith('/api/') or request.path.startswith('/scanner/'):
        return jsonify({'error': 'Please log in to access this feature.', 'action': 'login'}), 401
    return redirect(url_for('auth.login_page'))

# ✅ HEALTH CHECK ENDPOINT — Fast response for Render (DO NOT REMOVE)
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()}), 200

# ✅ LAZY DATABASE INITIALIZATION — Only runs when needed, not on startup
_db_initialized = False

def ensure_db_initialized():
    """Initialize database only when first needed, not on startup"""
    global _db_initialized
    if not _db_initialized:
        try:
            with app.app_context():
                db.create_all()
                _db_initialized = True
        except Exception as e:
            print(f"[DB] Initialization error (will retry): {e}")

@app.route('/api/server-time')
def server_time():
    from datetime import timedelta, timezone
    try:
        import zoneinfo
        eastern = zoneinfo.ZoneInfo('America/New_York')
        now = datetime.now(eastern)
    except Exception:
        from datetime import timedelta
        now = datetime.utcnow() - timedelta(hours=5)
    day = now.weekday()
    monday = now - timedelta(days=day)
    friday = monday + timedelta(days=4)
    week_range = f"{monday.strftime('%b %d')} — {friday.strftime('%b %d, %Y')}"
    hour = now.hour; minute = now.minute; is_weekday = day < 5
    market_open = is_weekday and (hour > 9 or (hour == 9 and minute >= 30)) and hour < 16
    pre_market = is_weekday and hour >= 4 and (hour < 9 or (hour == 9 and minute < 30))
    after_hours = is_weekday and hour >= 16 and hour < 20
    if market_open: market_status = 'MARKET OPEN'; status_color = '#00ff99'
    elif pre_market: market_status = 'PRE-MARKET'; status_color = '#ffe033'
    elif after_hours: market_status = 'AFTER HOURS'; status_color = '#ff7744'
    else: market_status = 'MARKET CLOSED'; status_color = '#ff4466'
    return jsonify({'date': now.strftime('%B %d, %Y'), 'time': now.strftime('%H:%M:%S') + (' EDT' if now.dst() else ' EST'),
        'day': now.strftime('%A'), 'week_range': week_range, 'market_status': market_status,
        'status_color': status_color, 'timestamp': now.isoformat()})

# ✅ REGISTER BLUEPRINTS WITH ERROR HANDLING
app.register_blueprint(auth_bp)

# Try to import optional blueprints, but don't crash if they don't exist
try:
    from payments import payments_bp
    app.register_blueprint(payments_bp)
except ImportError as e:
    print(f"[WARNING] payments blueprint not found: {e}")

try:
    from scanner import scanner_bp
    app.register_blueprint(scanner_bp)
except ImportError as e:
    print(f"[WARNING] scanner blueprint not found: {e}")

try:
    from forex import forex_bp
    app.register_blueprint(forex_bp)
except ImportError as e:
    print(f"[WARNING] forex blueprint not found: {e}")

# ✅ REMOVED: The blocking db.create_all() that was causing the timeout
# Old code removed:
# with app.app_context():
#     db.create_all()

# ═══════════════════════════════════════════════════════════════
# CANDLESTICK ENGINE — Real chart data using yfinance (FREE)
# ═══════════════════════════════════════════════════════════════

# Yahoo Finance symbol map for forex pairs
YF_MAP = {
    'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X', 'USD/JPY': 'USDJPY=X',
    'USD/CHF': 'USDCHF=X', 'AUD/USD': 'AUDUSD=X', 'USD/CAD': 'USDCAD=X',
    'NZD/USD': 'NZDUSD=X', 'EUR/GBP': 'EURGBP=X', 'EUR/JPY': 'EURJPY=X',
    'GBP/JPY': 'GBPJPY=X', 'XAU/USD': 'GC=F',    'DXY':     'DX-Y.NYB',
}

# Candle cache — avoid re-fetching on every request
_candle_cache = {}
_candle_cache_ttl = 300  # 5 minutes

def get_candles(pair, interval='1d', period='3mo'):
    cache_key = f"{pair}_{interval}_{period}"
    now = time.time()
    if cache_key in _candle_cache:
        cached = _candle_cache[cache_key]
        if now - cached['ts'] < _candle_cache_ttl:
            return cached['data']

    try:
        import yfinance as yf
        sym = YF_MAP.get(pair, pair.replace('/', '') + '=X')
        ticker = yf.Ticker(sym)
        df = ticker.history(interval=interval, period=period)
        if df.empty:
            return []
        candles = []
        for ts, row in df.iterrows():
            candles.append({
                'time': str(ts)[:10] if interval != '1h' else str(ts)[:16],
                'open':  round(float(row['Open']),  5),
                'high':  round(float(row['High']),  5),
                'low':   round(float(row['Low']),   5),
                'close': round(float(row['Close']), 5),
                'volume': int(row.get('Volume', 0))
            })
        _candle_cache[cache_key] = {'data': candles, 'ts': now}
        return candles
    except Exception as e:
        print(f'[Candles] {pair} {interval} error: {e}')
        return []

def calc_ema(closes, period):
    if len(closes) < period:
        return None
    k = 2.0 / (period + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = c * k + ema * (1 - k)
    return round(ema, 5)

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i-1]
        gains.append(max(0, delta))
        losses.append(max(0, -delta))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for i in range(period, len(closes)):
        delta = closes[i] - closes[i-1]
        avg_gain = (avg_gain * (period - 1) + max(0, delta)) / period
        avg_loss = (avg_loss * (period - 1) + max(0, -delta)) / period
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return round(100 - (100 / (1 + rs)), 2)

def calc_macd(closes):
    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    if ema12 is None or ema26 is None:
        return None, None
    macd_line = ema12 - ema26
    signal_line = calc_ema([macd_line] if isinstance(macd_line, (int, float)) else closes[-9:], 9)
    return round(macd_line, 5), round(signal_line, 5) if signal_line else None

@app.route('/api/candlestick/<pair>')
@login_required
def get_candle_data(pair):
    ensure_db_initialized()
    interval = request.args.get('interval', '1d')
    period = request.args.get('period', '3mo')
    candles = get_candles(pair, interval, period)
    return jsonify({'pair': pair, 'interval': interval, 'period': period, 'candles': candles})

# ═══════════════════════════════════════════════════════════════
# REST OF YOUR ROUTES (from original app.py)
# ═══════════════════════════════════════════════════════════════

@app.route('/api/options-flow', methods=['GET'])
@login_required
def options_flow():
    return jsonify({'status':'placeholder','message':'Live options flow coming soon — Unusual Whales API',
                    'preview':[
                        {'ticker':'SPY','type':'CALL','strike':'580C','expiry':'0DTE','size':5000,'premium':'$2.1M','sentiment':'BULLISH'},
                        {'ticker':'NVDA','type':'CALL','strike':'900C','expiry':'1DTE','size':1200,'premium':'$890K','sentiment':'BULLISH'},
                        {'ticker':'AAPL','type':'PUT','strike':'220P','expiry':'3DTE','size':800,'premium':'$340K','sentiment':'BEARISH'},
                    ]})

import uuid, threading
_byakugan_jobs = {}

def run_byakugan_job(job_id, scan_filter, date_str, user_id):
    try:
        ensure_db_initialized()
        _byakugan_jobs[job_id]['status'] = 'scanning'
        # ... your rest of the implementation
    except Exception as e:
        import traceback; print(traceback.format_exc())
        _byakugan_jobs[job_id]={'status':'error','error':str(e)}

@app.route('/api/byakugan-scan', methods=['POST'])
@login_required
@elite_required
def byakugan_scan_v2():
    try:
        data=request.get_json() or {}
        scan_filter=data.get('filter','ALL')
        date_str=datetime.now().strftime('%A, %B %d, %Y')
        job_id=str(uuid.uuid4())[:8]
        _byakugan_jobs[job_id]={'status':'starting'}
        t=threading.Thread(target=run_byakugan_job,args=(job_id,scan_filter,date_str,current_user.id),daemon=True)
        t.start()
        return jsonify({'job_id':job_id,'status':'starting'})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/byakugan-poll/<job_id>', methods=['GET'])
@login_required
@elite_required
def byakugan_poll(job_id):
    job=_byakugan_jobs.get(job_id)
    if not job: return jsonify({'status':'error','error':'Job not found'}),404
    if job['status']=='done':
        result=job.get('result',{}); result['status']='done'
        del _byakugan_jobs[job_id]; return jsonify(result)
    if job['status']=='error':
        return jsonify({'status':'error','error':job.get('error','Unknown error')}),500
    return jsonify({'status':job['status']})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
