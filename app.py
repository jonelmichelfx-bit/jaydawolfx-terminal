from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import anthropic
from flask_login import LoginManager, current_user, login_required
from models import db, User
from auth import auth_bp
from decorators import analysis_gate, basic_required, pro_required, elite_required, byakugan_required
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
    'byakugan': os.environ.get('STRIPE_BYAKUGAN_PRICE_ID', 'price_REPLACE_BYAKUGAN'),
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

app.register_blueprint(auth_bp)
from payments import payments_bp
app.register_blueprint(payments_bp)
from scanner import scanner_bp
app.register_blueprint(scanner_bp)
from forex import forex_bp
app.register_blueprint(forex_bp)

with app.app_context():
    db.create_all()

# ═══════════════════════════════════════════════════════════════
# CANDLESTICK ENGINE — Real chart data using yfinance (FREE)
# ═══════════════════════════════════════════════════════════════

# Yahoo Finance symbol map for forex pairs
YF_MAP = {
    'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X', 'USD/JPY': 'USDJPY=X',
    'USD/CHF': 'USDCHF=X', 'AUD/USD': 'AUDUSD=X', 'USD/CAD': 'USDCAD=X',
    'NZD/USD': 'NZDUSD=X', 'EUR/GBP': 'EURGBP=X', 'EUR/JPY': 'EURJPY=X',
    'GBP/JPY': 'GBPJPY=X', 'XAU/USD': 'GC=F',    'DXY':     'DX=F',
    'SPY':     'SPY',      'QQQ':     'QQQ',      'VIX':     '^VIX',
}

# Candle cache — avoid re-fetching on every request
_candle_cache = {}
_candle_cache_ttl = 900  # 15 minutes — reuse candles across scanner requests

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
    """Wilder's Smoothed RSI — same method as TradingView/MT4"""
    if len(closes) < period + 2:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    # First avg = simple average of first {period} values (Wilder's seed)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    # Then Wilder's smoothing for remaining values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)

def calc_macd(closes):
    if len(closes) < 26:
        return None, None
    ema12 = calc_ema(closes[-50:], 12)
    ema26 = calc_ema(closes[-50:], 26)
    if ema12 and ema26:
        macd = round(ema12 - ema26, 5)
        return macd, 'BULLISH' if macd > 0 else 'BEARISH'
    return None, None

def find_sr_levels(candles, current_price, lookback=50):
    if len(candles) < 5:
        return []

    recent = candles[-lookback:] if len(candles) > lookback else candles
    highs = [c['high'] for c in recent]
    lows  = [c['low']  for c in recent]
    closes = [c['close'] for c in recent]

    levels = []

    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            levels.append({'price': highs[i], 'type': 'swing_high', 'strength': 1})

    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            levels.append({'price': lows[i], 'type': 'swing_low', 'strength': 1})

    is_jpy = current_price > 50
    is_gold = current_price > 1000
    if is_gold:
        base = round(current_price / 50) * 50
        for i in range(-4, 5):
            levels.append({'price': base + i * 50, 'type': 'round_number', 'strength': 2})
    elif is_jpy:
        base = round(current_price)
        for i in range(-5, 6):
            if i % 50 == 0 or i % 100 == 0:
                levels.append({'price': base + i, 'type': 'round_number', 'strength': 2})
            elif i % 25 == 0:
                levels.append({'price': base + i, 'type': 'round_number', 'strength': 1})
    else:
        base = round(current_price * 100) / 100
        for i in [-200, -150, -100, -50, 0, 50, 100, 150, 200]:
            levels.append({'price': round(base + i * 0.0001, 4), 'type': 'round_number', 'strength': 1})

    threshold = current_price * 0.001
    clustered = []
    sorted_levels = sorted(levels, key=lambda x: x['price'])
    for lv in sorted_levels:
        merged = False
        for cl in clustered:
            if abs(lv['price'] - cl['price']) < threshold:
                cl['strength'] += lv['strength']
                merged = True
                break
        if not merged:
            clustered.append({'price': lv['price'], 'type': lv['type'], 'strength': lv['strength']})

    clustered.sort(key=lambda x: abs(x['price'] - current_price))
    result = []
    for lv in clustered[:8]:
        dist = lv['price'] - current_price
        is_jpy_pair = current_price > 50
        pip_size = 0.01 if is_jpy_pair else 0.0001
        pips = round(abs(dist) / pip_size)
        lv_type = 'RESISTANCE' if dist > 0 else 'SUPPORT'
        result.append({
            'type': lv_type,
            'price': round(lv['price'], 2 if is_jpy_pair or is_gold else 4),
            'distance_pips': pips,
            'strength': lv['strength'],
            'note': f"{'Strong' if lv['strength'] >= 3 else 'Moderate'} {lv_type.lower()} — {'swing ' + lv['type'].replace('_',' ') if 'swing' in lv['type'] else 'psychological level'}"
        })
    return result

def get_chart_analysis(pair, current_price):
    """
    FULL 5-timeframe chart analysis — wraps get_sage_chart_data.
    Safe to call sequentially. Candle cache (15min TTL) prevents redundant fetches.
    Wolf Scanner + Forex Scanner get identical real data as Sage Mode:
    - Wilder RSI, EMA9/20/50/200, market structure (HH/HL/LH/LL)
    - 16 candlestick patterns, real swing S/R, ADR, weekly range %, London levels
    """
    sage = get_sage_chart_data(pair, current_price)
    da   = sage.get("daily",   {})
    wk   = sage.get("weekly",  {})
    h1   = sage.get("hourly",  {})
    h4   = sage.get("h4",      {})
    m15  = sage.get("m15",     {})
    sr   = sage.get("sr_levels", [])

    w_trend  = wk.get("trend", "NEUTRAL")
    d_trend  = da.get("trend", "NEUTRAL")
    h_trend  = h1.get("trend", "NEUTRAL")
    h4_trend = h4.get("trend", "NEUTRAL")

    bull = sum(1 for t in [w_trend, d_trend, h_trend, h4_trend] if t == "BULLISH")
    bear = sum(1 for t in [w_trend, d_trend, h_trend, h4_trend] if t == "BEARISH")

    return {
        "pair":          pair,
        "current_price": current_price,
        "weekly":  wk,
        "daily":   da,
        "hourly":  h1,
        "h4":      h4,
        "m15":     m15,
        "sr_levels": sr,
        "adr":     sage.get("adr"),
        "weekly_range_pct": sage.get("weekly_range_pct"),
        "weekly_high": sage.get("weekly_high"),
        "weekly_low":  sage.get("weekly_low"),
        "trend_strength": sage.get("trend_strength", "UNKNOWN"),
        "trend_score":    sage.get("trend_score", 0),
        "d1_patterns":  sage.get("d1_patterns",  []),
        "h4_patterns":  sage.get("h4_patterns",  []),
        "m15_patterns": sage.get("m15_patterns", []),
        "indicators": {
            "ema9":      da.get("ema9"),
            "ema20":     da.get("ema20"),
            "ema50":     da.get("ema50"),
            "ema200":    da.get("ema200"),
            "rsi":       da.get("rsi"),
            "macd_bias": da.get("macd_bias"),
            "atr":       da.get("atr"),
            "bb_upper":  da.get("bb_upper"),
            "bb_mid":    da.get("bb_mid"),
            "bb_lower":  da.get("bb_lower"),
            "bb_position": da.get("bb_position"),
        },
        "trend_summary": {
            "weekly":  w_trend,
            "daily":   d_trend,
            "hourly":  h_trend,
            "h4":      h4_trend,
            "overall": "BULLISH" if bull >= 3 else "BEARISH" if bear >= 3 else "MIXED",
            "alignment": f"{bull}/4 bullish, {bear}/4 bearish"
        }
    }

def format_chart_analysis_for_prompt(ca):
    """Full real chart data formatted for AI prompt — Wolf Scanner + Forex Scanner"""
    if not ca:
        return "Chart data unavailable"

    pair = ca.get("pair", "?")
    cp   = ca.get("current_price", "?")
    da   = ca.get("daily",   {})
    wk   = ca.get("weekly",  {})
    h1   = ca.get("hourly",  {})
    h4   = ca.get("h4",      {})
    m15  = ca.get("m15",     {})
    ts   = ca.get("trend_summary", {})
    sr   = ca.get("sr_levels", [])
    sep  = "=" * 65

    lines = [
        sep,
        f"REAL CHART DATA — {pair} @ {cp}",
        f"Overall: {ts.get('overall','?')} | Trend Strength: {ca.get('trend_strength','?')}",
        f"TF Alignment: {ts.get('alignment','?')} | ADR: {ca.get('adr','?')} | Weekly Range Used: {ca.get('weekly_range_pct','?')}%",
        f"Weekly High: {ca.get('weekly_high','?')} | Weekly Low: {ca.get('weekly_low','?')}",
        sep,
        f"WEEKLY: Trend={wk.get('trend','?')} Structure={wk.get('structure','?')} Phase={wk.get('phase','?')}",
        f"  EMA20={wk.get('ema20','?')} RSI={wk.get('rsi','?')} Last3={wk.get('last3','?')}",
        f"  52wk High={wk.get('high52','?')} | 52wk Low={wk.get('low52','?')}",
        f"DAILY: Trend={da.get('trend','?')} Structure={da.get('structure','?')} Phase={da.get('phase','?')}",
        f"  {da.get('phase_desc','')}",
        f"  EMA9={da.get('ema9','?')} EMA20={da.get('ema20','?')} EMA50={da.get('ema50','?')} EMA200={da.get('ema200','?')}",
        f"  RSI={da.get('rsi','?')} MACD={da.get('macd_bias','?')} ATR={da.get('atr','?')} vs200EMA={da.get('vs_ema200','?')}",
        f"  BB: Upper={da.get('bb_upper','?')} Mid={da.get('bb_mid','?')} Lower={da.get('bb_lower','?')} Position={da.get('bb_position','?')}%",
        f"  Swing Highs: {da.get('swing_highs',[])} | Swing Lows: {da.get('swing_lows',[])}",
        f"  20d High={da.get('high20d','?')} | 20d Low={da.get('low20d','?')} | Last5={da.get('last5','?')}",
        f"H4: Trend={h4.get('trend','?')} Structure={h4.get('structure','?')} Phase={h4.get('phase','?')}",
        f"  EMA9={h4.get('ema9','?')} EMA20={h4.get('ema20','?')} RSI={h4.get('rsi','?')} MACD={h4.get('macd_bias','?')}",
        f"  48h High={h4.get('high48h','?')} | 48h Low={h4.get('low48h','?')}",
        f"H1: Trend={h1.get('trend','?')} Structure={h1.get('structure','?')}",
        f"  EMA9={h1.get('ema9','?')} EMA20={h1.get('ema20','?')} RSI={h1.get('rsi','?')} MACD={h1.get('macd_bias','?')}",
        f"  24h High={h1.get('high24h','?')} | 24h Low={h1.get('low24h','?')}",
        f"M15: Trend={m15.get('trend','?')} Structure={m15.get('structure','?')}",
        f"  EMA9={m15.get('ema9','?')} RSI={m15.get('rsi','?')}",
        f"  London High={m15.get('london_high','?')} | London Low={m15.get('london_low','?')}",
    ]

    if sr:
        lines.append("KEY S/R (from real swing highs/lows):")
        for lv in sr[:6]:
            lines.append(f"  {lv['type']}: {lv['price']} | {lv['distance_pips']} pips | {lv['note']}")

    h1_sr = h1.get("sr", [])
    if h1_sr:
        lines.append("INTRADAY S/R (H1):")
        for lv in h1_sr[:3]:
            lines.append(f"  {lv['type']}: {lv['price']} ({lv['distance_pips']} pips)")

    all_pats = (
        [("D1", p) for p in ca.get("d1_patterns",[])] +
        [("H4", p) for p in ca.get("h4_patterns",[])] +
        [("M15",p) for p in ca.get("m15_patterns",[])]
    )
    if all_pats:
        lines.append("CANDLESTICK PATTERNS:")
        for tf, p in all_pats:
            lines.append(f"  [{tf}] {p['pattern']} ({p['bias']}) — {p['note']}")

    lines.append(sep)
    return "\n".join(lines)


def get_multi_pair_chart_data(pairs, current_prices):
    results = {}
    def fetch_one(pair):
        price = current_prices.get(pair, {})
        cp = float(price.get('price', 1.0)) if price else 1.0
        return pair, get_chart_analysis(pair, cp)

    # Sequential fetch — prevents OOM crash on 2GB Render instance.
    # yfinance is network I/O bound, not CPU bound, so sequential is nearly
    # as fast while using 4x less RAM. Candle cache (15min TTL) means
    # repeated scans are instant.
    for pair in pairs:
        try:
            price = current_prices.get(pair, {})
            cp = float(price.get('price', 1.0)) if price else 1.0
            results[pair] = get_chart_analysis(pair, cp)
        except Exception as e:
            print(f'[ChartFetch] {pair}: {e}')
    return results

# ── Options helpers ──────────────────────────────────────────

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return None
    try:
        d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
        if option_type=='call':
            price=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2); delta=norm.cdf(d1)
            theta=(-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365
        else:
            price=K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1); delta=norm.cdf(d1)-1
            theta=(-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
        gamma=norm.pdf(d1)/(S*sigma*np.sqrt(T)); vega=S*norm.pdf(d1)*np.sqrt(T)/100
        rho=(K*T*np.exp(-r*T)*norm.cdf(d2)/100 if option_type=='call' else -K*T*np.exp(-r*T)*norm.cdf(-d2)/100)
        return {'price':round(float(price),4),'delta':round(float(delta),4),'gamma':round(float(gamma),6),
                'theta':round(float(theta),4),'vega':round(float(vega),4),'rho':round(float(rho),4)}
    except Exception as e: return {'error':str(e)}

def build_pnl_curve(S, K, T, r, sigma, option_type, premium_paid, days_held):
    price_range=np.linspace(S*0.70,S*1.30,80); T_remaining=max(T-(days_held/365),0.001); curve=[]
    for target in price_range:
        g=calculate_greeks(target,K,T_remaining,r,sigma,option_type)
        if g and 'price' in g: curve.append({'price':round(float(target),2),'pnl':round((g['price']-premium_paid)*100,2)})
    return curve

def fetch_live_data(ticker, expiration, strike, option_type):
    try:
        import yfinance as yf
        stock=yf.Ticker(ticker); hist=stock.history(period='1d')
        if hist.empty: return None,'Could not fetch stock price'
        stock_price=float(hist['Close'].iloc[-1])
        try:
            chain=stock.option_chain(expiration); contracts=chain.calls if option_type=='call' else chain.puts
            strike_f=float(strike); row=contracts.iloc[(contracts['strike']-strike_f).abs().argsort()[:1]]
            if not row.empty:
                iv=float(row['impliedVolatility'].iloc[0]); mark=float(row['lastPrice'].iloc[0])
                return {'stock_price':stock_price,'iv':iv,'mark':mark,'source':'yfinance'},None
        except: pass
        return {'stock_price':stock_price,'iv':None,'mark':None,'source':'price-only'},None
    except ImportError: return None,'yfinance not installed'
    except Exception as e: return None,str(e)

def fetch_stock_price_only(ticker):
    try:
        import yfinance as yf
        hist=yf.Ticker(ticker).history(period='1d')
        if not hist.empty: return float(hist['Close'].iloc[-1])
    except: pass
    return None

def fetch_option_expirations(ticker):
    try:
        import yfinance as yf
        return list(yf.Ticker(ticker).options),None
    except ImportError: return None,'yfinance not installed'
    except Exception as e: return None,str(e)

def fetch_option_strikes(ticker, expiration, option_type='call'):
    try:
        import yfinance as yf
        chain=yf.Ticker(ticker).option_chain(expiration)
        contracts=chain.calls if option_type=='call' else chain.puts
        return sorted(contracts['strike'].tolist()),None
    except ImportError: return None,'yfinance not installed'
    except Exception as e: return None,str(e)

# ── Pages ────────────────────────────────────────────────────

@app.route('/')
@login_required
def index(): return render_template('index.html')

@app.route('/login')
def login_page():
    if current_user.is_authenticated: return redirect(url_for('index'))
    return render_template('auth.html')

@app.route('/pricing')
def pricing(): return render_template('pricing.html')

@app.route('/ai-scanner')
@login_required
@basic_required
def ai_scanner(): return render_template('scanner.html')

@app.route('/ai-analysis')
@login_required
@elite_required
def ai_analysis(): return render_template('analysis.html')

@app.route('/wolf-elite')
@login_required
@elite_required
def wolf_elite(): return render_template('wolf_elite.html')

# ── Stripe ───────────────────────────────────────────────────

@app.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    plan=request.form.get('plan')
    if plan not in ('basic','pro','elite','byakugan'): flash('Invalid plan selected.','danger'); return redirect(url_for('pricing'))
    try:
        checkout=stripe.checkout.Session.create(payment_method_types=['card'],mode='subscription',
            line_items=[{'price':STRIPE_PRICES[plan],'quantity':1}],
            success_url=url_for('payment_success',plan=plan,_external=True),
            cancel_url=url_for('pricing',_external=True),
            client_reference_id=str(current_user.id),metadata={'plan':plan,'user_id':str(current_user.id)})
        return redirect(checkout.url,code=303)
    except Exception as e: flash(f'Payment error: {str(e)}','danger'); return redirect(url_for('pricing'))

@app.route('/payment-success')
@login_required
def payment_success():
    plan=request.args.get('plan','basic'); current_user.plan=plan; db.session.commit()
    flash(f"🐺 You're now on Wolf Elite {'Elite' if plan=='elite' else 'Basic'}! Let's get it.",'success')
    return redirect(url_for('index'))

@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    payload=request.get_data(as_text=True); sig_header=request.headers.get('Stripe-Signature')
    try: event=stripe.Webhook.construct_event(payload,sig_header,STRIPE_WEBHOOK_SECRET)
    except: return 'Invalid signature',400
    if event['type']=='checkout.session.completed':
        data=event['data']['object']; user_id=data.get('metadata',{}).get('user_id'); plan=data.get('metadata',{}).get('plan','basic')
        try:
            user=User.query.get(int(user_id))
            if user: user.plan=plan; user.stripe_customer_id=data.get('customer'); user.stripe_subscription_id=data.get('subscription'); db.session.commit()
        except Exception as e: print(f'[Webhook] DB error: {e}')
    elif event['type']=='customer.subscription.deleted':
        stripe_customer_id=event['data']['object'].get('customer')
        try:
            user=User.query.filter_by(stripe_customer_id=stripe_customer_id).first()
            if user: user.plan='trial'; db.session.commit()
        except Exception as e: print(f'[Webhook] DB error: {e}')
    return 'OK',200

# ── Options API ──────────────────────────────────────────────

@app.route('/api/autofill', methods=['POST'])
@analysis_gate
def autofill():
    ticker=request.json.get('ticker','').upper().strip()
    if not ticker: return jsonify({'error':'No ticker provided'})
    expirations,err=fetch_option_expirations(ticker)
    if err: return jsonify({'error':err})
    stock_price=fetch_stock_price_only(ticker)
    return jsonify({'ticker':ticker,'stock_price':round(stock_price,2) if stock_price else None,'expirations':expirations[:12] if expirations else []})

@app.route('/api/strikes', methods=['POST'])
@login_required
def strikes():
    data=request.json; strikes_list,err=fetch_option_strikes(data.get('ticker','').upper(),data.get('expiration',''),data.get('option_type','call'))
    if err: return jsonify({'error':err})
    return jsonify({'strikes':strikes_list})

@app.route('/api/contract', methods=['POST'])
@login_required
def contract():
    data=request.json; live,err=fetch_live_data(data.get('ticker','').upper(),data.get('expiration',''),data.get('strike',0),data.get('option_type','call'))
    if err: return jsonify({'error':err})
    return jsonify({'data':live})

@app.route('/api/greeks', methods=['POST'])
@analysis_gate
def greeks():
    data=request.json; ticker=data.get('ticker','AAPL').upper(); strike=float(data.get('strike',150))
    expiration=data.get('expiration',''); option_type=data.get('option_type','call')
    days_held=int(data.get('days_held',0)); r=float(data.get('r',0.045)); theta_alert=float(data.get('theta_alert',50))
    live_data=None
    if expiration: live_data,_=fetch_live_data(ticker,expiration,strike,option_type)
    if expiration:
        try:
            exp_date=datetime.strptime(expiration,'%Y-%m-%d'); T=max((exp_date-datetime.now()).days/365,0.001); dte_days=max((exp_date-datetime.now()).days,1)
        except: T=30/365; dte_days=30
    else: T=float(data.get('dte',30))/365; dte_days=int(data.get('dte',30))
    S=float(live_data['stock_price']) if live_data and live_data.get('stock_price') else float(data.get('stock_price',150))
    sigma=float(live_data['iv']) if live_data and live_data.get('iv') else float(data.get('iv',0.30))
    premium=float(live_data['mark']) if live_data and live_data.get('mark') else float(data.get('premium_paid',0))
    greeks_result=calculate_greeks(S,strike,T,r,sigma,option_type); pnl_curve=build_pnl_curve(S,strike,T,r,sigma,option_type,premium,days_held)
    daily_theta_d=(greeks_result['theta']*100) if greeks_result else 0
    return jsonify({'greeks':greeks_result,'live_data':live_data,'pnl_curve':pnl_curve,'stock_price':S,
        'premium_paid':premium,'sigma':sigma,'daily_theta_dollars':round(daily_theta_d,2),
        'theta_alert':abs(daily_theta_d)>theta_alert,'T':round(T*365,1),'dte_days':dte_days})

@app.route('/api/simulate', methods=['POST'])
@analysis_gate
def simulate():
    data=request.json
    S=float(data.get('stock_price',150));K=float(data.get('strike',150));T=float(data.get('dte',30))/365
    r=float(data.get('r',0.045));sigma=float(data.get('iv',0.30));option_type=data.get('option_type','call');premium=float(data.get('premium_paid',0))
    scenarios=[{'days':d,'curve':build_pnl_curve(S,K,T,r,sigma,option_type,premium,d)} for d in [0,5,10,15,20]]
    return jsonify({'scenarios':scenarios})

@app.route('/health')
def health(): return jsonify({'status':'online','terminal':'JAYDAWOLFX OPTIONS TERMINAL 🐺'}),200

@app.route('/api/track-pick', methods=['POST'])
@login_required
def track_pick():
    data=request.get_json(); tracker_file='wolf_tracker.json'
    try:
        if os.path.exists(tracker_file):
            with open(tracker_file,'r') as f: tracker=json.load(f)
        else: tracker={'picks':[]}
        tracker['picks'].append({'week':data.get('week'),'ticker':data.get('ticker'),'entry':data.get('entry'),
            'target':data.get('target'),'stop':data.get('stop'),'result':data.get('result','PENDING'),
            'pct_change':data.get('pct_change',0),'date_added':datetime.now().strftime('%Y-%m-%d')})
        with open(tracker_file,'w') as f: json.dump(tracker,f)
        return jsonify({'success':True}),200
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/tracker-stats', methods=['GET'])
@login_required
def tracker_stats():
    tracker_file='wolf_tracker.json'
    try:
        if not os.path.exists(tracker_file): return jsonify({'total':0,'wins':0,'losses':0,'win_rate':0,'picks':[]}),200
        with open(tracker_file,'r') as f: tracker=json.load(f)
        picks=tracker.get('picks',[]); completed=[p for p in picks if p['result'] in ['WIN','LOSS']]
        wins=len([p for p in completed if p['result']=='WIN']); losses=len([p for p in completed if p['result']=='LOSS'])
        win_rate=round((wins/len(completed)*100) if completed else 0)
        return jsonify({'total':len(picks),'wins':wins,'losses':losses,'win_rate':win_rate,'picks':picks[-20:]}),200
    except Exception as e: return jsonify({'error':str(e)}),500

# ═══════════════════════════════════════════════════════════════
# FOREX — LIVE DATA, CACHE, WOLF SCANNER
# ═══════════════════════════════════════════════════════════════

TWELVE_DATA_KEY = os.environ.get('TWELVE_DATA_API_KEY', '')
NEWS_API_KEY    = os.environ.get('NEWS_API_KEY', '')

FOREX_SESSIONS = {
    'TOKYO':    {'pairs': ['USD/JPY','EUR/JPY','GBP/JPY','AUD/USD','NZD/USD']},
    'LONDON':   {'pairs': ['EUR/USD','GBP/USD','EUR/GBP','USD/CHF','EUR/JPY']},
    'NEW YORK': {'pairs': ['EUR/USD','GBP/USD','USD/CAD','USD/JPY','XAU/USD']},
    'OVERLAP':  {'pairs': ['EUR/USD','GBP/USD','USD/JPY','XAU/USD']},
}

FALLBACK = {
    'EUR/USD':{'price':1.0380,'change':-0.0021,'pct':-0.20,'high':1.0412,'low':1.0361},
    'GBP/USD':{'price':1.2621,'change':-0.0018,'pct':-0.14,'high':1.2658,'low':1.2598},
    'USD/JPY':{'price':150.21,'change':0.38,'pct':0.25,'high':150.58,'low':149.81},
    'USD/CHF':{'price':0.8981,'change':0.0014,'pct':0.16,'high':0.9001,'low':0.8958},
    'AUD/USD':{'price':0.6241,'change':-0.0019,'pct':-0.30,'high':0.6271,'low':0.6221},
    'USD/CAD':{'price':1.4431,'change':0.0028,'pct':0.19,'high':1.4461,'low':1.4401},
    'NZD/USD':{'price':0.5681,'change':-0.0012,'pct':-0.21,'high':0.5701,'low':0.5661},
    'EUR/GBP':{'price':0.8221,'change':0.0008,'pct':0.10,'high':0.8241,'low':0.8201},
    'EUR/JPY':{'price':155.82,'change':0.21,'pct':0.13,'high':156.21,'low':155.41},
    'GBP/JPY':{'price':189.61,'change':0.42,'pct':0.22,'high':190.21,'low':189.01},
    'XAU/USD':{'price':2857.40,'change':12.30,'pct':0.43,'high':2868.20,'low':2841.10},
    'DXY':    {'price':107.82,'change':0.21,'pct':0.19,'high':108.11,'low':107.51},
}

_price_cache = {'prices': {}, 'fetched_at': 0, 'ttl': 60, 'live': False}
_er_cache = {'rates': {}, 'fetched_at': 0}

def get_session():
    from datetime import timezone, timedelta
    try:
        now = datetime.now(timezone.utc) + timedelta(hours=-5)
        h, day = now.hour, now.weekday()
        if day == 5: return 'CLOSED', []
        if day == 6 and h < 17: return 'CLOSED', []
        if day == 4 and h >= 17: return 'CLOSED', []
        if h >= 19 or h < 3:  return 'TOKYO', FOREX_SESSIONS['TOKYO']['pairs']
        if 3 <= h < 8:         return 'LONDON', FOREX_SESSIONS['LONDON']['pairs']
        if 8 <= h < 12:        return 'OVERLAP', FOREX_SESSIONS['OVERLAP']['pairs']
        if 12 <= h < 17:       return 'NEW YORK', FOREX_SESSIONS['NEW YORK']['pairs']
        return 'AFTER HOURS', []
    except: return 'UNKNOWN', []

def get_price(symbol):
    try:
        import yfinance as yf
        sym = YF_MAP.get(symbol, symbol.replace('/', '') + '=X')
        ticker = yf.Ticker(sym)
        df = ticker.history(period='2d', interval='1h')
        if not df.empty:
            latest = df.iloc[-1]
            prev   = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
            price  = round(float(latest['Close']), 5)
            prev_c = round(float(prev['Close']), 5)
            change = round(price - prev_c, 5)
            pct    = round((change / prev_c) * 100, 2) if prev_c else 0
            high   = round(float(df['High'].tail(8).max()), 5)
            low    = round(float(df['Low'].tail(8).min()),  5)
            return {
                'price': price, 'open': prev_c,
                'high': high,   'low':  low,
                'change': change, 'percent_change': pct,
                'symbol': symbol, 'live': True
            }
    except Exception as e:
        print(f'[yfinance price] {symbol}: {e}')

    try:
        sym = symbol.replace('/', '')
        r = http_requests.get(
            f'https://api.twelvedata.com/quote?symbol={sym}&apikey={TWELVE_DATA_KEY}',
            timeout=4)
        d = r.json()
        if 'close' in d and 'code' not in d:
            return {'price':float(d.get('close',0)),'open':float(d.get('open',0)),
                    'high':float(d.get('high',0)),'low':float(d.get('low',0)),
                    'change':float(d.get('change',0)),'percent_change':float(d.get('percent_change',0)),
                    'symbol':symbol,'live':True}
    except: pass

    fb = FALLBACK.get(symbol)
    if fb: return {'price':fb['price'],'open':fb['price'],'high':fb['high'],'low':fb['low'],
                   'change':fb['change'],'percent_change':fb['pct'],'symbol':symbol,'live':False}
    return None

def get_prices_parallel(pairs):
    results = {}
    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(get_price, p): p for p in pairs}
            for f in as_completed(futures, timeout=5):
                pair = futures[f]
                try:
                    q = f.result()
                    if q: results[pair] = q
                except:
                    fb = FALLBACK.get(pair)
                    if fb: results[pair] = {'price':fb['price'],'high':fb['high'],'low':fb['low'],'change':fb['change'],'percent_change':fb['pct'],'symbol':pair,'live':False}
    except:
        for p in pairs:
            fb = FALLBACK.get(p)
            if fb: results[p] = {'price':fb['price'],'high':fb['high'],'low':fb['low'],'change':fb['change'],'percent_change':fb['pct'],'symbol':p,'live':False}
    return results

def get_cached_prices():
    now = time.time()
    if now - _price_cache['fetched_at'] < _price_cache['ttl'] and _price_cache['prices']:
        return _price_cache['prices'], _price_cache['live']
    pairs = ['EUR/USD','GBP/USD','USD/JPY','USD/CHF','AUD/USD','USD/CAD',
             'NZD/USD','EUR/GBP','EUR/JPY','GBP/JPY','XAU/USD','DXY']
    fresh = get_prices_parallel(pairs)
    if fresh:
        _price_cache['prices'] = fresh
        _price_cache['fetched_at'] = now
        _price_cache['live'] = any(v.get('live', False) for v in fresh.values())
    return _price_cache['prices'], _price_cache['live']

def get_news(pair=''):
    try:
        if pair:
            q = pair.replace('/','+')
            url = f'https://newsapi.org/v2/everything?q={q}+forex&language=en&sortBy=publishedAt&pageSize=4&apiKey={NEWS_API_KEY}'
        else:
            url = f'https://newsapi.org/v2/everything?q=forex+Fed+ECB+central+bank&language=en&sortBy=publishedAt&pageSize=6&apiKey={NEWS_API_KEY}'
        r = http_requests.get(url, timeout=3)
        arts = r.json().get('articles', [])
        return [{'title':a.get('title',''),'source':a.get('source',{}).get('name',''),'published':a.get('publishedAt','')[:10]} for a in arts if a.get('title')]
    except: return []

def call_claude(prompt, max_tokens=2500):
    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    msg = client.messages.create(model='claude-sonnet-4-5', max_tokens=max_tokens,
                                  messages=[{'role':'user','content':prompt}])
    return msg.content[0].text

def parse_json_response(text):
    text = text.strip()
    if text.startswith('```'):
        parts = text.split('```')
        text = parts[1] if len(parts) > 1 else text
        if text.startswith('json'): text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    if start > 0: text = text[start:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        opens = text.count('{') - text.count('}')
        arrays = text.count('[') - text.count(']')
        for cutoff in [',\n    {', ', {', ',{']:
            last = text.rfind(cutoff)
            if last > len(text) * 0.7:
                text = text[:last]
                break
        text = text.rstrip(',').rstrip()
        text += ']' * max(0, arrays) + '}' * max(0, opens)
        try:
            return json.loads(text)
        except:
            raise json.JSONDecodeError("Could not parse AI response", text, 0)

# ── Forex pages ───────────────────────────────────────────────

@app.route('/forex')
@login_required
@pro_required
def forex(): return render_template('forex.html')

@app.route('/forex-wolf')
@login_required
def forex_wolf(): return render_template('forex_wolf.html')

@app.route('/api/forex-prices', methods=['GET'])
@login_required
def forex_prices():
    try:
        prices, is_live = get_cached_prices()
        session_name, session_pairs = get_session()
        return jsonify({'prices': prices, 'session': session_name,
                        'session_pairs': session_pairs, 'live': is_live,
                        'cached_at': datetime.now().strftime('%H:%M:%S')})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/forex-price', methods=['POST'])
@login_required
def forex_price():
    try:
        symbol = request.get_json().get('symbol', 'EUR/USD')
        q = get_price(symbol)
        return jsonify(q or {'error': 'Unavailable'})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/forex-news', methods=['POST'])
@login_required
def forex_news():
    try:
        pair = request.get_json().get('pair', '')
        return jsonify({'pair_news': get_news(pair), 'market_news': get_news()})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/forex-analyze', methods=['POST'])
@login_required
def forex_analyze():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 3000)
        pair = data.get('pair', '')

        live_ctx = ''
        if pair:
            q = get_price(pair)
            session_name, _ = get_session()
            if q:
                current_price = float(q['price'])
                live_ctx = f"\nLIVE PRICE: {pair} = {current_price} | H:{q['high']} L:{q['low']} | Change: {q.get('percent_change',0):+.2f}% | Session: {session_name}\n"

                chart = get_chart_analysis(pair, current_price)
                live_ctx += format_chart_analysis_for_prompt(chart)

            news = get_news(pair)
            if news:
                live_ctx += f"LATEST NEWS:\n" + '\n'.join([f"- {n['title']} ({n['source']})" for n in news[:3]]) + '\n\n'

        full_prompt = live_ctx + prompt if live_ctx else prompt
        text = call_claude(full_prompt, max_tokens)
        return jsonify({'content': text})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-scenarios', methods=['POST'])
@login_required
def forex_scenarios():
    try:
        date_str = datetime.now().strftime('%A, %B %d, %Y')
        session_name, _ = get_session()
        scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY']
        prices = get_prices_parallel(scan_pairs)
        news = get_news()

        chart_data = get_multi_pair_chart_data(scan_pairs, prices)

        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p,v in prices.items()])
        news_str = '\n'.join([f"- {n['title']}" for n in news[:5]]) or "- Markets await key economic data"

        chart_ctx = '\n'.join([format_chart_analysis_for_prompt(chart_data[p]) for p in scan_pairs if p in chart_data])

        prompt = f"""You are Wolf AI — professional forex trader. Today: {date_str}. Session: {session_name}.

LIVE PRICES:
{prices_str}

NEWS:
{news_str}

{chart_ctx}

Using the REAL chart data above (actual EMA, RSI, MACD, swing S/R levels), find the 7 BEST trades.
Only include pairs where 4+ timeframes align. Use the EXACT S/R levels from the chart data above.
Give BOTH buy AND sell scenario for each trade.

Respond ONLY in valid JSON (no markdown, no backticks):
{{"week":"{date_str}","session":"{session_name}","market_theme":"string","dxy_direction":"BULLISH or BEARISH","risk_sentiment":"RISK-ON or RISK-OFF","trades":[{{"rank":1,"pair":"EUR/USD","live_price":"1.0380","overall_bias":"BEARISH","timeframe_alignment":{{"monthly":"BEARISH","weekly":"BEARISH","daily":"BEARISH","h4":"BEARISH","h1":"NEUTRAL","m15":"BEARISH"}},"aligned_count":5,"confidence":82,"primary_direction":"SELL","thesis":"3-4 sentence thesis using real chart data","key_resistance":"1.0400","key_support":"1.0340","buy_scenario":{{"trigger":"string","entry":"1.0410","stop_loss":"1.0380","tp1":"1.0450","tp2":"1.0500","tp3":"1.0550","probability":30}},"sell_scenario":{{"trigger":"string","entry":"1.0360","stop_loss":"1.0390","tp1":"1.0320","tp2":"1.0280","tp3":"1.0240","probability":70}},"best_session":"LONDON","key_news_this_week":"string","invalidation":"string"}}]}}"""

        result = parse_json_response(call_claude(prompt, 5000))
        return jsonify(result)
    except json.JSONDecodeError as e: return jsonify({'error': f'AI returned invalid JSON: {str(e)}'}), 500
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-daily-picks', methods=['POST'])
@login_required
def forex_daily_picks():
    try:
        date_str = datetime.now().strftime('%A, %B %d, %Y')
        session_name, _ = get_session()
        scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','USD/CHF']
        prices = get_prices_parallel(scan_pairs)
        news = get_news()

        chart_data = get_multi_pair_chart_data(scan_pairs, prices)

        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p,v in prices.items()])
        news_str = '\n'.join([f"- {n['title']}" for n in news[:4]]) or "- Monitor key levels"
        chart_ctx = '\n'.join([format_chart_analysis_for_prompt(chart_data[p]) for p in scan_pairs if p in chart_data])

        prompt = f"""You are Wolf AI — professional intraday forex trader. Today: {date_str}. Session: {session_name}.

LIVE PRICES:
{prices_str}

NEWS:
{news_str}

{chart_ctx}

Using the REAL hourly chart data above (actual EMA, RSI, MACD, real S/R levels from swing highs/lows),
find the 3 BEST day trades for today's session. Use EXACT price levels from the chart data.

Respond ONLY in valid JSON (no markdown):
{{"date":"{date_str}","session":"{session_name}","dxy_bias":"BULLISH or BEARISH","risk_environment":"RISK-ON or RISK-OFF","picks":[{{"rank":1,"pair":"EUR/USD","direction":"SELL","entry":"1.0390","stop_loss":"1.0420","tp1":"1.0350","tp2":"1.0310","tp3":"1.0270","rr_ratio":"1:2.5","confidence":85,"sharingan_score":5,"thesis":"2-3 sentence thesis using real chart data","confluences":["Price below EMA200 daily","RSI 42 bearish","Hourly resistance at 1.0400"],"best_window":"London Open 3-5AM EST","key_news":"NFP Friday","invalidation":"Break above 1.0430","buy_scenario":"string","sell_scenario":"string"}}]}}"""

        result = parse_json_response(call_claude(prompt, 4000))
        return jsonify(result)
    except json.JSONDecodeError as e: return jsonify({'error': f'AI returned invalid JSON: {str(e)}'}), 500
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-weekly-picks', methods=['POST'])
@login_required
def forex_weekly_picks():
    try:
        date_str = datetime.now().strftime('%A, %B %d, %Y')
        session_name, _ = get_session()
        scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','NZD/USD']
        prices = get_prices_parallel(scan_pairs)
        news = get_news()

        chart_data = get_multi_pair_chart_data(scan_pairs, prices)

        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p,v in prices.items()])
        news_str = '\n'.join([f"- {n['title']}" for n in news[:4]]) or "- Monitor macro events"
        chart_ctx = '\n'.join([format_chart_analysis_for_prompt(chart_data[p]) for p in scan_pairs if p in chart_data])

        prompt = f"""You are Wolf AI — professional swing trader. Today: {date_str}.

LIVE PRICES:
{prices_str}

NEWS:
{news_str}

{chart_ctx}

Using the REAL weekly and daily chart data above (actual EMA, RSI, 52-week range, real S/R levels),
find the 3 BEST swing trades for this week (2-5 day holds). Use EXACT levels from real chart data.

Respond ONLY in valid JSON (no markdown):
{{"week":"{date_str}","weekly_theme":"Main macro theme","dxy_outlook":"BULLISH or BEARISH","central_bank_focus":"Key CB event this week","picks":[{{"rank":1,"pair":"GBP/USD","direction":"SELL","entry_zone":"1.2630-1.2650","stop_loss":"1.2700","tp1":"1.2570","tp2":"1.2500","tp3":"1.2420","rr_ratio":"1:2.8","confidence":80,"sharingan_score":4,"hold_days":"3-4","fundamental":"string","technical":"string using real EMA/RSI data","confluences":["Weekly bearish","Daily below EMA200","RSI 45 bearish"],"key_events":"BOE minutes","key_risk":"Surprise hawkish BOE","buy_scenario":"string","sell_scenario":"string"}}]}}"""

        result = parse_json_response(call_claude(prompt, 4000))
        return jsonify(result)
    except json.JSONDecodeError as e: return jsonify({'error': f'AI returned invalid JSON: {str(e)}'}), 500
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-scanner', methods=['POST'])
@login_required
def forex_scanner():
    try:
        data = request.get_json() or {}
        theme = data.get('theme', 'best setups today')
        date_str = datetime.now().strftime('%A, %B %d, %Y')
        session_name, _ = get_session()
        scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','USD/CHF','NZD/USD']
        prices = get_prices_parallel(scan_pairs)
        news = get_news()
        chart_data = get_multi_pair_chart_data(scan_pairs, prices)
        prices_str = "\n".join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v.get('percent_change',0):+.2f}%)" for p,v in prices.items()])
        news_str = "\n".join([f"- {n['title']}" for n in news[:4]]) or "- Monitor key levels"
        chart_ctx = "\n".join([format_chart_analysis_for_prompt(chart_data[p]) for p in scan_pairs if p in chart_data])
        prompt = (
            f"You are Wolf AI - elite forex scanner. Today: {date_str}. Session: {session_name}.\n"
            f"Theme requested: {theme}\n\n"
            f"LIVE PRICES:\n{prices_str}\n\n"
            f"NEWS:\n{news_str}\n\n"
            f"{chart_ctx}\n\n"
            f"Scan all pairs and find the 3 BEST setups matching the theme \"{theme}\".\n"
            f"Use REAL chart data above for S/R levels, EMAs, RSI. Be specific.\n\n"
            f'Respond ONLY in valid JSON (no markdown):\n'
            '{{"scan_theme":"{theme}","date":"{date_str}","session":"{session_name}","dxy_bias":"BULLISH or BEARISH","risk_environment":"RISK-ON or RISK-OFF","picks":[{{"rank":1,"pair":"EUR/USD","direction":"SELL","entry":"1.0390","stop_loss":"1.0420","tp1":"1.0350","tp2":"1.0310","tp3":"1.0270","rr_ratio":"1:2.5","confidence":85,"thesis":"2-3 sentence thesis using real chart data","confluences":["real level 1","real level 2"],"best_window":"London Open 3-5AM EST","invalidation":"Break above 1.0430","buy_scenario":"string","sell_scenario":"string"}}]}}'.format(theme=theme,date_str=date_str,session_name=session_name)
        )
        result = parse_json_response(call_claude(prompt, 3000))
        return jsonify(result)
    except json.JSONDecodeError as e: return jsonify({'error': f'AI returned invalid JSON: {str(e)}'}), 500
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-picks', methods=['POST'])
@login_required
def forex_picks():
    try:
        prompt = request.get_json().get('prompt', '')
        return jsonify({'content': call_claude(prompt, 2200)})
    except Exception as e: return jsonify({'error': str(e)}), 500

# ── Wolf Scan Job Store (in-memory) ──────────────────────────
import threading, uuid
def calculate_real_confidence(pair, direction, chart_data):
    """
    Calculate confidence score 0-100 from REAL chart data.
    Replaces AI-guessed confidence with data-driven score.
    direction: 'BUY' or 'SELL'
    """
    score = 0
    ca = chart_data.get(pair, {})
    if not ca:
        return 50  # no data — neutral

    d  = ca.get('daily', {})
    w  = ca.get('weekly', {})
    h  = ca.get('hourly', {})
    ts = ca.get('trend_summary', {})
    sr = ca.get('sr_levels', [])

    is_bull = direction.upper() in ('BUY', 'BULLISH')

    # ── 1. TIMEFRAME ALIGNMENT (30 pts) ──────────────────────
    w_trend = w.get('trend', 'NEUTRAL')
    d_trend = d.get('trend', 'NEUTRAL')
    h_trend = h.get('trend', 'NEUTRAL')

    tf_match = 0
    for t in [w_trend, d_trend, h_trend]:
        if is_bull and t == 'BULLISH':   tf_match += 1
        elif not is_bull and t == 'BEARISH': tf_match += 1

    score += tf_match * 10  # 0, 10, 20, or 30

    # ── 2. RSI ZONE (20 pts) ──────────────────────────────────
    d_rsi = d.get('rsi')
    h_rsi = h.get('rsi')

    for rsi in [d_rsi, h_rsi]:
        if rsi is None:
            continue
        if is_bull:
            if 40 <= rsi <= 60:   score += 8   # ideal momentum building
            elif 30 <= rsi < 40:  score += 10  # oversold bounce potential
            elif rsi < 30:        score += 6   # deep oversold — risky
            elif rsi > 70:        score -= 5   # overbought — bad for buy
        else:
            if 40 <= rsi <= 60:   score += 8
            elif 60 < rsi <= 70:  score += 10  # overbought rejection potential
            elif rsi > 70:        score += 6   # deep overbought — risky
            elif rsi < 30:        score -= 5   # oversold — bad for sell

    # ── 3. EMA STACK (20 pts) ─────────────────────────────────
    cp = ca.get('current_price', 0)
    ema20  = d.get('ema20')
    ema50  = d.get('ema50')
    ema200 = d.get('ema200')

    if cp and ema200:
        if is_bull and cp > ema200:   score += 8
        elif not is_bull and cp < ema200: score += 8

    if cp and ema50:
        if is_bull and cp > ema50:    score += 7
        elif not is_bull and cp < ema50:  score += 7

    if cp and ema20:
        if is_bull and cp > ema20:    score += 5
        elif not is_bull and cp < ema20:  score += 5

    # ── 4. MACD CONFIRMS (15 pts) ─────────────────────────────
    d_macd = d.get('macd_bias')
    h_macd = h.get('macd_bias')

    if d_macd:
        if is_bull and d_macd == 'BULLISH':   score += 8
        elif not is_bull and d_macd == 'BEARISH': score += 8

    if h_macd:
        if is_bull and h_macd == 'BULLISH':   score += 7
        elif not is_bull and h_macd == 'BEARISH': score += 7

    # ── 5. CLEAN S/R NEARBY (15 pts) ─────────────────────────
    if sr:
        close_levels = [lv for lv in sr if lv.get('distance_pips', 999) < 50]
        if close_levels:
            score += 8
        strong_levels = [lv for lv in sr if lv.get('strength', 0) >= 3 and lv.get('distance_pips', 999) < 80]
        if strong_levels:
            score += 7

    # ── CLAMP to 0-100 ────────────────────────────────────────
    return max(0, min(100, score))



# ── ASYNC JOB STORE ─────────────────────────────────────────────────────────
_async_jobs = {}

def _run_async_ai_job(job_id, prompt, max_tokens, pair=''):
    """Generic async AI job - prevents 502 Bad Gateway from Render 30s timeout"""
    try:
        _async_jobs[job_id] = {'status': 'running'}
        # Add live context if pair provided
        live_ctx = ''
        if pair:
            try:
                q = get_price(pair)
                session_name, _ = get_session()
                if q:
                    current_price = float(q['price'])
                    live_ctx = f"\nLIVE PRICE: {pair} = {current_price} | H:{q['high']} L:{q['low']} | Change: {q.get('percent_change',0):+.2f}% | Session: {session_name}\n"
                    chart = get_chart_analysis(pair, current_price)
                    live_ctx += format_chart_analysis_for_prompt(chart)
                news = get_news(pair)
                if news:
                    live_ctx += "LATEST NEWS:\n" + '\n'.join([f"- {n['title']} ({n['source']})" for n in news[:3]]) + '\n\n'
            except Exception as ctx_err:
                print(f'[AsyncAI] context error: {ctx_err}')
        full_prompt = live_ctx + prompt if live_ctx else prompt
        text = call_claude(full_prompt, max_tokens)
        _async_jobs[job_id] = {'status': 'done', 'content': text}
    except Exception as e:
        import traceback; print(traceback.format_exc())
        _async_jobs[job_id] = {'status': 'error', 'error': str(e)}

@app.route('/api/async-ai-start', methods=['POST'])
@login_required
def async_ai_start():
    """Start any AI prompt as background job - returns job_id immediately"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 3000)
        pair = data.get('pair', '')
        job_id = str(uuid.uuid4())[:8]
        _async_jobs[job_id] = {'status': 'starting'}
        t = threading.Thread(target=_run_async_ai_job, args=(job_id, prompt, max_tokens, pair), daemon=True)
        t.start()
        return jsonify({'job_id': job_id, 'status': 'starting'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/async-ai-poll/<job_id>', methods=['GET'])
@login_required
def async_ai_poll(job_id):
    """Poll async AI job"""
    job = _async_jobs.get(job_id)
    if not job: return jsonify({'status': 'error', 'error': 'Job not found'}), 404
    if job['status'] == 'done':
        del _async_jobs[job_id]
        return jsonify({'status': 'done', 'content': job['content']})
    if job['status'] == 'error':
        err = job.get('error', 'Unknown error')
        del _async_jobs[job_id]
        return jsonify({'status': 'error', 'error': err}), 500
    return jsonify({'status': job['status']})

# ── Wolf Stock Scanner page ───────────────────────────────────
@app.route('/byakugan')
@login_required
@elite_required
def byakugan_page():
    return render_template('wolf_stocks.html')

# ═══════════════════════════════════════════════════════════════
# NEW FEATURES — Position Sizing, Earnings, Sector Heatmap
# ═══════════════════════════════════════════════════════════════

@app.route('/api/position-size', methods=['POST'])
@login_required
def position_size():
    try:
        data = request.get_json() or {}
        account_size  = float(data.get('account_size', 10000))
        risk_pct      = float(data.get('risk_pct', 1.0))
        option_price  = float(data.get('option_price', 3.00))
        stop_price    = float(data.get('stop_price', 1.50))
        target_price  = float(data.get('target_price', 6.00))
        risk_per_trade    = account_size * (risk_pct / 100)
        risk_per_contract = (option_price - stop_price) * 100
        contracts = int(risk_per_trade / risk_per_contract) if risk_per_contract > 0 else 0
        contracts = max(1, min(contracts, 50))
        total_cost  = contracts * option_price * 100
        max_loss    = contracts * risk_per_contract
        max_gain    = contracts * (target_price - option_price) * 100
        reward_risk = round((target_price - option_price) / (option_price - stop_price), 2) if (option_price - stop_price) > 0 else 0
        win_rate_needed = round(1 / (1 + reward_risk) * 100, 1) if reward_risk > 0 else 50
        return jsonify({'contracts': contracts, 'total_cost': round(total_cost,2), 'max_loss': round(max_loss,2),
                        'max_gain': round(max_gain,2), 'reward_risk': reward_risk,
                        'risk_per_trade': round(risk_per_trade,2), 'win_rate_needed': win_rate_needed,
                        'account_size': account_size, 'risk_pct': risk_pct})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/earnings-check', methods=['POST'])
@login_required
def earnings_check():
    try:
        import yfinance as yf
        data     = request.get_json() or {}
        tickers  = data.get('tickers', ['AAPL','MSFT','NVDA','TSLA','AMD'])
        days_out = int(data.get('days_out', 14))
        results  = []
        for sym in tickers[:20]:
            try:
                cal = yf.Ticker(sym).calendar
                earn_date = None; days_away = None; safe = True
                if cal is not None and not cal.empty:
                    cols = cal.columns.tolist()
                    if cols:
                        earn_date = str(cols[0])[:10]
                        dt = datetime.strptime(earn_date, '%Y-%m-%d')
                        days_away = (dt - datetime.now()).days
                        safe = days_away > days_out or days_away < 0
                results.append({'ticker': sym, 'earn_date': earn_date, 'days_away': days_away, 'safe': safe,
                                'warning': f'Earnings in {days_away} days — IV crush risk!' if (days_away and 0 < days_away <= days_out) else None})
            except:
                results.append({'ticker': sym, 'earn_date': None, 'days_away': None, 'safe': True, 'warning': None})
        return jsonify({'results': results, 'checked_at': datetime.now().strftime('%Y-%m-%d %H:%M')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sector-heatmap', methods=['GET'])
@login_required
def sector_heatmap():
    try:
        import yfinance as yf
        sectors = {'Technology':'XLK','Financials':'XLF','Energy':'XLE','Healthcare':'XLV',
                   'Consumer Disc':'XLY','Industrials':'XLI','Materials':'XLB','Utilities':'XLU',
                   'Real Estate':'XLRE','Consumer Staples':'XLP','Communication':'XLC','Defense':'ITA'}
        results = []
        for name, etf in sectors.items():
            try:
                hist = yf.Ticker(etf).history(period='5d',interval='1d')
                if len(hist) >= 2:
                    today = float(hist['Close'].iloc[-1]); prev = float(hist['Close'].iloc[-2])
                    chg   = round(((today-prev)/prev)*100,2)
                    wk_chg= round(((today-float(hist['Close'].iloc[0]))/float(hist['Close'].iloc[0]))*100,2) if len(hist)>=5 else chg
                    results.append({'sector':name,'etf':etf,'price':round(today,2),'day_chg':chg,'week_chg':wk_chg,
                                    'signal':'HOT' if chg>1 else 'COLD' if chg<-1 else 'NEUTRAL'})
            except: pass
        results.sort(key=lambda x: x['day_chg'], reverse=True)
        return jsonify({'sectors':results,'updated':datetime.now().strftime('%H:%M EST')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        _byakugan_jobs[job_id]['status'] = 'scanning'
        if scan_filter=='TECH': universe=['AAPL','MSFT','NVDA','AMD','META','GOOGL','AMZN','NFLX','COIN','PLTR','SMCI','AVGO','MU','CRM']
        elif scan_filter=='MEME': universe=['TSLA','MARA','HOOD','SOFI','COIN','PLTR','SQ','PYPL','RIOT','NIO']
        elif scan_filter=='BLUE': universe=['AAPL','MSFT','GOOGL','AMZN','META','JPM','GS','BAC','XOM','CVX','DIS']
        elif scan_filter=='ETF': universe=['SPY','QQQ','IWM','GLD','TLT','XLK','XLF','XLE','XLV','ARKK']
        elif scan_filter=='DEFENSE': universe=['LMT','RTX','NOC','GD','HII','LHX','AXON','KTOS','PLTR']
        else: universe=['AAPL','MSFT','NVDA','TSLA','AMD','META','GOOGL','AMZN','COIN','PLTR','JPM','GS','XOM','SMCI','AVGO']  # 15 stocks — balanced speed vs coverage
        regime = get_market_regime()
        vix = float(regime.get('vix', 20))
        _byakugan_jobs[job_id]['status'] = 'scoring'
        scored = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(score_stock, sym): sym for sym in universe}
            for f in as_completed(futures, timeout=50):
                sym = futures[f]
                try:
                    s = f.result()
                    if s and s['score'] > 15: scored.append(s)
                except Exception as e: print(f'[Byakugan] {sym}: {e}')
        scored.sort(key=lambda x: x['score'], reverse=True)
        top5 = scored[:5]
        # Lower threshold if nothing scores > 15
        if not scored:
            print('[Byakugan] No stocks scored at all — retrying with lower threshold')
            scored = [s for s in [score_stock(sym) for sym in universe[:5]] if s]
        if not scored:
            _byakugan_jobs[job_id]={'status':'error','error':'Market data unavailable — try again in 30 seconds'}; return
        top5 = scored[:5]
        _byakugan_jobs[job_id]['status'] = 'news'
        for stock in top5:
            try: stock['news'] = get_news(stock['ticker'])[:3]
            except: stock['news'] = []
        _byakugan_jobs[job_id]['status'] = 'greeks'
        for stock in top5:
            try:
                S=stock['price']; K=round(S/5)*5; T=3/365; r=0.045
                iv=0.35 if not stock.get('iv_rank') else max(0.15,min(0.80,stock['iv_rank']/100))
                opt_type='call' if stock.get('direction')=='BULLISH' else 'put'
                stock['greeks']=calculate_greeks(S,K,T,r,iv,opt_type); stock['atm_strike']=K; stock['iv_estimate']=round(iv*100,1)
            except: stock['greeks']=None
        _byakugan_jobs[job_id]['status'] = 'analyzing'
        if vix>30: vix_guide="VIX EXTREME >30: Sell premium only."
        elif vix>20: vix_guide="VIX ELEVATED 20-30: Defined-risk spreads."
        elif vix<15: vix_guide="VIX LOW <15: Buy directional. 0DTE/1DTE viable SPY/QQQ."
        else: vix_guide="VIX NORMAL 15-20: Debit spreads 3-7 DTE ideal."
        stocks_ctx = ''
        for s in top5:
            g=s.get('greeks') or {}
            stocks_ctx += f"\n===\n{s['ticker']} ${s['price']} Score:{s['score']} {s['direction']}\nEMA20:{s['ema20']} EMA50:{s['ema50']} EMA200:{s['ema200']} RSI:{s['rsi']} MACD:{s['macd_bias']}\nIV:{s.get('iv_estimate','?')}% Vol:{s['vol_ratio']}x UOA:{s['unusual_activity']}x\nD:{g.get('delta','?')} G:{g.get('gamma','?')} T:{g.get('theta','?')} V:{g.get('vega','?')}\nSR:{[(l['type'],l['price']) for l in s['sr_levels'][:2]]}\nNEWS:{' | '.join([n['title'][:60] for n in s['news']]) if s['news'] else 'None'}\n==="
        prompt = f"""You are Byakugan — elite Wall Street options trader. Paul Tudor Jones + Tom Sosnoff + Jesse Livermore. Analyze so trader knows EXACTLY what to do tomorrow open.
TODAY:{date_str} MARKET:SPY:{regime['spy_price']} ({regime['spy_change']:+.2f}%) VIX:{vix} {regime['fear_greed']} {regime['regime']} VIX STRATEGY:{vix_guide}
{stocks_ctx}
RULES: Friday weeklies 1-5DTE stocks. 0DTE/1DTE SPY/QQQ. Delta 0.35-0.50 directional. Confidence clear=80-92% mixed=65-75%. Greeks required. Real S/R entries.
JSON only no markdown: {{"scan_date":"{date_str}","market_regime":{{"spy":"{regime['spy_price']}","spy_change":"{regime['spy_change']:+.2f}%","vix":"{vix}","sentiment":"{regime['fear_greed']}","regime":"{regime['regime']}","wolf_market_read":"2 sentences"}},"tomorrow_game_plan":"3 sentences","picks":[{{"rank":1,"ticker":"X","price":"0","wolf_score":80,"confidence":82,"direction":"BULLISH","sector":"Tech","thesis":"thesis","news_catalyst":"news","technical_setup":"setup","entry_zone":"X","key_support":"X","key_resistance":"Y","stop_loss":"X","target_1":"X","target_2":"Y","target_3":"Z","tomorrow_entry":"entry plan","options_play":{{"strategy":"LONG CALL","recommended_strike":"Xc","expiration":"This Friday (3 DTE)","entry_price":"X","max_risk":"$X","target_exit":"$X","stop_exit":"$X","greeks":{{"delta":0.42,"gamma":0.008,"theta":-0.85,"vega":0.45}},"iv_environment":"context","note":"note"}},"confluences":["c1"],"warnings":["w1"],"invalidation":"stop"}}]}}"""
        result = None; last_error = None
        for attempt in range(3):
            try:
                raw = call_claude(prompt, 5000)
                if not raw or not raw.strip(): raise ValueError('Empty response')
                result = parse_json_response(raw); break
            except Exception as retry_err:
                last_error = retry_err
                print(f'[Byakugan] Attempt {attempt+1} failed: {retry_err}')
                if attempt < 2: time.sleep(2)
        if result is None: raise Exception(f'AI failed after 3 attempts: {last_error}')
        for pick in result.get('picks',[]):
            match = next((s for s in top5 if s['ticker']==pick.get('ticker','')),None)
            if match:
                pick['real_score']=match['score']; pick['real_rsi']=match['rsi']
                pick['real_vol_ratio']=match['vol_ratio']; pick['real_iv_rank']=match['iv_rank']
                pick['real_signals']=match['signals']; pick['sr_levels']=match['sr_levels']
                pick['news']=match['news']; pick['real_greeks']=match.get('greeks')
        result['market_regime']=regime
        _byakugan_jobs[job_id]={'status':'done','result':result}
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
# ═══════════════════════════════════════════════════════════════
# AI INFRASTRUCTURE SCANNER — paste at bottom of app.py
# (before the "

# ═══════════════════════════════════════════════════════════════════
# SAGE MODE — ULTIMATE SCANNER ENGINE
# ═══════════════════════════════════════════════════════════════════

def calc_atr(candles, period=14):
    if len(candles) < period + 1: return None
    trs = []
    for i in range(1, len(candles)):
        h=candles[i]["high"]; l=candles[i]["low"]; pc=candles[i-1]["close"]
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    return round(sum(trs[-period:])/period, 5)

def calc_bollinger(closes, period=20, sd=2):
    if len(closes) < period: return None, None, None
    recent = closes[-period:]; mid = sum(recent)/period
    std = (sum((x-mid)**2 for x in recent)/period)**0.5
    return round(mid+sd*std,5), round(mid,5), round(mid-sd*std,5)

def detect_candle_patterns(candles):
    """Real candlestick pattern recognition — Steve Nison method"""
    if len(candles) < 3: return []
    patterns = []
    c0=candles[-1]; c1=candles[-2]; c2=candles[-3]
    # Core measurements
    body0=abs(c0["close"]-c0["open"]); range0=max(c0["high"]-c0["low"],0.0001)
    body1=abs(c1["close"]-c1["open"]); range1=max(c1["high"]-c1["low"],0.0001)
    lw0=min(c0["open"],c0["close"])-c0["low"]   # lower wick
    uw0=c0["high"]-max(c0["open"],c0["close"])   # upper wick
    lw1=min(c1["open"],c1["close"])-c1["low"]
    uw1=c1["high"]-max(c1["open"],c1["close"])
    bull0=c0["close"]>c0["open"]; bear0=c0["close"]<c0["open"]
    bull1=c1["close"]>c1["open"]; bear1=c1["close"]<c1["open"]
    avg_body=max((body0+body1)/2, 0.0001)

    # 1. DOJI — indecision, body < 10% of range
    if body0/range0 < 0.1:
        patterns.append({"pattern":"DOJI","bias":"NEUTRAL","note":"Market indecision at this level — breakout watch"})

    # 2. HAMMER — bullish reversal after downtrend (long lower wick)
    if lw0 > body0*2.5 and uw0 < body0*0.5 and bull0:
        patterns.append({"pattern":"HAMMER","bias":"BULLISH","note":"Strong buyer rejection of lows — bullish reversal"})

    # 3. INVERTED HAMMER — bullish after downtrend
    if uw0 > body0*2.5 and lw0 < body0*0.5 and bull0:
        patterns.append({"pattern":"INVERTED HAMMER","bias":"BULLISH","note":"Buyers testing highs — watch for follow-through"})

    # 4. SHOOTING STAR — bearish after uptrend (long upper wick)
    if uw0 > body0*2.5 and lw0 < body0*0.5 and bear0:
        patterns.append({"pattern":"SHOOTING STAR","bias":"BEARISH","note":"Seller rejection of highs — bearish reversal"})

    # 5. HANGING MAN — bearish after uptrend (same shape as hammer but at top)
    if lw0 > body0*2.5 and uw0 < body0*0.5 and bear0:
        patterns.append({"pattern":"HANGING MAN","bias":"BEARISH","note":"Failed buyers at high — distribution candle"})

    # 6. PIN BAR — long wick vs body (either direction, price action key pattern)
    if (lw0 > range0*0.6) or (uw0 > range0*0.6):
        direction = "BULLISH" if lw0 > uw0 else "BEARISH"
        patterns.append({"pattern":"PIN BAR","bias":direction,"note":"Strong price rejection — high probability reversal zone"})

    # 7. BULLISH ENGULFING — bears then bulls absorb
    if bear1 and bull0 and c0["open"] <= c1["close"] and c0["close"] >= c1["open"] and body0 > body1:
        patterns.append({"pattern":"BULLISH ENGULFING","bias":"BULLISH","note":"Bulls fully engulf prior bearish candle"})

    # 8. BEARISH ENGULFING
    if bull1 and bear0 and c0["open"] >= c1["close"] and c0["close"] <= c1["open"] and body0 > body1:
        patterns.append({"pattern":"BEARISH ENGULFING","bias":"BEARISH","note":"Bears fully engulf prior bullish candle"})

    # 9. INSIDE BAR — consolidation / compression before big move
    if c0["high"] <= c1["high"] and c0["low"] >= c1["low"]:
        patterns.append({"pattern":"INSIDE BAR","bias":"NEUTRAL","note":"Price compressed inside prior candle — breakout building"})

    # 10. MORNING STAR — 3-candle bullish reversal
    if bear1 and body0 > avg_body*0.5 and bull0 and c0["close"] > (c2["open"]+c2["close"])/2:
        patterns.append({"pattern":"MORNING STAR","bias":"BULLISH","note":"3-candle bottom reversal — bulls taking control"})

    # 11. EVENING STAR — 3-candle bearish reversal
    if bull1 and body0 > avg_body*0.5 and bear0 and c0["close"] < (c2["open"]+c2["close"])/2:
        patterns.append({"pattern":"EVENING STAR","bias":"BEARISH","note":"3-candle top reversal — bears taking control"})

    # 12. THREE WHITE SOLDIERS (check last 3 candles all bullish, higher closes)
    if len(candles) >= 3:
        last3 = candles[-3:]
        if all(c["close"] > c["open"] for c in last3) and            last3[2]["close"] > last3[1]["close"] > last3[0]["close"]:
            patterns.append({"pattern":"THREE WHITE SOLDIERS","bias":"BULLISH","note":"3 consecutive bullish candles — strong uptrend momentum"})

    # 13. THREE BLACK CROWS
    if len(candles) >= 3:
        last3 = candles[-3:]
        if all(c["close"] < c["open"] for c in last3) and            last3[2]["close"] < last3[1]["close"] < last3[0]["close"]:
            patterns.append({"pattern":"THREE BLACK CROWS","bias":"BEARISH","note":"3 consecutive bearish candles — strong downtrend momentum"})

    # 14. TWEEZER TOP — two candles with same high (resistance rejection)
    if abs(c0["high"] - c1["high"]) < range0*0.05 and bear0:
        patterns.append({"pattern":"TWEEZER TOP","bias":"BEARISH","note":"Double rejection at same resistance — sellers in control"})

    # 15. TWEEZER BOTTOM — two candles with same low (support acceptance)
    if abs(c0["low"] - c1["low"]) < range0*0.05 and bull0:
        patterns.append({"pattern":"TWEEZER BOTTOM","bias":"BULLISH","note":"Double bounce off same support — buyers in control"})

    # 16. MARUBOZU — full body candle (no wicks) = strong momentum
    if body0/range0 > 0.92:
        bias = "BULLISH" if bull0 else "BEARISH"
        patterns.append({"pattern":"MARUBOZU","bias":bias,"note":"Strong momentum candle — trend continuation expected"})

    return patterns[:4]  # return top 4 most recent patterns

def detect_market_structure(candles):
    """
    Real market structure detection — higher highs/lows = uptrend
    Based on Dow Theory + price action (Sam Seiden / ICT style)
    """
    if len(candles) < 10:
        return {"structure":"UNKNOWN","phase":"UNKNOWN","description":"Not enough data"}

    highs  = [c["high"]  for c in candles[-20:]]
    lows   = [c["low"]   for c in candles[-20:]]
    closes = [c["close"] for c in candles[-20:]]

    # Find swing highs and lows (local maxima/minima over 3-bar window)
    swing_highs, swing_lows = [], []
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i]  < lows[i-1]  and lows[i]  < lows[i-2]  and lows[i]  < lows[i+1]  and lows[i]  < lows[i+2]:
            swing_lows.append(lows[i])

    structure = "RANGING"
    description = "No clear directional structure"

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1] > swing_highs[-2]   # higher high
        hl = swing_lows[-1]  > swing_lows[-2]    # higher low
        lh = swing_highs[-1] < swing_highs[-2]   # lower high
        ll = swing_lows[-1]  < swing_lows[-2]    # lower low

        if hh and hl:
            structure = "UPTREND"
            description = "Higher Highs + Higher Lows = clean uptrend. Bulls in full control."
        elif lh and ll:
            structure = "DOWNTREND"
            description = "Lower Highs + Lower Lows = clean downtrend. Bears in full control."
        elif hh and ll:
            structure = "RANGING"
            description = "Higher Highs but Lower Lows = expanded range. Breakout pending."
        elif lh and hl:
            structure = "RANGING"
            description = "Lower Highs + Higher Lows = compression triangle. Coiling for breakout."
        else:
            structure = "RANGING"
            description = "Mixed swing structure — no clear directional bias."

    # Detect impulse vs ABC correction
    # Simple: if last 5 candles all same direction = impulse, mixed = correction
    last5 = candles[-5:] if len(candles) >= 5 else candles
    bull_count = sum(1 for c in last5 if c["close"] > c["open"])
    bear_count = sum(1 for c in last5 if c["close"] < c["open"])

    if bull_count >= 4:
        phase = "IMPULSE BULLISH"
    elif bear_count >= 4:
        phase = "IMPULSE BEARISH"
    elif bull_count == 3 and bear_count == 2:
        phase = "CORRECTION — possible ABC bull trap in downtrend"
    elif bear_count == 3 and bull_count == 2:
        phase = "CORRECTION — possible ABC bear trap in uptrend"
    else:
        phase = "CONSOLIDATION — no momentum"

    return {"structure": structure, "phase": phase, "description": description,
            "swing_highs": [round(h,5) for h in swing_highs[-3:]] if swing_highs else [],
            "swing_lows":  [round(l,5) for l in swing_lows[-3:]]  if swing_lows  else []}


def calc_adr(candles, days=10):
    """Average Daily Range — how far price typically moves in a day"""
    if len(candles) < 3:
        return None
    recent = candles[-days:] if len(candles) >= days else candles
    ranges = [c["high"] - c["low"] for c in recent]
    return round(sum(ranges)/len(ranges), 5)


def calc_weekly_range_pct(candles):
    """How much of the weekly average range has been used — Gemini strategy key metric"""
    if len(candles) < 6:
        return None, None, None
    # Weekly = 5 trading days
    week_candles = candles[-5:] if len(candles) >= 5 else candles
    week_high = max(c["high"] for c in week_candles)
    week_low  = min(c["low"]  for c in week_candles)
    week_range = week_high - week_low

    # Average weekly range over past 4 weeks
    if len(candles) >= 20:
        avg_weekly_ranges = []
        for i in range(4):
            start = -(5*(i+1))
            end   = -(5*i) if i > 0 else None
            wc = candles[start:end]
            if wc:
                avg_weekly_ranges.append(max(c["high"] for c in wc) - min(c["low"] for c in wc))
        avg_range = sum(avg_weekly_ranges)/len(avg_weekly_ranges) if avg_weekly_ranges else week_range
    else:
        avg_range = week_range

    pct_used = round((week_range / avg_range * 100) if avg_range > 0 else 0, 1)
    return round(week_high, 5), round(week_low, 5), pct_used


def detect_trend_strength(candles_d1, candles_h4):
    """Trend strength: how aligned are EMAs across timeframes"""
    if not candles_d1 or not candles_h4:
        return "UNKNOWN", 0
    dc  = [c["close"] for c in candles_d1]
    h4c = [c["close"] for c in candles_h4]
    cp  = dc[-1] if dc else 0

    e9_d  = calc_ema(dc,  9);  e21_d = calc_ema(dc, 21);  e50_d = calc_ema(dc, 50)
    e9_h4 = calc_ema(h4c, 9);  e21_h4= calc_ema(h4c,21);  e50_h4= calc_ema(h4c,50)

    score = 0
    # Daily alignment
    if e9_d and e21_d and e50_d:
        if cp > e9_d > e21_d > e50_d:   score += 3  # perfect bull stack
        elif cp < e9_d < e21_d < e50_d: score -= 3  # perfect bear stack
        elif cp > e21_d: score += 1
        elif cp < e21_d: score -= 1
    # H4 alignment
    if e9_h4 and e21_h4 and e50_h4:
        if cp > e9_h4 > e21_h4 > e50_h4:   score += 2
        elif cp < e9_h4 < e21_h4 < e50_h4: score -= 2
        elif cp > e21_h4: score += 1
        elif cp < e21_h4: score -= 1

    abs_score = abs(score)
    if abs_score >= 4:
        strength = "STRONG " + ("BULLISH" if score > 0 else "BEARISH")
    elif abs_score >= 2:
        strength = "MODERATE " + ("BULLISH" if score > 0 else "BEARISH")
    else:
        strength = "WEAK / RANGING"

    return strength, score


def get_sage_chart_data(pair, current_price):
    """
    Full real chart analysis engine — pulls live OHLC from yfinance.
    Calculates real RSI (Wilder), real EMAs, real S/R from swing highs/lows,
    real candlestick patterns, market structure, ADR, weekly range usage.
    """
    data = {
        "pair": pair, "price": current_price,
        "weekly": {}, "daily": {}, "h4": {}, "hourly": {}, "m15": {},
        "sr_levels": [], "d1_patterns": [], "h4_patterns": [], "m15_patterns": [],
        "market_structure": {}, "trend_strength": "UNKNOWN",
        "adr": None, "weekly_range_pct": None,
        "weekly_high": None, "weekly_low": None,
        "london_levels": {}, "ny_levels": {}
    }
    try:
        # ── WEEKLY — big picture context ──────────────────────────────
        wk = get_candles(pair, "1wk", "1y")
        if wk and len(wk) >= 10:
            wc  = [c["close"] for c in wk]
            wk_high52 = max(c["high"] for c in wk)
            wk_low52  = min(c["low"]  for c in wk)
            wk_rsi    = calc_rsi(wc)
            wk_ema20  = calc_ema(wc, 20)
            wk_struct = detect_market_structure(wk[-16:])
            wk_adr    = calc_adr(wk, 8)
            wk_h, wk_l, wk_pct = calc_weekly_range_pct(wk)
            # Trend vs EMA alignment
            wk_trend = "BULLISH" if (wk_ema20 and current_price > wk_ema20) else "BEARISH"
            data["weekly"] = {
                "trend":    wk_trend,
                "ema20":    wk_ema20,
                "rsi":      wk_rsi,
                "high52":   round(wk_high52, 5),
                "low52":    round(wk_low52, 5),
                "structure": wk_struct["structure"],
                "phase":    wk_struct["phase"],
                "adr":      wk_adr,
                "last3":    "{} bull, {} bear".format(
                    sum(1 for c in wk[-3:] if c["close"]>c["open"]),
                    sum(1 for c in wk[-3:] if c["close"]<=c["open"]))
            }
            data["weekly_high"]    = wk_h
            data["weekly_low"]     = wk_l
            data["weekly_range_pct"] = wk_pct

        # ── DAILY — trade direction bias ──────────────────────────────
        d1 = get_candles(pair, "1d", "6mo")
        if d1 and len(d1) >= 20:
            dc  = [c["close"] for c in d1]
            bbu, bbm, bbl = calc_bollinger(dc)
            e9  = calc_ema(dc, 9)
            e20 = calc_ema(dc, 20)
            e50 = calc_ema(dc, 50)
            e200= calc_ema(dc, min(200, len(dc)))
            d1_rsi  = calc_rsi(dc)
            d1_macd = calc_macd(dc)[1]
            d1_atr  = calc_atr(d1)
            d1_struct = detect_market_structure(d1[-30:])
            d1_adr  = calc_adr(d1, 10)
            # Price location relative to Bollinger
            if bbu and bbl:
                bb_range = bbu - bbl
                bb_pos = round(((current_price - bbl) / bb_range * 100), 1) if bb_range > 0 else 50
            else:
                bb_pos = 50
            # Daily trend — require TWO confirmations
            bull_signals = sum([
                1 if (e9  and current_price > e9)   else 0,
                1 if (e20 and current_price > e20)  else 0,
                1 if (e50 and current_price > e50)  else 0,
                1 if (d1_macd == "BULLISH")         else 0,
            ])
            d1_trend = "BULLISH" if bull_signals >= 3 else "BEARISH" if bull_signals <= 1 else "MIXED"
            data["daily"] = {
                "trend":    d1_trend,
                "ema9":     e9,  "ema20": e20, "ema50": e50, "ema200": e200,
                "rsi":      d1_rsi,
                "macd_bias": d1_macd,
                "atr":      d1_atr,
                "adr":      d1_adr,
                "bb_upper": bbu, "bb_mid": bbm, "bb_lower": bbl,
                "bb_position": bb_pos,
                "vs_ema200": "ABOVE" if (e200 and current_price > e200) else "BELOW",
                "structure": d1_struct["structure"],
                "phase":    d1_struct["phase"],
                "phase_desc": d1_struct["description"],
                "swing_highs": d1_struct.get("swing_highs", []),
                "swing_lows":  d1_struct.get("swing_lows",  []),
                "high20d":  round(max(c["high"] for c in d1[-20:]), 5),
                "low20d":   round(min(c["low"]  for c in d1[-20:]), 5),
                "last5":    "{} bull, {} bear".format(
                    sum(1 for c in d1[-5:] if c["close"]>c["open"]),
                    sum(1 for c in d1[-5:] if c["close"]<=c["open"]))
            }
            data["sr_levels"]   = find_sr_levels(d1, current_price, lookback=60)
            data["d1_patterns"] = detect_candle_patterns(d1[-10:])
            data["adr"]         = d1_adr

        # ── H4 — trade direction confirmation ────────────────────────
        h4 = get_candles(pair, "4h", "30d")
        if h4 and len(h4) >= 10:
            h4c = [c["close"] for c in h4]
            h4_e9   = calc_ema(h4c, 9)
            h4_e20  = calc_ema(h4c, 20)
            h4_e50  = calc_ema(h4c, 50)
            h4_rsi  = calc_rsi(h4c)
            h4_macd = calc_macd(h4c)[1]
            h4_atr  = calc_atr(h4)
            h4_struct = detect_market_structure(h4[-20:])
            bull_h4 = sum([
                1 if (h4_e9  and current_price > h4_e9)  else 0,
                1 if (h4_e20 and current_price > h4_e20) else 0,
                1 if (h4_macd == "BULLISH") else 0,
            ])
            h4_trend = "BULLISH" if bull_h4 >= 2 else "BEARISH" if bull_h4 == 0 else "MIXED"
            data["h4"] = {
                "trend":    h4_trend,
                "ema9":     h4_e9, "ema20": h4_e20, "ema50": h4_e50,
                "rsi":      h4_rsi,
                "macd_bias": h4_macd,
                "atr":      h4_atr,
                "structure": h4_struct["structure"],
                "phase":    h4_struct["phase"],
                "high48h":  round(max(c["high"] for c in h4[-12:]), 5),
                "low48h":   round(min(c["low"]  for c in h4[-12:]), 5),
                "last4":    "{} bull, {} bear".format(
                    sum(1 for c in h4[-4:] if c["close"]>c["open"]),
                    sum(1 for c in h4[-4:] if c["close"]<=c["open"]))
            }
            data["h4_patterns"] = detect_candle_patterns(h4[-10:])

        # ── H1 — entry timeframe ──────────────────────────────────────
        h1 = get_candles(pair, "1h", "5d")
        if h1 and len(h1) >= 20:
            h1c = [c["close"] for c in h1]
            h1_e9   = calc_ema(h1c, 9)
            h1_e20  = calc_ema(h1c, 20)
            h1_e50  = calc_ema(h1c, 50)
            h1_rsi  = calc_rsi(h1c)
            h1_macd = calc_macd(h1c)[1]
            h1_atr  = calc_atr(h1)
            h1_struct = detect_market_structure(h1[-24:])
            data["hourly"] = {
                "trend":    "BULLISH" if (h1_e20 and current_price > h1_e20) else "BEARISH",
                "ema9":     h1_e9, "ema20": h1_e20, "ema50": h1_e50,
                "rsi":      h1_rsi,
                "macd_bias": h1_macd,
                "atr":      h1_atr,
                "structure": h1_struct["structure"],
                "phase":    h1_struct["phase"],
                "high24h":  round(max(c["high"] for c in h1[-24:]), 5),
                "low24h":   round(min(c["low"]  for c in h1[-24:]), 5),
                "sr":       find_sr_levels(h1, current_price, lookback=40)[:5]
            }

        # ── M15 — trigger / entry confirmation ───────────────────────
        m15 = get_candles(pair, "15m", "3d")
        if m15 and len(m15) >= 14:
            m15c = [c["close"] for c in m15]
            m15_e9  = calc_ema(m15c, 9)
            m15_e20 = calc_ema(m15c, 20)
            m15_rsi = calc_rsi(m15c)
            m15_struct = detect_market_structure(m15[-20:])
            # London session range (approx last 8 candles of M15 from 3AM-8AM ET)
            london_range_h = max(c["high"] for c in m15[-32:-8]) if len(m15) >= 40 else None
            london_range_l = min(c["low"]  for c in m15[-32:-8]) if len(m15) >= 40 else None
            data["m15"] = {
                "trend":    "BULLISH" if (m15_e9 and current_price > m15_e9) else "BEARISH",
                "ema9":     m15_e9, "ema20": m15_e20,
                "rsi":      m15_rsi,
                "structure": m15_struct["structure"],
                "phase":    m15_struct["phase"],
                "london_high": round(london_range_h, 5) if london_range_h else None,
                "london_low":  round(london_range_l, 5) if london_range_l else None,
                "last4":    "{} bull, {} bear".format(
                    sum(1 for c in m15[-4:] if c["close"]>c["open"]),
                    sum(1 for c in m15[-4:] if c["close"]<=c["open"]))
            }
            data["m15_patterns"] = detect_candle_patterns(m15[-10:])

        # ── OVERALL TREND STRENGTH (multi-TF EMA alignment) ──────────
        d1_ref = get_candles(pair, "1d", "6mo") if not d1 else d1
        h4_ref = get_candles(pair, "4h", "30d") if not h4 else h4
        strength, score = detect_trend_strength(d1_ref, h4_ref)
        data["trend_strength"] = strength
        data["trend_score"]    = score

    except Exception as e:
        print("[SageChart] {}: {}".format(pair, e))
    return data


def format_sage_chart(d):
    """
    Formats all real chart data into a structured text block for the AI.
    Every number here comes from real live OHLC candles — no estimates.
    """
    wk  = d.get("weekly",  {})
    da  = d.get("daily",   {})
    h4  = d.get("h4",      {})
    h1  = d.get("hourly",  {})
    m15 = d.get("m15",     {})
    sep = "=" * 70

    lines = [
        sep,
        "LIVE REAL CHART DATA — {} @ {}".format(d["pair"], d["price"]),
        "Overall Trend Strength: {} (score: {})".format(
            d.get("trend_strength","?"), d.get("trend_score","?")),
        "ADR (Avg Daily Range): {} | Weekly Range Used: {}%".format(
            d.get("adr","?"), d.get("weekly_range_pct","?")),
        "Weekly High: {} | Weekly Low: {}".format(
            d.get("weekly_high","?"), d.get("weekly_low","?")),
        sep,

        "── WEEKLY CONTEXT (Big Picture) ──",
        "Trend={} | Structure={} | Phase={}".format(
            wk.get("trend","?"), wk.get("structure","?"), wk.get("phase","?")),
        "EMA20={} | RSI={} | ADR={} | Last3={}".format(
            wk.get("ema20","?"), wk.get("rsi","?"), wk.get("adr","?"), wk.get("last3","?")),
        "52wk High={} | 52wk Low={}".format(wk.get("high52","?"), wk.get("low52","?")),

        "── DAILY (Trade Direction) ──",
        "Trend={} | Structure={} | Phase={}".format(
            da.get("trend","?"), da.get("structure","?"), da.get("phase","?")),
        "Price Action: {}".format(da.get("phase_desc","?")),
        "EMA9={} | EMA20={} | EMA50={} | EMA200={}".format(
            da.get("ema9","?"), da.get("ema20","?"), da.get("ema50","?"), da.get("ema200","?")),
        "RSI={} | MACD={} | ATR={} | vs200EMA={}".format(
            da.get("rsi","?"), da.get("macd_bias","?"), da.get("atr","?"), da.get("vs_ema200","?")),
        "Bollinger: Upper={} | Mid={} | Lower={} | Position in BB={}%".format(
            da.get("bb_upper","?"), da.get("bb_mid","?"), da.get("bb_lower","?"), da.get("bb_position","?")),
        "20d High={} | 20d Low={} | Last5={}".format(
            da.get("high20d","?"), da.get("low20d","?"), da.get("last5","?")),
        "Swing Highs (D1): {} | Swing Lows (D1): {}".format(
            da.get("swing_highs",[]), da.get("swing_lows",[])),

        "── H4 (Trade Direction Confirmation) ──",
        "Trend={} | Structure={} | Phase={}".format(
            h4.get("trend","?"), h4.get("structure","?"), h4.get("phase","?")),
        "EMA9={} | EMA20={} | EMA50={} | RSI={} | MACD={} | ATR={}".format(
            h4.get("ema9","?"), h4.get("ema20","?"), h4.get("ema50","?"),
            h4.get("rsi","?"), h4.get("macd_bias","?"), h4.get("atr","?")),
        "48h High={} | 48h Low={} | Last4={}".format(
            h4.get("high48h","?"), h4.get("low48h","?"), h4.get("last4","?")),

        "── H1 (Entry Timeframe) ──",
        "Trend={} | Structure={} | Phase={}".format(
            h1.get("trend","?"), h1.get("structure","?"), h1.get("phase","?")),
        "EMA9={} | EMA20={} | EMA50={} | RSI={} | MACD={} | ATR={}".format(
            h1.get("ema9","?"), h1.get("ema20","?"), h1.get("ema50","?"),
            h1.get("rsi","?"), h1.get("macd_bias","?"), h1.get("atr","?")),
        "24h High={} | 24h Low={}".format(h1.get("high24h","?"), h1.get("low24h","?")),

        "── M15 (Entry Trigger) ──",
        "Trend={} | Structure={} | Phase={}".format(
            m15.get("trend","?"), m15.get("structure","?"), m15.get("phase","?")),
        "EMA9={} | EMA20={} | RSI={} | Last4={}".format(
            m15.get("ema9","?"), m15.get("ema20","?"), m15.get("rsi","?"), m15.get("last4","?")),
        "London Session High={} | London Session Low={}".format(
            m15.get("london_high","?"), m15.get("london_low","?")),
    ]

    # Support/Resistance
    sr = d.get("sr_levels", [])
    if sr:
        lines.append("── KEY S/R LEVELS (from real swing highs/lows + round numbers) ──")
        for lv in sr[:8]:
            lines.append("  {}: {} | {} pips away | Strength={} | {}".format(
                lv["type"], lv["price"], lv["distance_pips"], lv["strength"], lv["note"]))

    # Intraday S/R from H1
    h1_sr = h1.get("sr", [])
    if h1_sr:
        lines.append("── INTRADAY S/R (H1 level) ──")
        for lv in h1_sr[:4]:
            lines.append("  {}: {} | {} pips".format(lv["type"], lv["price"], lv["distance_pips"]))

    # Candlestick patterns
    all_pats = (
        [("D1", p) for p in d.get("d1_patterns",[])] +
        [("H4", p) for p in d.get("h4_patterns",[])] +
        [("M15",p) for p in d.get("m15_patterns",[])]
    )
    if all_pats:
        lines.append("── CANDLESTICK PATTERNS (Steve Nison method) ──")
        for tf, p in all_pats:
            lines.append("  [{}] {} ({}) — {}".format(tf, p["pattern"], p["bias"], p["note"]))
    else:
        lines.append("CANDLESTICK PATTERNS: No high-probability patterns on current candle")

    lines.append(sep)
    return "\n".join(lines)


def call_claude_with_search(prompt, max_tokens=600):
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        msg = client.messages.create(model="claude-sonnet-4-5", max_tokens=max_tokens,
            tools=[{"type":"web_search_20250305","name":"web_search"}],
            messages=[{"role":"user","content":prompt}])
        return "".join(b.text for b in msg.content if hasattr(b,"text") and b.type=="text").strip() or "No news available."
    except Exception as e:
        print("[SageSearch] {}".format(e)); return "News search unavailable."

_sage_jobs = {}

def _run_sage_job(job_id, pair, mode):
    try:
        _sage_jobs[job_id]={"status":"running","step":"Fetching live price..."}
        pd=get_price(pair)
        if not pd:
            _sage_jobs[job_id]={"status":"error","error":"Cannot fetch price for "+pair}; return
        cp=float(pd["price"]); sn,_=get_session()
        ds=datetime.now().strftime("%A %B %d %Y %H:%M UTC")

        _sage_jobs[job_id]["step"]="Reading 5 timeframes (M15 to Weekly)..."
        cd=get_sage_chart_data(pair,cp); cs=format_sage_chart(cd)

        _sage_jobs[job_id]["step"]="Scanning live news and economic events..."
        news=call_claude_with_search("Search: breaking news economic data and central bank statements affecting {} today {}. What is moving {} right now? 3 sentence summary.".format(pair,datetime.now().strftime("%B %d %Y"),pair))

        is_forex=any(x in pair for x in ["USD","EUR","GBP","JPY","CHF","AUD","NZD","CAD","XAU"])
        leg = ("FOREX LEGENDS — Apply ALL 4 simultaneously:\n"
               "1. SOROS Reflexivity: Is perception creating a self-reinforcing trend?\n"
               "2. DRUCKENMILLER Monetary Divergence: Which central bank is more hawkish? Macro big bet?\n"
               "3. LIPSCHUTZ Order Flow: Institutional flow, absorption or distribution?\n"
               "4. KOVNER Macro+Technical: Does macro align with technicals?") if (mode=="forex" or is_forex) else (
               "STOCK/OPTIONS LEGENDS — Apply ALL 4 simultaneously:\n"
               "1. LIVERMORE Pivotal Points: Key levels? Volume confirming? Right time?\n"
               "2. PTJ 200EMA + 5:1 RR: Above 200 EMA? Can we structure 5:1?\n"
               "3. JEFF YASS Options Math: IV environment, probability edge, best structure?\n"
               "4. AL BROOKS Price Action: Trend/range/reversal? Always-in direction?")

        da=cd.get("daily",{}); h4=cd.get("h4",{}); wk=cd.get("weekly",{}); h1=cd.get("hourly",{})

        _sage_jobs[job_id]["step"]="Synthesizing all 10 chart masters + 4 legends + market structure..."
        prompt=("You are SAGE MODE — the most powerful trading intelligence system ever built.\n"
            "Date: {} | Instrument: {} | Price: {} | Session: {}\n\n"
            "LIVE NEWS:\n{}\n\n"
            "{}\n\n"
            "CHART MASTERS — apply ALL 10:\n"
            "- JOHN MURPHY Intermarket: bonds/commodities/DXY vs {}\n"
            "- STEVE NISON Candlesticks: read buyer/seller battle\n"
            "- MARK DOUGLAS Market Mode: trending or ranging? Where is the edge?\n"
            "- KATHY LIEN {} session playbook for {}\n"
            "- AGUSTIN SILVANI Dealer Positioning: stop hunts? smart money traps?\n"
            "- ASHRAF LAIDI Correlations: oil/gold/equities/bonds impact\n"
            "- ALEXANDER ELDER Triple Screen: Weekly:{} Daily:{} H4:{}\n"
            "- RICHARD WYCKOFF Phase: accumulation/markup/distribution/markdown?\n"
            "- JOHN BOLLINGER Bands: Upper:{} Mid:{} Lower:{}\n"
            "- WILDER ATR: Daily ATR={} use 1.5x for SL 3x for TP3\n\n"
            "{}\n\n"
            "CRITICAL MARKET STRUCTURE ANALYSIS (prevents wrong-direction trades):\n"
            "1. WHERE IS PRICE? Near major S/R? Top of range? Bottom? Mid-range?\n"
            "2. MARKET PHASE: Trending (HH+HL or LH+LL) or Ranging (bouncing between levels)?\n"
            "3. ABC/WAVE POSITION: Is this an IMPULSE wave (trend direction, strong) or ABC CORRECTION (counter-trend trap)?\n"
            "4. KEY S/R ZONES: 2 nearest resistance levels above. 2 nearest support levels below. Rejecting any?\n"
            "5. TREND STRENGTH: ADX-equivalent — strong trend (25+) or consolidating/ranging (<20)?\n"
            "6. HIGHER TF CONTEXT: Weekly and Daily structure — respecting Daily EMA? In a weekly range?\n"
            "7. RULE: Return WAIT if: at major S/R without breakout, ABC correction likely, or ranging with no edge.\n\n"
            "MANDATE: 30-40 pip minimum. SL behind real SR. Min 2:1 RR. High TF=direction Low TF=entry.\n\n"
            "Return ONLY valid JSON (no markdown, no extra text):\n"
            '{{\"verdict\":\"BUY or SELL or WAIT\",\"confidence\":0,\"session\":\"{}\",\"entry\":\"{}\",\"stop_loss\":\"0\",\"sl_pips\":0,'
            '\"tp1\":\"0\",\"tp1_pips\":0,\"tp2\":\"0\",\"tp2_pips\":0,\"tp3\":\"0\",\"tp3_pips\":0,\"rr_ratio\":\"0:1\",'
            '\"timeframe_alignment\":{{\"weekly\":\"?\",\"daily\":\"?\",\"h4\":\"?\",\"h1\":\"?\",\"m15\":\"?\"}},'
            '\"legend_consensus\":{{\"soros\":\"\",\"druckenmiller\":\"\",\"lipschutz\":\"\",\"kovner\":\"\"}},'
            '\"chart_masters\":{{\"murphy\":\"\",\"nison\":\"\",\"douglas\":\"\",\"kathy_lien\":\"\",\"silvani\":\"\",\"laidi\":\"\",\"elder\":\"\",\"wyckoff\":\"\",\"bollinger\":\"\",\"wilder\":\"\"}},'
            '\"key_levels\":{{\"nearest_support\":\"\",\"nearest_resistance\":\"\",\"stop_zone\":\"\"}},'
            '\"market_structure\":{{\"phase\":\"TRENDING or RANGING or BREAKOUT or REVERSAL\",\"abc_position\":\"IMPULSE WAVE or ABC CORRECTION or UNKNOWN\",\"price_location\":\"AT RESISTANCE or AT SUPPORT or MID-RANGE or BREAKOUT ZONE\",\"trend_strength\":\"STRONG TREND or MODERATE TREND or RANGING or CHOPPY\",\"higher_tf_context\":\"\",\"sr_above\":\"\",\"sr_below\":\"\"}},'
            '\"candlestick_signal\":\"\",\"news_impact\":\"\",\"geopolitical_risk\":\"\",\"sage_says\":\"\",\"invalidation\":\"\"}}'
        ).format(ds,pair,cp,sn,news,leg,pair,sn,pair,
                 wk.get("trend","?"),da.get("trend","?"),h4.get("trend","?"),
                 da.get("bb_upper","?"),da.get("bb_mid","?"),da.get("bb_lower","?"),
                 da.get("atr","?"),cs,sn,cp)

        raw=call_claude(prompt,max_tokens=4000)
        result=parse_json_response(raw)
        result.update({"pair":pair,"price":cp,"mode":mode,"analyzed_at":datetime.now().strftime("%H:%M UTC")})
        _sage_jobs[job_id]={"status":"done","result":result}
    except Exception as e:
        import traceback; print("[SageMode] {}".format(traceback.format_exc()))
        _sage_jobs[job_id]={"status":"error","error":str(e)}

@app.route("/sage-mode")
@login_required
@byakugan_required
def sage_mode_page():
    return render_template("sage.html")

@app.route("/api/sage-start", methods=["POST"])
@login_required
@byakugan_required
def sage_start():
    try:
        data=request.get_json() or {}
        pair=data.get("pair","EUR/USD").upper().strip()
        mode=data.get("mode","forex")
        import uuid as _uuid
        job_id=str(_uuid.uuid4())[:8]
        _sage_jobs[job_id]={"status":"starting","step":"Initializing Sage Mode..."}
        threading.Thread(target=_run_sage_job,args=(job_id,pair,mode),daemon=True).start()
        return jsonify({"job_id":job_id,"status":"starting"})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/sage-poll/<job_id>", methods=["GET"])
@login_required
def sage_poll(job_id):
    job=_sage_jobs.get(job_id)
    if not job: return jsonify({"status":"error","error":"Job not found"}),404
    if job["status"]=="done":
        result=dict(job["result"]); result["status"]="done"
        _sage_jobs.pop(job_id,None); return jsonify(result)
    if job["status"]=="error":
        err=job.get("error","Unknown"); _sage_jobs.pop(job_id,None)
        return jsonify({"status":"error","error":err}),500
    return jsonify({"status":job["status"],"step":job.get("step","Processing...")})



@app.route("/api/sage-scanner", methods=["POST"])
@login_required
@byakugan_required
def sage_scanner():
    """Scan 8-10 forex pairs simultaneously and return trending ones"""
    try:
        import uuid as _uuid
        job_id = str(_uuid.uuid4())[:8]
        _sage_jobs[job_id] = {"status": "running", "step": "Scanning 10 pairs for trending setups..."}
        threading.Thread(target=_run_sage_scanner_job, args=(job_id,), daemon=True).start()
        return jsonify({"job_id": job_id, "status": "starting"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/sage-scanner-poll/<job_id>", methods=["GET"])
@login_required
def sage_scanner_poll(job_id):
    job = _sage_jobs.get(job_id)
    if not job: return jsonify({"status": "error", "error": "Job not found"}), 404
    if job["status"] == "done":
        result = dict(job["result"]); result["status"] = "done"
        _sage_jobs.pop(job_id, None); return jsonify(result)
    if job["status"] == "error":
        err = job.get("error", "Unknown"); _sage_jobs.pop(job_id, None)
        return jsonify({"status": "error", "error": err}), 500
    return jsonify({"status": job["status"], "step": job.get("step", "Scanning...")})

def _run_sage_scanner_job(job_id):
    try:
        SCAN_PAIRS = ["EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","NZD/USD","USD/CHF","EUR/JPY","GBP/JPY","XAU/USD"]
        results = []
        for i, pair in enumerate(SCAN_PAIRS):
            _sage_jobs[job_id]["step"] = "Scanning {}/10: {}...".format(i+1, pair)
            try:
                pd = get_price(pair)
                if not pd: continue
                cp = float(pd["price"])
                cd = get_sage_chart_data(pair, cp)
                da = cd.get("daily", {}); h4 = cd.get("h4", {}); wk = cd.get("weekly", {})
                chart_str = format_sage_chart(cd)
                verdict_prompt = (
                    'Analyze ' + pair + ' at ' + str(cp) + '. '
                    'Real chart data:\n' + chart_str + '\n\n'
                    'Based ONLY on the real data above, return compact JSON (no markdown):\n'
                    '{"pair":"' + pair + '",'
                    '"verdict":"BUY or SELL or WAIT",'
                    '"confidence":0,'
                    '"trend_strength":"STRONG BULLISH or STRONG BEARISH or MODERATE BULLISH or MODERATE BEARISH or RANGING",'
                    '"market_structure":"UPTREND or DOWNTREND or RANGING",'
                    '"abc_position":"IMPULSE or CORRECTION or CONSOLIDATION",'
                    '"direction":"UP or DOWN or SIDEWAYS",'
                    '"reason":"max 15 words using real S/R and EMA data",'
                    '"entry":"' + str(cp) + '",'
                    '"sl":"nearest real S/R level",'
                    '"tp1":"next real S/R level",'
                    '"key_pattern":"candlestick pattern or none"}'
                )
                prompt = verdict_prompt
                raw = call_claude(prompt, max_tokens=400)
                parsed = parse_json_response(raw)
                if parsed and parsed.get("verdict") in ["BUY","SELL"]:
                    parsed["pair"] = pair
                    parsed["price"] = cp
                    results.append(parsed)
            except Exception as pe:
                print("[SageScanner] {} error: {}".format(pair, pe))
                continue

        # Sort by confidence descending
        results.sort(key=lambda x: int(x.get("confidence", 0)), reverse=True)
        # Return top 5 with real signals, rest as WAIT
        top = [r for r in results if r.get("verdict") in ["BUY","SELL"]][:5]
        # Add WAIT pairs
        all_pairs_result = []
        for pair in SCAN_PAIRS:
            found = next((r for r in results if r.get("pair") == pair), None)
            if found:
                all_pairs_result.append(found)
            else:
                all_pairs_result.append({"pair": pair, "verdict": "WAIT", "confidence": 0, "trend_strength": "UNKNOWN", "direction": "SIDEWAYS", "reason": "No data or ranging market"})

        _sage_jobs[job_id] = {
            "status": "done",
            "result": {
                "pairs": all_pairs_result,
                "top_picks": top,
                "scanned_at": datetime.now().strftime("%H:%M UTC")
            }
        }
    except Exception as e:
        import traceback; print("[SageScanner] {}".format(traceback.format_exc()))
        _sage_jobs[job_id] = {"status": "error", "error": str(e)}


AI_INFRA_UNIVERSE = {
    'ALL': [
        'MRVL','INTC','SMCI','CRDO',
        'APLD','IREN','NBIS',
        'OKLO','CEG','VST',
        'MOD','STRL','CLS',
        'PATH','TER',
        'PLTR','AI',
    ],
    'CHIPS':      ['MRVL','INTC','AMD','AVGO','CRDO','MU'],
    'SERVERS':    ['SMCI','CLS','JBL','DELL'],
    'DATACENTER': ['APLD','IREN','NBIS','DLR'],
    'POWER':      ['OKLO','CEG','VST','ETN'],
    'ROBOTICS':   ['PATH','TER','ISRG'],
    'DEEP_VALUE': ['SMCI','INTC','MRVL','APLD','IREN','AI','PATH'],
}

TICKER_CATEGORY = {
    'MRVL':'CHIPS',    'INTC':'CHIPS',    'AMD':'CHIPS',
    'AVGO':'CHIPS',    'CRDO':'NETWORKING','MU':'CHIPS',
    'SMCI':'SERVERS',  'CLS':'MANUFACTURING','JBL':'MANUFACTURING','DELL':'SERVERS',
    'APLD':'DATA CENTER','IREN':'DATA CENTER','NBIS':'DATA CENTER','DLR':'DATA CENTER',
    'OKLO':'POWER',    'CEG':'POWER',     'VST':'POWER',    'ETN':'POWER',
    'MOD':'COOLING',   'STRL':'CONSTRUCTION',
    'PATH':'ROBOTICS', 'TER':'ROBOTICS',  'ISRG':'ROBOTICS',
    'PLTR':'AI SOFTWARE','AI':'AI SOFTWARE',
}

AI_ROLE_MAP = {
    'MRVL': 'Custom AI ASICs for hyperscalers — direct NVIDIA ASIC competitor',
    'INTC': '18A foundry + $350M SambaNova — US chip manufacturing turnaround',
    'AMD':  'MI300X GPU competing with NVIDIA H100/H200 for AI training',
    'AVGO': 'Custom AI XPU chips (Google TPU, Meta) + AI networking',
    'CRDO': 'High-speed Active Electrical Cables for AI data center interconnects',
    'MU':   'HBM3E memory stacked on NVIDIA GPUs for AI training',
    'SMCI': 'AI server racks with direct liquid cooling — deeply discounted',
    'CLS':  'Contract mfg: AI servers and networking gear for hyperscalers',
    'JBL':  'AI server racks, liquid-cooling, networking switches',
    'DELL': 'AI server infrastructure + PowerEdge AI, Microsoft/NVIDIA partner',
    'APLD': 'HPC & AI data center operator — 150MW CoreWeave deal',
    'IREN': 'Ex-BTC miner converting to AI GPU cloud — NVIDIA Blackwell ordered',
    'NBIS': 'AI cloud infra — Meta + Microsoft contracts, scaling fast',
    'DLR':  'Largest data center REIT — AI colocation demand surge',
    'OKLO': 'Small modular nuclear reactors for AI data center power',
    'CEG':  'Nuclear operator with Microsoft AI data center energy deals',
    'VST':  'Power generation play on AI electricity demand surge',
    'ETN':  'Electrical components for AI data center buildout',
    'MOD':  'Data center chillers — revenue +119% YoY, $2B target by 2028',
    'STRL': 'Builds AI data center facilities — $2.6B backlog +64% YoY',
    'PATH': 'Agentic AI automation — software robots for enterprise workflows',
    'TER':  'Semiconductor test equipment + Universal Robots',
    'ISRG': 'AI-guided da Vinci surgical robots — market leader',
    'PLTR': 'AI Platform (AIP) — US gov + enterprise AI analytics',
    'AI':   'Enterprise AI apps — pure-play beaten down, direct AI exposure',
}

def score_ai_infra_stock(ticker_sym):
    base = score_stock(ticker_sym)
    if not base:
        return None
    try:
        import yfinance as yf
        import concurrent.futures as cf
        ticker = yf.Ticker(ticker_sym)
        # hist already pulled in score_stock but we need 52w data
        try:
            hist = ticker.history(period='1y', interval='1d', timeout=8)
        except Exception:
            hist = None
        price  = base['price']
        week52_high = round(float(hist['High'].tail(252).max()), 2) if hist is not None and not hist.empty else None
        week52_low  = round(float(hist['Low'].tail(252).min()),  2) if hist is not None and not hist.empty else None
        vs_52w_high = round(((price - week52_high) / week52_high) * 100, 1) if week52_high else None
        analyst_target = None; analyst_rating = None; num_analysts = None
        pe_ratio = None; revenue_growth = None; market_cap = None
        # Wrap ticker.info in strict 5-second timeout — never block the whole job
        def _get_info():
            return ticker.info
        try:
            with cf.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_get_info)
                info = fut.result(timeout=5)
            analyst_target  = info.get('targetMeanPrice')
            analyst_rating  = (info.get('recommendationKey','') or '').replace('_',' ').title() or None
            num_analysts    = info.get('numberOfAnalystOpinions')
            pe_ratio        = info.get('trailingPE')
            revenue_growth  = info.get('revenueGrowth')
            mc = info.get('marketCap', 0)
            if mc > 1e12:   market_cap = f"{round(mc/1e12,1)}T"
            elif mc > 1e9:  market_cap = f"{round(mc/1e9,1)}B"
            elif mc > 1e6:  market_cap = f"{round(mc/1e6,1)}M"
        except Exception as e:
            print(f'[AIInfra info SKIP {ticker_sym}] {e}')
        upside_pct = round(((analyst_target - price) / price) * 100, 1) if analyst_target and price else None
        score = base['score']
        if vs_52w_high:
            if vs_52w_high < -40:   score += 20
            elif vs_52w_high < -25: score += 12
            elif vs_52w_high < -15: score += 6
        if upside_pct:
            if upside_pct > 50:   score += 15
            elif upside_pct > 30: score += 10
            elif upside_pct > 15: score += 5
            elif upside_pct < 0:  score -= 10
        if upside_pct and upside_pct >= 20 and vs_52w_high and vs_52w_high <= -15:
            signal = 'BUY'
        elif upside_pct and upside_pct > 0:
            signal = 'WATCH'
        else:
            signal = 'AVOID'
        base.update({
            'week52_high':    week52_high,
            'week52_low':     week52_low,
            'vs_52w_high':    vs_52w_high,
            'analyst_target': round(analyst_target, 2) if analyst_target else None,
            'analyst_rating': analyst_rating,
            'num_analysts':   num_analysts,
            'upside_pct':     upside_pct,
            'market_cap':     market_cap,
            'pe_ratio':       round(pe_ratio, 1) if pe_ratio else None,
            'revenue_growth': (f"+{round(revenue_growth*100,1)}%" if revenue_growth and revenue_growth > 0
                               else f"{round(revenue_growth*100,1)}%" if revenue_growth else None),
            'ai_role':        AI_ROLE_MAP.get(ticker_sym, ''),
            'category':       TICKER_CATEGORY.get(ticker_sym, 'AI'),
            'signal':         signal,
            'score':          max(0, min(100, score)),
        })
        return base
    except Exception as e:
        print(f'[ScoreAIInfra {ticker_sym}] {e}')
        return base

_ai_infra_jobs = {}

def _run_ai_infra_job(job_id, scan_filter, date_str):
    try:
        _ai_infra_jobs[job_id] = {'status': 'scanning'}
        universe = AI_INFRA_UNIVERSE.get(scan_filter, AI_INFRA_UNIVERSE['ALL'])
        regime   = get_market_regime()
        _ai_infra_jobs[job_id] = {'status': 'scoring'}
        scored = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(score_ai_infra_stock, sym): sym for sym in universe}
            for f in as_completed(futures, timeout=90):
                sym = futures[f]
                try:
                    s = f.result()
                    if s: scored.append(s)
                except Exception as e:
                    print(f'[AIInfra score] {sym}: {e}')
        scored.sort(key=lambda x: (
            0 if x.get('signal')=='BUY' else 1 if x.get('signal')=='WATCH' else 2,
            -(x.get('upside_pct') or 0)
        ))
        top5 = scored[:5]
        # If scoring failed entirely, use raw universe with base prices so we always return something
        if not scored:
            print(f'[AIInfra] WARNING: zero stocks scored for {scan_filter} — using fallback')
            for sym in universe[:5]:
                try:
                    import yfinance as yf
                    t = yf.Ticker(sym)
                    h = t.history(period='5d', timeout=8)
                    price = round(float(h['Close'].iloc[-1]), 2) if not h.empty else 0
                    scored.append({'ticker':sym,'price':price,'score':50,'signal':'WATCH',
                                   'rsi':50,'vol_ratio':1.0,'signals':[],'sr_levels':{},
                                   'ema20':None,'ema50':None,'ema200':None,'macd_bias':'NEUTRAL',
                                   'near_earnings':False,'news':[],
                                   'ai_role':AI_ROLE_MAP.get(sym,''),'category':TICKER_CATEGORY.get(sym,'AI'),
                                   'week52_high':None,'week52_low':None,'vs_52w_high':None,
                                   'analyst_target':None,'analyst_rating':None,'num_analysts':None,
                                   'upside_pct':None,'market_cap':None,'pe_ratio':None,'revenue_growth':None})
                except Exception as fe:
                    print(f'[AIInfra fallback {sym}] {fe}')
        if not scored:
            _ai_infra_jobs[job_id] = {'status':'error','error':'No stocks scored'}; return
        top5 = scored[:5]
        _ai_infra_jobs[job_id] = {'status': 'news'}
        for stock in top5:
            try: stock['news'] = get_news(stock['ticker'])[:3]
            except: stock['news'] = []
        _ai_infra_jobs[job_id] = {'status': 'analyzing'}
        vix = float(regime.get('vix', 20))
        stocks_ctx = ''
        for s in top5:
            stocks_ctx += f"""
---
{s['ticker']} | ${s['price']} | Signal:{s.get('signal','WATCH')} | Score:{s['score']}/100
Category:{s.get('category','')} | AI Role:{s.get('ai_role','')}
EMA20:{s.get('ema20','?')} EMA50:{s.get('ema50','?')} EMA200:{s.get('ema200','?')}
RSI:{s.get('rsi','?')} MACD:{s.get('macd_bias','?')} Volume:{s.get('vol_ratio','?')}x
52W HIGH:${s.get('week52_high','?')} CURRENT:${s['price']} 52W LOW:${s.get('week52_low','?')}
Discount from high:{s.get('vs_52w_high','?')}% | Analyst target:${s.get('analyst_target','?')} Upside:{s.get('upside_pct','?')}% Rating:{s.get('analyst_rating','?')} ({s.get('num_analysts','?')} analysts)
Market cap:{s.get('market_cap','?')} P/E:{s.get('pe_ratio','?')} Rev growth:{s.get('revenue_growth','?')}
News:{' | '.join([n['title'][:60] for n in s.get('news',[])]) or 'None'}
---"""
        prompt = f"""You are Wolf AI — elite AI infrastructure investor.
Peter Lynch + Druckenmiller + Cathie Wood methodology applied to AI picks & shovels.
TODAY:{date_str} | SPY:${regime['spy_price']} ({regime.get('spy_change',0):+.2f}%) VIX:{vix} {regime['fear_greed']} REGIME:{regime['regime']} | FILTER:{scan_filter}

REAL STOCK DATA FROM YFINANCE:
{stocks_ctx}

For each stock give the EXACT reason to buy or avoid NOW based on AI infrastructure role, value vs fair value, specific catalyst, and time horizon.

Respond ONLY in valid JSON (no markdown):
{{"scan_date":"{date_str}","filter":"{scan_filter}","sector_read":"2-sentence AI infrastructure sector read","picks":[{{"rank":1,"ticker":"X","current_price":"0.00","signal":"BUY","confidence":82,"wolf_score":75,"category":"CHIPS","ai_role":"specific role","thesis":"3-sentence thesis using real data","infrastructure_role":"what breaks in AI if this fails","catalyst":"near-term catalyst max 12 words","risk":"key risk max 10 words","verdict":"one decisive sentence","why_now":"what specifically changed","time_horizon":"3-6 months","entry_strategy":"entry plan","exit_strategy":"exit plan","week52_high":"0","week52_low":"0","vs_52w_high":-20,"analyst_target":"0","upside_pct":25,"analyst_rating":"Strong Buy","num_analysts":20,"market_cap":"10B","pe_ratio":25,"revenue_growth":"+25%","ai_revenue_pct":"~50%","confluences":["reason1","reason2","reason3"],"ai_edge":["edge1","edge2"],"warnings":["risk1"],"invalidation":"what makes this wrong"}}]}}"""
        result = None; last_error = None
        for attempt in range(3):
            try:
                raw = call_claude(prompt, 6000)
                if not raw or not raw.strip(): raise ValueError('Empty response')
                result = parse_json_response(raw); break
            except Exception as retry_err:
                last_error = retry_err
                print(f'[AIInfra] Attempt {attempt+1} failed: {retry_err}')
                if attempt < 2: time.sleep(2)
        if result is None: raise Exception(f'Claude failed after 3 attempts: {last_error}')
        for pick in result.get('picks', []):
            match = next((s for s in top5 if s['ticker']==pick.get('ticker','')), None)
            if match:
                pick['real_score']     = match['score']
                pick['real_rsi']       = match['rsi']
                pick['real_vol_ratio'] = match['vol_ratio']
                pick['real_signals']   = match['signals']
                pick['sr_levels']      = match['sr_levels']
                pick['news']           = match['news']
                pick['ema20']          = match.get('ema20')
                pick['ema50']          = match.get('ema50')
                pick['ema200']         = match.get('ema200')
                pick['macd_bias']      = match.get('macd_bias')
                pick['near_earnings']  = match.get('near_earnings', False)
                if match.get('week52_high'):              pick['week52_high']   = str(match['week52_high'])
                if match.get('week52_low'):               pick['week52_low']    = str(match['week52_low'])
                if match.get('vs_52w_high') is not None:  pick['vs_52w_high']   = match['vs_52w_high']
                if match.get('analyst_target'):           pick['analyst_target']= str(match['analyst_target'])
                if match.get('upside_pct') is not None:  pick['upside_pct']    = match['upside_pct']
                if match.get('analyst_rating'):           pick['analyst_rating']= match['analyst_rating']
                if match.get('num_analysts'):             pick['num_analysts']  = match['num_analysts']
                if match.get('market_cap'):               pick['market_cap']    = match['market_cap']
        result['market_regime'] = regime
        _ai_infra_jobs[job_id] = {'status': 'done', 'result': result}
    except Exception as e:
        import traceback; print(traceback.format_exc())
        _ai_infra_jobs[job_id] = {'status': 'error', 'error': str(e)}


@app.route('/education')
def education_page():
    return render_template('education.html')

@app.route('/legends')
@login_required
@byakugan_required
def legends_page():
    return render_template('legends.html')

@app.route('/ai-infra')
@login_required
@byakugan_required
def ai_infra_page():
    return render_template('ai_infra.html')


@app.route('/api/ai-infra-scan', methods=['POST'])
@login_required
@elite_required
def ai_infra_scan():
    try:
        data        = request.get_json() or {}
        scan_filter = data.get('filter', 'ALL')
        date_str    = datetime.now().strftime('%A, %B %d, %Y')
        job_id      = str(uuid.uuid4())[:8]
        _ai_infra_jobs[job_id] = {'status': 'starting'}
        t = threading.Thread(target=_run_ai_infra_job, args=(job_id, scan_filter, date_str), daemon=True)
        t.start()
        return jsonify({'job_id': job_id, 'status': 'starting'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-infra-poll/<job_id>', methods=['GET'])
@login_required
@elite_required
def ai_infra_poll(job_id):
    job = _ai_infra_jobs.get(job_id)
    if not job: return jsonify({'status':'error','error':'Job not found'}), 404
    if job['status'] == 'done':
        result = job.get('result', {}); result['status'] = 'done'
        _ai_infra_jobs.pop(job_id, None); return jsonify(result)
    if job['status'] == 'error':
        return jsonify({'status':'error','error':job.get('error','Unknown')}), 500
    return jsonify({'status': job['status']})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
