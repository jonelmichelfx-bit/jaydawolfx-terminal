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
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
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
    analysis = {
        'pair': pair,
        'current_price': current_price,
        'daily': {}, 'weekly': {}, 'hourly': {},
        'sr_levels': [],
        'indicators': {},
        'trend_summary': {}
    }

    try:
        daily = get_candles(pair, '1d', '3mo')
        if daily and len(daily) >= 20:
            d_closes = [c['close'] for c in daily]
            d_highs  = [c['high']  for c in daily]
            d_lows   = [c['low']   for c in daily]

            ema20  = calc_ema(d_closes, 20)
            ema50  = calc_ema(d_closes, 50)
            ema200 = calc_ema(d_closes, min(200, len(d_closes)))
            rsi    = calc_rsi(d_closes)
            macd_val, macd_bias = calc_macd(d_closes)

            price_vs_ema200 = 'ABOVE' if (ema200 and current_price > ema200) else 'BELOW'
            daily_trend = 'BULLISH' if current_price > (ema50 or current_price) else 'BEARISH'

            last5 = daily[-5:]
            bullish_candles = sum(1 for c in last5 if c['close'] > c['open'])
            bearish_candles = 5 - bullish_candles

            analysis['daily'] = {
                'trend': daily_trend,
                'ema20': ema20, 'ema50': ema50, 'ema200': ema200,
                'rsi': rsi, 'macd': macd_val, 'macd_bias': macd_bias,
                'price_vs_ema200': price_vs_ema200,
                'last_5_candles': f"{bullish_candles} bullish, {bearish_candles} bearish",
                'recent_high': round(max(d_highs[-20:]), 5),
                'recent_low':  round(min(d_lows[-20:]),  5),
                '3mo_high': round(max(d_highs), 5),
                '3mo_low':  round(min(d_lows),  5),
            }

            analysis['sr_levels'] = find_sr_levels(daily, current_price, lookback=60)

        weekly = get_candles(pair, '1wk', '1y')
        if weekly and len(weekly) >= 10:
            w_closes = [c['close'] for c in weekly]
            w_highs  = [c['high']  for c in weekly]
            w_lows   = [c['low']   for c in weekly]
            w_ema20  = calc_ema(w_closes, min(20, len(w_closes)))
            w_rsi    = calc_rsi(w_closes)

            weekly_trend = 'BULLISH' if current_price > (w_ema20 or current_price) else 'BEARISH'
            last3_weekly = weekly[-3:]
            w_bull = sum(1 for c in last3_weekly if c['close'] > c['open'])

            analysis['weekly'] = {
                'trend': weekly_trend,
                'ema20': w_ema20,
                'rsi': w_rsi,
                'last_3_candles': f"{w_bull} bullish, {3-w_bull} bearish",
                '52wk_high': round(max(w_highs), 5),
                '52wk_low':  round(min(w_lows),  5),
            }

        hourly = get_candles(pair, '1h', '5d')
        if hourly and len(hourly) >= 20:
            h_closes = [c['close'] for c in hourly]
            h_ema20  = calc_ema(h_closes, 20)
            h_ema50  = calc_ema(h_closes, 50)
            h_rsi    = calc_rsi(h_closes)
            h_macd, h_macd_bias = calc_macd(h_closes)

            hourly_trend = 'BULLISH' if current_price > (h_ema20 or current_price) else 'BEARISH'
            last4_h = hourly[-4:]
            h_bull = sum(1 for c in last4_h if c['close'] > c['open'])

            h_sr = find_sr_levels(hourly, current_price, lookback=40)

            analysis['hourly'] = {
                'trend': hourly_trend,
                'ema20': h_ema20, 'ema50': h_ema50,
                'rsi': h_rsi, 'macd_bias': h_macd_bias,
                'last_4_candles': f"{h_bull} bullish, {4-h_bull} bearish",
                'recent_high': round(max(c['high'] for c in hourly[-24:]), 5),
                'recent_low':  round(min(c['low']  for c in hourly[-24:]), 5),
                'sr_levels': h_sr[:4]
            }

        w_trend = analysis['weekly'].get('trend', 'NEUTRAL')
        d_trend = analysis['daily'].get('trend', 'NEUTRAL')
        h_trend = analysis['hourly'].get('trend', 'NEUTRAL')

        bull_count = sum(1 for t in [w_trend, d_trend, h_trend] if t == 'BULLISH')
        bear_count = sum(1 for t in [w_trend, d_trend, h_trend] if t == 'BEARISH')

        analysis['trend_summary'] = {
            'weekly': w_trend, 'daily': d_trend, 'hourly': h_trend,
            'overall': 'BULLISH' if bull_count >= 2 else 'BEARISH' if bear_count >= 2 else 'MIXED',
            'alignment': f"{bull_count}/3 bullish, {bear_count}/3 bearish"
        }

    except Exception as e:
        print(f'[ChartAnalysis] {pair} error: {e}')

    return analysis

def format_chart_analysis_for_prompt(ca):
    if not ca:
        return "Chart data unavailable"

    d = ca.get('daily', {})
    w = ca.get('weekly', {})
    h = ca.get('hourly', {})
    ts = ca.get('trend_summary', {})
    sr = ca.get('sr_levels', [])

    lines = [f"\n{'='*60}",
             f"REAL CHART DATA FOR {ca['pair']} — Price: {ca['current_price']}",
             f"{'='*60}"]

    lines.append(f"\nTREND ALIGNMENT: {ts.get('overall','?')} ({ts.get('alignment','?')})")
    lines.append(f"  Weekly: {ts.get('weekly','?')} | Daily: {ts.get('daily','?')} | Hourly: {ts.get('hourly','?')}")

    if w:
        lines.append(f"\nWEEKLY CHART (1 Year of data):")
        lines.append(f"  Trend: {w.get('trend','?')} | EMA20: {w.get('ema20','?')} | RSI: {w.get('rsi','?')}")
        lines.append(f"  52-Week High: {w.get('52wk_high','?')} | 52-Week Low: {w.get('52wk_low','?')}")
        lines.append(f"  Last 3 candles: {w.get('last_3_candles','?')}")

    if d:
        lines.append(f"\nDAILY CHART (3 Months of data):")
        lines.append(f"  Trend: {d.get('trend','?')} | EMA20: {d.get('ema20','?')} | EMA50: {d.get('ema50','?')} | EMA200: {d.get('ema200','?')}")
        lines.append(f"  Price vs EMA200: {d.get('price_vs_ema200','?')} | RSI: {d.get('rsi','?')} | MACD: {d.get('macd_bias','?')}")
        lines.append(f"  3-Month High: {d.get('3mo_high','?')} | 3-Month Low: {d.get('3mo_low','?')}")
        lines.append(f"  Last 5 candles: {d.get('last_5_candles','?')}")

    if h:
        lines.append(f"\nHOURLY CHART (5 Days of data):")
        lines.append(f"  Trend: {h.get('trend','?')} | EMA20: {h.get('ema20','?')} | EMA50: {h.get('ema50','?')}")
        lines.append(f"  RSI: {h.get('rsi','?')} | MACD: {h.get('macd_bias','?')}")
        lines.append(f"  24hr High: {h.get('recent_high','?')} | 24hr Low: {h.get('recent_low','?')}")
        lines.append(f"  Last 4 candles: {h.get('last_4_candles','?')}")

    if sr:
        lines.append(f"\nREAL SUPPORT & RESISTANCE (from actual swing highs/lows):")
        for lv in sr[:6]:
            lines.append(f"  {lv['type']}: {lv['price']} — {lv['note']} ({lv['distance_pips']} pips away)")

    h_sr = h.get('sr_levels', [])
    if h_sr:
        lines.append(f"\nHOURLY S/R (intraday levels):")
        for lv in h_sr[:4]:
            lines.append(f"  {lv['type']}: {lv['price']} ({lv['distance_pips']} pips)")

    lines.append(f"{'='*60}\n")
    return '\n'.join(lines)

def get_multi_pair_chart_data(pairs, current_prices):
    results = {}
    def fetch_one(pair):
        price = current_prices.get(pair, {})
        cp = float(price.get('price', 1.0)) if price else 1.0
        return pair, get_chart_analysis(pair, cp)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fetch_one, p): p for p in pairs}
        for f in as_completed(futures, timeout=30):
            try:
                pair, analysis = f.result()
                results[pair] = analysis
            except Exception as e:
                print(f'Chart fetch error: {e}')
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
    if plan not in ('basic','pro','elite'): flash('Invalid plan selected.','danger'); return redirect(url_for('pricing'))
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

@app.route('/wolf-scanner')
@login_required
@elite_required
def wolf_scanner_page(): return render_template('wolf_scanner.html')

# ── Forex API ─────────────────────────────────────────────────

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


_wolf_scan_jobs = {}

def _run_wolf_scan_job(job_id, scan_filter):
    try:
        _wolf_scan_jobs[job_id] = {'status': 'running'}
        date_str = datetime.now().strftime('%A, %B %d, %Y')
        session_name, _ = get_session()
        prices, is_live = get_cached_prices()

        if scan_filter == 'MAJORS':   scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','USD/CHF','AUD/USD','USD/CAD']
        elif scan_filter == 'GOLD':   scan_pairs = ['XAU/USD','EUR/USD','USD/JPY','AUD/USD','NZD/USD']
        elif scan_filter == 'ASIAN':  scan_pairs = ['USD/JPY','EUR/JPY','GBP/JPY','AUD/USD','NZD/USD']
        elif scan_filter == 'LONDON': scan_pairs = ['EUR/USD','GBP/USD','EUR/GBP','USD/CHF','EUR/JPY']
        else: scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','NZD/USD','USD/CHF']

        prices_lines = []
        for p in scan_pairs:
            q = prices.get(p)
            if q:
                dp = 2 if 'JPY' in p or p == 'XAU/USD' else 5
                prices_lines.append(f"{p}: {float(q['price']):.{dp}f} (H:{float(q['high']):.{dp}f} L:{float(q['low']):.{dp}f} Chg:{float(q.get('percent_change',0)):+.2f}%)")
        prices_str = '\n'.join(prices_lines)

        news_items = get_news()
        news_str = '\n'.join([f"- {n['title']} ({n['source']}, {n['published']})" for n in news_items[:6]]) or "- Monitor key economic events"

        chart_data = get_multi_pair_chart_data(scan_pairs, prices)
        chart_ctx = '\n'.join([format_chart_analysis_for_prompt(chart_data[p]) for p in scan_pairs if p in chart_data])

        prompt = f"""You are Wolf AI — professional forex trader using Soros, Druckenmiller, Kovner, Paul Tudor Jones methodology.

TODAY: {date_str} | SESSION: {session_name}
PRICES: {'LIVE' if is_live else 'REFERENCE'}

LIVE PRICES:
{prices_str}

NEWS:
{news_str}

{chart_ctx}

CRITICAL: Use the REAL chart data above. The S/R levels, EMA values, RSI readings are from ACTUAL candles.
DO NOT make up levels — use the exact swing highs/lows provided above.

Find the 5 BEST trades:
STEP 1 — DXY direction drives everything (Soros)
STEP 2 — Central bank divergence (Druckenmiller)
STEP 3 — 4+ timeframe confluence required (Kovner)
STEP 4 — Trade WITH the trend only
STEP 5 — Use ONLY real S/R levels from chart data above
STEP 6 — Both BUY and SELL scenarios per trade
STEP 7 — Economic calendar warnings

Respond ONLY in valid JSON (no markdown, no backticks):
{{"scan_date":"{date_str}","session":"{session_name}","market_theme":"string","dxy_bias":"BULLISH or BEARISH","risk_sentiment":"RISK-ON or RISK-OFF","wolf_commentary":"2-3 sentences","trades":[{{"rank":1,"pair":"EUR/USD","current_price":"1.0380","trend":"DOWNTREND","primary_direction":"SELL","wolf_score":8.5,"confidence":85,"aligned_count":5,"thesis":"3-4 sentence thesis citing real EMA/RSI/S/R data","timeframe_alignment":{{"monthly":"BEARISH","weekly":"BEARISH","daily":"BEARISH","h4":"BEARISH","h1":"NEUTRAL","m15":"BEARISH"}},"confluences":["Price below EMA200 1.0520","RSI 38 bearish momentum","Real resistance at 1.0412 (swing high)","DXY bullish divergence"],"key_levels":[{{"type":"RESISTANCE","price":"1.0412","note":"Real swing high from chart data","distance_pips":32}},{{"type":"CURRENT","price":"1.0380","note":"Current price","distance_pips":0}},{{"type":"SUPPORT","price":"1.0340","note":"Real swing low from chart data","distance_pips":40}}],"buy_scenario":{{"trigger":"Break above real resistance 1.0412 on H4","entry":"1.0418","stop_loss":"1.0390","tp1":"1.0460","tp2":"1.0510","tp3":"1.0560","rr":"1:2.5","probability":25}},"sell_scenario":{{"trigger":"Reject real resistance 1.0400 break below 1.0360","entry":"1.0355","stop_loss":"1.0390","tp1":"1.0310","tp2":"1.0270","tp3":"1.0220","rr":"1:2.8","probability":75}},"warnings":[{{"level":"HIGH","text":"US CPI Thursday 8:30AM EST — wait for release"}}],"relevant_news":["string"]}}]}}"""

        # Auto-retry up to 3 times if AI returns empty or bad JSON
        result = None
        last_error = None
        for attempt in range(3):
            try:
                raw = call_claude(prompt, 6000)
                if not raw or not raw.strip():
                    raise ValueError('Empty response from AI')
                result = parse_json_response(raw)
                break  # success — stop retrying
            except Exception as retry_err:
                last_error = retry_err
                print(f'[WolfScan] Attempt {attempt+1} failed: {retry_err}')
                if attempt < 2:
                    time.sleep(2)
        if result is None:
            raise Exception(f'AI failed after 3 attempts: {last_error}')


        for trade in result.get('trades', []):
            pair = trade.get('pair', '')
            q = prices.get(pair)
            if q:
                dp = 2 if 'JPY' in pair or pair == 'XAU/USD' else 4
                trade['current_price'] = f"{float(q['price']):.{dp}f}"
            if pair in chart_data and chart_data[pair].get('sr_levels'):
                trade['real_sr_levels'] = chart_data[pair]['sr_levels'][:6]

            # ── OVERWRITE AI confidence with real data-driven score ──
            direction = trade.get('primary_direction', 'BUY')
            real_conf = calculate_real_confidence(pair, direction, chart_data)
            trade['confidence'] = real_conf
            # wolf_score out of 10 — derived from same real data
            trade['wolf_score'] = round(real_conf / 10, 1)

        _wolf_scan_jobs[job_id] = {'status': 'done', 'result': result}
    except Exception as e:
        import traceback; print(traceback.format_exc())
        _wolf_scan_jobs[job_id] = {'status': 'error', 'error': str(e)}

@app.route('/api/wolf-scan', methods=['POST'])
@login_required
def wolf_scan():
    data = request.get_json() or {}
    scan_filter = data.get('filter', 'ALL')
    job_id = str(uuid.uuid4())
    t = threading.Thread(target=_run_wolf_scan_job, args=(job_id, scan_filter))
    t.daemon = True
    t.start()
    return jsonify({'job_id': job_id, 'status': 'running'})

@app.route('/api/wolf-scan-result/<job_id>', methods=['GET'])
@login_required
def wolf_scan_result(job_id):
    job = _wolf_scan_jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not_found'}), 404
    if job['status'] == 'running':
        return jsonify({'status': 'running'})
    if job['status'] == 'error':
        return jsonify({'status': 'error', 'error': job.get('error', 'Unknown error')}), 500
    result = job['result']
    result['status'] = 'done'
    # Clean up old job
    _wolf_scan_jobs.pop(job_id, None)
    return jsonify(result)

# ── Admin ─────────────────────────────────────────────────────
@app.route('/make-me-admin/<secret>')
@login_required
def make_me_admin(secret):
    if secret != os.environ.get('ADMIN_SECRET', 'wolfadmin2026'):
        return 'Wrong secret', 403
    current_user.plan = 'admin'
    db.session.commit()
    return f'✅ {current_user.email} is now ADMIN — full access unlocked! <a href="/">Go to terminal</a>'
# ═══════════════════════════════════════════════════════════════
# WOLF STOCK SCANNER — Add these to app.py
# ═══════════════════════════════════════════════════════════════

# ── Stock Scanner watchlist (high volume, optionable stocks) ──
SCAN_UNIVERSE = [
    'AAPL','MSFT','NVDA','TSLA','AMD','META','GOOGL','AMZN',
    'NFLX','COIN','PLTR','MARA','HOOD','UBER','DIS','JPM',
    'GS','XOM','SMCI','AVGO','MU','CRM','SQ','PYPL'
]

# ── Market regime from SPY + VIX ─────────────────────────────
def get_market_regime():
    try:
        import yfinance as yf
        spy = yf.Ticker('SPY').history(period='5d', interval='1d', timeout=6)
        vix = yf.Ticker('^VIX').history(period='5d', interval='1d', timeout=6)

        spy_close = float(spy['Close'].iloc[-1])
        spy_prev  = float(spy['Close'].iloc[-2])
        spy_change = round(((spy_close - spy_prev) / spy_prev) * 100, 2)

        vix_level = float(vix['Close'].iloc[-1]) if not vix.empty else 20.0

        if vix_level > 30:   fear = 'EXTREME FEAR'; regime = 'BEARISH'
        elif vix_level > 20: fear = 'FEAR';          regime = 'CAUTION'
        elif vix_level > 15: fear = 'NEUTRAL';       regime = 'NEUTRAL'
        else:                fear = 'GREED';          regime = 'BULLISH'

        spy_trend = 'BULLISH' if spy_change > 0 else 'BEARISH'

        return {
            'spy_price': round(spy_close, 2),
            'spy_change': spy_change,
            'spy_trend': spy_trend,
            'vix': round(vix_level, 1),
            'fear_greed': fear,
            'regime': regime
        }
    except Exception as e:
        print(f'[MarketRegime] {e}')
        return {'spy_price': 0, 'spy_change': 0, 'spy_trend': 'UNKNOWN', 'vix': 20, 'fear_greed': 'NEUTRAL', 'regime': 'NEUTRAL'}

# ── Score a single stock ──────────────────────────────────────
def score_stock(ticker_sym):
    """
    Score a stock 0-100 based on:
    - Trend alignment (EMA20/50/200)
    - RSI momentum
    - Volume surge
    - IV Rank (options attractiveness)
    - Unusual options activity (vol/OI ratio)
    - No earnings within 3 days
    Returns dict with score and all data, or None if fails
    """
    try:
        import yfinance as yf
        import numpy as np
        from datetime import datetime, timedelta

        ticker = yf.Ticker(ticker_sym)

        # ── Price + Volume history ────────────────────────────
        hist = ticker.history(period='1y', interval='1d', timeout=8)
        if hist is None or hist.empty or len(hist) < 50:
            return None

        closes  = hist['Close'].tolist()
        volumes = hist['Volume'].tolist()
        current_price = round(closes[-1], 2)

        # ── EMAs ─────────────────────────────────────────────
        ema20  = calc_ema(closes, 20)
        ema50  = calc_ema(closes, 50)
        ema200 = calc_ema(closes, min(200, len(closes)))
        rsi    = calc_rsi(closes)
        macd_val, macd_bias = calc_macd(closes)

        # ── Volume surge ──────────────────────────────────────
        avg_vol_20 = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else 1
        today_vol  = volumes[-1]
        vol_ratio  = round(today_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0

        # ── IV Rank (from options chain) ──────────────────────
        iv_rank = None
        unusual_activity = 0.0
        call_vol = 0; put_vol = 0; total_oi = 0
        options_strategy = 'LONG CALL'

        try:
            import concurrent.futures as _cf
            def _get_opts():
                e = ticker.options
                if e: return ticker.option_chain(e[0])
                return None
            with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(_get_opts)
                _chain = _fut.result(timeout=5)
            if _chain:
                chain = _chain
                exps = ['dummy']  # signal that we got chain
            else:
                raise Exception('no options')
            if exps:
                # Use nearest expiration for IV scan
                calls = chain.calls
                puts  = chain.puts

                # IV from ATM options
                atm_calls = calls[abs(calls['strike'] - current_price) < current_price * 0.05]
                if not atm_calls.empty:
                    iv_now = float(atm_calls['impliedVolatility'].mean())
                else:
                    iv_now = float(calls['impliedVolatility'].mean()) if not calls.empty else 0.3

                # Volume and OI
                call_vol = int(calls['volume'].sum()) if not calls.empty else 0
                put_vol  = int(puts['volume'].sum())  if not puts.empty else 0
                call_oi  = int(calls['openInterest'].sum()) if not calls.empty else 1
                put_oi   = int(puts['openInterest'].sum())  if not puts.empty else 1
                total_oi = call_oi + put_oi

                # Unusual activity = volume/OI ratio (>1.0 = unusual)
                unusual_activity = round((call_vol + put_vol) / max(total_oi, 1), 3)

                # IV Rank approximation (compare to 52wk high/low IV proxy via price range)
                high_52 = max(hist['High'].tail(252).tolist())
                low_52  = min(hist['Low'].tail(252).tolist())
                price_range = high_52 - low_52
                iv_rank_approx = round(((current_price - low_52) / price_range) * 100) if price_range > 0 else 50
                iv_rank = iv_rank_approx

                # Strategy selection based on IV rank
                if iv_rank > 60:
                    options_strategy = 'CREDIT SPREAD'
                elif iv_rank < 30:
                    options_strategy = 'LONG CALL/PUT'
                else:
                    options_strategy = 'DEBIT SPREAD'

        except Exception as ex:
            print(f'[Options {ticker_sym}] {ex}')

        # ── Earnings check ────────────────────────────────────
        near_earnings = False
        try:
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                # earnings date
                earn_dates = cal.columns.tolist() if hasattr(cal, 'columns') else []
                if earn_dates:
                    earn_dt = datetime.strptime(str(earn_dates[0])[:10], '%Y-%m-%d')
                    days_to_earn = abs((earn_dt - datetime.now()).days)
                    near_earnings = days_to_earn <= 3
        except:
            pass

        # ── Sector ───────────────────────────────────────────
        sector = 'Unknown'
        try:
            info = ticker.fast_info
            sector = getattr(info, 'sector', 'Unknown') or 'Unknown'
        except:
            pass

        # ── Scoring ───────────────────────────────────────────
        score = 0
        direction = 'NEUTRAL'
        signals = []

        # Trend alignment (40 pts max)
        above_ema20  = current_price > (ema20  or 0)
        above_ema50  = current_price > (ema50  or 0)
        above_ema200 = current_price > (ema200 or 0)

        if above_ema200:
            score += 15; signals.append('Above EMA200 (bullish structure)')
        else:
            score -= 5;  signals.append('Below EMA200 (bearish structure)')

        if above_ema50:
            score += 12; signals.append(f'Above EMA50 ({round(ema50,2)})')
        if above_ema20:
            score += 8;  signals.append(f'Above EMA20 ({round(ema20,2)})')

        # RSI (20 pts)
        if rsi:
            if 45 <= rsi <= 65:
                score += 20; signals.append(f'RSI {rsi} — momentum building')
            elif 35 <= rsi < 45:
                score += 12; signals.append(f'RSI {rsi} — oversold bounce potential')
            elif rsi > 70:
                score -= 10; signals.append(f'RSI {rsi} — overbought warning')
            elif rsi < 30:
                score += 8;  signals.append(f'RSI {rsi} — deep oversold')

        # Volume surge (20 pts)
        if vol_ratio >= 2.5:
            score += 20; signals.append(f'Volume {vol_ratio}x average — strong conviction')
        elif vol_ratio >= 1.5:
            score += 12; signals.append(f'Volume {vol_ratio}x average — elevated')
        elif vol_ratio >= 1.2:
            score += 6;  signals.append(f'Volume {vol_ratio}x average — slightly elevated')

        # Unusual options activity (15 pts)
        if unusual_activity >= 1.5:
            score += 15; signals.append(f'Unusual options activity {unusual_activity}x — institutional interest')
        elif unusual_activity >= 0.8:
            score += 8;  signals.append(f'Options activity {unusual_activity}x — moderate')

        # MACD (5 pts)
        if macd_bias == 'BULLISH':
            score += 5; signals.append('MACD bullish crossover')
        else:
            signals.append('MACD bearish')

        # Earnings penalty
        if near_earnings:
            score -= 20; signals.append('⚠️ EARNINGS WITHIN 3 DAYS — high IV crush risk')

        # Direction
        bull_signals = sum([above_ema20, above_ema50, above_ema200, macd_bias == 'BULLISH'])
        if bull_signals >= 3:   direction = 'BULLISH'
        elif bull_signals <= 1: direction = 'BEARISH'
        else:                   direction = 'NEUTRAL'

        # Recommended option
        if direction == 'BULLISH':
            rec_option = 'BUY CALLS'
        elif direction == 'BEARISH':
            rec_option = 'BUY PUTS'
        else:
            rec_option = 'WAIT FOR SETUP'

        # S/R from daily candles
        candles = []
        for i, (ts, row) in enumerate(hist.tail(60).iterrows()):
            candles.append({'high': float(row['High']), 'low': float(row['Low']),
                           'open': float(row['Open']), 'close': float(row['Close'])})
        sr_levels = find_sr_levels(candles, current_price, lookback=60) if candles else []

        return {
            'ticker': ticker_sym,
            'price': current_price,
            'score': max(0, min(100, score)),
            'direction': direction,
            'rec_option': rec_option,
            'options_strategy': options_strategy,
            'signals': signals,
            'ema20': round(ema20, 2) if ema20 else None,
            'ema50': round(ema50, 2) if ema50 else None,
            'ema200': round(ema200, 2) if ema200 else None,
            'rsi': rsi,
            'macd_bias': macd_bias,
            'vol_ratio': vol_ratio,
            'iv_rank': iv_rank,
            'unusual_activity': unusual_activity,
            'call_vol': call_vol,
            'put_vol': put_vol,
            'near_earnings': near_earnings,
            'sector': sector,
            'sr_levels': sr_levels[:4],
            'options_strategy': options_strategy,
        }

    except Exception as e:
        print(f'[ScoreStock {ticker_sym}] {e}')
        return None

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
# (before the "if __name__ == '__main__':" line)
# ═══════════════════════════════════════════════════════════════

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

@app.route('/ai-infra')
@login_required
@elite_required
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
