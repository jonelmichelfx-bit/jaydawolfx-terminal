from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import anthropic
from flask_login import LoginManager, current_user, login_required
from models import db, User
from auth import auth_bp
from decorators import analysis_gate, basic_required, elite_required
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
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///jaydawolfx.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_DURATION'] = 60 * 60 * 24 * 30
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24 * 30
app.config['SESSION_PERMANENT'] = True

stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PRICES = {
    'basic': os.environ.get('STRIPE_BASIC_PRICE_ID', 'price_REPLACE_BASIC'),
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
    return redirect(url_for('login_page'))

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
    """
    Fetch real OHLC candles from Yahoo Finance — completely free.
    interval: '1h' (hourly), '1d' (daily), '1wk' (weekly)
    period:   '5d','1mo','3mo','6mo','1y'
    Returns list of candle dicts or empty list on failure.
    """
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
    """Calculate EMA from list of closes"""
    if len(closes) < period:
        return None
    k = 2.0 / (period + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = c * k + ema * (1 - k)
    return round(ema, 5)

def calc_rsi(closes, period=14):
    """Calculate RSI"""
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
    """Calculate MACD line - EMA12 minus EMA26"""
    if len(closes) < 26:
        return None, None
    ema12 = calc_ema(closes[-50:], 12)
    ema26 = calc_ema(closes[-50:], 26)
    if ema12 and ema26:
        macd = round(ema12 - ema26, 5)
        return macd, 'BULLISH' if macd > 0 else 'BEARISH'
    return None, None

def find_sr_levels(candles, current_price, lookback=50):
    """
    Find real support and resistance from actual swing highs/lows.
    Returns levels sorted by distance to current price.
    """
    if len(candles) < 5:
        return []

    recent = candles[-lookback:] if len(candles) > lookback else candles
    highs = [c['high'] for c in recent]
    lows  = [c['low']  for c in recent]
    closes = [c['close'] for c in recent]

    levels = []

    # Find swing highs (local maxima)
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            levels.append({'price': highs[i], 'type': 'swing_high', 'strength': 1})

    # Find swing lows (local minima)
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            levels.append({'price': lows[i], 'type': 'swing_low', 'strength': 1})

    # Add round numbers (psychological levels)
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

    # Cluster nearby levels (within 0.1% of each other)
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

    # Sort by distance to current price and take closest 6
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
    Full technical analysis from real candle data.
    Returns structured analysis for Claude to use.
    """
    analysis = {
        'pair': pair,
        'current_price': current_price,
        'daily': {}, 'weekly': {}, 'hourly': {},
        'sr_levels': [],
        'indicators': {},
        'trend_summary': {}
    }

    try:
        # ── DAILY candles (3 months) ──────────────────────────
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

            # Daily trend
            price_vs_ema200 = 'ABOVE' if (ema200 and current_price > ema200) else 'BELOW'
            daily_trend = 'BULLISH' if current_price > (ema50 or current_price) else 'BEARISH'

            # Recent daily candles summary
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

            # S/R from daily candles
            analysis['sr_levels'] = find_sr_levels(daily, current_price, lookback=60)

        # ── WEEKLY candles (1 year) ───────────────────────────
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

        # ── HOURLY candles (5 days) ───────────────────────────
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

            # Hourly S/R (closer levels)
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

        # ── Overall trend summary ─────────────────────────────
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
    """Format chart analysis into clean text for Claude prompt"""
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

    # Trend summary
    lines.append(f"\nTREND ALIGNMENT: {ts.get('overall','?')} ({ts.get('alignment','?')})")
    lines.append(f"  Weekly: {ts.get('weekly','?')} | Daily: {ts.get('daily','?')} | Hourly: {ts.get('hourly','?')}")

    # Weekly
    if w:
        lines.append(f"\nWEEKLY CHART (1 Year of data):")
        lines.append(f"  Trend: {w.get('trend','?')} | EMA20: {w.get('ema20','?')} | RSI: {w.get('rsi','?')}")
        lines.append(f"  52-Week High: {w.get('52wk_high','?')} | 52-Week Low: {w.get('52wk_low','?')}")
        lines.append(f"  Last 3 candles: {w.get('last_3_candles','?')}")

    # Daily
    if d:
        lines.append(f"\nDAILY CHART (3 Months of data):")
        lines.append(f"  Trend: {d.get('trend','?')} | EMA20: {d.get('ema20','?')} | EMA50: {d.get('ema50','?')} | EMA200: {d.get('ema200','?')}")
        lines.append(f"  Price vs EMA200: {d.get('price_vs_ema200','?')} | RSI: {d.get('rsi','?')} | MACD: {d.get('macd_bias','?')}")
        lines.append(f"  3-Month High: {d.get('3mo_high','?')} | 3-Month Low: {d.get('3mo_low','?')}")
        lines.append(f"  Last 5 candles: {d.get('last_5_candles','?')}")

    # Hourly
    if h:
        lines.append(f"\nHOURLY CHART (5 Days of data):")
        lines.append(f"  Trend: {h.get('trend','?')} | EMA20: {h.get('ema20','?')} | EMA50: {h.get('ema50','?')}")
        lines.append(f"  RSI: {h.get('rsi','?')} | MACD: {h.get('macd_bias','?')}")
        lines.append(f"  24hr High: {h.get('recent_high','?')} | 24hr Low: {h.get('recent_low','?')}")
        lines.append(f"  Last 4 candles: {h.get('last_4_candles','?')}")

    # S/R levels
    if sr:
        lines.append(f"\nREAL SUPPORT & RESISTANCE (from actual swing highs/lows):")
        for lv in sr[:6]:
            lines.append(f"  {lv['type']}: {lv['price']} — {lv['note']} ({lv['distance_pips']} pips away)")

    # Hourly S/R
    h_sr = h.get('sr_levels', [])
    if h_sr:
        lines.append(f"\nHOURLY S/R (intraday levels):")
        for lv in h_sr[:4]:
            lines.append(f"  {lv['type']}: {lv['price']} ({lv['distance_pips']} pips)")

    lines.append(f"{'='*60}\n")
    return '\n'.join(lines)

def get_multi_pair_chart_data(pairs, current_prices):
    """Fetch chart data for multiple pairs in parallel"""
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
    if plan not in ('basic','elite'): flash('Invalid plan selected.','danger'); return redirect(url_for('pricing'))
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
    """
    Get live price using yfinance — same library as candlesticks.
    100% free, accurate, no API key needed.
    """
    try:
        import yfinance as yf
        sym = YF_MAP.get(symbol, symbol.replace('/', '') + '=X')
        ticker = yf.Ticker(sym)
        # Get today's data
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

    # Fallback to Twelve Data if yfinance fails
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

    # Last resort — fallback prices
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
    msg = client.messages.create(model='claude-opus-4-6', max_tokens=max_tokens,
                                  messages=[{'role':'user','content':prompt}])
    return msg.content[0].text

def parse_json_response(text):
    """Parse JSON — handles markdown fences and truncation gracefully"""
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
def forex(): return render_template('forex.html')

@app.route('/forex-wolf')
@login_required
def forex_wolf(): return render_template('forex_wolf.html')

@app.route('/wolf-scanner')
@login_required
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
    """Deep analysis with REAL candlestick data"""
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

                # Add REAL chart analysis
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
    """7 best trades with REAL chart data"""
    try:
        date_str = datetime.now().strftime('%A, %B %d, %Y')
        session_name, _ = get_session()
        scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY']
        prices = get_prices_parallel(scan_pairs)
        news = get_news()

        # Fetch real chart data for all pairs in parallel
        chart_data = get_multi_pair_chart_data(scan_pairs, prices)

        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p,v in prices.items()])
        news_str = '\n'.join([f"- {n['title']}" for n in news[:5]]) or "- Markets await key economic data"

        # Build real chart context for each pair
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
    """Top 3 day trades with REAL hourly candle data"""
    try:
        date_str = datetime.now().strftime('%A, %B %d, %Y')
        session_name, _ = get_session()
        scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','USD/CHF']
        prices = get_prices_parallel(scan_pairs)
        news = get_news()

        # Fetch real chart data
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
    """Top 3 swing trades with REAL weekly/daily candle data"""
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

@app.route('/api/forex-picks', methods=['POST'])
@login_required
def forex_picks():
    try:
        prompt = request.get_json().get('prompt', '')
        return jsonify({'content': call_claude(prompt, 2200)})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/wolf-scan', methods=['POST'])
@login_required
def wolf_scan():
    """Wolf Scanner with REAL candlestick data — most accurate analysis"""
    try:
        data = request.get_json() or {}
        scan_filter = data.get('filter', 'ALL')
        date_str = datetime.now().strftime('%A, %B %d, %Y')
        session_name, _ = get_session()
        prices, is_live = get_cached_prices()

        if scan_filter == 'MAJORS':   scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','USD/CHF','AUD/USD','USD/CAD']
        elif scan_filter == 'GOLD':   scan_pairs = ['XAU/USD','EUR/USD','USD/JPY','AUD/USD','NZD/USD']
        elif scan_filter == 'ASIAN':  scan_pairs = ['USD/JPY','EUR/JPY','GBP/JPY','AUD/USD','NZD/USD']
        elif scan_filter == 'LONDON': scan_pairs = ['EUR/USD','GBP/USD','EUR/GBP','USD/CHF','EUR/JPY']
        else: scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','NZD/USD','USD/CHF']

        # Build price string
        prices_lines = []
        for p in scan_pairs:
            q = prices.get(p)
            if q:
                dp = 2 if 'JPY' in p or p == 'XAU/USD' else 5
                prices_lines.append(f"{p}: {float(q['price']):.{dp}f} (H:{float(q['high']):.{dp}f} L:{float(q['low']):.{dp}f} Chg:{float(q.get('percent_change',0)):+.2f}%)")
        prices_str = '\n'.join(prices_lines)

        news_items = get_news()
        news_str = '\n'.join([f"- {n['title']} ({n['source']}, {n['published']})" for n in news_items[:6]]) or "- Monitor key economic events"

        # Fetch REAL chart data for all scan pairs
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

        result = parse_json_response(call_claude(prompt, 6000))

        # Override prices with confirmed live data
        for trade in result.get('trades', []):
            pair = trade.get('pair', '')
            q = prices.get(pair)
            if q:
                dp = 2 if 'JPY' in pair or pair == 'XAU/USD' else 4
                trade['current_price'] = f"{float(q['price']):.{dp}f}"
            # Inject real S/R levels if chart data available
            if pair in chart_data and chart_data[pair].get('sr_levels'):
                trade['real_sr_levels'] = chart_data[pair]['sr_levels'][:6]

        return jsonify(result)
    except json.JSONDecodeError as e: return jsonify({'error': f'AI analysis error — try again ({str(e)[:50]})'}), 500
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# ── Admin ─────────────────────────────────────────────────────
@app.route('/make-me-admin/<secret>')
@login_required
def make_me_admin(secret):
    if secret != os.environ.get('ADMIN_SECRET', 'wolfadmin2026'):
        return 'Wrong secret', 403
    current_user.plan = 'admin'
    db.session.commit()
    return f'✅ {current_user.email} is now ADMIN — full access unlocked! <a href="/">Go to terminal</a>'

if __name__ == '__main__':
    app.run(debug=True, port=5000)
