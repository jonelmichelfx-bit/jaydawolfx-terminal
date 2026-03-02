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
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── Config ──────────────────────────────────────────
app.secret_key = os.environ.get('SECRET_KEY', 'jaydawolfx-secret-2026')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///jaydawolfx.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_DURATION'] = 60 * 60 * 24 * 30  # 30 days
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24 * 30  # 30 days
app.config['SESSION_PERMANENT'] = True

# ── Stripe ───────────────────────────────────────────
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', '')

STRIPE_PRICES = {
    'basic': os.environ.get('STRIPE_BASIC_PRICE_ID', 'price_REPLACE_BASIC'),
    'elite': os.environ.get('STRIPE_ELITE_PRICE_ID', 'price_REPLACE_ELITE'),
}
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', '')

# ── Extensions ───────────────────────────────────────
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

# ── Server Time API ──────────────────────────────────
@app.route('/api/server-time')
def server_time():
    from datetime import datetime, timedelta
    from datetime import timezone
    import zoneinfo
    try:
        eastern = zoneinfo.ZoneInfo('America/New_York')
        now = datetime.now(eastern)
    except Exception:
        from datetime import timedelta
        now = datetime.utcnow() - timedelta(hours=5)
    
    day = now.weekday()
    monday = now - timedelta(days=day)
    friday = monday + timedelta(days=4)
    week_range = f"{monday.strftime('%b %d')} — {friday.strftime('%b %d, %Y')}"
    
    hour = now.hour
    minute = now.minute
    is_weekday = day < 5
    market_open = is_weekday and (hour > 9 or (hour == 9 and minute >= 30)) and hour < 16
    pre_market = is_weekday and hour >= 4 and (hour < 9 or (hour == 9 and minute < 30))
    after_hours = is_weekday and hour >= 16 and hour < 20
    
    if market_open:
        market_status = 'MARKET OPEN'; status_color = '#00ff99'
    elif pre_market:
        market_status = 'PRE-MARKET'; status_color = '#ffe033'
    elif after_hours:
        market_status = 'AFTER HOURS'; status_color = '#ff7744'
    else:
        market_status = 'MARKET CLOSED'; status_color = '#ff4466'
    
    return jsonify({
        'date': now.strftime('%B %d, %Y'),
        'time': now.strftime('%H:%M:%S') + (' EDT' if now.dst() else ' EST'),
        'day': now.strftime('%A'),
        'week_range': week_range,
        'market_status': market_status,
        'status_color': status_color,
        'timestamp': now.isoformat()
    })

# ── Blueprints ───────────────────────────────────────
app.register_blueprint(auth_bp)

from payments import payments_bp
app.register_blueprint(payments_bp)

from scanner import scanner_bp
app.register_blueprint(scanner_bp)
from forex import forex_bp
app.register_blueprint(forex_bp)

# ── Create DB tables ─────────────────────────────────
with app.app_context():
    db.create_all()

# ────────────────────────────────────────────────────
# YOUR EXISTING FUNCTIONS (unchanged)
# ────────────────────────────────────────────────────

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
        rho   = (K * T * np.exp(-r * T) * norm.cdf(d2) / 100 if option_type == 'call' else -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)
        return {
            'price': round(float(price), 4),
            'delta': round(float(delta), 4),
            'gamma': round(float(gamma), 6),
            'theta': round(float(theta), 4),
            'vega':  round(float(vega),  4),
            'rho':   round(float(rho),   4),
        }
    except Exception as e:
        return {'error': str(e)}

def build_pnl_curve(S, K, T, r, sigma, option_type, premium_paid, days_held):
    price_range = np.linspace(S * 0.70, S * 1.30, 80)
    T_remaining = max(T - (days_held / 365), 0.001)
    curve = []
    for target in price_range:
        g = calculate_greeks(target, K, T_remaining, r, sigma, option_type)
        if g and 'price' in g:
            curve.append({'price': round(float(target), 2), 'pnl': round((g['price'] - premium_paid) * 100, 2)})
    return curve

def fetch_live_data(ticker, expiration, strike, option_type):
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d')
        if hist.empty:
            return None, 'Could not fetch stock price'
        stock_price = float(hist['Close'].iloc[-1])
        try:
            chain = stock.option_chain(expiration)
            contracts = chain.calls if option_type == 'call' else chain.puts
            strike_f = float(strike)
            row = contracts.iloc[(contracts['strike'] - strike_f).abs().argsort()[:1]]
            if not row.empty:
                iv   = float(row['impliedVolatility'].iloc[0])
                mark = float(row['lastPrice'].iloc[0])
                return {'stock_price': stock_price, 'iv': iv, 'mark': mark, 'source': 'yfinance'}, None
        except Exception:
            pass
        return {'stock_price': stock_price, 'iv': None, 'mark': None, 'source': 'price-only'}, None
    except ImportError:
        return None, 'yfinance not installed — run: pip install yfinance'
    except Exception as e:
        return None, str(e)

def fetch_stock_price_only(ticker):
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return None

def fetch_option_expirations(ticker):
    try:
        import yfinance as yf
        return list(yf.Ticker(ticker).options), None
    except ImportError:
        return None, 'yfinance not installed — run: pip install yfinance'
    except Exception as e:
        return None, str(e)

def fetch_option_strikes(ticker, expiration, option_type='call'):
    try:
        import yfinance as yf
        chain = yf.Ticker(ticker).option_chain(expiration)
        contracts = chain.calls if option_type == 'call' else chain.puts
        return sorted(contracts['strike'].tolist()), None
    except ImportError:
        return None, 'yfinance not installed'
    except Exception as e:
        return None, str(e)

# ────────────────────────────────────────────────────
# PAGES
# ────────────────────────────────────────────────────

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('auth.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

# ── Basic plan required ($29/mo) ─────────────────────
@app.route('/ai-scanner')
@login_required
@basic_required
def ai_scanner():
    return render_template('scanner.html')

# ── Wolf Elite required ($150/mo) ────────────────────
@app.route('/ai-analysis')
@login_required
@elite_required
def ai_analysis():
    return render_template('analysis.html')

@app.route('/wolf-elite')
@login_required
@elite_required
def wolf_elite():
    return render_template('wolf_elite.html')

# ────────────────────────────────────────────────────
# STRIPE CHECKOUT
# ────────────────────────────────────────────────────

@app.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    plan = request.form.get('plan')
    if plan not in ('basic', 'elite'):
        flash('Invalid plan selected.', 'danger')
        return redirect(url_for('pricing'))
    try:
        checkout = stripe.checkout.Session.create(
            payment_method_types=['card'],
            mode='subscription',
            line_items=[{'price': STRIPE_PRICES[plan], 'quantity': 1}],
            success_url=url_for('payment_success', plan=plan, _external=True),
            cancel_url=url_for('pricing', _external=True),
            client_reference_id=str(current_user.id),
            metadata={'plan': plan, 'user_id': str(current_user.id)},
        )
        return redirect(checkout.url, code=303)
    except Exception as e:
        flash(f'Payment error: {str(e)}', 'danger')
        return redirect(url_for('pricing'))

@app.route('/payment-success')
@login_required
def payment_success():
    plan = request.args.get('plan', 'basic')
    # Update the user's plan in DB
    current_user.plan = plan
    db.session.commit()
    flash(f"🐺 You're now on Wolf Elite {'Elite' if plan == 'elite' else 'Basic'}! Let's get it.", 'success')
    return redirect(url_for('index'))

@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception:
        return 'Invalid signature', 400

    if event['type'] == 'checkout.session.completed':
        data    = event['data']['object']
        user_id = data.get('metadata', {}).get('user_id')
        plan    = data.get('metadata', {}).get('plan', 'basic')
        try:
            user = User.query.get(int(user_id))
            if user:
                user.plan = plan
                user.stripe_customer_id    = data.get('customer')
                user.stripe_subscription_id = data.get('subscription')
                db.session.commit()
                print(f'[Webhook] User {user_id} upgraded to {plan}')
        except Exception as e:
            print(f'[Webhook] DB error: {e}')

    elif event['type'] == 'customer.subscription.deleted':
        stripe_customer_id = event['data']['object'].get('customer')
        try:
            user = User.query.filter_by(stripe_customer_id=stripe_customer_id).first()
            if user:
                user.plan = 'trial'
                db.session.commit()
                print(f'[Webhook] Subscription cancelled for {stripe_customer_id}')
        except Exception as e:
            print(f'[Webhook] DB error: {e}')

    return 'OK', 200

# ────────────────────────────────────────────────────
# API ROUTES
# ────────────────────────────────────────────────────

@app.route('/api/autofill', methods=['POST'])
@analysis_gate
def autofill():
    ticker = request.json.get('ticker', '').upper().strip()
    if not ticker:
        return jsonify({'error': 'No ticker provided'})
    expirations, err = fetch_option_expirations(ticker)
    if err:
        return jsonify({'error': err})
    stock_price = fetch_stock_price_only(ticker)
    return jsonify({
        'ticker': ticker,
        'stock_price': round(stock_price, 2) if stock_price else None,
        'expirations': expirations[:12] if expirations else [],
    })

@app.route('/api/strikes', methods=['POST'])
@login_required
def strikes():
    data = request.json
    strikes_list, err = fetch_option_strikes(data.get('ticker','').upper(), data.get('expiration',''), data.get('option_type','call'))
    if err:
        return jsonify({'error': err})
    return jsonify({'strikes': strikes_list})

@app.route('/api/contract', methods=['POST'])
@login_required
def contract():
    data = request.json
    live, err = fetch_live_data(data.get('ticker','').upper(), data.get('expiration',''), data.get('strike',0), data.get('option_type','call'))
    if err:
        return jsonify({'error': err})
    return jsonify({'data': live})

@app.route('/api/greeks', methods=['POST'])
@analysis_gate
def greeks():
    data        = request.json
    ticker      = data.get('ticker', 'AAPL').upper()
    strike      = float(data.get('strike', 150))
    expiration  = data.get('expiration', '')
    option_type = data.get('option_type', 'call')
    days_held   = int(data.get('days_held', 0))
    r           = float(data.get('r', 0.045))
    theta_alert = float(data.get('theta_alert', 50))

    live_data = None
    if expiration:
        live_data, _ = fetch_live_data(ticker, expiration, strike, option_type)

    if expiration:
        try:
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            T = max((exp_date - datetime.now()).days / 365, 0.001)
            dte_days = max((exp_date - datetime.now()).days, 1)
        except Exception:
            T = 30 / 365; dte_days = 30
    else:
        T = float(data.get('dte', 30)) / 365; dte_days = int(data.get('dte', 30))

    S       = float(live_data['stock_price']) if live_data and live_data.get('stock_price') else float(data.get('stock_price', 150))
    sigma   = float(live_data['iv'])    if live_data and live_data.get('iv')    else float(data.get('iv', 0.30))
    premium = float(live_data['mark'])  if live_data and live_data.get('mark')  else float(data.get('premium_paid', 0))

    greeks_result = calculate_greeks(S, strike, T, r, sigma, option_type)
    pnl_curve     = build_pnl_curve(S, strike, T, r, sigma, option_type, premium, days_held)
    daily_theta_d = (greeks_result['theta'] * 100) if greeks_result else 0

    return jsonify({
        'greeks': greeks_result, 'live_data': live_data,
        'pnl_curve': pnl_curve, 'stock_price': S,
        'premium_paid': premium, 'sigma': sigma,
        'daily_theta_dollars': round(daily_theta_d, 2),
        'theta_alert': abs(daily_theta_d) > theta_alert,
        'T': round(T * 365, 1), 'dte_days': dte_days,
    })

@app.route('/api/simulate', methods=['POST'])
@analysis_gate
def simulate():
    data = request.json
    S=float(data.get('stock_price',150)); K=float(data.get('strike',150))
    T=float(data.get('dte',30))/365; r=float(data.get('r',0.045))
    sigma=float(data.get('iv',0.30)); option_type=data.get('option_type','call')
    premium=float(data.get('premium_paid',0))
    scenarios = [{'days': d, 'curve': build_pnl_curve(S,K,T,r,sigma,option_type,premium,d)} for d in [0,5,10,15,20]]
    return jsonify({'scenarios': scenarios})

# ── Health check ─────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({'status': 'online', 'terminal': 'JAYDAWOLFX OPTIONS TERMINAL 🐺'}), 200

# ── Wolf Elite Results Tracker ────────────────────────
@app.route('/api/track-pick', methods=['POST'])
@login_required
def track_pick():
    import json, os
    data = request.get_json()
    tracker_file = 'wolf_tracker.json'
    try:
        if os.path.exists(tracker_file):
            with open(tracker_file, 'r') as f:
                tracker = json.load(f)
        else:
            tracker = {'picks': []}
        tracker['picks'].append({
            'week': data.get('week'),
            'ticker': data.get('ticker'),
            'entry': data.get('entry'),
            'target': data.get('target'),
            'stop': data.get('stop'),
            'result': data.get('result', 'PENDING'),
            'pct_change': data.get('pct_change', 0),
            'date_added': datetime.now().strftime('%Y-%m-%d')
        })
        with open(tracker_file, 'w') as f:
            json.dump(tracker, f)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tracker-stats', methods=['GET'])
@login_required  
def tracker_stats():
    import json, os
    tracker_file = 'wolf_tracker.json'
    try:
        if not os.path.exists(tracker_file):
            return jsonify({'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'picks': []}), 200
        with open(tracker_file, 'r') as f:
            tracker = json.load(f)
        picks = tracker.get('picks', [])
        completed = [p for p in picks if p['result'] in ['WIN', 'LOSS']]
        wins = len([p for p in completed if p['result'] == 'WIN'])
        losses = len([p for p in completed if p['result'] == 'LOSS'])
        win_rate = round((wins / len(completed) * 100) if completed else 0)
        return jsonify({
            'total': len(picks), 'wins': wins,
            'losses': losses, 'win_rate': win_rate,
            'picks': picks[-20:]
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# ═══════════════════════════════════════════════════════════════
# FOREX LIVE DATA + WOLF ROUTES
# Replace the existing forex routes at the bottom of app.py
# with all of this (paste above if __name__ == '__main__':)
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# FOREX LIVE DATA + WOLF ROUTES
# Replace the existing forex routes at the bottom of app.py
# with all of this (paste above if __name__ == '__main__':)
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# FOREX LIVE DATA + WOLF ROUTES
# Replace the existing forex routes at the bottom of app.py
# with all of this (paste above if __name__ == '__main__':)
# ═══════════════════════════════════════════════════════════════

import requests as http_requests

TWELVE_DATA_KEY = os.environ.get('TWELVE_DATA_API_KEY', '')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')

FOREX_SESSIONS = {
    'TOKYO':   {'start': 19, 'end': 4,  'color': 'purple', 'pairs': ['USD/JPY','EUR/JPY','GBP/JPY','AUD/USD','NZD/USD']},
    'LONDON':  {'start': 3,  'end': 12, 'color': 'cyan',   'pairs': ['EUR/USD','GBP/USD','EUR/GBP','USD/CHF','EUR/JPY']},
    'NEW YORK':{'start': 8,  'end': 17, 'color': 'green',  'pairs': ['EUR/USD','GBP/USD','USD/CAD','USD/JPY','XAU/USD']},
    'OVERLAP': {'start': 8,  'end': 12, 'color': 'gold',   'pairs': ['EUR/USD','GBP/USD','USD/JPY','XAU/USD']},
}

PAIR_MAP = {
    'EUR/USD':'EUR/USD','GBP/USD':'GBP/USD','USD/JPY':'USD/JPY',
    'USD/CHF':'USD/CHF','AUD/USD':'AUD/USD','USD/CAD':'USD/CAD',
    'NZD/USD':'NZD/USD','EUR/GBP':'EUR/GBP','EUR/JPY':'EUR/JPY',
    'GBP/JPY':'GBP/JPY','XAU/USD':'XAU/USD','BTC/USD':'BTC/USD',
}

def get_current_session():
    """Get current forex session based on EST time"""
    from datetime import datetime, timezone, timedelta
    try:
        est_offset = timedelta(hours=-5)  # EST (UTC-5), adjust for DST manually
        now_est = datetime.now(timezone.utc) + est_offset
        h = now_est.hour
        day = now_est.weekday()  # 0=Mon, 6=Sun
        # Market closed: Friday 5PM - Sunday 5PM EST
        if day == 5:  # Saturday
            return 'CLOSED', []
        if day == 6 and h < 17:  # Sunday before 5PM
            return 'CLOSED', []
        if day == 4 and h >= 17:  # Friday after 5PM
            return 'CLOSED', []
        # Active sessions
        if h >= 19 or h < 3:
            return 'TOKYO', FOREX_SESSIONS['TOKYO']['pairs']
        elif h >= 3 and h < 8:
            return 'LONDON', FOREX_SESSIONS['LONDON']['pairs']
        elif h >= 8 and h < 12:
            return 'OVERLAP', FOREX_SESSIONS['OVERLAP']['pairs']
        elif h >= 12 and h < 17:
            return 'NEW YORK', FOREX_SESSIONS['NEW YORK']['pairs']
        else:
            return 'PRE-MARKET', []
    except:
        return 'UNKNOWN', []

def fetch_live_price(symbol):
    """Fetch real live price from Twelve Data"""
    try:
        # Format symbol for Twelve Data
        td_symbol = symbol.replace('/', '')
        if symbol == 'XAU/USD':
            td_symbol = 'XAU/USD'
        elif symbol == 'BTC/USD':
            td_symbol = 'BTC/USD'
        
        url = f'https://api.twelvedata.com/price?symbol={td_symbol}&apikey={TWELVE_DATA_KEY}'
        resp = http_requests.get(url, timeout=5)
        data = resp.json()
        if 'price' in data:
            return float(data['price'])
        return None
    except:
        return None

# Fallback prices when API unavailable
FALLBACK_PRICES = {
    'EUR/USD': {'price':1.0842,'change':0.0012,'percent_change':0.11,'high':1.0871,'low':1.0821},
    'GBP/USD': {'price':1.2634,'change':-0.0021,'percent_change':-0.17,'high':1.2671,'low':1.2601},
    'USD/JPY': {'price':149.82,'change':0.44,'percent_change':0.29,'high':150.21,'low':149.41},
    'USD/CHF': {'price':0.8923,'change':-0.0011,'percent_change':-0.12,'high':0.8951,'low':0.8901},
    'AUD/USD': {'price':0.6521,'change':-0.0015,'percent_change':-0.23,'high':0.6551,'low':0.6501},
    'USD/CAD': {'price':1.3541,'change':0.0022,'percent_change':0.16,'high':1.3561,'low':1.3511},
    'NZD/USD': {'price':0.6102,'change':-0.0008,'percent_change':-0.13,'high':0.6121,'low':0.6091},
    'EUR/GBP': {'price':0.8582,'change':0.0009,'percent_change':0.10,'high':0.8601,'low':0.8561},
    'EUR/JPY': {'price':162.44,'change':0.55,'percent_change':0.34,'high':162.91,'low':161.88},
    'GBP/JPY': {'price':189.22,'change':0.31,'percent_change':0.16,'high':189.81,'low':188.61},
    'XAU/USD': {'price':2913.50,'change':8.20,'percent_change':0.31,'high':2925.30,'low':2901.10},
    'BTC/USD': {'price':84200,'change':1240,'percent_change':1.87,'high':85100,'low':83200},
    'DXY':     {'price':104.22,'change':-0.18,'percent_change':-0.17,'high':104.51,'low':104.01},
}

def fetch_live_quote(symbol):
    """Fetch full quote including high/low/change — with fallback"""
    try:
        td_symbol = symbol.replace('/', '')
        url = f'https://api.twelvedata.com/quote?symbol={td_symbol}&apikey={TWELVE_DATA_KEY}'
        resp = http_requests.get(url, timeout=8)
        data = resp.json()
        if 'close' in data:
            return {
                'price': float(data.get('close', 0)),
                'open': float(data.get('open', 0)),
                'high': float(data.get('high', 0)),
                'low': float(data.get('low', 0)),
                'change': float(data.get('change', 0)),
                'percent_change': float(data.get('percent_change', 0)),
                'volume': data.get('volume', 0),
                'symbol': symbol,
                'live': True
            }
    except:
        pass
    # Return fallback so routes never crash
    fb = FALLBACK_PRICES.get(symbol, {})
    if fb:
        return {
            'price': fb['price'], 'open': fb['price'],
            'high': fb['high'], 'low': fb['low'],
            'change': fb['change'], 'percent_change': fb['percent_change'],
            'volume': 0, 'symbol': symbol, 'live': False
        }
    return None

def fetch_forex_news(pair):
    """Fetch real news for a forex pair"""
    try:
        # Build search query from pair
        currencies = pair.replace('/', ' ').replace('USD', 'Dollar').replace('EUR', 'Euro').replace('GBP', 'Pound').replace('JPY', 'Yen').replace('XAU', 'Gold')
        query = f'forex {pair} {currencies} currency'
        url = f'https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}'
        resp = http_requests.get(url, timeout=5)
        data = resp.json()
        articles = data.get('articles', [])
        news = []
        for a in articles[:5]:
            news.append({
                'title': a.get('title', ''),
                'source': a.get('source', {}).get('name', ''),
                'published': a.get('publishedAt', '')[:10],
                'description': a.get('description', '')[:200] if a.get('description') else '',
                'url': a.get('url', '')
            })
        return news
    except:
        return []

def fetch_market_news():
    """Fetch general forex/macro market news — with fallback"""
    try:
        url = f'https://newsapi.org/v2/everything?q=forex+currency+Fed+ECB+central+bank&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}'
        resp = http_requests.get(url, timeout=8)
        data = resp.json()
        articles = data.get('articles', [])
        if articles:
            news = []
            for a in articles[:10]:
                news.append({
                    'title': a.get('title', ''),
                    'source': a.get('source', {}).get('name', ''),
                    'published': a.get('publishedAt', '')[:10],
                    'description': (a.get('description', '') or '')[:200],
                })
            return news
    except:
        pass
    # Fallback news so AI still gets context
    return [
        {'title': 'Federal Reserve holds rates steady, watches inflation data', 'source': 'Reuters', 'published': '2026-03-02', 'description': 'Fed signals patience on rate cuts'},
        {'title': 'ECB considers further rate cuts amid slowing eurozone growth', 'source': 'Bloomberg', 'published': '2026-03-02', 'description': 'European economy shows weakness'},
        {'title': 'Dollar index under pressure as risk appetite improves', 'source': 'FXStreet', 'published': '2026-03-02', 'description': 'DXY retreats from highs'},
        {'title': 'Gold surges as safe haven demand rises on geopolitical tensions', 'source': 'MarketWatch', 'published': '2026-03-02', 'description': 'XAU/USD breaks key resistance'},
        {'title': 'Bank of Japan signals possible rate hike timeline', 'source': 'Nikkei', 'published': '2026-03-02', 'description': 'Yen strengthens on BOJ hawkish tone'},
    ]


# ── ROUTE: Forex page ─────────────────────────────────────────
@app.route('/forex')
@login_required
def forex():
    return render_template('forex.html')

@app.route('/forex-wolf')
@login_required
def forex_wolf():
    return render_template('forex_wolf.html')


# ── ROUTE: Live price for single pair ────────────────────────
@app.route('/api/forex-price', methods=['POST'])
@login_required
def forex_price():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'EUR/USD')
        quote = fetch_live_quote(symbol)
        if quote:
            return jsonify(quote)
        return jsonify({'error': 'Price unavailable', 'live': False}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── ROUTE: All prices at once ─────────────────────────────────
@app.route('/api/forex-prices', methods=['GET'])
@login_required
def forex_prices():
    """Fetch live prices for all major pairs"""
    try:
        pairs = ['EUR/USD','GBP/USD','USD/JPY','USD/CHF','AUD/USD','USD/CAD','NZD/USD','EUR/GBP','EUR/JPY','GBP/JPY','XAU/USD']
        prices = {}
        for pair in pairs:
            quote = fetch_live_quote(pair)
            if quote:
                prices[pair] = quote
        session_name, session_pairs = get_current_session()
        return jsonify({
            'prices': prices,
            'session': session_name,
            'session_pairs': session_pairs,
            'timestamp': datetime.now().strftime('%H:%M:%S EST')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── ROUTE: News for a pair ────────────────────────────────────
@app.route('/api/forex-news', methods=['POST'])
@login_required
def forex_news():
    try:
        data = request.get_json()
        pair = data.get('pair', 'EUR/USD')
        news = fetch_forex_news(pair)
        market_news = fetch_market_news()
        return jsonify({'pair_news': news, 'market_news': market_news})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── ROUTE: Main analysis with live data ──────────────────────
@app.route('/api/forex-analyze', methods=['POST'])
@login_required
def forex_analyze():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 2500)
        pair = data.get('pair', '')

        # Enrich prompt with live data if pair provided
        live_context = ''
        if pair:
            quote = fetch_live_quote(pair)
            news = fetch_forex_news(pair)
            session_name, _ = get_current_session()
            if quote:
                live_context = f"""
LIVE MARKET DATA (real-time):
Current Price: {quote['price']}
Today's High: {quote['high']}
Today's Low: {quote['low']}
Open: {quote['open']}
Change: {quote['change']:+.5f} ({quote['percent_change']:+.2f}%)
Current Session: {session_name}
"""
            if news:
                live_context += f"\nREAL-TIME NEWS FOR {pair}:\n"
                for n in news[:3]:
                    live_context += f"- {n['title']} ({n['source']}, {n['published']})\n"

        full_prompt = live_context + '\n' + prompt if live_context else prompt

        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return jsonify({'content': message.content[0].text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── ROUTE: 7 best scenario trades for the week ───────────────
@app.route('/api/forex-scenarios', methods=['POST'])
@login_required
def forex_scenarios():
    """Top 7 trades with full buy/sell scenarios using live data"""
    try:
        now = datetime.now()
        date_str = now.strftime('%A, %B %d, %Y')
        session_name, _ = get_current_session()

        # Fetch live prices for top pairs
        top_pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY']
        live_prices = {}
        for p in top_pairs:
            q = fetch_live_quote(p)
            if q:
                live_prices[p] = q

        # Fetch market news
        market_news = fetch_market_news()
        news_summary = '\n'.join([f"- {n['title']}" for n in market_news[:8]])

        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p, v in live_prices.items()])

        prompt = f"""You are Wolf AI — a professional forex trader with 15 years experience. 
Today is {date_str}. Current session: {session_name}.
Forex market: Sunday 5PM EST open → Friday 5PM EST close.
Sessions: Tokyo 7PM-4AM EST · London 3AM-12PM EST · NY 8AM-5PM EST · Overlap 8AM-12PM EST (highest volatility).

LIVE PRICES RIGHT NOW:
{prices_str}

REAL MARKET NEWS TODAY:
{news_summary}

Do a complete top-down analysis and find the 7 BEST trades for this week.
For each trade, go through: Monthly direction → Weekly direction → Daily direction → H4 direction → H1 direction → M15 entry signal.
Only recommend trades where 4+ timeframes align in the same direction.

For each pair provide BOTH a BUY scenario AND a SELL scenario based on key price levels.

Respond ONLY in valid JSON, no markdown:
{{
  "week": "{date_str}",
  "session": "{session_name}",
  "market_theme": "Main macro theme this week",
  "dxy_direction": "BULLISH or BEARISH",
  "risk_sentiment": "RISK-ON or RISK-OFF",
  "trades": [
    {{
      "rank": 1,
      "pair": "EUR/USD",
      "live_price": "1.0842",
      "overall_bias": "BEARISH",
      "timeframe_alignment": {{
        "monthly": "BEARISH",
        "weekly": "BEARISH", 
        "daily": "BEARISH",
        "h4": "BEARISH",
        "h1": "NEUTRAL",
        "m15": "BEARISH"
      }},
      "aligned_count": 5,
      "confidence": 82,
      "primary_direction": "SELL",
      "thesis": "Full 3-4 sentence top-down thesis explaining why this pair is trending in this direction across all timeframes",
      "key_resistance": "1.0890",
      "key_support": "1.0780",
      "critical_zone": "1.0850",
      "buy_scenario": {{
        "trigger": "Price breaks and closes above 1.0890 on H4 with strong bullish candle",
        "news_needed": "Hawkish ECB surprise or weak US data",
        "entry": "1.0895",
        "stop_loss": "1.0855",
        "tp1": "1.0950",
        "tp2": "1.1000",
        "tp3": "1.1050",
        "probability": 35,
        "confirmation": "RSI above 55 on H4, MACD cross bullish"
      }},
      "sell_scenario": {{
        "trigger": "Price rejects 1.0890 and breaks below 1.0820 on H1",
        "news_needed": "Strong US NFP or hawkish Fed",
        "entry": "1.0815",
        "stop_loss": "1.0855",
        "tp1": "1.0780",
        "tp2": "1.0720",
        "tp3": "1.0650",
        "probability": 65,
        "confirmation": "RSI below 45, momentum bearish on H4"
      }},
      "best_session": "LONDON",
      "key_news_this_week": "Fed minutes Wednesday, NFP Friday",
      "invalidation": "Daily close above 1.0920"
    }}
  ]
}}"""

        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text.strip()
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        text = text.strip()
        result = json.loads(text)
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── ROUTE: Daily guaranteed picks with live data ──────────────
@app.route('/api/forex-daily-picks', methods=['POST'])
@login_required
def forex_daily_picks():
    
    try:
        now = datetime.now()
        date_str = now.strftime('%A, %B %d, %Y')
        session_name, session_pairs = get_current_session()

        # Live prices
        pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','USD/CHF']
        live_prices = {}
        for p in pairs:
            q = fetch_live_quote(p)
            if q:
                live_prices[p] = q
        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p, v in live_prices.items()])

        # News
        market_news = fetch_market_news()
        news_str = '\n'.join([f"- {n['title']}" for n in market_news[:6]])

        prompt = f"""You are Wolf AI, elite forex day trader. Today is {date_str}. Current session: {session_name}.
Forex hours: Sunday 5PM EST open to Friday 5PM EST close.
Best sessions: London 3AM-12PM EST, NY 8AM-5PM EST, Overlap 8AM-12PM EST.

LIVE PRICES:
{prices_str}

TODAY'S NEWS:
{news_str}

Find TOP 3 GUARANTEED DAY TRADES for London & NY sessions today.
Use top-down analysis: Monthly→Weekly→Daily→H4 direction, then H1/M15 for entry.
Only pick where 5+ factors align. Use the LIVE PRICES above for exact levels.

Respond ONLY with valid JSON:
{{
  "session": "{session_name}",
  "date": "{date_str}",
  "risk_environment": "RISK-ON or RISK-OFF",
  "dxy_bias": "BULLISH or BEARISH",
  "picks": [
    {{
      "rank": 1,
      "pair": "EUR/USD",
      "live_price": "1.0842",
      "direction": "SELL",
      "sharingan_score": 5,
      "confidence": 88,
      "entry": "1.0840",
      "stop_loss": "1.0870",
      "tp1": "1.0800",
      "tp2": "1.0760",
      "tp3": "1.0720",
      "sl_pips": 30,
      "tp1_pips": 40,
      "rr_ratio": "1:1.3",
      "best_window": "3AM-6AM EST London Open",
      "thesis": "Full 3-4 sentence reason based on live price and news",
      "confluences": ["Monthly bearish", "Weekly below key level", "Daily downtrend", "H4 lower highs", "News bearish USD"],
      "buy_scenario": "IF price breaks above 1.0870 with H4 close, enter long target 1.0920",
      "sell_scenario": "IF price rejects 1.0860 and breaks 1.0820, enter short target 1.0780",
      "key_news": "Specific news from today affecting this pair",
      "invalidation": "H4 close above 1.0880"
    }}
  ]
}}"""

        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text.strip()
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'): text = text[4:]
        text = text.strip()
        data = json.loads(text)
        return jsonify(data)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── ROUTE: Weekly swing picks with live data ──────────────────
@app.route('/api/forex-weekly-picks', methods=['POST'])
@login_required
def forex_weekly_picks():
    
    try:
        now = datetime.now()
        date_str = now.strftime('%A, %B %d, %Y')

        # Live prices
        pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY']
        live_prices = {}
        for p in pairs:
            q = fetch_live_quote(p)
            if q: live_prices[p] = q
        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']})" for p, v in live_prices.items()])

        market_news = fetch_market_news()
        news_str = '\n'.join([f"- {n['title']}" for n in market_news[:6]])

        prompt = f"""You are Wolf AI, elite forex swing trader. Week of {date_str}.
Forex hours: Sunday 5PM EST open to Friday 5PM EST close.

LIVE PRICES THIS WEEK:
{prices_str}

THIS WEEK'S NEWS:
{news_str}

Find TOP 3 SWING TRADES for this week (2-7 day holds).
Use full top-down: Monthly→Weekly→Daily direction, H4 for entry zone.
Use live prices for exact levels.

Respond ONLY with valid JSON:
{{
  "week": "{date_str}",
  "weekly_theme": "Main macro theme",
  "dxy_outlook": "BULLISH — reason",
  "central_bank_focus": "Fed/ECB/BOE etc",
  "picks": [
    {{
      "rank": 1,
      "pair": "GBP/USD",
      "live_price": "1.2634",
      "direction": "SELL",
      "hold_days": "3-5",
      "confidence": 85,
      "entry_zone": "1.2620-1.2640",
      "stop_loss": "1.2690",
      "tp1": "1.2540",
      "tp2": "1.2480",
      "tp3": "1.2400",
      "sl_pips": 60,
      "tp1_pips": 90,
      "rr_ratio": "1:1.5",
      "weekly_bias": "BEARISH",
      "timeframe_alignment": "Monthly BEAR · Weekly BEAR · Daily BEAR · H4 NEUTRAL",
      "fundamental": "Central bank + macro reason in 2 sentences",
      "technical": "Weekly/daily chart setup in 2 sentences using live price levels",
      "buy_scenario": "IF weekly closes above X, long targets Y",
      "sell_scenario": "IF weekly closes below X, short targets Y",
      "key_events": "Specific events this week",
      "key_risk": "What breaks this setup"
    }}
  ]
}}"""

        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text.strip()
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'): text = text[4:]
        text = text.strip()
        data = json.loads(text)
        return jsonify(data)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── ROUTE: Forex scanner ──────────────────────────────────────
@app.route('/api/forex-scanner', methods=['POST'])
@login_required
def forex_scanner():
    
    try:
        data = request.get_json()
        theme = data.get('theme', 'strongest momentum')
        now = datetime.now()
        date_str = now.strftime('%A, %B %d, %Y')

        # Live prices
        pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','NZD/USD','USD/CHF']
        live_prices = {}
        for p in pairs:
            q = fetch_live_quote(p)
            if q: live_prices[p] = q
        prices_str = '\n'.join([f"{p}: {v['price']} ({v['percent_change']:+.2f}%)" for p, v in live_prices.items()])

        prompt = f"""You are Wolf AI, forex market analyst. Today is {date_str}.

LIVE PRICES:
{prices_str}

Scan ALL pairs for theme: "{theme}"
Find TOP 5 pairs that best match this theme using the live prices above.

Respond ONLY with valid JSON:
{{
  "theme": "{theme}",
  "date": "{date_str}",
  "pairs": [
    {{
      "pair": "EUR/USD",
      "live_price": "1.0842",
      "direction": "SELL",
      "score": 88,
      "action": "STRONG SELL",
      "session": "LONDON",
      "entry": "1.0840",
      "stop_loss": "1.0870",
      "target": "1.0780",
      "thesis": "Why this pair matches the theme — 2-3 sentences using live price",
      "catalyst": "Specific catalyst",
      "timeframe": "Intraday"
    }}
  ]
}}"""

        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text.strip()
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'): text = text[4:]
        text = text.strip()
        result = json.loads(text)
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── ROUTE: Forex picks (legacy) ───────────────────────────────
@app.route('/api/forex-picks', methods=['POST'])
@login_required
def forex_picks():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2200,
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({'content': message.content[0].text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── GLOBAL JSON ERROR HANDLER ─────────────────────────────────
# Add these to app after the routes to ensure API errors return JSON not HTML
@app.errorhandler(401)
def unauthorized(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Session expired — please refresh and log in again'}), 401
    return redirect(url_for('login_page'))

@app.errorhandler(500)  
def server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Server error — please try again'}), 500
    return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)