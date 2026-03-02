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
# ═══════════════════════════════════════════════════════════════
# FOREX LIVE DATA + WOLF ROUTES — CLEAN VERSION
# Paste above if __name__ == '__main__': in app.py
# ═══════════════════════════════════════════════════════════════

import requests as http_requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

TWELVE_DATA_KEY = os.environ.get('TWELVE_DATA_API_KEY', '')
NEWS_API_KEY    = os.environ.get('NEWS_API_KEY', '')

FOREX_SESSIONS = {
    'TOKYO':    {'pairs': ['USD/JPY','EUR/JPY','GBP/JPY','AUD/USD','NZD/USD']},
    'LONDON':   {'pairs': ['EUR/USD','GBP/USD','EUR/GBP','USD/CHF','EUR/JPY']},
    'NEW YORK': {'pairs': ['EUR/USD','GBP/USD','USD/CAD','USD/JPY','XAU/USD']},
    'OVERLAP':  {'pairs': ['EUR/USD','GBP/USD','USD/JPY','XAU/USD']},
}

# Fallback prices — used when Twelve Data unreachable
FALLBACK = {
    'EUR/USD':{'price':1.0842,'change':0.0012,'pct':0.11,'high':1.0871,'low':1.0821},
    'GBP/USD':{'price':1.2634,'change':-0.0021,'pct':-0.17,'high':1.2671,'low':1.2601},
    'USD/JPY':{'price':149.82,'change':0.44,'pct':0.29,'high':150.21,'low':149.41},
    'USD/CHF':{'price':0.8923,'change':-0.0011,'pct':-0.12,'high':0.8951,'low':0.8901},
    'AUD/USD':{'price':0.6521,'change':-0.0015,'pct':-0.23,'high':0.6551,'low':0.6501},
    'USD/CAD':{'price':1.3541,'change':0.0022,'pct':0.16,'high':1.3561,'low':1.3511},
    'NZD/USD':{'price':0.6102,'change':-0.0008,'pct':-0.13,'high':0.6121,'low':0.6091},
    'EUR/GBP':{'price':0.8582,'change':0.0009,'pct':0.10,'high':0.8601,'low':0.8561},
    'EUR/JPY':{'price':162.44,'change':0.55,'pct':0.34,'high':162.91,'low':161.88},
    'GBP/JPY':{'price':189.22,'change':0.31,'pct':0.16,'high':189.81,'low':188.61},
    'XAU/USD':{'price':2913.50,'change':8.20,'pct':0.31,'high':2925.30,'low':2901.10},
    'DXY':    {'price':104.22,'change':-0.18,'pct':-0.17,'high':104.51,'low':104.01},
}

def get_session():
    """Get current forex session"""
    from datetime import datetime, timezone, timedelta
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
    except:
        return 'UNKNOWN', []

def get_price(symbol):
    """Fetch single price — fast, 4s timeout, falls back gracefully"""
    try:
        sym = symbol.replace('/', '')
        r = http_requests.get(
            f'https://api.twelvedata.com/quote?symbol={sym}&apikey={TWELVE_DATA_KEY}',
            timeout=4
        )
        d = r.json()
        if 'close' in d:
            return {
                'price': float(d.get('close', 0)),
                'open':  float(d.get('open', 0)),
                'high':  float(d.get('high', 0)),
                'low':   float(d.get('low', 0)),
                'change': float(d.get('change', 0)),
                'percent_change': float(d.get('percent_change', 0)),
                'symbol': symbol, 'live': True
            }
    except:
        pass
    fb = FALLBACK.get(symbol)
    if fb:
        return {'price':fb['price'],'open':fb['price'],'high':fb['high'],
                'low':fb['low'],'change':fb['change'],'percent_change':fb['pct'],
                'symbol':symbol,'live':False}
    return None

def get_prices_parallel(pairs):
    """Fetch multiple prices in parallel — max 5s total"""
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
        # All failed — use fallbacks
        for p in pairs:
            fb = FALLBACK.get(p)
            if fb: results[p] = {'price':fb['price'],'high':fb['high'],'low':fb['low'],'change':fb['change'],'percent_change':fb['pct'],'symbol':p,'live':False}
    return results

def get_news(pair=''):
    """Fetch news — 3s timeout, returns empty list on failure"""
    try:
        if pair:
            q = pair.replace('/', '+')
            url = f'https://newsapi.org/v2/everything?q={q}+forex&language=en&sortBy=publishedAt&pageSize=4&apiKey={NEWS_API_KEY}'
        else:
            url = f'https://newsapi.org/v2/everything?q=forex+Fed+ECB+central+bank&language=en&sortBy=publishedAt&pageSize=6&apiKey={NEWS_API_KEY}'
        r = http_requests.get(url, timeout=3)
        arts = r.json().get('articles', [])
        return [{'title':a.get('title',''),'source':a.get('source',{}).get('name',''),'published':a.get('publishedAt','')[:10]} for a in arts if a.get('title')]
    except:
        return []

def call_claude(prompt, max_tokens=2500):
    """Call Claude — clean, simple"""
    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    msg = client.messages.create(
        model='claude-opus-4-6',
        max_tokens=max_tokens,
        messages=[{'role':'user','content':prompt}]
    )
    return msg.content[0].text

def parse_json_response(text):
    """Safely parse JSON from Claude response"""
    text = text.strip()
    if text.startswith('```'):
        parts = text.split('```')
        text = parts[1] if len(parts) > 1 else text
        if text.startswith('json'):
            text = text[4:]
    return json.loads(text.strip())

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/forex')
@login_required
def forex():
    return render_template('forex.html')

@app.route('/forex-wolf')
@login_required
def forex_wolf():
    return render_template('forex_wolf.html')

@app.route('/api/forex-prices', methods=['GET'])
@login_required
def forex_prices():
    try:
        pairs = ['EUR/USD','GBP/USD','USD/JPY','USD/CHF','AUD/USD','USD/CAD','NZD/USD','EUR/GBP','EUR/JPY','GBP/JPY','XAU/USD']
        prices = get_prices_parallel(pairs)
        # Also get DXY
        dxy = get_price('DXY')
        if dxy: prices['DXY'] = dxy
        session_name, session_pairs = get_session()
        return jsonify({'prices': prices, 'session': session_name, 'session_pairs': session_pairs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-price', methods=['POST'])
@login_required
def forex_price():
    try:
        symbol = request.get_json().get('symbol', 'EUR/USD')
        q = get_price(symbol)
        return jsonify(q or {'error': 'Unavailable'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-news', methods=['POST'])
@login_required
def forex_news():
    try:
        pair = request.get_json().get('pair', '')
        news = get_news(pair)
        return jsonify({'pair_news': news, 'market_news': get_news()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-analyze', methods=['POST'])
@login_required
def forex_analyze():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 2500)
        pair = data.get('pair', '')

        # Enrich with live data
        live_ctx = ''
        if pair:
            q = get_price(pair)
            session_name, _ = get_session()
            if q:
                live_ctx = f"""
LIVE MARKET DATA:
Pair: {pair} | Price: {q['price']} | High: {q['high']} | Low: {q['low']}
Change: {q['change']:+.5f} ({q['percent_change']:+.2f}%) | Session: {session_name}
Data: {'LIVE' if q.get('live') else 'REFERENCE'}

"""
            news = get_news(pair)
            if news:
                live_ctx += f"LATEST NEWS FOR {pair}:\n"
                for n in news[:3]:
                    live_ctx += f"- {n['title']} ({n['source']})\n"
                live_ctx += "\n"

        full_prompt = live_ctx + prompt if live_ctx else prompt
        text = call_claude(full_prompt, max_tokens)
        return jsonify({'content': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-scenarios', methods=['POST'])
@login_required
def forex_scenarios():
    try:
        now = datetime.now()
        date_str = now.strftime('%A, %B %d, %Y')
        session_name, _ = get_session()

        pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY']
        prices = get_prices_parallel(pairs)
        news = get_news()

        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p,v in prices.items()])
        news_str = '\n'.join([f"- {n['title']}" for n in news[:5]]) or "- Markets await key economic data"

        prompt = f"""You are Wolf AI — professional forex trader. Today: {date_str}. Session: {session_name}.

LIVE PRICES:
{prices_str}

NEWS:
{news_str}

Analyze all pairs top-down (Monthly→Weekly→Daily→H4→H1→M15) and find the 7 BEST trades.
Only include pairs where 4+ timeframes align. Give BOTH buy AND sell scenario for each.

Respond ONLY in valid JSON (no markdown, no backticks):
{{
  "week": "{date_str}",
  "session": "{session_name}",
  "market_theme": "string",
  "dxy_direction": "BULLISH or BEARISH",
  "risk_sentiment": "RISK-ON or RISK-OFF",
  "trades": [
    {{
      "rank": 1,
      "pair": "EUR/USD",
      "live_price": "1.0842",
      "overall_bias": "BEARISH",
      "timeframe_alignment": {{"monthly":"BEARISH","weekly":"BEARISH","daily":"BEARISH","h4":"BEARISH","h1":"NEUTRAL","m15":"BEARISH"}},
      "aligned_count": 5,
      "confidence": 82,
      "primary_direction": "SELL",
      "thesis": "3-4 sentence thesis",
      "key_resistance": "1.0890",
      "key_support": "1.0780",
      "buy_scenario": {{"trigger":"string","entry":"1.0895","stop_loss":"1.0855","tp1":"1.0950","tp2":"1.1000","tp3":"1.1050","probability":35}},
      "sell_scenario": {{"trigger":"string","entry":"1.0815","stop_loss":"1.0855","tp1":"1.0780","tp2":"1.0720","tp3":"1.0650","probability":65}},
      "best_session": "LONDON",
      "key_news_this_week": "string",
      "invalidation": "string"
    }}
  ]
}}"""

        text = call_claude(prompt, 4000)
        result = parse_json_response(text)
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'AI returned invalid JSON: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-daily-picks', methods=['POST'])
@login_required
def forex_daily_picks():
    try:
        now = datetime.now()
        date_str = now.strftime('%A, %B %d, %Y')
        session_name, _ = get_session()

        pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','USD/CHF']
        prices = get_prices_parallel(pairs)
        news = get_news()

        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p,v in prices.items()])
        news_str = '\n'.join([f"- {n['title']}" for n in news[:4]]) or "- Monitor key levels"

        prompt = f"""You are Wolf AI — professional intraday forex trader. Today: {date_str}. Session: {session_name}.

LIVE PRICES:
{prices_str}

NEWS:
{news_str}

Find the 3 BEST day trades for today's London and NY sessions.
Only pairs with strong momentum and clear entry points.

Respond ONLY in valid JSON (no markdown):
{{
  "date": "{date_str}",
  "session": "{session_name}",
  "dxy_bias": "BULLISH or BEARISH",
  "risk_environment": "RISK-ON or RISK-OFF",
  "picks": [
    {{
      "rank": 1,
      "pair": "EUR/USD",
      "direction": "SELL",
      "entry": "1.0880",
      "stop_loss": "1.0910",
      "tp1": "1.0840",
      "tp2": "1.0800",
      "tp3": "1.0760",
      "rr_ratio": "1:2.5",
      "confidence": 85,
      "sharingan_score": 5,
      "thesis": "2-3 sentence thesis",
      "confluences": ["Daily bearish engulfing","Below 200 EMA","DXY bullish","London session momentum"],
      "best_window": "London Open 3-5AM EST",
      "key_news": "NFP Friday",
      "invalidation": "Break above 1.0920",
      "buy_scenario": "If breaks above 1.0920 with strong candle, look for 1.0950",
      "sell_scenario": "If rejects 1.0880, target 1.0840 with tight stop"
    }}
  ]
}}"""

        text = call_claude(prompt, 3000)
        result = parse_json_response(text)
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'AI returned invalid JSON: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-weekly-picks', methods=['POST'])
@login_required
def forex_weekly_picks():
    try:
        now = datetime.now()
        date_str = now.strftime('%A, %B %d, %Y')
        session_name, _ = get_session()

        pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','NZD/USD']
        prices = get_prices_parallel(pairs)
        news = get_news()

        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p,v in prices.items()])
        news_str = '\n'.join([f"- {n['title']}" for n in news[:4]]) or "- Monitor macro events"

        prompt = f"""You are Wolf AI — professional swing trader. Today: {date_str}.

LIVE PRICES:
{prices_str}

NEWS:
{news_str}

Find the 3 BEST swing trades for this week (2-5 day holds).
Focus on pairs with strong weekly/daily momentum and key level setups.

Respond ONLY in valid JSON (no markdown):
{{
  "week": "{date_str}",
  "weekly_theme": "Main macro theme",
  "dxy_outlook": "BULLISH or BEARISH",
  "central_bank_focus": "Key CB event this week",
  "picks": [
    {{
      "rank": 1,
      "pair": "GBP/USD",
      "direction": "SELL",
      "entry_zone": "1.2650-1.2670",
      "stop_loss": "1.2720",
      "tp1": "1.2580",
      "tp2": "1.2500",
      "tp3": "1.2420",
      "rr_ratio": "1:2.8",
      "confidence": 80,
      "sharingan_score": 4,
      "hold_days": "3-4",
      "fundamental": "2-3 sentence fundamental reason",
      "technical": "2-3 sentence technical reason",
      "confluences": ["Weekly bearish","Daily below EMA","BOE dovish","DXY strength"],
      "key_events": "BOE minutes Thursday",
      "key_risk": "Surprise hawkish BOE",
      "buy_scenario": "If breaks above 1.2720, bullish to 1.2800",
      "sell_scenario": "If rejects 1.2660, target 1.2500"
    }}
  ]
}}"""

        text = call_claude(prompt, 3000)
        result = parse_json_response(text)
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'AI returned invalid JSON: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-scanner', methods=['POST'])
@login_required
def forex_scanner():
    try:
        data = request.get_json()
        theme = data.get('theme', 'SAFE_HAVEN')
        session_name, _ = get_session()

        theme_prompts = {
            'SAFE_HAVEN': 'safe haven flows — USD, JPY, Gold, CHF strength/weakness',
            'RISK_ON': 'risk-on environment — AUD, NZD, GBP momentum plays',
            'DOLLAR': 'USD strength/weakness plays across all major pairs',
            'GOLD': 'XAU/USD and commodity currency correlations',
            'CENTRAL_BANK': 'central bank divergence trades — rate differential plays',
            'BREAKOUT': 'technical breakout setups — pairs at key levels',
        }
        theme_desc = theme_prompts.get(theme, theme_prompts['SAFE_HAVEN'])

        pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','USD/CHF','EUR/JPY','GBP/JPY']
        prices = get_prices_parallel(pairs)
        prices_str = '\n'.join([f"{p}: {v['price']} ({v['percent_change']:+.2f}%)" for p,v in prices.items()])

        prompt = f"""You are Wolf AI scanner. Session: {session_name}. Theme: {theme_desc}.

LIVE PRICES:
{prices_str}

Scan all pairs for {theme_desc} opportunities. Find 4-5 best setups.

Respond ONLY in valid JSON (no markdown):
{{
  "theme": "{theme}",
  "session": "{session_name}",
  "theme_analysis": "2-3 sentence overall theme analysis",
  "setups": [
    {{
      "pair": "USD/JPY",
      "direction": "BUY",
      "setup_type": "Trend continuation",
      "entry": "149.50",
      "stop_loss": "148.80",
      "tp1": "150.50",
      "tp2": "151.50",
      "confidence": 78,
      "reason": "2 sentence reason this fits the theme",
      "urgency": "HIGH or MEDIUM or LOW"
    }}
  ]
}}"""

        text = call_claude(prompt, 2000)
        result = parse_json_response(text)
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'AI returned invalid JSON: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forex-picks', methods=['POST'])
@login_required
def forex_picks():
    try:
        prompt = request.get_json().get('prompt', '')
        text = call_claude(prompt, 2200)
        return jsonify({'content': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# ═══════════════════════════════════════════════════════════════
# WOLF SCANNER ROUTE + SERVER-SIDE PRICE CACHE
# Add this to forex_routes_final.py (paste before if __name__)
# ═══════════════════════════════════════════════════════════════
# The cache fetches prices ONCE every 60 seconds server-side.
# All users share the same cached prices = 1440 API calls/day
# instead of 1440 × number_of_users. Saves 99% of API credits.

import time

# ── PRICE CACHE ────────────────────────────────────────────────
_price_cache = {
    'prices': {},
    'fetched_at': 0,
    'ttl': 60,  # seconds — fetch fresh prices every 60s
    'live': False
}

def get_cached_prices():
    """Return cached prices — only fetches from API if cache is stale"""
    now = time.time()
    if now - _price_cache['fetched_at'] < _price_cache['ttl'] and _price_cache['prices']:
        return _price_cache['prices'], _price_cache['live']

    # Cache stale — fetch fresh
    pairs = ['EUR/USD','GBP/USD','USD/JPY','USD/CHF','AUD/USD','USD/CAD',
             'NZD/USD','EUR/GBP','EUR/JPY','GBP/JPY','XAU/USD','DXY']
    fresh = get_prices_parallel(pairs)

    if fresh:
        _price_cache['prices'] = fresh
        _price_cache['fetched_at'] = now
        # Check if any are live (not fallback)
        _price_cache['live'] = any(v.get('live', False) for v in fresh.values())

    return _price_cache['prices'], _price_cache['live']


# ── SCANNER PAGE ───────────────────────────────────────────────
@app.route('/wolf-scanner')
@login_required
def wolf_scanner_page():
    return render_template('wolf_scanner.html')


# ── UPDATED PRICES ROUTE (uses cache) ─────────────────────────
@app.route('/api/forex-prices', methods=['GET'])
@login_required
def forex_prices():
    try:
        prices, is_live = get_cached_prices()
        session_name, session_pairs = get_session()
        return jsonify({
            'prices': prices,
            'session': session_name,
            'session_pairs': session_pairs,
            'live': is_live,
            'cached_at': datetime.now().strftime('%H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── WOLF SCAN ROUTE ────────────────────────────────────────────
@app.route('/api/wolf-scan', methods=['POST'])
@login_required
def wolf_scan():
    """The main scanner — returns 5 best trades with full analysis"""
    try:
        data = request.get_json() or {}
        scan_filter = data.get('filter', 'ALL')
        now = datetime.now()
        date_str = now.strftime('%A, %B %d, %Y')
        session_name, _ = get_session()

        # Get cached prices (shared across all users)
        prices, is_live = get_cached_prices()

        # Filter pairs based on selection
        all_pairs = list(prices.keys())
        if scan_filter == 'MAJORS':
            scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','USD/CHF','AUD/USD','USD/CAD']
        elif scan_filter == 'GOLD':
            scan_pairs = ['XAU/USD','EUR/USD','USD/JPY','AUD/USD','NZD/USD']
        elif scan_filter == 'ASIAN':
            scan_pairs = ['USD/JPY','EUR/JPY','GBP/JPY','AUD/USD','NZD/USD']
        elif scan_filter == 'LONDON':
            scan_pairs = ['EUR/USD','GBP/USD','EUR/GBP','USD/CHF','EUR/JPY']
        else:
            scan_pairs = ['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD',
                         'USD/CAD','GBP/JPY','EUR/JPY','NZD/USD','USD/CHF']

        # Build prices string
        prices_lines = []
        for p in scan_pairs:
            q = prices.get(p)
            if q:
                dp = 2 if (p.includes('JPY') if hasattr(p,'includes') else 'JPY' in p) or p=='XAU/USD' else 5
                prices_lines.append(
                    f"{p}: {float(q['price']):.{dp}f} "
                    f"(High: {float(q['high']):.{dp}f} "
                    f"Low: {float(q['low']):.{dp}f} "
                    f"Change: {float(q.get('percent_change',0)):+.2f}%)"
                )
        prices_str = '\n'.join(prices_lines)

        # Get news
        news_items = get_news()
        news_str = '\n'.join([f"- {n['title']} ({n['source']}, {n['published']})"
                              for n in news_items[:6]]) or "- Monitor key economic events this week"

        # The Wolf prompt — Soros/Kovner/Druckenmiller methodology
        prompt = f"""You are Wolf AI — a professional forex trader trained in the methodologies of George Soros, Stanley Druckenmiller, Bruce Kovner, and Paul Tudor Jones.

TODAY: {date_str} | SESSION: {session_name}
DATA SOURCE: {'LIVE REAL-TIME' if is_live else 'REFERENCE (API limit — use as guide)'}

LIVE MARKET PRICES:
{prices_str}

LATEST MARKET NEWS:
{news_str}

YOUR TASK: Scan all pairs above and find the 5 BEST trades using this exact methodology:

STEP 1 — SOROS TOP-DOWN: Start with DXY direction. Dollar drives 80% of forex moves.
STEP 2 — DRUCKENMILLER DIVERGENCE: Find central bank policy differences (Fed vs ECB vs BOJ vs BOE). Rate divergence = biggest edge.
STEP 3 — KOVNER TIMEFRAME CONFLUENCE: Only recommend trades where 4+ timeframes (Monthly/Weekly/Daily/H4/H1/M15) align.
STEP 4 — TREND IDENTIFICATION: Is this pair in clear uptrend or downtrend? Trade WITH the trend only.
STEP 5 — S/R LEVELS: Find key support and resistance levels NEAR the current price (within 50-100 pips). These are the actual levels to watch.
STEP 6 — SCENARIOS: For each trade give both a BUY scenario AND a SELL scenario based on what price does at key levels.
STEP 7 — WARNINGS: Note any upcoming economic data, central bank meetings, or news events that could affect the trade. Tell trader to WAIT if data is imminent.

IMPORTANT RULES:
- Trade trends, not reversals
- Only levels NEAR current price matter (within 100 pips for majors, 5 for gold)
- Be specific with exact price levels
- Warn about economic calendar events (NFP, CPI, FOMC, BOE, ECB meetings)
- Use real technical analysis: EMA200, RSI, MACD, key S/R zones

Respond ONLY in valid JSON (absolutely no markdown, no backticks, no extra text):
{{
  "scan_date": "{date_str}",
  "session": "{session_name}",
  "market_theme": "One sentence describing main macro theme today",
  "dxy_bias": "BULLISH or BEARISH",
  "risk_sentiment": "RISK-ON or RISK-OFF",
  "wolf_commentary": "2-3 sentences from Wolf AI on overall market conditions right now",
  "trades": [
    {{
      "rank": 1,
      "pair": "EUR/USD",
      "current_price": "1.0380",
      "trend": "DOWNTREND",
      "primary_direction": "SELL",
      "wolf_score": 8.5,
      "confidence": 85,
      "aligned_count": 5,
      "thesis": "3-4 sentence explanation using Soros/Druckenmiller/Kovner logic. Why is this the best trade right now? What macro factors, what technical factors, what central bank divergence?",
      "timeframe_alignment": {{
        "monthly": "BEARISH",
        "weekly": "BEARISH",
        "daily": "BEARISH",
        "h4": "BEARISH",
        "h1": "NEUTRAL",
        "m15": "BEARISH"
      }},
      "confluences": [
        "Price below 200 EMA on Daily",
        "Fed hawkish vs ECB dovish divergence",
        "DXY bullish — dollar strength",
        "Weekly bearish engulfing candle",
        "RSI below 50 on H4"
      ],
      "key_levels": [
        {{"type": "RESISTANCE", "price": "1.0450", "note": "Previous daily high — key ceiling", "distance_pips": 70}},
        {{"type": "RESISTANCE", "price": "1.0400", "note": "48hr high — immediate resistance", "distance_pips": 20}},
        {{"type": "CURRENT",    "price": "1.0380", "note": "Current price", "distance_pips": 0}},
        {{"type": "SUPPORT",    "price": "1.0340", "note": "Yesterday low — first support", "distance_pips": 40}},
        {{"type": "SUPPORT",    "price": "1.0280", "note": "Weekly S/R zone", "distance_pips": 100}}
      ],
      "buy_scenario": {{
        "trigger": "If price breaks and closes ABOVE 1.0450 on H4 with strong bullish candle — momentum shift confirmed",
        "entry": "1.0460",
        "stop_loss": "1.0410",
        "tp1": "1.0510",
        "tp2": "1.0570",
        "tp3": "1.0640",
        "rr": "1:3.0",
        "probability": 25
      }},
      "sell_scenario": {{
        "trigger": "If price rejects 1.0400 resistance and breaks below 1.0360 — continuation of downtrend",
        "entry": "1.0355",
        "stop_loss": "1.0400",
        "tp1": "1.0310",
        "tp2": "1.0260",
        "tp3": "1.0200",
        "rr": "1:2.5",
        "probability": 75
      }},
      "warnings": [
        {{"level": "HIGH", "text": "US CPI data Thursday 8:30AM EST — WAIT for release before entering, expect 50-80 pip spike"}},
        {{"level": "MEDIUM", "text": "ECB President speaks Wednesday — could cause EUR volatility"}}
      ],
      "relevant_news": [
        "Fed holds rates — signals no cuts until inflation cools further",
        "ECB mulls rate cuts as eurozone growth slows"
      ]
    }}
  ]
}}"""

        text = call_claude(prompt, 4500)
        result = parse_json_response(text)

        # Inject live prices into results for accuracy
        for trade in result.get('trades', []):
            pair = trade.get('pair', '')
            q = prices.get(pair)
            if q:
                dp = 2 if 'JPY' in pair or pair == 'XAU/USD' else 4
                trade['current_price'] = f"{float(q['price']):.{dp}f}"
                trade['live_price_confirmed'] = True

        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({'error': f'AI analysis error — please try again ({str(e)[:50]})'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)