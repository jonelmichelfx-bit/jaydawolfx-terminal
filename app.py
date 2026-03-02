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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
# Add these routes to your app.py
# Paste them anywhere after your existing routes

@app.route('/api/forex-analyze', methods=['POST'])
@login_required
def forex_analyze():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({'content': message.content[0].text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
# ═══════════════════════════════════════════════════════════
# FOREX WOLF ROUTES — Add these to app.py
# Paste BEFORE the final "if __name__ == '__main__':" line
# ═══════════════════════════════════════════════════════════

@app.route('/forex-wolf')
@login_required
def forex_wolf():
    return render_template('forex_wolf.html')

@app.route('/api/forex-daily-picks', methods=['POST'])
@login_required
def forex_daily_picks():
    """Top 3 guaranteed day trades — London & NY session"""
    import json
    from datetime import datetime
    try:
        now = datetime.now()
        date_str = now.strftime('%A, %B %d, %Y')
        time_str = now.strftime('%I:%M %p EST')

        prompt = f"""You are Wolf AI, elite forex day trader. Today is {date_str}, {time_str}.

Find the TOP 3 GUARANTEED DAY TRADES for today's London (3AM-12PM EST) and NY (8AM-5PM EST) sessions.

Scan ALL pairs: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY, GBP/JPY, XAU/USD
Only pick where MINIMUM 5 factors align. Be ruthlessly selective.

Respond ONLY with valid JSON, no markdown, no extra text:
{{
  "session": "LONDON/NY",
  "date": "{date_str}",
  "risk_environment": "RISK-ON or RISK-OFF",
  "dxy_bias": "BULLISH or BEARISH",
  "picks": [
    {{
      "rank": 1,
      "pair": "EUR/USD",
      "direction": "BUY",
      "sharingan_score": 5,
      "confidence": 88,
      "entry": "1.0840",
      "stop_loss": "1.0810",
      "tp1": "1.0880",
      "tp2": "1.0920",
      "tp3": "1.0960",
      "sl_pips": 30,
      "tp1_pips": 40,
      "rr_ratio": "1:1.3",
      "best_window": "3AM-6AM EST",
      "thesis": "Full 3-4 sentence reason why this is the best setup today",
      "confluences": ["Factor 1", "Factor 2", "Factor 3", "Factor 4", "Factor 5"],
      "buy_scenario": "IF price breaks above 1.0855 with strong London candle close, enter long",
      "sell_scenario": "IF price rejects 1.0855 and breaks below 1.0830, enter short",
      "key_news": "Any news affecting this pair today",
      "invalidation": "What exactly cancels this trade"
    }}
  ]
}}"""

        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text
        # Clean JSON
        text = text.strip()
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        text = text.strip()
        data = json.loads(text)
        return jsonify(data)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON parse error: {str(e)}', 'raw': text[:500]}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forex-weekly-picks', methods=['POST'])
@login_required
def forex_weekly_picks():
    """Top 3 swing trades for the week"""
    import json
    from datetime import datetime
    try:
        now = datetime.now()
        date_str = now.strftime('%A, %B %d, %Y')

        prompt = f"""You are Wolf AI, elite forex swing trader. Week of {date_str}.

Find the TOP 3 SWING TRADES for this week (2-7 day holds) across ALL major pairs.
Deep research: Fed/ECB/BOE/BOJ stance, DXY weekly direction, weekly/daily chart setups, key events this week.

Respond ONLY with valid JSON, no markdown, no extra text:
{{
  "week": "{date_str}",
  "weekly_theme": "Main macro theme driving forex this week",
  "dxy_outlook": "BULLISH or BEARISH — brief reason",
  "central_bank_focus": "Which central bank is market focused on this week",
  "picks": [
    {{
      "rank": 1,
      "pair": "GBP/USD",
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
      "fundamental": "2 sentences on central bank and macro reason",
      "technical": "2 sentences on weekly/daily chart setup",
      "buy_scenario": "IF price breaks above X weekly close, long targets Y",
      "sell_scenario": "IF price breaks below X weekly close, short targets Y",
      "key_events": "Events this week that could be catalysts",
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
        text = message.content[0].text
        text = text.strip()
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        text = text.strip()
        data = json.loads(text)
        return jsonify(data)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON parse error: {str(e)}', 'raw': text[:500]}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forex-scanner', methods=['POST'])
@login_required
def forex_scanner():
    """Scan forex pairs for a specific theme/setup"""
    import json
    from datetime import datetime
    try:
        data = request.get_json()
        theme = data.get('theme', 'strongest momentum')
        now = datetime.now()
        date_str = now.strftime('%A, %B %d, %Y')

        prompt = f"""You are Wolf AI, forex market analyst. Today is {date_str}.

Scan ALL major forex pairs for: "{theme}"
Find the TOP 5 pairs that best match this theme right now.

Respond ONLY with valid JSON, no markdown:
{{
  "theme": "{theme}",
  "date": "{date_str}",
  "pairs": [
    {{
      "pair": "EUR/USD",
      "direction": "BUY",
      "score": 88,
      "action": "STRONG BUY",
      "session": "LONDON",
      "entry": "1.0840",
      "stop_loss": "1.0800",
      "target": "1.0920",
      "thesis": "Why this pair matches the theme — 2-3 sentences",
      "catalyst": "Specific catalyst driving this move",
      "timeframe": "Intraday / 2-3 days / Weekly"
    }}
  ]
}}"""

        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text
        text = text.strip()
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        text = text.strip()
        result = json.loads(text)
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON parse error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
