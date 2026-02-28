from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, current_user, login_required
from models import db, User
from auth import auth_bp
from decorators import analysis_gate
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

# ── Extensions ───────────────────────────────────────
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Authentication required.', 'action': 'login'}), 401
    return redirect(url_for('login_page'))

# ── Blueprints ───────────────────────────────────────
app.register_blueprint(auth_bp)

from payments import payments_bp
app.register_blueprint(payments_bp)

from scanner import scanner_bp
app.register_blueprint(scanner_bp)

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
@login_required
def pricing():
    return render_template('pricing.html')

@app.route('/ai-scanner')
@login_required
def ai_scanner():
    return render_template('scanner.html')

# ────────────────────────────────────────────────────
# API ROUTES — now protected with @analysis_gate
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
