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

# Fallback prices — March 2026
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

# ── Server-side price cache (all users share this) ────────────
_price_cache = {'prices': {}, 'fetched_at': 0, 'ttl': 60, 'live': False}

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

# ── Free price source: open.er-api.com (1500/day free, no key) ──
_er_cache = {'rates': {}, 'fetched_at': 0}

def _get_er_rates():
    """Fetch all FX rates from free API — cached 60s"""
    now = time.time()
    if now - _er_cache['fetched_at'] < 60 and _er_cache['rates']:
        return _er_cache['rates']
    try:
        r = http_requests.get('https://open.er-api.com/v6/latest/USD', timeout=5)
        d = r.json()
        if d.get('rates'):
            _er_cache['rates'] = d['rates']
            _er_cache['fetched_at'] = now
            return d['rates']
    except: pass
    return {}

def get_price(symbol):
    """Get price — tries free API first, then Twelve Data, then fallback"""
    # Special case: DXY not available on free FX APIs, use Twelve Data or fallback
    if symbol == 'DXY':
        try:
            r = http_requests.get(f'https://api.twelvedata.com/quote?symbol=DXY&apikey={TWELVE_DATA_KEY}', timeout=4)
            d = r.json()
            if 'close' in d and 'code' not in d:
                return {'price':float(d.get('close',0)),'open':float(d.get('open',0)),
                        'high':float(d.get('high',0)),'low':float(d.get('low',0)),
                        'change':float(d.get('change',0)),'percent_change':float(d.get('percent_change',0)),
                        'symbol':'DXY','live':True}
        except: pass
        fb = FALLBACK.get('DXY')
        if fb: return {'price':fb['price'],'open':fb['price'],'high':fb['high'],'low':fb['low'],
                       'change':fb['change'],'percent_change':fb['pct'],'symbol':'DXY','live':False}
        return None

    # Try free exchange rate API
    try:
        rates = _get_er_rates()
        if rates:
            base, quote = symbol.split('/')
            if base == 'USD' and quote in rates:
                price = float(rates[quote])
                # Estimate high/low as ±0.2% (ER API only gives close)
                return {'price':price,'open':price,'high':round(price*1.002,5),
                        'low':round(price*0.998,5),'change':0.0,'percent_change':0.0,
                        'symbol':symbol,'live':True}
            elif quote == 'USD' and base in rates:
                price = round(1.0 / float(rates[base]), 5)
                return {'price':price,'open':price,'high':round(price*1.002,5),
                        'low':round(price*0.998,5),'change':0.0,'percent_change':0.0,
                        'symbol':symbol,'live':True}
            elif base in rates and quote in rates:
                price = round(float(rates[quote]) / float(rates[base]), 5)
                return {'price':price,'open':price,'high':round(price*1.002,5),
                        'low':round(price*0.998,5),'change':0.0,'percent_change':0.0,
                        'symbol':symbol,'live':True}
            # XAU/USD — gold not in FX rates, fall through to Twelve Data
    except: pass

    # Twelve Data fallback (uses credits — only if free API failed)
    try:
        sym = symbol.replace('/', '')
        r = http_requests.get(f'https://api.twelvedata.com/quote?symbol={sym}&apikey={TWELVE_DATA_KEY}', timeout=4)
        d = r.json()
        if 'close' in d and 'code' not in d:
            return {'price':float(d.get('close',0)),'open':float(d.get('open',0)),
                    'high':float(d.get('high',0)),'low':float(d.get('low',0)),
                    'change':float(d.get('change',0)),'percent_change':float(d.get('percent_change',0)),
                    'symbol':symbol,'live':True}
    except: pass

    # Last resort: fallback prices
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
    """One fetch per 60s shared across ALL users — saves 99% of API credits"""
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
    text = text.strip()
    if text.startswith('```'):
        parts = text.split('```')
        text = parts[1] if len(parts) > 1 else text
        if text.startswith('json'): text = text[4:]
    return json.loads(text.strip())

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

# ── Forex API (single forex_prices using cache) ───────────────

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
        prompt = data.get('prompt', ''); max_tokens = data.get('max_tokens', 2500); pair = data.get('pair', '')
        live_ctx = ''
        if pair:
            q = get_price(pair); session_name, _ = get_session()
            if q:
                live_ctx = f"\nLIVE MARKET DATA:\nPair: {pair} | Price: {q['price']} | High: {q['high']} | Low: {q['low']}\nChange: {q['change']:+.5f} ({q['percent_change']:+.2f}%) | Session: {session_name}\nData: {'LIVE' if q.get('live') else 'REFERENCE'}\n\n"
            news = get_news(pair)
            if news:
                live_ctx += f"LATEST NEWS FOR {pair}:\n" + '\n'.join([f"- {n['title']} ({n['source']})" for n in news[:3]]) + '\n\n'
        text = call_claude((live_ctx + prompt) if live_ctx else prompt, max_tokens)
        return jsonify({'content': text})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/forex-scenarios', methods=['POST'])
@login_required
def forex_scenarios():
    try:
        date_str = datetime.now().strftime('%A, %B %d, %Y'); session_name, _ = get_session()
        prices = get_prices_parallel(['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY'])
        news = get_news()
        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p,v in prices.items()])
        news_str = '\n'.join([f"- {n['title']}" for n in news[:5]]) or "- Markets await key economic data"
        prompt = f"""You are Wolf AI — professional forex trader. Today: {date_str}. Session: {session_name}.
LIVE PRICES:\n{prices_str}\nNEWS:\n{news_str}
Analyze all pairs top-down (Monthly→Weekly→Daily→H4→H1→M15) and find the 7 BEST trades.
Only include pairs where 4+ timeframes align. Give BOTH buy AND sell scenario for each.
Respond ONLY in valid JSON (no markdown, no backticks):
{{"week":"{date_str}","session":"{session_name}","market_theme":"string","dxy_direction":"BULLISH or BEARISH","risk_sentiment":"RISK-ON or RISK-OFF","trades":[{{"rank":1,"pair":"EUR/USD","live_price":"1.0380","overall_bias":"BEARISH","timeframe_alignment":{{"monthly":"BEARISH","weekly":"BEARISH","daily":"BEARISH","h4":"BEARISH","h1":"NEUTRAL","m15":"BEARISH"}},"aligned_count":5,"confidence":82,"primary_direction":"SELL","thesis":"3-4 sentence thesis","key_resistance":"1.0400","key_support":"1.0340","buy_scenario":{{"trigger":"string","entry":"1.0410","stop_loss":"1.0380","tp1":"1.0450","tp2":"1.0500","tp3":"1.0550","probability":30}},"sell_scenario":{{"trigger":"string","entry":"1.0360","stop_loss":"1.0390","tp1":"1.0320","tp2":"1.0280","tp3":"1.0240","probability":70}},"best_session":"LONDON","key_news_this_week":"string","invalidation":"string"}}]}}"""
        result = parse_json_response(call_claude(prompt, 4000))
        return jsonify(result)
    except json.JSONDecodeError as e: return jsonify({'error': f'AI returned invalid JSON: {str(e)}'}), 500
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/forex-daily-picks', methods=['POST'])
@login_required
def forex_daily_picks():
    try:
        date_str = datetime.now().strftime('%A, %B %d, %Y'); session_name, _ = get_session()
        prices = get_prices_parallel(['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','USD/CHF'])
        news = get_news()
        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p,v in prices.items()])
        news_str = '\n'.join([f"- {n['title']}" for n in news[:4]]) or "- Monitor key levels"
        prompt = f"""You are Wolf AI — professional intraday forex trader. Today: {date_str}. Session: {session_name}.
LIVE PRICES:\n{prices_str}\nNEWS:\n{news_str}
Find the 3 BEST day trades for today. Only pairs with strong momentum and clear entry points.
Respond ONLY in valid JSON (no markdown):
{{"date":"{date_str}","session":"{session_name}","dxy_bias":"BULLISH or BEARISH","risk_environment":"RISK-ON or RISK-OFF","picks":[{{"rank":1,"pair":"EUR/USD","direction":"SELL","entry":"1.0390","stop_loss":"1.0420","tp1":"1.0350","tp2":"1.0310","tp3":"1.0270","rr_ratio":"1:2.5","confidence":85,"sharingan_score":5,"thesis":"2-3 sentence thesis","confluences":["Daily bearish","Below 200 EMA","DXY bullish"],"best_window":"London Open 3-5AM EST","key_news":"NFP Friday","invalidation":"Break above 1.0430","buy_scenario":"string","sell_scenario":"string"}}]}}"""
        result = parse_json_response(call_claude(prompt, 3000))
        return jsonify(result)
    except json.JSONDecodeError as e: return jsonify({'error': f'AI returned invalid JSON: {str(e)}'}), 500
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/forex-weekly-picks', methods=['POST'])
@login_required
def forex_weekly_picks():
    try:
        date_str = datetime.now().strftime('%A, %B %d, %Y'); session_name, _ = get_session()
        prices = get_prices_parallel(['EUR/USD','GBP/USD','USD/JPY','XAU/USD','AUD/USD','USD/CAD','GBP/JPY','EUR/JPY','NZD/USD'])
        news = get_news()
        prices_str = '\n'.join([f"{p}: {v['price']} (H:{v['high']} L:{v['low']} {v['percent_change']:+.2f}%)" for p,v in prices.items()])
        news_str = '\n'.join([f"- {n['title']}" for n in news[:4]]) or "- Monitor macro events"
        prompt = f"""You are Wolf AI — professional swing trader. Today: {date_str}.
LIVE PRICES:\n{prices_str}\nNEWS:\n{news_str}
Find the 3 BEST swing trades for this week (2-5 day holds).
Respond ONLY in valid JSON (no markdown):
{{"week":"{date_str}","weekly_theme":"Main macro theme","dxy_outlook":"BULLISH or BEARISH","central_bank_focus":"Key CB event this week","picks":[{{"rank":1,"pair":"GBP/USD","direction":"SELL","entry_zone":"1.2630-1.2650","stop_loss":"1.2700","tp1":"1.2570","tp2":"1.2500","tp3":"1.2420","rr_ratio":"1:2.8","confidence":80,"sharingan_score":4,"hold_days":"3-4","fundamental":"string","technical":"string","confluences":["Weekly bearish","Daily below EMA"],"key_events":"BOE minutes","key_risk":"Surprise hawkish BOE","buy_scenario":"string","sell_scenario":"string"}}]}}"""
        result = parse_json_response(call_claude(prompt, 3000))
        return jsonify(result)
    except json.JSONDecodeError as e: return jsonify({'error': f'AI returned invalid JSON: {str(e)}'}), 500
    except Exception as e: return jsonify({'error': str(e)}), 500

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

        prices_lines = []
        for p in scan_pairs:
            q = prices.get(p)
            if q:
                dp = 2 if 'JPY' in p or p == 'XAU/USD' else 5
                prices_lines.append(f"{p}: {float(q['price']):.{dp}f} (High: {float(q['high']):.{dp}f} Low: {float(q['low']):.{dp}f} Change: {float(q.get('percent_change',0)):+.2f}%)")
        prices_str = '\n'.join(prices_lines)

        news_items = get_news()
        news_str = '\n'.join([f"- {n['title']} ({n['source']}, {n['published']})" for n in news_items[:6]]) or "- Monitor key economic events"

        prompt = f"""You are Wolf AI — a professional forex trader using Soros, Druckenmiller, Kovner, and Paul Tudor Jones methodology.

TODAY: {date_str} | SESSION: {session_name}
DATA: {'LIVE REAL-TIME' if is_live else 'REFERENCE PRICES'}

LIVE PRICES:
{prices_str}

NEWS:
{news_str}

Find the 5 BEST trades using this methodology:
STEP 1 — SOROS: DXY direction drives everything
STEP 2 — DRUCKENMILLER: Central bank divergence (Fed vs ECB vs BOJ vs BOE)
STEP 3 — KOVNER: Only trades with 4+ timeframe confluence (Monthly/Weekly/Daily/H4/H1/M15)
STEP 4 — TREND: Trade WITH the trend only
STEP 5 — S/R: Key levels within 100 pips of current price only
STEP 6 — SCENARIOS: Both BUY and SELL scenario per trade
STEP 7 — WARNINGS: Economic calendar events, wait signals

Respond ONLY in valid JSON (no markdown, no backticks):
{{"scan_date":"{date_str}","session":"{session_name}","market_theme":"string","dxy_bias":"BULLISH or BEARISH","risk_sentiment":"RISK-ON or RISK-OFF","wolf_commentary":"2-3 sentences on current market","trades":[{{"rank":1,"pair":"EUR/USD","current_price":"1.0380","trend":"DOWNTREND","primary_direction":"SELL","wolf_score":8.5,"confidence":85,"aligned_count":5,"thesis":"3-4 sentence Soros/Druckenmiller/Kovner thesis","timeframe_alignment":{{"monthly":"BEARISH","weekly":"BEARISH","daily":"BEARISH","h4":"BEARISH","h1":"NEUTRAL","m15":"BEARISH"}},"confluences":["Below 200 EMA Daily","Fed hawkish vs ECB dovish","DXY bullish","Weekly bearish engulfing","RSI below 50 H4"],"key_levels":[{{"type":"RESISTANCE","price":"1.0400","note":"48hr high","distance_pips":20}},{{"type":"CURRENT","price":"1.0380","note":"Current price","distance_pips":0}},{{"type":"SUPPORT","price":"1.0340","note":"Yesterday low","distance_pips":40}}],"buy_scenario":{{"trigger":"Break above 1.0420 on H4","entry":"1.0425","stop_loss":"1.0395","tp1":"1.0460","tp2":"1.0510","tp3":"1.0560","rr":"1:2.5","probability":25}},"sell_scenario":{{"trigger":"Reject 1.0400 and break below 1.0360","entry":"1.0355","stop_loss":"1.0390","tp1":"1.0310","tp2":"1.0270","tp3":"1.0220","rr":"1:2.8","probability":75}},"warnings":[{{"level":"HIGH","text":"US CPI Thursday 8:30AM EST — wait for release"}},{{"level":"MEDIUM","text":"ECB speak Wednesday — EUR volatility"}}],"relevant_news":["Fed signals no cuts until inflation cools","ECB mulls rate cut timeline"]}}]}}"""

        result = parse_json_response(call_claude(prompt, 4500))

        for trade in result.get('trades', []):
            pair = trade.get('pair', '')
            q = prices.get(pair)
            if q:
                dp = 2 if 'JPY' in pair or pair == 'XAU/USD' else 4
                trade['current_price'] = f"{float(q['price']):.{dp}f}"

        return jsonify(result)
    except json.JSONDecodeError as e: return jsonify({'error': f'AI analysis error — try again ({str(e)[:50]})'}), 500
    except Exception as e: return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
