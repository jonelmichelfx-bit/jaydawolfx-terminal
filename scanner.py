import os
import re
import json
from datetime import datetime, date, timedelta
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
import anthropic

scanner_bp = Blueprint('scanner', __name__, url_prefix='/scanner')
client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

_scan_cache = {'date': None, 'results': None}


def get_stock_prices(tickers):
    """Get real current prices for a list of tickers."""
    try:
        import yfinance as yf
        data = {}
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period='2d')
                if len(hist) >= 1:
                    current = float(hist['Close'].iloc[-1])
                    prev = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else current
                    change_pct = ((current - prev) / prev) * 100
                    data[ticker] = {
                        'price': round(current, 2),
                        'change_pct': round(change_pct, 2),
                        'prev': round(prev, 2)
                    }
            except Exception:
                pass
        return data
    except Exception:
        return {}


def get_market_snapshot():
    """Get broad market context."""
    tickers = ['SPY', 'QQQ', 'VIX', 'NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'META', 'AMD', 'GOOGL']
    return get_stock_prices(tickers)


def call_claude(prompt, max_tokens=2000):
    """Call Claude AI and return parsed JSON."""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    text = message.content[0].text.strip()

    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    if not text.startswith('['):
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            text = match.group(0)

    return json.loads(text)


@scanner_bp.route('/analyze', methods=['POST'])
@login_required
def analyze():
    """Swing trader - theme or daily scan."""
    data = request.get_json()
    prompt_text = data.get('prompt', '')
    if not prompt_text:
        return jsonify({'error': 'No prompt provided.'}), 400

    try:
        # First get market snapshot for real prices
        market = get_market_snapshot()
        today = datetime.now().strftime('%B %d, %Y')

        # Add real price context to prompt
        price_context = "CURRENT REAL MARKET PRICES:\n"
        for ticker, info in market.items():
            price_context += f"{ticker}: ${info['price']} ({'+' if info['change_pct'] >= 0 else ''}{info['change_pct']}%)\n"

        enhanced_prompt = prompt_text + f"\n\n{price_context}\n\nIMPORTANT: Use these REAL current prices when giving price targets and entry points. Do not use outdated prices."

        stocks = call_claude(enhanced_prompt)

        # Enrich with real prices where available
        for stock in stocks:
            ticker = stock.get('ticker', '')
            if ticker in market:
                stock['current_price'] = market[ticker]['price']
                stock['change_pct'] = market[ticker]['change_pct']

        return jsonify({'stocks': stocks}), 200

    except json.JSONDecodeError as e:
        return jsonify({'error': f'AI parse error. Try again.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@scanner_bp.route('/daytrader', methods=['POST'])
@login_required
def daytrader():
    """Day trader - options expiring today or tomorrow."""
    try:
        import yfinance as yf

        today = datetime.now().strftime('%B %d, %Y')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%B %d, %Y')
        day_of_week = datetime.now().strftime('%A')

        # Get real market data for day trading candidates
        day_trade_tickers = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AAPL', 'AMD', 'META', 'AMZN', 'MSFT', 'GOOGL', 'SMCI', 'PLTR', 'COIN', 'MSTR', 'ARM']
        market = get_stock_prices(day_trade_tickers)

        price_context = "REAL-TIME PRICES RIGHT NOW:\n"
        for ticker, info in market.items():
            price_context += f"{ticker}: ${info['price']} ({'+' if info['change_pct'] >= 0 else ''}{info['change_pct']}% today)\n"

        prompt = f"""You are an elite options day trader. Today is {today} ({day_of_week}).

{price_context}

Find the TOP 5 best options trades for TODAY and TOMORROW expiration.

Focus on:
- High liquidity options (SPY, QQQ, NVDA, TSLA, AAPL, AMD, META have the most volume)
- Clear technical setups (support/resistance, momentum)
- Options expiring {today} or {tomorrow}
- Realistic entry/exit based on the REAL prices above

For each trade give:
- CALL or PUT
- Exact strike price (realistic based on current price)
- Entry price range for the option contract
- Stop loss (when to exit if wrong)
- Take profit target
- Probability of success
- Best time to enter (market open, mid-day, etc)

Respond ONLY with JSON array (no markdown):
[{{
  "ticker": "SPY",
  "company": "S&P 500 ETF",
  "direction": "CALL",
  "current_price": 580.50,
  "strike": 582,
  "expiry": "Today",
  "option_entry": "$1.20-$1.50",
  "stop_loss": "$0.60",
  "take_profit": "$2.50-$3.00",
  "score": 85,
  "probability": "72%",
  "why": "SPY holding above key support at 580, momentum building for push to 583",
  "entry_time": "Market open 9:30-10:00 AM or pullback to support",
  "catalyst": "Specific reason for move today",
  "risk": "Main risk factor",
  "risk_reward": "1:2"
}}]"""

        trades = call_claude(prompt)

        # Add real current prices
        for trade in trades:
            ticker = trade.get('ticker', '')
            if ticker in market:
                trade['current_price'] = market[ticker]['price']
                trade['change_pct'] = market[ticker]['change_pct']

        return jsonify({'trades': trades, 'date': today}), 200

    except json.JSONDecodeError:
        return jsonify({'error': 'AI parse error. Try again.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@scanner_bp.route('/daily', methods=['GET'])
@login_required
def daily_scan():
    """Cached daily swing scan."""
    global _scan_cache
    today = date.today().isoformat()

    if _scan_cache['date'] == today and _scan_cache['results']:
        return jsonify({'stocks': _scan_cache['results'], 'cached': True, 'date': today}), 200

    market = get_market_snapshot()
    today_str = datetime.now().strftime('%B %d, %Y')
    price_context = "\n".join([f"{t}: ${i['price']} ({'+' if i['change_pct']>=0 else ''}{i['change_pct']}%)" for t, i in market.items()])

    prompt = f"""You are an elite hedge fund analyst. Today is {today_str}.

REAL CURRENT PRICES:
{price_context}

Find TOP 5 best stocks to BUY now based on current macro trends, AI boom, sector momentum, upcoming catalysts.
Use the REAL prices above for your price targets.

Respond ONLY with JSON array:
[{{"ticker":"","company":"","score":90,"action":"STRONG BUY","theme":"","type":"1st Derivative","why":"","catalyst":"","price_target":"$X by Month Year","entry":"Buy at $X-$X","risk":"","timeframe":""}}]"""

    try:
        stocks = call_claude(prompt)
        for stock in stocks:
            t = stock.get('ticker', '')
            if t in market:
                stock['current_price'] = market[t]['price']
                stock['change_pct'] = market[t]['change_pct']
        _scan_cache = {'date': today, 'results': stocks}
        return jsonify({'stocks': stocks, 'cached': False, 'date': today}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@scanner_bp.route('/analyze-chart', methods=['POST'])
@login_required
def analyze_chart():
    """AI chart analysis for a specific ticker."""
    data = request.get_json()
    ticker = data.get('ticker', '').upper().strip()

    if not ticker:
        return jsonify({'error': 'No ticker provided.'}), 400

    try:
        import yfinance as yf

        # Get real price data
        t = yf.Ticker(ticker)
        hist = t.history(period='60d')
        info = {}
        try:
            info = t.info
        except Exception:
            pass

        if hist.empty:
            return jsonify({'error': f'Could not find data for {ticker}. Check the ticker symbol.'}), 400

        current_price = round(float(hist['Close'].iloc[-1]), 2)
        prev_price = round(float(hist['Close'].iloc[-2]), 2)
        change_pct = round(((current_price - prev_price) / prev_price) * 100, 2)

        # Calculate basic technicals
        closes = hist['Close'].tolist()
        high_20 = round(max(hist['High'].tail(20).tolist()), 2)
        low_20 = round(min(hist['Low'].tail(20).tolist()), 2)
        high_5 = round(max(hist['High'].tail(5).tolist()), 2)
        low_5 = round(min(hist['Low'].tail(5).tolist()), 2)
        avg_vol_20 = round(hist['Volume'].tail(20).mean())
        today_vol = int(hist['Volume'].iloc[-1])

        # Simple moving averages
        ma20 = round(sum(closes[-20:]) / 20, 2) if len(closes) >= 20 else None
        ma50 = round(sum(closes[-50:]) / 50, 2) if len(closes) >= 50 else None

        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'Unknown')

        today = datetime.now().strftime('%B %d, %Y')

        prompt = f"""You are an elite technical analyst and options trader. Today is {today}.

REAL MARKET DATA FOR {ticker} ({company_name}):
- Current Price: ${current_price}
- Change Today: {'+' if change_pct >= 0 else ''}{change_pct}%
- 20-Day High: ${high_20}
- 20-Day Low: ${low_20}
- 5-Day High: ${high_5}
- 5-Day Low: ${low_5}
- 20-Day MA: ${ma20}
- 50-Day MA: ${ma50}
- Today's Volume: {today_vol:,}
- Avg 20-Day Volume: {avg_vol_20:,}
- Sector: {sector}

Based on this REAL data, provide a complete technical analysis.

Respond ONLY with JSON (no markdown):
{{
  "ticker": "{ticker}",
  "company": "{company_name}",
  "current_price": {current_price},
  "change_pct": {change_pct},
  "signal": "BULLISH",
  "confidence": 82,
  "signal_reason": "One sentence why bullish/bearish/neutral",
  "trend": "Detailed trend analysis 2-3 sentences. Is it uptrend downtrend sideways. Where is it relative to MAs.",
  "resistance": {high_20},
  "support": {low_20},
  "price_target": 0.00,
  "stop_loss": 0.00,
  "prediction": "What you expect in next 5-10 days and why based on the data",
  "risks": "Key risks that could invalidate this analysis",
  "call_option": "Strike: $X | Expiry: X weeks out | Why: reason<br>Best if price breaks above $X resistance",
  "put_option": "Strike: $X | Expiry: X weeks out | Why: reason<br>Best if price breaks below $X support",
  "summary": "2-3 sentence Wolf AI summary of the overall picture and what a trader should do"
}}"""

        result = call_claude(prompt)

        # Make sure it's a dict not list
        if isinstance(result, list):
            result = result[0]

        return jsonify(result), 200

    except json.JSONDecodeError:
        return jsonify({'error': 'AI parse error. Try again.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
