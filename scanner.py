import os
import re
import json
from datetime import datetime, date
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from decorators import pro_required
import anthropic

scanner_bp = Blueprint('scanner', __name__, url_prefix='/scanner')

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

# Cache to store daily scan results (resets each day)
_scan_cache = {
    'date': None,
    'results': None
}


def get_market_context():
    """Get current stock prices for context using yfinance."""
    try:
        import yfinance as yf
        tickers = ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'META', 'GOOGL', 'AMD']
        data = {}
        for ticker in tickers:
            try:
                hist = yf.Ticker(ticker).history(period='2d')
                if len(hist) >= 2:
                    current = float(hist['Close'].iloc[-1])
                    prev = float(hist['Close'].iloc[-2])
                    change_pct = ((current - prev) / prev) * 100
                    data[ticker] = {
                        'price': round(current, 2),
                        'change_pct': round(change_pct, 2)
                    }
            except Exception:
                pass
        return data
    except Exception:
        return {}


def run_ai_scan(theme=None):
    """Run AI analysis to find top stock opportunities."""
    
    market_data = get_market_context()
    
    today = datetime.now().strftime('%B %d, %Y')
    
    if theme:
        prompt = f"""You are an expert stock analyst and financial researcher. Today is {today}.

A trader is asking about stocks related to this theme: "{theme}"

Current market snapshot:
{json.dumps(market_data, indent=2)}

Your job:
1. Analyze the "{theme}" theme and find the TOP 5 stocks most likely to benefit
2. For each stock, explain WHY it will benefit from this theme
3. Rate each stock's opportunity score from 0-100
4. Consider both direct plays (1st derivative) and indirect plays (2nd derivative)

Respond ONLY with a JSON array, no other text:
[
  {{
    "ticker": "NVDA",
    "company": "NVIDIA Corporation", 
    "score": 92,
    "theme": "AI Chips",
    "why": "Direct beneficiary of AI boom - makes the GPUs that power all AI training",
    "catalyst": "Data center revenue up 400% YoY, new Blackwell chip demand exceeding supply",
    "risk": "High valuation, China export restrictions",
    "timeframe": "3-6 months",
    "type": "1st Derivative"
  }}
]"""
    else:
        prompt = f"""You are an expert stock analyst and financial researcher. Today is {today}.

Your job is to scan the market and find the TOP 5 stock opportunities RIGHT NOW based on:
- Current macro trends (AI, energy, geopolitics, interest rates)
- Sector momentum
- Recent catalysts
- Undervalued opportunities

Current market snapshot:
{json.dumps(market_data, indent=2)}

Find stocks that are "about to move" because of what's happening in the world RIGHT NOW.
Think about: AI infrastructure, energy transition, defense, biotech breakthroughs, reshoring/manufacturing.

Respond ONLY with a JSON array, no other text:
[
  {{
    "ticker": "NVDA",
    "company": "NVIDIA Corporation",
    "score": 92,
    "theme": "AI Infrastructure",
    "why": "Direct beneficiary of AI boom - makes the GPUs that power all AI training",
    "catalyst": "Data center revenue up 400% YoY, new Blackwell chip demand exceeding supply",
    "risk": "High valuation, China export restrictions",
    "timeframe": "3-6 months",
    "type": "1st Derivative"
  }}
]"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        # Clean up response - remove markdown if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        stocks = json.loads(response_text)
        
        # Add current price data if available
        for stock in stocks:
            ticker = stock.get('ticker', '')
            if ticker in market_data:
                stock['current_price'] = market_data[ticker]['price']
                stock['change_pct'] = market_data[ticker]['change_pct']
        
        return stocks, None
        
    except json.JSONDecodeError as e:
        return None, f"AI response parsing error: {str(e)}"
    except Exception as e:
        return None, str(e)


@scanner_bp.route('/daily', methods=['GET'])
@pro_required
def daily_scan():
    """Returns today's top stock picks - cached daily."""
    global _scan_cache
    
    today = date.today().isoformat()
    
    # Return cached results if already ran today
    if _scan_cache['date'] == today and _scan_cache['results']:
        return jsonify({
            'stocks': _scan_cache['results'],
            'cached': True,
            'date': today,
            'message': 'Daily scan results'
        }), 200
    
    # Run fresh scan
    stocks, error = run_ai_scan()
    
    if error:
        return jsonify({'error': error}), 500
    
    # Cache results
    _scan_cache['date'] = today
    _scan_cache['results'] = stocks
    
    return jsonify({
        'stocks': stocks,
        'cached': False,
        'date': today,
        'message': 'Fresh daily scan complete'
    }), 200


@scanner_bp.route('/theme', methods=['POST'])
@pro_required
def theme_scan():
    """Search stocks by theme."""
    data = request.get_json()
    theme = data.get('theme', '').strip()
    
    if not theme:
        return jsonify({'error': 'Please enter a theme to search.'}), 400
    
    if len(theme) > 100:
        return jsonify({'error': 'Theme too long. Keep it under 100 characters.'}), 400
    
    stocks, error = run_ai_scan(theme=theme)
    
    if error:
        return jsonify({'error': error}), 500
    
    return jsonify({
        'stocks': stocks,
        'theme': theme,
        'date': date.today().isoformat(),
        'message': f'Top stocks for theme: {theme}'
    }), 200


@scanner_bp.route('/refresh', methods=['POST'])
@pro_required  
def refresh_scan():
    """Force refresh the daily scan."""
    global _scan_cache
    
    stocks, error = run_ai_scan()
    
    if error:
        return jsonify({'error': error}), 500
    
    today = date.today().isoformat()
    _scan_cache['date'] = today
    _scan_cache['results'] = stocks
    
    return jsonify({
        'stocks': stocks,
        'cached': False,
        'date': today,
        'message': 'Scan refreshed'
    }), 200


@scanner_bp.route('/analyze', methods=['POST'])
@login_required
def analyze():
    """Generic AI analysis endpoint - called from browser."""
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided.'}), 400

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = message.content[0].text.strip()
        
        # Try to extract JSON from response
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        # Try to find JSON array in text using regex
        if not text.startswith('['):
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                text = match.group(0)
        
        stocks = json.loads(text)
        return jsonify({'stocks': stocks}), 200
        
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Parse error: {str(e)}. Raw: {text[:200]}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
