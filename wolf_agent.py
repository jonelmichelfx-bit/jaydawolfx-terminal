"""
WOLF AGENT - JayDaWolfX Terminal
Real chart reading engine. No price prediction. Just structure, S/R, EMA, pattern.
Trained on: Volman (Price Action), BabyPips (Technical Analysis), Fidelity (Chart Patterns)
"""

import os, json, time, threading, requests
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request, render_template
from anthropic import Anthropic

wolf_bp = Blueprint('wolf', __name__)

TWELVE_DATA_KEY = os.environ.get('TWELVE_DATA_API_KEY', '')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')

# ─── Job store ──────────────────────────────────────────────────────────────
_wolf_jobs = {}

# ─── Instruments to scan ────────────────────────────────────────────────────
FOREX_PAIRS = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD',
    'USD/CAD', 'USD/CHF', 'NZD/USD', 'EUR/GBP',
    'GBP/JPY', 'EUR/JPY', 'AUD/JPY', 'EUR/CAD',
    'GBP/CAD', 'USD/MXN'
]
STOCKS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'AMD', 'GOOGL']
OPTIONS_INSTRUMENTS = ['SPY', 'QQQ', 'SPX']  # 0DTE targets

# ─── TwelveData helpers ──────────────────────────────────────────────────────

def td_symbol(pair):
    """Convert EUR/USD to EUR/USD format TwelveData understands."""
    return pair.replace('/', '/')

def fetch_ohlc(symbol, interval='1day', outputsize=35):
    """Fetch real OHLC candles from TwelveData."""
    try:
        # TwelveData uses different format for forex vs stocks
        sym = symbol.replace('/', '')  # EURUSD for forex
        if '/' in symbol:
            url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_DATA_KEY}"
        else:
            url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_DATA_KEY}"

        r = requests.get(url, timeout=10)
        data = r.json()

        if 'values' not in data:
            return None

        candles = []
        for v in reversed(data['values']):  # oldest first
            candles.append({
                'date': v['datetime'],
                'open':  float(v['open']),
                'high':  float(v['high']),
                'low':   float(v['low']),
                'close': float(v['close']),
                'volume': float(v.get('volume', 0))
            })
        return candles
    except Exception as e:
        return None

def fetch_current_price(symbol):
    """Get real-time quote."""
    try:
        url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_DATA_KEY}"
        r = requests.get(url, timeout=5)
        data = r.json()
        return float(data.get('price', 0))
    except:
        return None

# ─── Technical Analysis Engine ──────────────────────────────────────────────

def calc_ema(closes, period):
    """Calculate EMA from close prices."""
    if len(closes) < period:
        return []
    k = 2.0 / (period + 1)
    ema = [sum(closes[:period]) / period]
    for price in closes[period:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return ema

def calc_atr(candles, period=14):
    """Average True Range for SL sizing."""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]['high']
        l = candles[i]['low']
        pc = candles[i-1]['close']
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[-period:]) / period
    return round(atr, 5)

def calc_adx(candles, period=14):
    """ADX for trend strength. >25 = trending, <20 = ranging."""
    if len(candles) < period * 2:
        return None
    try:
        plus_dm, minus_dm, tr_list = [], [], []
        for i in range(1, len(candles)):
            h, l = candles[i]['high'], candles[i]['low']
            ph, pl, pc = candles[i-1]['high'], candles[i-1]['low'], candles[i-1]['close']
            up_move = h - ph
            down_move = pl - l
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
            tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))

        def smooth(lst, p):
            s = sum(lst[:p])
            result = [s]
            for x in lst[p:]:
                result.append(result[-1] - result[-1]/p + x)
            return result

        tr_s = smooth(tr_list, period)
        pdm_s = smooth(plus_dm, period)
        mdm_s = smooth(minus_dm, period)

        dx_list = []
        for i in range(len(tr_s)):
            pdi = 100 * pdm_s[i] / tr_s[i] if tr_s[i] else 0
            mdi = 100 * mdm_s[i] / tr_s[i] if tr_s[i] else 0
            dx = 100 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) else 0
            dx_list.append(dx)

        if len(dx_list) < period:
            return None
        adx = sum(dx_list[-period:]) / period
        return round(adx, 1)
    except:
        return None

def find_sr_zones(candles, sensitivity=0.003):
    """
    Find REAL S/R zones using pivot highs/lows with minimum 2 touches.
    Volman principle: only trade levels that have been TESTED at least twice.
    """
    if len(candles) < 10:
        return [], []

    # Find pivot highs (local max) and pivot lows (local min)
    pivots_high = []
    pivots_low = []
    lookback = 3

    for i in range(lookback, len(candles) - lookback):
        h = candles[i]['high']
        l = candles[i]['low']

        is_ph = all(candles[i]['high'] >= candles[j]['high'] for j in range(i-lookback, i+lookback+1) if j != i)
        is_pl = all(candles[i]['low'] <= candles[j]['low'] for j in range(i-lookback, i+lookback+1) if j != i)

        if is_ph:
            pivots_high.append({'price': h, 'date': candles[i]['date'], 'idx': i})
        if is_pl:
            pivots_low.append({'price': l, 'date': candles[i]['date'], 'idx': i})

    current_price = candles[-1]['close']
    margin = current_price * sensitivity

    def cluster_levels(pivots):
        """Group nearby pivots into zones, count touches."""
        if not pivots:
            return []
        zones = []
        used = set()

        for i, p in enumerate(pivots):
            if i in used:
                continue
            cluster = [p['price']]
            dates = [p['date']]
            used.add(i)

            for j, q in enumerate(pivots):
                if j not in used and abs(p['price'] - q['price']) <= margin * 2:
                    cluster.append(q['price'])
                    dates.append(q['date'])
                    used.add(j)

            if len(cluster) >= 2:  # MINIMUM 2 TOUCHES - Volman rule
                zone_price = sum(cluster) / len(cluster)
                zones.append({
                    'price': round(zone_price, 5),
                    'touches': len(cluster),
                    'last_touch': max(dates),
                    'strength': 'STRONG' if len(cluster) >= 3 else 'MODERATE'
                })

        return sorted(zones, key=lambda z: abs(z['price'] - current_price))

    resistance_zones = cluster_levels(pivots_high)
    support_zones = cluster_levels(pivots_low)

    # Filter: above price = resistance, below = support
    resistances = [z for z in resistance_zones if z['price'] > current_price][:4]
    supports = [z for z in support_zones if z['price'] < current_price][:4]

    return supports, resistances

def detect_trend_structure(candles):
    """
    BabyPips + Volman: real trend = higher highs + higher lows (bull) 
    or lower highs + lower lows (bear). No indicator needed.
    """
    if len(candles) < 10:
        return 'UNKNOWN', 0

    recent = candles[-20:]  # Last 20 candles
    highs = [c['high'] for c in recent]
    lows = [c['low'] for c in recent]

    # Count HH/HL vs LH/LL transitions
    hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    hl = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
    lh = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
    ll = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])

    bull_score = hh + hl
    bear_score = lh + ll
    total = bull_score + bear_score

    if total == 0:
        return 'SIDEWAYS', 0

    bull_pct = (bull_score / total) * 100
    bear_pct = (bear_score / total) * 100

    if bull_pct >= 60:
        return 'UPTREND', round(bull_pct)
    elif bear_pct >= 60:
        return 'DOWNTREND', round(bear_pct)
    else:
        return 'SIDEWAYS', round(max(bull_pct, bear_pct))

def score_pair_for_trend(candles):
    """
    Score a pair 0-100 for trend quality.
    High ADX + clear structure + price away from EMA center = trending.
    """
    if not candles or len(candles) < 30:
        return 0

    closes = [c['close'] for c in candles]
    adx = calc_adx(candles) or 0
    trend, strength = detect_trend_structure(candles)

    ema100 = calc_ema(closes, 100)
    ema_score = 0
    if ema100:
        current = closes[-1]
        ema_val = ema100[-1]
        # Price far from EMA = trending, at EMA = ranging
        pct_away = abs(current - ema_val) / ema_val * 100
        ema_score = min(pct_away * 5, 30)  # max 30 pts

    adx_score = min(adx, 50)  # max 50 pts
    structure_score = strength * 0.2  # max 20 pts (0-100% * 0.2)

    if trend == 'SIDEWAYS':
        return int(adx_score * 0.3)  # Penalize sideways hard

    total = adx_score + structure_score + ema_score
    return int(min(total, 100))

# ─── News Check ──────────────────────────────────────────────────────────────

def check_news_risk(symbol):
    """
    Check NewsAPI for high-impact news in the next 24 hours.
    Returns warning string if risk found, else None.
    """
    try:
        if not NEWS_API_KEY:
            return None

        # Extract currency/stock from symbol
        terms = symbol.replace('/', ' ').replace('USD', 'dollar').replace('EUR', 'euro')
        terms = terms.replace('GBP', 'pound').replace('JPY', 'yen')

        # Check for economic events keywords
        economic_keywords = ['NFP', 'CPI', 'Fed', 'FOMC', 'interest rate', 'jobs report',
                           'inflation', 'GDP', 'retail sales', 'earnings']

        url = (f"https://newsapi.org/v2/everything?q={symbol[:3]}+forex+OR+{symbol[:3]}+economy"
               f"&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}")
        r = requests.get(url, timeout=5)
        articles = r.json().get('articles', [])

        high_impact = []
        for a in articles:
            title = (a.get('title', '') + ' ' + a.get('description', '')).lower()
            for kw in economic_keywords:
                if kw.lower() in title:
                    high_impact.append(kw)
                    break

        if high_impact:
            return f"⚠️ HIGH IMPACT NEWS RISK: {', '.join(set(high_impact[:2]))} found in recent news — trade with caution or avoid"
        return None
    except:
        return None

# ─── Wolf Agent Claude Prompt ────────────────────────────────────────────────

WOLF_SYSTEM_PROMPT = """You are WOLF AGENT — a professional market analyst and chart reader trained in:
- Bob Volman's Price Action (Forex Price Action Scalping / Understanding Price Action)
- BabyPips School of Pipsology (Technical Analysis, S/R, Candlesticks, Trend)  
- Fidelity Chart Patterns (Double Top/Bottom, H&S, Triangles, Flags, Wedges)
- 0DTE Options (SPY/QQQ directional plays, credit spreads at S/R)

YOUR MISSION:
Read the REAL price data provided. Analyze market structure. Give a professional trading signal.

STRICT RULES — NO EXCEPTIONS:
1. ZERO price prediction. You read what IS happening, not what WILL happen.
2. Only recommend BUY if price is ABOVE the 100 EMA and in uptrend structure
3. Only recommend SELL if price is BELOW the 100 EMA and in downtrend structure
4. S/R zones are only valid if they have 2+ REAL price touches (provided in data)
5. Signal requires: trend + EMA alignment + price at tested S/R zone + no high-impact news
6. If ANY condition is missing → signal = WAIT, never force a trade
7. SL = beyond the nearest S/R zone + ATR buffer (never arbitrary)
8. TP1 = next S/R zone, TP2 = next zone after that, TP3 = major swing high/low

VOLMAN PRINCIPLES TO APPLY:
- Buildup at S/R (consolidation before break) = HIGH CONFIDENCE entry
- Price shooting through S/R with no buildup = SKIP (false break risk)
- False break (wick through level, close back inside) = trap, do NOT enter
- Double pressure (both bulls and bears same direction) = strongest moves
- EMA 100 is your trend FILTER, not your entry trigger

BABYPIPS PRINCIPLES:
- Higher highs + higher lows = uptrend confirmed
- Lower highs + lower lows = downtrend confirmed  
- Price at 50% retracement of last swing = higher probability bounce
- Volume spike at S/R = confirmation (if available)
- ADX > 25 = trending, < 20 = ranging (don't trade ranging pairs)

FIDELITY PATTERNS:
- Pattern NOT valid until price CLOSES beyond the level (no wick entries)
- Double top/bottom at tested S/R = high probability reversal
- Bull/bear flag = continuation after pullback to 38-50% retracement
- Ascending/descending triangle at S/R = compression before breakout
- H&S neckline break = strongest reversal signal

0DTE OPTIONS RULES (SPY/QQQ only):
- Only enter AFTER opening range established (9:45-10:15 AM ET)
- Use previous day high/low as key S/R levels
- BUY CALLS: price above premarket high + above 100 EMA + ADX > 25
- BUY PUTS: price below premarket low + below 100 EMA + ADX > 25
- Strike: ATM for strong trend, OTM when waiting for retest
- NEVER trade 0DTE when: news in 2 hours, ADX < 20, price in middle of range
- Best 0DTE entry: 10:15-11:00 AM ET window after direction confirmed

RESPOND ONLY IN THIS EXACT JSON FORMAT — no text outside the JSON:
{
  "signal": "BUY" | "SELL" | "WAIT",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "bias": "BULLISH" | "BEARISH" | "NEUTRAL",
  "entry_zone": {"low": 0.0, "high": 0.0},
  "stop_loss": 0.0,
  "tp1": 0.0,
  "tp2": 0.0,
  "tp3": 0.0,
  "risk_reward": "1:2.0",
  "ema100_status": "price above/below 100 EMA — exact value",
  "trend_structure": "exact structure observed from price data",
  "key_level": "nearest S/R zone with touch count and type",
  "pattern_detected": "pattern name or NONE",
  "buildup_present": true | false,
  "analysis": "2-3 sentence professional chart reading based ONLY on data provided. No predictions.",
  "news_warning": null | "warning string",
  "option_play": null | {"type": "CALL/PUT", "strike_guidance": "ATM/OTM", "rationale": "..."}
}"""

def run_wolf_analysis(symbol, candles, current_price, supports, resistances,
                      trend, trend_strength, adx, ema100_val, atr, news_warning, is_option=False):
    """Feed real data to Claude for chart analysis. No prediction, pure reading."""

    # Build data payload
    recent_candles = candles[-10:]  # Last 10 candles for Claude to read
    candle_summary = []
    for c in recent_candles:
        body = abs(c['close'] - c['open'])
        wick_up = c['high'] - max(c['open'], c['close'])
        wick_dn = min(c['open'], c['close']) - c['low']
        direction = 'BULL' if c['close'] > c['open'] else 'BEAR'
        candle_summary.append({
            'date': c['date'],
            'O': round(c['open'], 5), 'H': round(c['high'], 5),
            'L': round(c['low'], 5),  'C': round(c['close'], 5),
            'direction': direction,
            'body_size': round(body, 5),
            'upper_wick': round(wick_up, 5),
            'lower_wick': round(wick_dn, 5)
        })

    # 30-day high/low for context
    highs_30 = [c['high'] for c in candles]
    lows_30 = [c['low'] for c in candles]
    high_30d = round(max(highs_30), 5) if highs_30 else 0
    low_30d = round(min(lows_30), 5) if lows_30 else 0
    mid_30d = round((high_30d + low_30d) / 2, 5)

    # Price position
    price_pct = round(((current_price - low_30d) / (high_30d - low_30d) * 100) if (high_30d - low_30d) > 0 else 50, 1)

    user_message = f"""SYMBOL: {symbol}
CURRENT PRICE: {current_price}
30-DAY RANGE: High={high_30d} | Low={low_30d} | Mid={mid_30d}
PRICE POSITION: {price_pct}% of 30-day range (0%=low, 100%=high)

100 EMA: {round(ema100_val, 5) if ema100_val else 'NOT AVAILABLE'}
PRICE vs 100 EMA: {'ABOVE' if ema100_val and current_price > ema100_val else 'BELOW' if ema100_val else 'UNKNOWN'}
ADX (Trend Strength): {adx if adx else 'NOT AVAILABLE'} {'(TRENDING)' if adx and adx > 25 else '(RANGING/WEAK)' if adx else ''}
ATR (Volatility): {atr}

TREND STRUCTURE: {trend} ({trend_strength}% score)

REAL SUPPORT ZONES (minimum 2 touches each):
{json.dumps(supports[:3], indent=2) if supports else 'None found in 30-day data'}

REAL RESISTANCE ZONES (minimum 2 touches each):
{json.dumps(resistances[:3], indent=2) if resistances else 'None found in 30-day data'}

LAST 10 CANDLES (newest last):
{json.dumps(candle_summary, indent=2)}

NEWS RISK: {news_warning if news_warning else 'NONE DETECTED'}

{'OPTIONS CONTEXT: This is a 0DTE options analysis request for SPY/QQQ. Apply 0DTE rules.' if is_option else ''}

TASK: Read this data like a professional chart analyst. Apply Volman, BabyPips, and Fidelity principles.
Give me the trade signal based purely on what this data shows. No forecasting. No fluff."""

    client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1000,
        system=WOLF_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()

    # Parse JSON
    try:
        # Strip any markdown fences
        if '```' in raw:
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        result = json.loads(raw.strip())
        result['symbol'] = symbol
        result['current_price'] = current_price
        result['atr'] = atr
        result['adx'] = adx
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        return result
    except Exception as e:
        return {
            'symbol': symbol,
            'signal': 'ERROR',
            'error': f'Parse error: {str(e)}',
            'raw': raw[:200],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        }

# ─── Main Wolf Scan Job ──────────────────────────────────────────────────────

def wolf_scan_job(job_id, scan_type='all'):
    """
    Full Wolf Agent scan:
    1. Find top 3 trending forex pairs
    2. Analyze top 3 trending stocks  
    3. Best 0DTE options setup
    4. Give clean signals for each
    """
    job = _wolf_jobs[job_id]

    try:
        results = {
            'forex': [],
            'stocks': [],
            'options': [],
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'market_session': get_market_session()
        }

        # ── PHASE 1: Scan forex pairs for trending ──────────────────────────
        if scan_type in ('all', 'forex'):
            job['step'] = '🔍 Scanning 14 forex pairs for trends...'
            scored_forex = []

            for pair in FOREX_PAIRS:
                candles = fetch_ohlc(pair, interval='1day', outputsize=35)
                if candles and len(candles) >= 20:
                    score = score_pair_for_trend(candles)
                    trend, strength = detect_trend_structure(candles)
                    adx = calc_adx(candles)
                    if trend != 'SIDEWAYS' and score > 40:
                        scored_forex.append({
                            'pair': pair, 'score': score,
                            'trend': trend, 'adx': adx, 'candles': candles
                        })
                time.sleep(0.3)  # Rate limit

            # Top 3 trending pairs
            top_forex = sorted(scored_forex, key=lambda x: x['score'], reverse=True)[:3]

            job['step'] = f'✅ Found {len(top_forex)} trending pairs. Running deep analysis...'

            for item in top_forex:
                pair = item['pair']
                candles = item['candles']
                job['step'] = f'📊 Analyzing {pair}...'

                closes = [c['close'] for c in candles]
                current_price = fetch_current_price(pair) or closes[-1]
                ema100_list = calc_ema(closes, 100)
                ema100_val = ema100_list[-1] if ema100_list else None
                atr = calc_atr(candles)
                adx = item['adx']
                trend, strength = detect_trend_structure(candles)
                supports, resistances = find_sr_zones(candles)
                news = check_news_risk(pair)

                analysis = run_wolf_analysis(
                    pair, candles, current_price, supports, resistances,
                    trend, strength, adx, ema100_val, atr, news, is_option=False
                )
                analysis['trend_score'] = item['score']
                results['forex'].append(analysis)
                time.sleep(1)

        # ── PHASE 2: Scan stocks ─────────────────────────────────────────────
        if scan_type in ('all', 'stocks'):
            job['step'] = '📈 Scanning stocks for trends...'
            scored_stocks = []

            for stock in STOCKS:
                candles = fetch_ohlc(stock, interval='1day', outputsize=35)
                if candles and len(candles) >= 20:
                    score = score_pair_for_trend(candles)
                    trend, _ = detect_trend_structure(candles)
                    adx = calc_adx(candles)
                    if trend != 'SIDEWAYS' and score > 35:
                        scored_stocks.append({
                            'symbol': stock, 'score': score,
                            'trend': trend, 'adx': adx, 'candles': candles
                        })
                time.sleep(0.3)

            top_stocks = sorted(scored_stocks, key=lambda x: x['score'], reverse=True)[:3]
            job['step'] = f'✅ Found {len(top_stocks)} trending stocks. Analyzing...'

            for item in top_stocks:
                symbol = item['symbol']
                candles = item['candles']
                job['step'] = f'📊 Analyzing {symbol}...'

                closes = [c['close'] for c in candles]
                current_price = fetch_current_price(symbol) or closes[-1]
                ema100_list = calc_ema(closes, 100)
                ema100_val = ema100_list[-1] if ema100_list else None
                atr = calc_atr(candles)
                adx = item['adx']
                trend, strength = detect_trend_structure(candles)
                supports, resistances = find_sr_zones(candles)
                news = check_news_risk(symbol)

                analysis = run_wolf_analysis(
                    symbol, candles, current_price, supports, resistances,
                    trend, strength, adx, ema100_val, atr, news, is_option=False
                )
                analysis['trend_score'] = item['score']
                results['stocks'].append(analysis)
                time.sleep(1)

        # ── PHASE 3: 0DTE Options ─────────────────────────────────────────────
        if scan_type in ('all', 'options'):
            job['step'] = '⚡ Analyzing 0DTE options setups (SPY/QQQ)...'

            for opt_sym in OPTIONS_INSTRUMENTS[:2]:  # SPY and QQQ
                job['step'] = f'⚡ 0DTE setup for {opt_sym}...'

                # Use 15min candles for intraday 0DTE context
                candles_15m = fetch_ohlc(opt_sym, interval='15min', outputsize=40)
                candles_1d = fetch_ohlc(opt_sym, interval='1day', outputsize=35)

                if not candles_1d:
                    continue

                closes_1d = [c['close'] for c in candles_1d]
                current_price = fetch_current_price(opt_sym) or closes_1d[-1]
                ema100_list = calc_ema(closes_1d, 100)
                ema100_val = ema100_list[-1] if ema100_list else None
                atr = calc_atr(candles_1d)
                adx = calc_adx(candles_1d)
                trend, strength = detect_trend_structure(candles_1d)
                supports, resistances = find_sr_zones(candles_1d)

                # Add previous day levels as critical 0DTE S/R
                prev_day = candles_1d[-2] if len(candles_1d) >= 2 else None
                if prev_day:
                    supports.insert(0, {
                        'price': round(prev_day['low'], 3),
                        'touches': 1,
                        'last_touch': prev_day['date'],
                        'strength': 'PREV-DAY-LOW',
                        'note': 'Previous Day Low'
                    })
                    resistances.insert(0, {
                        'price': round(prev_day['high'], 3),
                        'touches': 1,
                        'last_touch': prev_day['date'],
                        'strength': 'PREV-DAY-HIGH',
                        'note': 'Previous Day High'
                    })

                news = check_news_risk(opt_sym)

                analysis = run_wolf_analysis(
                    opt_sym, candles_1d, current_price, supports, resistances,
                    trend, strength, adx, ema100_val, atr, news, is_option=True
                )
                analysis['instrument_type'] = '0DTE'
                results['options'].append(analysis)
                time.sleep(1)

        job['status'] = 'done'
        job['result'] = results
        job['step'] = '✅ Wolf Agent analysis complete'

    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)
        job['step'] = f'❌ Error: {str(e)}'

def get_market_session():
    """Returns current trading session."""
    now = datetime.utcnow()
    hour = now.hour
    if 22 <= hour or hour < 7:
        return 'ASIA'
    elif 7 <= hour < 12:
        return 'LONDON'
    elif 13 <= hour < 17:
        return 'NEW YORK'
    elif 12 <= hour < 13:
        return 'LONDON/NY OVERLAP'
    else:
        return 'AFTER HOURS'

# ─── Routes ──────────────────────────────────────────────────────────────────

@wolf_bp.route('/wolf-agent')
def wolf_agent_page():
    return render_template('wolf_agent.html')

@wolf_bp.route('/api/wolf-scan', methods=['POST'])
def wolf_scan():
    data = request.get_json() or {}
    scan_type = data.get('scan_type', 'all')

    job_id = f"wolf_{int(time.time() * 1000)}"
    _wolf_jobs[job_id] = {
        'status': 'running',
        'step': '🐺 Wolf Agent initializing...',
        'result': None
    }

    t = threading.Thread(target=wolf_scan_job, args=(job_id, scan_type), daemon=True)
    t.start()

    return jsonify({'job_id': job_id, 'status': 'starting'})

@wolf_bp.route('/api/wolf-poll/<job_id>')
def wolf_poll(job_id):
    job = _wolf_jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not_found'}), 404

    if job['status'] == 'done':
        result = job['result']
        del _wolf_jobs[job_id]
        return jsonify({'status': 'done', 'result': result})

    if job['status'] == 'error':
        error = job.get('error', 'Unknown error')
        del _wolf_jobs[job_id]
        return jsonify({'status': 'error', 'error': error})

    return jsonify({'status': 'running', 'step': job.get('step', 'Processing...')})
