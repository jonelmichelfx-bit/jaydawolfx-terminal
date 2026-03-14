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

def calc_rsi(closes, period=14):
    """Wilder's Smoothed RSI — same formula as TradingView/MT4."""
    if len(closes) < period + 2:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 1)

def calc_macd(closes):
    """EMA12 - EMA26. Returns (macd_val, bias)."""
    if len(closes) < 26:
        return None, None
    ema12 = calc_ema(closes[-50:], 12)
    ema26 = calc_ema(closes[-50:], 26)
    if ema12 and ema26:
        macd = round(ema12 - ema26, 5)
        return macd, 'BULLISH' if macd > 0 else 'BEARISH'
    return None, None

def calc_bollinger(closes, period=20, num_std=2):
    """Bollinger Bands — upper, mid, lower."""
    if len(closes) < period:
        return None, None, None
    recent = closes[-period:]
    mid = sum(recent) / period
    variance = sum((x - mid) ** 2 for x in recent) / period
    std = variance ** 0.5
    upper = round(mid + num_std * std, 5)
    lower = round(mid - num_std * std, 5)
    return upper, round(mid, 5), lower

def find_sr_simple(candles, current_price):
    """Real S/R from swing highs/lows — min 2 touches (Volman rule)."""
    if len(candles) < 10:
        return [], []
    highs = [c['high'] for c in candles]
    lows  = [c['low']  for c in candles]
    margin = current_price * 0.003
    levels = []
    for i in range(2, len(candles) - 2):
        if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
            levels.append({'price': highs[i], 'type': 'R'})
        if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
            levels.append({'price': lows[i], 'type': 'S'})
    clustered = []
    for lv in levels:
        merged = False
        for cl in clustered:
            if abs(lv['price'] - cl['price']) < margin:
                cl['touches'] = cl.get('touches', 1) + 1
                merged = True
                break
        if not merged:
            clustered.append({'price': round(lv['price'], 5), 'type': lv['type'], 'touches': 1})
    valid = [l for l in clustered if l['touches'] >= 2]
    if not valid:
        valid = sorted(clustered, key=lambda x: abs(x['price'] - current_price))[:6]
    supports    = [l for l in valid if l['price'] < current_price]
    resistances = [l for l in valid if l['price'] > current_price]
    return (sorted(supports,    key=lambda x: x['price'], reverse=True)[:3],
            sorted(resistances, key=lambda x: x['price'])[:3])


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

def fetch_wolf_chart_data(symbol, current_price, daily_candles=None):
    """
    Full 5-timeframe chart analysis for Wolf Agent.
    Accepts pre-fetched daily candles to avoid duplicate API calls.
    Only fetches Weekly/H4/H1/M15 fresh — reuses daily from scan job.
    """
    data = {'symbol': symbol, 'price': current_price,
            'weekly': {}, 'daily': {}, 'h4': {}, 'h1': {}, 'm15': {},
            'supports': [], 'resistances': [], 'patterns': {}}

    def tf_block(candles):
        if not candles or len(candles) < 10:
            return {}
        closes = [c['close'] for c in candles]
        highs  = [c['high']  for c in candles]
        lows   = [c['low']   for c in candles]
        ema9   = calc_ema(closes, min(9,   len(closes)))
        ema20  = calc_ema(closes, min(20,  len(closes)))
        ema50  = calc_ema(closes, min(50,  len(closes)))
        ema100 = calc_ema(closes, min(100, len(closes)))
        ema200 = calc_ema(closes, min(200, len(closes)))
        rsi    = calc_rsi(closes)
        _, macd_bias = calc_macd(closes)
        atr    = calc_atr(candles)
        adx    = calc_adx(candles)
        trend, strength = detect_trend_structure(candles)
        bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
        bull = sum([
            1 if ema9   and current_price > ema9   else 0,
            1 if ema20  and current_price > ema20  else 0,
            1 if ema50  and current_price > ema50  else 0,
            1 if macd_bias == 'BULLISH' else 0,
        ])
        tf_trend = 'BULLISH' if bull >= 3 else 'BEARISH' if bull <= 1 else 'MIXED'
        return {
            'trend': tf_trend, 'structure': trend, 'strength': strength,
            'ema9':   round(ema9,   5) if ema9   else None,
            'ema20':  round(ema20,  5) if ema20  else None,
            'ema50':  round(ema50,  5) if ema50  else None,
            'ema100': round(ema100, 5) if ema100 else None,
            'ema200': round(ema200, 5) if ema200 else None,
            'rsi': rsi, 'macd': macd_bias, 'atr': atr, 'adx': adx,
            'adx_signal': 'TRENDING' if adx and adx > 25 else 'RANGING' if adx and adx < 20 else 'WEAK',
            'vs_ema100': 'ABOVE' if ema100 and current_price > ema100 else 'BELOW',
            'vs_ema200': 'ABOVE' if ema200 and current_price > ema200 else 'BELOW',
            'bb_upper': bb_upper, 'bb_mid': bb_mid, 'bb_lower': bb_lower,
            'high': round(max(highs[-20:]), 5),
            'low':  round(min(lows[-20:]),  5),
            'candles': len(candles),
        }

    # ── Daily — reuse pre-fetched candles from scan job (no extra API call) ──
    d1_c = daily_candles
    if not d1_c:
        try:
            d1_c = fetch_ohlc(symbol, interval='1day', outputsize=60)
        except: pass
    if d1_c:
        data['daily'] = tf_block(d1_c)
        sup, res = find_sr_simple(d1_c, current_price)
        data['supports']    = sup
        data['resistances'] = res

    # ── Weekly ────────────────────────────────────────────────────────────────
    try:
        wk_c = fetch_ohlc(symbol, interval='1week', outputsize=52)
        data['weekly'] = tf_block(wk_c) if wk_c else {}
    except: pass
    time.sleep(0.15)

    # ── H4 ────────────────────────────────────────────────────────────────────
    try:
        h4_c = fetch_ohlc(symbol, interval='4h', outputsize=60)
        data['h4'] = tf_block(h4_c) if h4_c else {}
    except: pass
    time.sleep(0.15)

    # ── H1 ────────────────────────────────────────────────────────────────────
    try:
        h1_c = fetch_ohlc(symbol, interval='1h', outputsize=48)
        data['h1'] = tf_block(h1_c) if h1_c else {}
    except: pass
    time.sleep(0.15)

    # ── M15 ───────────────────────────────────────────────────────────────────
    try:
        m15_c = fetch_ohlc(symbol, interval='15min', outputsize=40)
        data['m15'] = tf_block(m15_c) if m15_c else {}
    except: pass

    return data


def format_wolf_chart(d):
    """Format 5-TF data block for Claude prompt."""
    sep = '=' * 60
    wk = d.get('weekly', {}); da = d.get('daily', {})
    h4 = d.get('h4', {});     h1 = d.get('h1', {})
    m15 = d.get('m15', {})
    sup = d.get('supports', []); res = d.get('resistances', [])

    def tf_line(label, tf):
        if not tf:
            return f'{label}: no data'
        return (f"{label}: {tf.get('trend','?')} | Structure={tf.get('structure','?')} ({tf.get('strength','?')}%)"
                f" | EMA9={tf.get('ema9','?')} EMA20={tf.get('ema20','?')} EMA50={tf.get('ema50','?')}"
                f" EMA100={tf.get('ema100','?')} EMA200={tf.get('ema200','?')}"
                f" | RSI={tf.get('rsi','?')} MACD={tf.get('macd','?')}"
                f" | ADX={tf.get('adx','?')} ({tf.get('adx_signal','?')})"
                f" | ATR={tf.get('atr','?')}"
                f" | {tf.get('vs_ema100','?')} EMA100 | {tf.get('vs_ema200','?')} EMA200"
                f" | BB Upper={tf.get('bb_upper','?')} Mid={tf.get('bb_mid','?')} Lower={tf.get('bb_lower','?')}"
                f" | Range High={tf.get('high','?')} Low={tf.get('low','?')}")

    lines = [
        sep,
        f"5-TIMEFRAME CHART DATA — {d['symbol']} @ {d['price']}",
        sep,
        tf_line('WEEKLY', wk),
        tf_line('DAILY',  da),
        tf_line('H4',     h4),
        tf_line('H1',     h1),
        tf_line('M15',    m15),
        sep,
    ]
    if sup:
        lines.append('REAL SUPPORT ZONES (min 2 touches):')
        for s in sup:
            lines.append(f"  SUPPORT @ {s['price']} — {s['touches']} touches")
    if res:
        lines.append('REAL RESISTANCE ZONES (min 2 touches):')
        for r in res:
            lines.append(f"  RESISTANCE @ {r['price']} — {r['touches']} touches")
    lines.append(sep)
    return '\n'.join(lines)



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

FIDELITY CHART PATTERN SKILLS (FULL — from Kirkpatrick CMT):

PATTERN RULES (APPLY TO ALL):
- A pattern is NOT complete until price CLOSES beyond the breakout level — wick entries are traps
- Patterns are fractal — they work on any timeframe
- Always apply a confirmation filter before acting on any breakout
- FALSE BREAKOUT: price breaks level but immediately returns inside — DO NOT chase
- FAILED BREAKOUT (TRAP): false breakout + price then breaks opposite direction — this IS your entry
- Protective stop: place just outside the breakout bar, opposite side

MULTI-BAR REVERSAL PATTERNS:
- DOUBLE TOP: Two peaks at same resistance separated by a trough. Valid ONLY on close below the neckline (trough). Target = height of pattern subtracted from neckline breakout price.
- DOUBLE BOTTOM: Two troughs at same support separated by a peak. Valid on close above neckline. Target = height added to neckline breakout. 
- TRIPLE TOP/BOTTOM: Three touches at exact same level = strongest confirmation. Same target method. Best triple bottom performance after sustained decline.
- HEAD & SHOULDERS (TOP): Left shoulder + higher head + right shoulder at same level. LOWEST FAILURE RATE of all patterns. Pattern ONLY complete on CLOSE below neckline. Target = distance from head to neckline, projected down from breakout. Throwback to neckline after break = second entry.
- INVERSE H&S: Identical but inverted at bottoms. Valid on neckline break upward.
- RECTANGLE: Horizontal support + resistance bounding price. Many false breakouts — always confirm close. "Shortfall" (price fails to reach opposite boundary) = early signal of true breakout direction. Target = height of rectangle added/subtracted from breakout.
- CUP & HANDLE: Rounded bottom (not V-shaped) + small handle (flag pattern). Complete on break above both lips. Target = cup depth added to breakout price.
- PIPE BOTTOM: Two tall bars at end of downtrend. Second bar closes upper half of range. Most reliable on weekly data. Target = taller bar height added to top.

TRIANGLE PATTERNS:
- SYMMETRICAL TRIANGLE: Converging upper + lower trendlines. Each trendline touched minimum TWICE. Best performance on upward breakout. Target = pattern height from breakout.
- ASCENDING TRIANGLE: Horizontal resistance + rising support. Breaks more commonly UPWARD. Above-average downside performance if it breaks down. Target = height from breakout.
- DESCENDING TRIANGLE: Horizontal support + falling resistance. Breaks more commonly DOWNWARD. Above-average performance on upside break. Target = height from breakout.
- RISING WEDGE: Both trendlines slope UP. Need 5 touches minimum (3 on one side, 2 on other). Bearish — breaks DOWN from climax peak. Below-average performance. Target = lowest trough in pattern.
- FALLING WEDGE: Both trendlines slope DOWN. Need 5 touches. Bullish — breaks UP. Target = height added to breakout.

CANDLESTICK PATTERNS (use WITH trend context — alone they are weak):
- DOJI: Open = close, equal wicks. Indecision only. NOT a signal alone — needs S/R context.
- HARAMI: Large body + small opposite body COMPLETELY INSIDE large body. Can break either way — wait for confirmation.
- HAMMER (at bottom): Long lower wick, small body at top of candle. Potential reversal at support. Below-average alone.
- HANGING MAN (at top): Same shape as hammer but at resistance. Below-average alone. Needs confirmation.
- SHOOTING STAR: Long upper wick, small body at bottom at resistance. Potential reversal down. Average performance.
- ENGULFING BULL: Small black candle + large white candle COMPLETELY engulfs it. At support in downtrend = strongest reversal signal.
- ENGULFING BEAR: Small white candle + large black candle engulfs it. At resistance in uptrend = strong sell.
- DARK CLOUD COVER: White candle + black candle that opens above white high then closes below white midpoint. Bearish reversal.
- PIERCING LINE: Black candle + white candle that opens below black low then closes above black midpoint. Bullish reversal.

SHORT-TERM PATTERNS:
- FLAG/PENNANT: After sharp move (the flagpole), small consolidation slightly OPPOSITE the trend. Target = flagpole height added to breakout from flag. Continuation pattern — trade WITH the prior trend.
- NARROW RANGE (NR4): Day 4 range is narrower than days 1-3. Low volatility compression = breakout imminent. Buy on break above NR4 high, sell on break below NR4 low.
- INSIDE BAR: Second bar completely inside first bar range. Same compression signal as NR4. Buy on break above inside bar close, sell on break below.
- GAP TRADING: Profitable on breakouts from S/R. Explosion gap pivot method: gap occurs → wait for throwback → if throwback COVERS the gap = skip, invalid. If throwback STOPS before gap = pivot low formed → enter above gap bar high. Protective stop initially at gap low.

PATTERN TARGET CALCULATIONS:
- All H&S/Double/Triple: measure height of pattern, project from breakout level
- All Triangles/Wedges: measure height at widest point, add to breakout
- Flags/Pennants: measure entire flagpole, add to flag breakout point
- Cup & Handle: measure cup depth, add to breakout above lip

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

WOLF_WEEKLY_PROMPT = """You are WOLF AGENT performing a WEEKLY MARKET OUTLOOK — the professional trader's Friday ritual.

You have been trained on:
- Fidelity Chart Patterns (all 20+ patterns with target calculation methods)
- Bob Volman Price Action (buildup, false breaks, double pressure)
- BabyPips Technical Analysis (top-down framework, confluence, S/R)
- ICT / Institutional thinking (liquidity, previous week highs/lows, session bias)

YOUR FRAMEWORK (apply in this exact order):
1. RETROSPECTIVE: What happened last week? Was it a trend week or range week? What was the narrative (technical break OR news catalyst)?
2. TOP-DOWN: Weekly structure → Daily pattern → H4 entry zone. Use 100 EMA as macro filter.
3. INDICATORS: ADX (>25 = trend, <20 = ranging). RSI zone. Price vs 50/200 MA.
4. PATTERN DETECTION: Look for any completing patterns on weekly or daily timeframe from the full Fidelity library.
5. GAMEPLAN: Specific entry trigger + level + stop + target. Not vague — real prices from the data.
6. RISK SENTIMENT: Is market risk-on (AUD/NZD rising, stocks up) or risk-off (JPY/Gold/USD rising)?

STRICT RULES:
- Only use the REAL price data provided. No invented levels.
- S/R zones only valid with 2+ real touches from the provided data.
- Pattern must be CONFIRMED by price — no wick entries.
- Give a GAMEPLAN with real numbers, not "wait and see" vagueness.
- If ADX < 20, say ranging and reduce position size recommendation.

RESPOND ONLY IN THIS EXACT JSON FORMAT:
{
  "symbol": "string",
  "week_reviewed": "YYYY-MM-DD to YYYY-MM-DD",
  "last_week_move_pips": 0,
  "last_week_direction": "BULLISH | BEARISH | RANGE",
  "last_week_narrative": "1-2 sentence story of what happened last week — was it a trend break, range, news-driven?",
  "key_level_broken": "price level that broke last week or NONE",
  "key_level_held": "price level that held last week or NONE",
  "current_trend": "UPTREND | DOWNTREND | RANGING",
  "ema100_status": "price above/below 100 EMA — exact value",
  "adx_reading": 0,
  "adx_signal": "TRENDING | RANGING",
  "pattern_detected": "pattern name + timeframe or NONE",
  "pattern_target": "price target from pattern or N/A",
  "key_support": 0.0,
  "key_resistance": 0.0,
  "upcoming_news_risk": "any high-impact events this week or NONE",
  "risk_sentiment": "RISK-ON | RISK-OFF | NEUTRAL",
  "gameplan": {
    "bias": "BULLISH | BEARISH | WAIT",
    "entry_trigger": "specific condition that must be met before entering",
    "entry_zone": {"low": 0.0, "high": 0.0},
    "stop_loss": 0.0,
    "tp1": 0.0,
    "tp2": 0.0,
    "tp3": 0.0,
    "invalidation": "what makes this gameplan wrong"
  },
  "wolf_verdict": "1 punchy sentence. The Wolf's final read on this pair for the week."
}"""

def run_wolf_analysis(symbol, candles, current_price, supports, resistances,
                      trend, trend_strength, adx, ema100_val, atr, news_warning, is_option=False):
    """Feed real 5-TF data to Claude. No prediction — pure chart reading."""

    # ── Fetch full 5-TF data — reuse daily candles from scan job ─────────────
    chart = fetch_wolf_chart_data(symbol, current_price, daily_candles=candles)
    chart_block = format_wolf_chart(chart)

    # Fallback context from daily candles (passed in from scan job)
    recent_candles = candles[-10:]
    candle_summary = []
    for c in recent_candles:
        body = abs(c['close'] - c['open'])
        direction = 'BULL' if c['close'] > c['open'] else 'BEAR'
        candle_summary.append({
            'date': c['date'],
            'O': round(c['open'], 5), 'H': round(c['high'], 5),
            'L': round(c['low'], 5),  'C': round(c['close'], 5),
            'direction': direction,
            'body_size': round(body, 5),
            'upper_wick': round(c['high'] - max(c['open'], c['close']), 5),
            'lower_wick': round(min(c['open'], c['close']) - c['low'], 5),
        })

    highs_30 = [c['high'] for c in candles]
    lows_30  = [c['low']  for c in candles]
    high_30d = round(max(highs_30), 5) if highs_30 else 0
    low_30d  = round(min(lows_30),  5) if lows_30  else 0
    price_pct = round(((current_price - low_30d) / (high_30d - low_30d) * 100)
                      if (high_30d - low_30d) > 0 else 50, 1)

    # Use richer daily data from 5-TF fetch if available
    da  = chart.get('daily', {})
    adx_real    = da.get('adx')    or adx
    ema100_real = da.get('ema100') or ema100_val
    atr_real    = da.get('atr')    or atr
    trend_real  = da.get('structure') or trend
    strength_real = da.get('strength') or trend_strength

    user_message = f"""SYMBOL: {symbol}
CURRENT PRICE: {current_price}
30-DAY RANGE: High={high_30d} | Low={low_30d} | Price at {price_pct}% of range

{chart_block}

LAST 10 DAILY CANDLES (newest last):
{json.dumps(candle_summary, indent=2)}

NEWS RISK: {news_warning if news_warning else 'NONE DETECTED'}

{'OPTIONS CONTEXT: 0DTE analysis. Apply ORB + VWAP + PDH/PDL rules.' if is_option else ''}

TASK: Read ALL 5 timeframes above like a professional chart analyst.
Apply Volman buildup rules, Fidelity pattern detection, BabyPips structure.
Give the trade signal based purely on what the data shows. No forecasting. No fluff.
ADX RULE: Only signal BUY/SELL if ADX > 25. If ADX < 20 = WAIT.
EMA100 RULE: BUY only if price ABOVE EMA100. SELL only if price BELOW EMA100."""

    client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1000,
        system=WOLF_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()
    try:
        if '```' in raw:
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        result = json.loads(raw.strip())
        result['symbol']        = symbol
        result['current_price'] = current_price
        result['atr']           = atr_real
        result['adx']           = adx_real
        result['timestamp']     = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        return result
    except Exception as e:
        return {
            'symbol': symbol, 'signal': 'ERROR',
            'error': f'Parse error: {str(e)}', 'raw': raw[:200],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        }

def run_weekly_analysis(symbol, candles_daily, candles_weekly, current_price,
                        supports, resistances, trend, trend_strength, adx, ema100_val, atr, news_warning):
    """Run weekly outlook analysis using Claude with real data."""

    # Last week's candles (most recent 5 daily candles)
    last_week = candles_daily[-6:-1] if len(candles_daily) >= 6 else candles_daily[:-1]
    this_week = candles_daily[-5:] if len(candles_daily) >= 5 else candles_daily

    # Weekly move calculation
    # Forex: multiply by pip value to get pips (EUR/USD = x10000, JPY = x100)
    # Stocks: use raw price difference in dollars/points
    is_stock = '/' not in symbol
    if is_stock:
        pip = 1          # stocks → dollar/point move
        unit = 'pts'
    elif any(x in symbol for x in ['JPY', 'HUF', 'KRW']):
        pip = 100        # JPY pairs → pips
        unit = 'pips'
    else:
        pip = 10000      # standard forex → pips
        unit = 'pips'

    week_open = last_week[0]['open'] if last_week else candles_daily[0]['open']
    week_close = last_week[-1]['close'] if last_week else candles_daily[-1]['close']
    last_week_pips = round((week_close - week_open) * pip)
    last_week_dir = 'BULLISH' if last_week_pips > 0 else 'BEARISH' if last_week_pips < 0 else 'RANGE'
    move_label = f"{last_week_pips:+d} {unit}"

    # 30-day range
    highs = [c['high'] for c in candles_daily]
    lows  = [c['low']  for c in candles_daily]
    high_30 = round(max(highs), 5)
    low_30  = round(min(lows),  5)
    price_pct = round(((current_price - low_30) / (high_30 - low_30) * 100) if (high_30 - low_30) > 0 else 50, 1)

    # Weekly candle summaries
    def summarize(candles, label):
        out = []
        for c in candles[-6:]:
            body = abs(c['close'] - c['open'])
            direction = 'BULL' if c['close'] > c['open'] else 'BEAR'
            out.append(f"{c['date']} | {direction} | O:{round(c['open'],5)} H:{round(c['high'],5)} L:{round(c['low'],5)} C:{round(c['close'],5)} | body:{round(body,5)}")
        return label + '\n' + '\n'.join(out)

    last_week_summary = summarize(last_week, 'LAST WEEK CANDLES:')
    this_week_summary = summarize(this_week, 'THIS WEEK CANDLES (so far):')

    user_message = f"""WEEKLY MARKET OUTLOOK REQUEST
SYMBOL: {symbol}
INSTRUMENT TYPE: {'STOCK — moves measured in dollars/points (pts), NOT pips' if is_stock else 'FOREX PAIR — moves measured in pips'}
CURRENT PRICE: {current_price}
LAST WEEK MOVE: {move_label} (open: {round(week_open,5)} → close: {round(week_close,5)})
30-DAY RANGE: High={high_30} | Low={low_30} | Price at {price_pct}% of range

100 EMA: {round(ema100_val, 5) if ema100_val else 'N/A'}
PRICE vs 100 EMA: {'ABOVE' if ema100_val and current_price > ema100_val else 'BELOW' if ema100_val else 'UNKNOWN'}
ADX: {adx if adx else 'N/A'} {'(TRENDING)' if adx and adx > 25 else '(RANGING)' if adx else ''}
ATR: {atr}
TREND STRUCTURE: {trend} ({trend_strength}% score)

REAL SUPPORT ZONES (2+ touches each):
{json.dumps(supports[:3], indent=2) if supports else 'None found'}

REAL RESISTANCE ZONES (2+ touches each):
{json.dumps(resistances[:3], indent=2) if resistances else 'None found'}

{last_week_summary}

{this_week_summary}

NEWS RISK: {news_warning if news_warning else 'NONE DETECTED'}

TASK: Apply the full weekly outlook framework. Read last week's price action narrative, find any completing patterns, and give me a real gameplan for next week with specific price levels. For stocks use dollar/point moves. For forex use pips."""

    client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1200,
        system=WOLF_WEEKLY_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()
    try:
        if '```' in raw:
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        result = json.loads(raw.strip())
        result['symbol'] = symbol
        result['current_price'] = current_price
        result['last_week_pips'] = last_week_pips
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        return result
    except Exception as e:
        return {
            'symbol': symbol, 'signal': 'ERROR',
            'error': f'Parse error: {str(e)}', 'raw': raw[:200],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        }


def wolf_weekly_job(job_id):
    """
    Weekly Outlook job:
    1. Scan all pairs/stocks for trend quality
    2. Pick top 3 forex + top 2 stocks
    3. Run weekly retrospective + next-week gameplan for each
    """
    job = _wolf_jobs[job_id]
    try:
        results = {'forex': [], 'stocks': [], 'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

        # ── FOREX: find top 3 trending ───────────────────────────────────────
        job['step'] = '🔍 Scanning forex pairs for weekly trends...'
        scored_forex = []
        for pair in FOREX_PAIRS:
            candles = fetch_ohlc(pair, interval='1day', outputsize=40)
            if candles and len(candles) >= 20:
                score = score_pair_for_trend(candles)
                trend, strength = detect_trend_structure(candles)
                adx = calc_adx(candles)
                if trend != 'SIDEWAYS' and score > 35:
                    scored_forex.append({'pair': pair, 'score': score, 'trend': trend, 'adx': adx, 'candles': candles})
            time.sleep(0.3)

        top_forex = sorted(scored_forex, key=lambda x: x['score'], reverse=True)[:3]
        job['step'] = f'✅ Top {len(top_forex)} trending pairs found. Building weekly outlooks...'

        for item in top_forex:
            pair  = item['pair']
            candles = item['candles']
            job['step'] = f'📅 Weekly outlook: {pair}...'
            closes = [c['close'] for c in candles]
            current_price = fetch_current_price(pair) or closes[-1]
            ema100_val = (calc_ema(closes, 100) or [None])[-1]
            atr   = calc_atr(candles)
            adx   = item['adx']
            trend, strength = detect_trend_structure(candles)
            supports, resistances = find_sr_simple(candles, current_price)
            news  = check_news_risk(pair)
            analysis = run_weekly_analysis(pair, candles, [], current_price, supports, resistances,
                                           trend, strength, adx, ema100_val, atr, news)
            analysis['trend_score'] = item['score']
            results['forex'].append(analysis)
            time.sleep(1)

        # ── STOCKS: top 2 trending ───────────────────────────────────────────
        job['step'] = '📈 Scanning stocks for weekly trends...'
        scored_stocks = []
        for stock in STOCKS:
            candles = fetch_ohlc(stock, interval='1day', outputsize=40)
            if candles and len(candles) >= 20:
                score = score_pair_for_trend(candles)
                trend, _ = detect_trend_structure(candles)
                adx = calc_adx(candles)
                if trend != 'SIDEWAYS' and score > 35:
                    scored_stocks.append({'symbol': stock, 'score': score, 'trend': trend, 'adx': adx, 'candles': candles})
            time.sleep(0.3)

        top_stocks = sorted(scored_stocks, key=lambda x: x['score'], reverse=True)[:2]
        job['step'] = f'✅ Top {len(top_stocks)} trending stocks. Building weekly outlooks...'

        for item in top_stocks:
            symbol  = item['symbol']
            candles = item['candles']
            job['step'] = f'📅 Weekly outlook: {symbol}...'
            closes = [c['close'] for c in candles]
            current_price = fetch_current_price(symbol) or closes[-1]
            ema100_val = (calc_ema(closes, 100) or [None])[-1]
            atr   = calc_atr(candles)
            adx   = item['adx']
            trend, strength = detect_trend_structure(candles)
            supports, resistances = find_sr_simple(candles, current_price)
            news  = check_news_risk(symbol)
            analysis = run_weekly_analysis(symbol, candles, [], current_price, supports, resistances,
                                           trend, strength, adx, ema100_val, atr, news)
            analysis['trend_score'] = item['score']
            results['stocks'].append(analysis)
            time.sleep(1)

        job['status'] = 'done'
        job['result'] = results
        job['step'] = '✅ Weekly outlook complete'

    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)
        job['step'] = f'❌ Error: {str(e)}'


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
                supports, resistances = find_sr_simple(candles, current_price)
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
                supports, resistances = find_sr_simple(candles, current_price)
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
                supports, resistances = find_sr_simple(candles_1d, current_price)

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


@wolf_bp.route('/api/wolf-weekly', methods=['POST'])
def wolf_weekly():
    """Start a weekly outlook job. Top trending pairs only."""
    job_id = f"weekly_{int(time.time() * 1000)}"
    _wolf_jobs[job_id] = {
        'status': 'running',
        'step': '📅 Wolf Weekly Outlook starting...',
        'result': None
    }
    t = threading.Thread(target=wolf_weekly_job, args=(job_id,), daemon=True)
    t.start()
    return jsonify({'job_id': job_id, 'status': 'starting'})


@wolf_bp.route('/api/wolf-weekly-poll/<job_id>')
def wolf_weekly_poll(job_id):
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
