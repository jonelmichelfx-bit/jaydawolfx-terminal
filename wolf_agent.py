"""
WOLF AGENT v2.0 — JayDaWolfX Terminal
Professional chart reading. Real data. No fluff.

0DTE METHOD: Opening Range Breakout + VWAP + Prev Day Levels + 100 EMA
FOREX METHOD: Trend + S/R zones + EMA stack + Volman buildup
STOCKS METHOD: Same as forex with daily structure

Trained on:
- Bob Volman — Understanding Price Action
- BabyPips — School of Pipsology  
- Fidelity — Identifying Chart Patterns
- ORB / VWAP 0DTE methodology (professional intraday)
"""

import os, json, time, threading, requests
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request, render_template
from anthropic import Anthropic

wolf_bp  = Blueprint('wolf', __name__)
client   = Anthropic()

TWELVE_DATA_KEY = os.environ.get('TWELVE_DATA_API_KEY', '')
NEWS_API_KEY    = os.environ.get('NEWS_API_KEY', '')

_wolf_jobs = {}

FOREX_PAIRS = [
    'EUR/USD','GBP/USD','USD/JPY','AUD/USD','USD/CAD',
    'USD/CHF','NZD/USD','EUR/GBP','GBP/JPY','EUR/JPY',
    'AUD/JPY','EUR/CAD','GBP/CAD','USD/MXN'
]
STOCKS      = ['SPY','QQQ','AAPL','MSFT','NVDA','TSLA','AMZN','META','AMD','GOOGL']
ODTE_SYMS   = ['SPY','QQQ']

# ─── DATA FETCHING ────────────────────────────────────────────────────────────

def fetch_ohlc(symbol, interval='1day', outputsize=35):
    try:
        url = (f"https://api.twelvedata.com/time_series"
               f"?symbol={symbol}&interval={interval}"
               f"&outputsize={outputsize}&apikey={TWELVE_DATA_KEY}")
        r = requests.get(url, timeout=12)
        data = r.json()
        if 'values' not in data:
            return None
        candles = []
        for v in reversed(data['values']):
            candles.append({
                'date':   v['datetime'],
                'open':   float(v['open']),
                'high':   float(v['high']),
                'low':    float(v['low']),
                'close':  float(v['close']),
                'volume': float(v.get('volume', 0))
            })
        return candles
    except:
        return None

def fetch_price(symbol):
    try:
        r = requests.get(
            f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_DATA_KEY}",
            timeout=5)
        return float(r.json().get('price', 0)) or None
    except:
        return None

# ─── TECHNICAL CALCULATIONS ──────────────────────────────────────────────────

def calc_ema(closes, period):
    if len(closes) < period:
        return []
    k   = 2.0 / (period + 1)
    ema = [sum(closes[:period]) / period]
    for p in closes[period:]:
        ema.append(p * k + ema[-1] * (1 - k))
    return ema

def calc_vwap(candles):
    """VWAP for intraday candles — uses all candles provided as single session."""
    total_tp_vol = 0
    total_vol    = 0
    for c in candles:
        tp  = (c['high'] + c['low'] + c['close']) / 3
        vol = c['volume'] if c['volume'] > 0 else 1
        total_tp_vol += tp * vol
        total_vol    += vol
    return total_tp_vol / total_vol if total_vol > 0 else None

def calc_atr(candles, period=14):
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]['high'], candles[i]['low'], candles[i-1]['close']
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    return round(sum(trs[-period:]) / period, 5)

def calc_adx(candles, period=14):
    if len(candles) < period * 2:
        return None
    try:
        plus_dm, minus_dm, tr_list = [], [], []
        for i in range(1, len(candles)):
            h, l   = candles[i]['high'], candles[i]['low']
            ph, pl, pc = candles[i-1]['high'], candles[i-1]['low'], candles[i-1]['close']
            up   = h - ph
            down = pl - l
            plus_dm.append(up   if up   > down and up   > 0 else 0)
            minus_dm.append(down if down > up   and down > 0 else 0)
            tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))

        def smooth(lst, p):
            s = sum(lst[:p]); r = [s]
            for x in lst[p:]: r.append(r[-1] - r[-1]/p + x)
            return r

        trs  = smooth(tr_list, period)
        pdms = smooth(plus_dm, period)
        mdms = smooth(minus_dm, period)
        dx   = []
        for i in range(len(trs)):
            pdi = 100 * pdms[i] / trs[i] if trs[i] else 0
            mdi = 100 * mdms[i] / trs[i] if trs[i] else 0
            dx.append(100 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) else 0)
        if len(dx) < period:
            return None
        return round(sum(dx[-period:]) / period, 1)
    except:
        return None

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    if len(gains) < period:
        return None
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
    if avg_l == 0:
        return 100
    rs = avg_g / avg_l
    return round(100 - 100 / (1 + rs), 1)

def detect_opening_range(candles_15m):
    """
    Opening Range = first 2 candles of session (15m = first 30 min).
    Returns ORB high, ORB low, and midpoint.
    """
    if not candles_15m or len(candles_15m) < 2:
        return None
    orb_high = max(c['high']  for c in candles_15m[:2])
    orb_low  = min(c['low']   for c in candles_15m[:2])
    return {
        'high': round(orb_high, 4),
        'low':  round(orb_low,  4),
        'mid':  round((orb_high + orb_low) / 2, 4),
        'width': round(orb_high - orb_low, 4)
    }

def find_sr_zones(candles, sensitivity=0.003):
    """Real S/R from pivot highs/lows. Min 2 touches = valid zone."""
    if len(candles) < 10:
        return [], []

    pivots_high, pivots_low = [], []
    lb = 3
    for i in range(lb, len(candles) - lb):
        is_ph = all(candles[i]['high'] >= candles[j]['high']
                    for j in range(i-lb, i+lb+1) if j != i)
        is_pl = all(candles[i]['low']  <= candles[j]['low']
                    for j in range(i-lb, i+lb+1) if j != i)
        if is_ph: pivots_high.append({'price': candles[i]['high'], 'date': candles[i]['date']})
        if is_pl: pivots_low.append( {'price': candles[i]['low'],  'date': candles[i]['date']})

    cp     = candles[-1]['close']
    margin = cp * sensitivity

    def cluster(pivots):
        zones, used = [], set()
        for i, p in enumerate(pivots):
            if i in used: continue
            cluster = [p['price']]; dates = [p['date']]; used.add(i)
            for j, q in enumerate(pivots):
                if j not in used and abs(p['price'] - q['price']) <= margin * 2:
                    cluster.append(q['price']); dates.append(q['date']); used.add(j)
            # Relaxed: allow single-touch zones too (but mark them WEAK)
            strength = 'STRONG' if len(cluster) >= 3 else 'MODERATE' if len(cluster) == 2 else 'WEAK'
            zones.append({
                'price':    round(sum(cluster) / len(cluster), 5),
                'touches':  len(cluster),
                'last_touch': max(dates),
                'strength': strength
            })
        return sorted(zones, key=lambda z: abs(z['price'] - cp))

    R = cluster(pivots_high)
    S = cluster(pivots_low)
    return [z for z in S if z['price'] < cp][:5], [z for z in R if z['price'] > cp][:5]

def detect_trend(candles):
    if len(candles) < 10: return 'UNKNOWN', 50
    recent = candles[-20:]
    highs  = [c['high']  for c in recent]
    lows   = [c['low']   for c in recent]
    hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    hl = sum(1 for i in range(1, len(lows))  if lows[i]  > lows[i-1])
    lh = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
    ll = sum(1 for i in range(1, len(lows))  if lows[i]  < lows[i-1])
    bull = hh + hl; bear = lh + ll; total = bull + bear
    if total == 0: return 'SIDEWAYS', 0
    bp = (bull / total) * 100; brp = (bear / total) * 100
    if bp  >= 55: return 'UPTREND',   round(bp)
    if brp >= 55: return 'DOWNTREND', round(brp)
    return 'SIDEWAYS', round(max(bp, brp))

def detect_candlestick_pattern(candles):
    """Detect key candlestick patterns from last 3 candles."""
    if len(candles) < 3:
        return 'NONE'
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    body3  = abs(c3['close'] - c3['open'])
    range3 = c3['high'] - c3['low']
    body2  = abs(c2['close'] - c2['open'])

    # Doji
    if range3 > 0 and body3 / range3 < 0.1:
        return 'DOJI — indecision'
    # Engulfing bull
    if (c2['close'] < c2['open'] and c3['close'] > c3['open']
            and c3['open'] <= c2['close'] and c3['close'] >= c2['open']):
        return 'BULLISH ENGULFING — reversal signal'
    # Engulfing bear
    if (c2['close'] > c2['open'] and c3['close'] < c3['open']
            and c3['open'] >= c2['close'] and c3['close'] <= c2['open']):
        return 'BEARISH ENGULFING — reversal signal'
    # Hammer (long lower wick, small body near top)
    lower_wick = min(c3['open'], c3['close']) - c3['low']
    upper_wick = c3['high'] - max(c3['open'], c3['close'])
    if range3 > 0 and lower_wick / range3 > 0.55 and body3 / range3 < 0.35:
        return 'HAMMER — bullish reversal'
    if range3 > 0 and upper_wick / range3 > 0.55 and body3 / range3 < 0.35:
        return 'SHOOTING STAR — bearish reversal'
    # Strong bullish bar
    if c3['close'] > c3['open'] and body3 / range3 > 0.7 if range3 > 0 else False:
        return 'STRONG BULL BAR — momentum'
    if c3['close'] < c3['open'] and body3 / range3 > 0.7 if range3 > 0 else False:
        return 'STRONG BEAR BAR — momentum'
    return 'NONE'

def score_trend(candles):
    if not candles or len(candles) < 20: return 0
    closes = [c['close'] for c in candles]
    adx    = calc_adx(candles) or 0
    trend, strength = detect_trend(candles)
    ema100 = calc_ema(closes, min(100, len(closes)))
    ema_score = 0
    if ema100:
        pct = abs(closes[-1] - ema100[-1]) / ema100[-1] * 100
        ema_score = min(pct * 5, 25)
    adx_score = min(adx, 50)
    str_score = strength * 0.25
    if trend == 'SIDEWAYS': return int(adx_score * 0.3)
    return int(min(adx_score + str_score + ema_score, 100))

def check_news(symbol):
    try:
        if not NEWS_API_KEY: return None
        kws = ['NFP','CPI','Fed','FOMC','rate decision','jobs report','inflation','GDP','earnings']
        sym_clean = symbol.split('/')[0]
        url = (f"https://newsapi.org/v2/everything?q={sym_clean}&sortBy=publishedAt"
               f"&pageSize=5&apiKey={NEWS_API_KEY}")
        arts = requests.get(url, timeout=5).json().get('articles', [])
        hits = []
        for a in arts:
            t = (a.get('title','') + ' ' + a.get('description','')).lower()
            for kw in kws:
                if kw.lower() in t:
                    hits.append(kw); break
        if hits: return f"⚠️ HIGH IMPACT NEWS: {', '.join(set(hits[:2]))} — trade with extra caution"
        return None
    except:
        return None

# ─── WOLF SYSTEM PROMPT ──────────────────────────────────────────────────────

WOLF_SYSTEM = """You are WOLF AGENT — JayDaWolfX's professional AI chart reader and market analyst.

You read REAL price data and give professional trading signals. You are trained in:

== FOREX & STOCK CHART READING ==
- Bob Volman Price Action: S/R zones from real pivot tests, buildup before breakout, double pressure, false breaks
- BabyPips: HH/HL trend structure, candlestick patterns, EMA as trend filter, volume confirmation
- Fidelity: Double tops/bottoms, H&S, flags, triangles, wedges — only valid on candle CLOSE beyond level
- 100 EMA = primary trend filter (above = bullish bias, below = bearish bias)
- RSI: Overbought >70 (watch for reversal), oversold <30 (watch for bounce)

== 0DTE OPTIONS METHOD (ORB + VWAP) ==
This is how professional 0DTE traders read the chart:

1. PRE-MARKET BIAS: Check if price is above/below previous day's close. Above = bullish bias. Below = bearish.
2. KEY LEVELS to mark BEFORE open:
   - Previous Day High (PDH) and Previous Day Low (PDL) — most important levels
   - Pre-market high/low if available
   - Major weekly/monthly S/R from daily chart
3. OPENING RANGE (ORB): First 15-30 minutes after 9:30 AM ET forms the range
   - ORB High and ORB Low = the day's key breakout levels
   - DO NOT trade during the first 15 minutes — wait for range to form
4. VWAP: Volume-weighted average price
   - Price above VWAP = bullish intraday bias → look for CALL setups
   - Price below VWAP = bearish intraday bias → look for PUT setups
   - VWAP + ORB agreement = highest probability setup
5. ENTRY TRIGGERS:
   - CALL: Price breaks AND closes ABOVE ORB High + price above VWAP + price above 100 EMA daily
   - PUT: Price breaks AND closes BELOW ORB Low + price below VWAP + price below 100 EMA daily
   - Best entry window: 9:45 AM – 11:30 AM ET (after range established, before lunch chop)
   - Second entry window: 2:00 PM – 3:30 PM ET (afternoon trend resumption)
6. RETEST ENTRY (higher probability, lower risk):
   - Wait for price to break ORB, then pull back and RETEST the broken level
   - If level holds, enter in direction of original breakout
   - This confirms the level flipped from resistance to support (or vice versa)
7. STRIKE SELECTION:
   - Strong trending day: ATM (at-the-money) for maximum gamma
   - Retest entry: Slightly OTM for better R/R
   - Never go deep OTM — too much decay risk
8. PROFIT TARGET: 50-100% gain on the option, or 1x-2x the ORB width measured from entry
9. STOP LOSS: 50% loss on the option, OR price closes back inside the ORB

== SIGNAL RELAXATION RULES ==
Do NOT be overly strict. Give a signal when you see a REASONABLE setup:
- If 2 out of 3 conditions align (trend + EMA + level), give a signal with MEDIUM confidence
- If all 3 align, give HIGH confidence
- Only say WAIT if: price is in middle of range with no clear setup, OR high-impact news in <2 hours
- A WEAK/MODERATE S/R level is still worth noting — just lower the confidence
- Don't require perfect buildup — if the level is right and trend is right, signal it

== ANALYSIS STYLE ==
Read the chart like a professional. Be concise. Tell me:
1. Where price IS right now (above/below what key level)
2. What the overall bias is (bull/bear/neutral)
3. What the cleanest trade setup looks like
4. Exact entry, stop, targets

RESPOND ONLY IN THIS JSON (no text outside):
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
  "ema100_status": "price ABOVE/BELOW 100 EMA at [value]",
  "trend_structure": "brief structure description",
  "key_level": "specific price level and why it matters",
  "pattern_detected": "pattern name or NONE",
  "candlestick": "pattern name or NONE",
  "rsi_reading": "RSI value and what it means",
  "vwap_status": "price above/below VWAP at [value] or N/A",
  "orb_status": "ORB high/low values and current price position or N/A",
  "analysis": "3-4 sentence professional chart read. What the chart is SHOWING right now.",
  "news_warning": null,
  "option_play": null | {
    "type": "CALL" | "PUT",
    "strike": "ATM or OTM",
    "entry_trigger": "exact trigger description",
    "profit_target": "% gain target",
    "stop": "exit condition",
    "best_window": "time window",
    "rationale": "why this setup"
  }
}"""

def run_analysis(symbol, candles_daily, candles_intraday, current_price,
                 supports, resistances, trend, adx, ema100_val, atr, rsi,
                 vwap, orb, news, is_0dte=False):
    """Feed real structured data to Wolf Agent for chart reading."""

    closes = [c['close'] for c in candles_daily]
    H30    = round(max(c['high']  for c in candles_daily), 5)
    L30    = round(min(c['low']   for c in candles_daily), 5)
    rng    = H30 - L30
    pos    = round((current_price - L30) / rng * 100, 1) if rng > 0 else 50

    # Last 5 candles for structure reading
    last5 = candles_daily[-5:]
    candle_data = [{
        'date': c['date'], 'O': round(c['open'],5), 'H': round(c['high'],5),
        'L': round(c['low'],5), 'C': round(c['close'],5),
        'direction': 'BULL' if c['close'] > c['open'] else 'BEAR',
        'body': round(abs(c['close']-c['open']),5)
    } for c in last5]

    prev_day = candles_daily[-2] if len(candles_daily) >= 2 else None
    pdh = round(prev_day['high'],  4) if prev_day else None
    pdl = round(prev_day['low'],   4) if prev_day else None
    pdc = round(prev_day['close'], 4) if prev_day else None

    # Intraday last 8 candles (if available)
    intraday_str = 'NOT AVAILABLE'
    if candles_intraday and len(candles_intraday) >= 4:
        last_intra = candles_intraday[-8:]
        intraday_str = json.dumps([{
            'time': c['date'][-8:], 'O': round(c['open'],3),
            'H': round(c['high'],3), 'L': round(c['low'],3),
            'C': round(c['close'],3)
        } for c in last_intra])

    msg = f"""INSTRUMENT: {symbol}  |  TYPE: {'0DTE OPTIONS TARGET' if is_0dte else 'FOREX/STOCK'}
CURRENT PRICE: {current_price}
30-DAY RANGE: High={H30} | Low={L30} | Price at {pos}% of range

=== KEY EMAs ===
100 EMA: {round(ema100_val,5) if ema100_val else 'N/A'}
Price vs 100 EMA: {'ABOVE ✅' if ema100_val and current_price > ema100_val else 'BELOW ⚠️' if ema100_val else 'UNKNOWN'}

=== INDICATORS ===
ADX: {adx} ({'TRENDING 🔥' if adx and adx >= 25 else 'WEAK/RANGING' if adx else 'N/A'})
RSI(14): {rsi} ({'OVERBOUGHT' if rsi and rsi >= 70 else 'OVERSOLD' if rsi and rsi <= 30 else 'NEUTRAL' if rsi else 'N/A'})
ATR: {atr}

=== INTRADAY VWAP & ORB ===
VWAP: {round(vwap,4) if vwap else 'N/A'}  |  Price vs VWAP: {'ABOVE 🟢' if vwap and current_price > vwap else 'BELOW 🔴' if vwap else 'N/A'}
Opening Range: {json.dumps(orb) if orb else 'N/A (use daily structure)'}
Price vs ORB: {'ABOVE ORB HIGH 🟢' if orb and current_price > orb['high'] else 'BELOW ORB LOW 🔴' if orb and current_price < orb['low'] else 'INSIDE RANGE ⚪' if orb else 'N/A'}

=== PREVIOUS DAY LEVELS ===
Prev Day High (PDH): {pdh}  |  Prev Day Low (PDL): {pdl}  |  Prev Day Close: {pdc}
Price vs Prev Close: {'ABOVE — bullish pre-bias' if pdc and current_price > pdc else 'BELOW — bearish pre-bias' if pdc else 'N/A'}

=== TREND STRUCTURE (30-day) ===
{trend[0]} — strength {trend[1]}%

=== SUPPORT ZONES (real pivot clusters) ===
{json.dumps(supports[:3]) if supports else 'None detected in range'}

=== RESISTANCE ZONES (real pivot clusters) ===
{json.dumps(resistances[:3]) if resistances else 'None detected in range'}

=== LAST 5 DAILY CANDLES ===
{json.dumps(candle_data, indent=2)}

=== RECENT INTRADAY (15min) ===
{intraday_str}

NEWS RISK: {news if news else 'NONE DETECTED ✅'}
{"=== 0DTE CONTEXT ===" if is_0dte else ""}
{"Apply full ORB + VWAP methodology. Identify the ATM strike and entry window." if is_0dte else ""}

READ THIS CHART. Give me the professional signal."""

    resp = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1200,
        system=WOLF_SYSTEM,
        messages=[{"role": "user", "content": msg}]
    )

    raw = resp.content[0].text.strip()
    try:
        clean = raw
        if '```' in clean:
            clean = clean.split('```')[1]
            if clean.startswith('json'): clean = clean[4:]
        result = json.loads(clean.strip())
        result.update({
            'symbol':        symbol,
            'current_price': current_price,
            'atr':           atr,
            'adx':           adx,
            'rsi':           rsi,
            'vwap':          round(vwap, 4) if vwap else None,
            'orb':           orb,
            'pdh':           pdh,
            'pdl':           pdl,
            'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        })
        return result
    except Exception as e:
        return {
            'symbol': symbol, 'signal': 'ERROR', 'error': str(e),
            'current_price': current_price,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        }

# ─── MAIN JOB ─────────────────────────────────────────────────────────────────

def wolf_job(job_id, scan_type):
    job = _wolf_jobs[job_id]
    results = {
        'forex':   [],
        'stocks':  [],
        'options': [],
        'scan_time':     datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
        'market_session': get_session()
    }

    try:
        # ── FOREX ──────────────────────────────────────────────────────────
        if scan_type in ('all', 'forex'):
            job['step'] = '🔍 Scanning 14 forex pairs...'
            scored = []
            for pair in FOREX_PAIRS:
                c = fetch_ohlc(pair, '1day', 35)
                if c and len(c) >= 20:
                    score = score_trend(c)
                    t, _  = detect_trend(c)
                    scored.append({'pair': pair, 'score': score, 'trend': t, 'candles': c})
                time.sleep(0.25)

            top = sorted(scored, key=lambda x: x['score'], reverse=True)[:3]
            job['step'] = f'📊 Analyzing top {len(top)} forex pairs...'

            for item in top:
                pair = item['pair']
                job['step'] = f'📊 Reading {pair} chart...'
                c1d  = item['candles']
                c15m = fetch_ohlc(pair, '15min', 30) or []
                price = fetch_price(pair) or c1d[-1]['close']
                closes = [c['close'] for c in c1d]
                ema100 = calc_ema(closes, min(100, len(closes)))
                atr    = calc_atr(c1d)
                adx    = calc_adx(c1d)
                rsi    = calc_rsi(closes)
                trend  = detect_trend(c1d)
                S, R   = find_sr_zones(c1d)
                vwap   = calc_vwap(c15m) if c15m else None
                orb    = detect_opening_range(c15m) if c15m else None
                news   = check_news(pair)
                sig    = run_analysis(pair, c1d, c15m, price, S, R, trend,
                                      adx, ema100[-1] if ema100 else None,
                                      atr, rsi, vwap, orb, news, False)
                sig['trend_score'] = item['score']
                results['forex'].append(sig)
                time.sleep(1)

        # ── STOCKS ─────────────────────────────────────────────────────────
        if scan_type in ('all', 'stocks'):
            job['step'] = '📈 Scanning stocks...'
            scored = []
            for sym in STOCKS:
                c = fetch_ohlc(sym, '1day', 35)
                if c and len(c) >= 20:
                    score = score_trend(c)
                    t, _  = detect_trend(c)
                    scored.append({'sym': sym, 'score': score, 'trend': t, 'candles': c})
                time.sleep(0.25)

            top = sorted(scored, key=lambda x: x['score'], reverse=True)[:3]
            job['step'] = f'📊 Analyzing top {len(top)} stocks...'

            for item in top:
                sym = item['sym']
                job['step'] = f'📊 Reading {sym} chart...'
                c1d  = item['candles']
                c15m = fetch_ohlc(sym, '15min', 30) or []
                price = fetch_price(sym) or c1d[-1]['close']
                closes = [c['close'] for c in c1d]
                ema100 = calc_ema(closes, min(100, len(closes)))
                atr    = calc_atr(c1d)
                adx    = calc_adx(c1d)
                rsi    = calc_rsi(closes)
                trend  = detect_trend(c1d)
                S, R   = find_sr_zones(c1d)
                vwap   = calc_vwap(c15m) if c15m else None
                orb    = detect_opening_range(c15m) if c15m else None
                news   = check_news(sym)
                sig    = run_analysis(sym, c1d, c15m, price, S, R, trend,
                                      adx, ema100[-1] if ema100 else None,
                                      atr, rsi, vwap, orb, news, False)
                sig['trend_score'] = item['score']
                results['stocks'].append(sig)
                time.sleep(1)

        # ── 0DTE OPTIONS ───────────────────────────────────────────────────
        if scan_type in ('all', 'options'):
            for sym in ODTE_SYMS:
                job['step'] = f'⚡ 0DTE analysis: {sym}...'
                c1d  = fetch_ohlc(sym, '1day', 35) or []
                c15m = fetch_ohlc(sym, '15min', 40) or []
                if not c1d: continue
                price  = fetch_price(sym) or c1d[-1]['close']
                closes = [c['close'] for c in c1d]
                ema100 = calc_ema(closes, min(100, len(closes)))
                atr    = calc_atr(c1d)
                adx    = calc_adx(c1d)
                rsi    = calc_rsi(closes)
                trend  = detect_trend(c1d)
                S, R   = find_sr_zones(c1d)
                vwap   = calc_vwap(c15m) if c15m else calc_vwap(c1d[-5:])
                orb    = detect_opening_range(c15m) if c15m else None
                news   = check_news(sym)
                sig    = run_analysis(sym, c1d, c15m, price, S, R, trend,
                                      adx, ema100[-1] if ema100 else None,
                                      atr, rsi, vwap, orb, news, True)
                sig['instrument_type'] = '0DTE'
                results['options'].append(sig)
                time.sleep(1)

        job['status'] = 'done'
        job['result'] = results
        job['step']   = '✅ Wolf Agent complete'

    except Exception as e:
        job['status'] = 'error'
        job['error']  = str(e)
        job['step']   = f'❌ {str(e)}'

def get_session():
    h = datetime.utcnow().hour
    if h < 7:   return 'ASIA SESSION'
    if h < 12:  return 'LONDON SESSION'
    if h < 13:  return 'LONDON/NY OVERLAP'
    if h < 17:  return 'NEW YORK SESSION'
    return 'AFTER HOURS'

# ─── ROUTES ──────────────────────────────────────────────────────────────────

@wolf_bp.route('/wolf-agent')
def wolf_agent_page():
    return render_template('wolf_agent.html')

@wolf_bp.route('/api/wolf-scan', methods=['POST'])
def wolf_scan():
    data = request.get_json() or {}
    scan_type = data.get('scan_type', 'all')
    job_id = f"wolf_{int(time.time()*1000)}"
    _wolf_jobs[job_id] = {'status': 'running', 'step': '🐺 Initializing...', 'result': None}
    threading.Thread(target=wolf_job, args=(job_id, scan_type), daemon=True).start()
    return jsonify({'job_id': job_id, 'status': 'starting'})

@wolf_bp.route('/api/wolf-poll/<job_id>')
def wolf_poll(job_id):
    job = _wolf_jobs.get(job_id)
    if not job: return jsonify({'status': 'not_found'}), 404
    if job['status'] == 'done':
        r = job['result']; del _wolf_jobs[job_id]
        return jsonify({'status': 'done', 'result': r})
    if job['status'] == 'error':
        e = job.get('error','Unknown'); del _wolf_jobs[job_id]
        return jsonify({'status': 'error', 'error': e})
    return jsonify({'status': 'running', 'step': job.get('step','Processing...')})
