# OPTIONS TERMINAL
A professional options Greeks calculator and P&L dashboard with live Robinhood integration.

---

## SETUP (5 minutes)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up your environment
```bash
cp .env.example .env
# Edit .env and set a random SECRET_KEY
```

### 3. Run the server
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## HOW TO USE

### Manual Mode (no Robinhood login needed)
- Click **"USE MANUAL MODE"** on the login screen
- Fill in the contract inputs in the left sidebar:
  - **Ticker** — stock symbol (AAPL, TSLA, SPY, etc.)
  - **Strike Price** — the strike of your option
  - **Stock Price** — current stock price
  - **Expiration Date** — contract expiration
  - **IV** — implied volatility (e.g. 0.30 for 30%)
  - **Risk-Free Rate** — current rate (e.g. 0.045 for 4.5%)
  - **Option Type** — CALL or PUT toggle
  - **Premium Paid** — what you paid for the contract
- Click **RUN ANALYSIS**

### Live Mode (Robinhood connected)
- Enter your Robinhood email + password
- If you have 2FA enabled (you should), enter the code when prompted
- Your **open positions** appear in the sidebar automatically
- The dashboard pulls **live IV and stock price** from Robinhood
- Click any position to auto-load it into the analyzer

---

## FEATURES

### Greeks Panel
Real-time Black-Scholes calculations for:
- **Delta** — how much the option moves per $1 stock move
- **Gamma** — how fast delta accelerates
- **Theta** — daily time decay (in dollars when multiplied by 100)
- **Vega** — sensitivity to 1% IV change
- **Rho** — interest rate sensitivity
- **Theoretical Price** — what Black-Scholes says the option is worth

### P&L Curve
- Live chart showing profit/loss across a range of stock prices
- **Target Price Slider** — drag to see P&L at any price
- **Days Held Slider** — see how time decay erodes value

### Time Decay Scenarios
Multi-line chart showing your P&L curve at Day 0, 5, 10, 15, and 20 — so you can visualize theta eating your position over time.

### Theta Watchdog
- Set a dollar threshold (e.g. $50/day)
- If Theta decay exceeds that threshold, the panel turns red and an alert fires

---

## ESTIMATING PROFIT

**Simple estimate:**
```
P&L = (Current Option Price - Premium Paid) × 100
```

**With time:**
```
P&L = (Delta × Stock Move) - (|Theta| × Days Held) × 100
```
*Note: This is a linear estimate. The actual P&L accounts for Gamma (acceleration) which makes the curve nonlinear.*

---

## SECURITY NOTES

- Credentials are sent only to your local Flask server and then to Robinhood's API
- Sessions are stored server-side only
- `robin_stocks` caches a login token locally in `~/robinhood.pickle` after first login
- **Never hardcode credentials** — use the login form or .env file
- This app has **no buy/sell functionality** — read-only access only

---

## DISCLAIMER

This tool is for **informational and educational purposes only**.
It is NOT financial advice. Options trading involves significant risk of loss.
Past performance does not predict future results.
Always consult a licensed financial advisor before trading.
