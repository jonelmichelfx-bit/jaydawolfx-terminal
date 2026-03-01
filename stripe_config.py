"""
stripe_config.py  —  Wolf Elite Options Terminal
Replace the PRICE_ID values with your actual Stripe Price IDs from dashboard.stripe.com
"""

import os

STRIPE_SECRET_KEY      = os.environ.get("STRIPE_SECRET_KEY", "sk_live_REPLACE_ME")
STRIPE_WEBHOOK_SECRET  = os.environ.get("STRIPE_WEBHOOK_SECRET", "whsec_REPLACE_ME")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "pk_live_REPLACE_ME")

# ── Price IDs (create these in Stripe Dashboard → Products) ───────────────────
PRICES = {
    "basic": {
        "price_id":    os.environ.get("STRIPE_BASIC_PRICE_ID", "price_REPLACE_BASIC"),
        "amount":      2900,   # cents
        "label":       "Basic",
        "interval":    "month",
        "features": [
            "Greeks Calculator",
            "AI Options Scanner (Swing + Day Trader)",
            "20-day free trial included",
        ],
    },
    "elite": {
        "price_id":    os.environ.get("STRIPE_ELITE_PRICE_ID", "price_REPLACE_ELITE"),
        "amount":      15000,  # cents
        "label":       "Wolf Elite",
        "interval":    "month",
        "features": [
            "Everything in Basic",
            "AI Chart Analysis",
            "Wolf Elite Weekly Picks + Tracker",
            "Priority support",
            "Early access to Forex Terminal",
        ],
    },
}

def get_price_id(plan: str) -> str:
    return PRICES[plan]["price_id"]
