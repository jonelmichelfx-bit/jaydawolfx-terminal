from functools import wraps
from flask import redirect, url_for, flash, session
from datetime import datetime

# ─────────────────────────────────────────────
#  Plan hierarchy:  trial < basic < elite
# ─────────────────────────────────────────────
PLAN_RANK = {
    "trial": 0,
    "basic": 1,
    "elite": 2,
}

def _get_user_plan():
    """
    Returns the effective plan string for the current session user.
    Expected session keys (set at login/subscription webhook):
        session['plan']          → 'trial' | 'basic' | 'elite'
        session['trial_end']     → datetime  (only for trial users)
        session['logged_in']     → True
    """
    if not session.get("logged_in"):
        return None

    plan = session.get("plan", "trial")

    # Check trial expiry
    if plan == "trial":
        trial_end = session.get("trial_end")
        if trial_end and datetime.utcnow() > trial_end:
            return "expired"

    return plan


def _require_plan(min_plan: str, redirect_target: str = "pricing"):
    """Factory that creates a decorator requiring at least `min_plan`."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            plan = _get_user_plan()

            if plan is None:
                flash("Please log in to access this page.", "warning")
                return redirect(url_for("login_page"))

            if plan == "expired":
                flash("Your free trial has expired. Upgrade to continue.", "warning")
                return redirect(url_for(redirect_target))

            if PLAN_RANK.get(plan, -1) < PLAN_RANK[min_plan]:
                flash(
                    f"This feature requires the "
                    f"{'Wolf Elite' if min_plan == 'elite' else 'Basic'} plan. "
                    f"Upgrade to unlock it! 🐺",
                    "upgrade",
                )
                return redirect(url_for(redirect_target))

            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ─────────────────────────────────────────────
#  Public decorators — use these on your routes
# ─────────────────────────────────────────────

def login_required(f):
    """Any logged-in user (including active trial)."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        plan = _get_user_plan()
        if plan is None:
            flash("Please log in.", "warning")
            return redirect(url_for("login_page"))
        if plan == "expired":
            flash("Your free trial has expired. Upgrade to continue.", "warning")
            return redirect(url_for("pricing"))
        return f(*args, **kwargs)
    return decorated_function


# Trial: Greeks calculator only
trial_required = login_required   # alias — any active session works

# Basic ($29/mo): Greeks + AI Scanner
basic_required = _require_plan("basic")

# Wolf Elite ($150/mo): Everything
elite_required = _require_plan("elite")

# ─────────────────────────────────────────────
#  analysis_gate — used on API routes
#  Blocks if trial expired or not logged in
#  (returns JSON error instead of redirect)
# ─────────────────────────────────────────────
from flask import jsonify
from flask_login import current_user

def analysis_gate(f):
    """Protects API routes — returns JSON error instead of redirect."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Please log in to access this feature.', 'action': 'login'}), 401
        if current_user.effective_plan == 'expired':
            return jsonify({'error': 'Your trial has expired. Upgrade to continue.', 'action': 'upgrade'}), 403
        return f(*args, **kwargs)
    return decorated_function
