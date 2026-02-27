from functools import wraps
from flask import jsonify
from flask_login import current_user, login_required


def subscription_required(f):
    """Decorator: requires active subscription (trial or paid)."""
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.has_active_subscription:
            plan = current_user.effective_plan
            if plan == 'expired' and current_user.plan == 'trial':
                return jsonify({
                    'error': 'Your 20-day free trial has ended.',
                    'action': 'upgrade',
                    'message': 'Subscribe for $20/month (Basic) or $40/month (Pro) to keep trading. 🐺',
                    'plans': {
                        'basic': {'price': 20, 'analyses_per_day': 5},
                        'pro': {'price': 40, 'analyses_per_day': 'unlimited', 'extras': ['trade_journal', 'alerts']}
                    }
                }), 402
            return jsonify({
                'error': 'Subscription required.',
                'action': 'upgrade',
            }), 402
        return f(*args, **kwargs)
    return decorated


def pro_required(f):
    """Decorator: requires Pro plan."""
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if current_user.effective_plan != 'pro':
            return jsonify({
                'error': 'This feature requires the Pro plan.',
                'action': 'upgrade_to_pro',
                'message': 'Upgrade to Pro ($40/month) for unlimited analyses, trade journal, and alerts. 🐺',
            }), 403
        return f(*args, **kwargs)
    return decorated


def analysis_gate(f):
    """Decorator: checks subscription + daily analysis limit."""
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        can_run, error = current_user.can_run_analysis()
        if not can_run:
            return jsonify({
                'error': error,
                'action': 'upgrade',
                'analyses_today': current_user.analyses_today,
                'daily_limit': current_user.daily_analysis_limit,
            }), 402
        
        # Run the analysis
        result = f(*args, **kwargs)
        
        # Only record if successful (2xx response)
        if hasattr(result, 'status_code') and 200 <= result.status_code < 300:
            current_user.record_analysis()
        elif isinstance(result, tuple) and len(result) == 2 and 200 <= result[1] < 300:
            current_user.record_analysis()
        
        return result
    return decorated
