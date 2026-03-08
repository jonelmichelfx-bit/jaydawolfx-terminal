# decorators.py — Wolf Terminal
# encoding: utf-8

from functools import wraps
from flask import redirect, url_for, flash, jsonify, request
from flask_login import current_user


def _is_api_request():
    return (
        request.path.startswith('/api/')
        or request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        or request.accept_mimetypes.best == 'application/json'
    )


def _plan_rank(plan):
    """trial=1, basic=2, pro=3, elite=4, byakugan=5, admin=99
    Admin ALWAYS bypasses everything — never touches gates."""
    return {
        'trial':    1,
        'basic':    2,
        'pro':      3,
        'elite':    4,
        'byakugan': 5,
        'admin':    99
    }.get(plan, 0)


def _is_admin():
    """Admin bypasses ALL restrictions."""
    return current_user.is_authenticated and current_user.plan == 'admin'


def analysis_gate(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            if _is_api_request():
                return jsonify({'error': 'Login required', 'redirect': url_for('auth.login_page')}), 401
            flash('Please log in to access the terminal.', 'warning')
            return redirect(url_for('auth.login_page'))
        if _is_admin(): return f(*args, **kwargs)
        if current_user.plan == 'expired' or (current_user.plan == 'trial' and not current_user.is_trial_active):
            if _is_api_request():
                return jsonify({'error': 'Trial expired. Please upgrade.', 'redirect': url_for('pricing')}), 403
            flash('Your free trial has ended. Upgrade to keep trading.', 'warning')
            return redirect(url_for('pricing'))
        return f(*args, **kwargs)
    return decorated


def basic_required(f):
    """Wolf Basic ($29/mo) — Academy and basic features."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            if _is_api_request():
                return jsonify({'error': 'Login required'}), 401
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('auth.login_page'))
        if _is_admin(): return f(*args, **kwargs)
        if _plan_rank(current_user.plan) < _plan_rank('basic'):
            if _is_api_request():
                return jsonify({'error': 'Wolf Basic plan required.', 'redirect': url_for('pricing')}), 403
            flash('Upgrade to Wolf Basic ($29/mo) to unlock this feature.', 'warning')
            return redirect(url_for('pricing'))
        return f(*args, **kwargs)
    return decorated


def pro_required(f):
    """Legacy pro gate — maps to elite level."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            if _is_api_request():
                return jsonify({'error': 'Login required'}), 401
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('auth.login_page'))
        if _is_admin(): return f(*args, **kwargs)
        if _plan_rank(current_user.plan) < _plan_rank('pro'):
            if _is_api_request():
                return jsonify({'error': 'Wolf Elite plan required.', 'redirect': url_for('pricing')}), 403
            flash('Upgrade to Wolf Elite to unlock this feature.', 'warning')
            return redirect(url_for('pricing'))
        return f(*args, **kwargs)
    return decorated


def elite_required(f):
    """Wolf Elite ($97/mo) — Forex terminal, War Room, SPY Signal, Scanners."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            if _is_api_request():
                return jsonify({'error': 'Login required'}), 401
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('auth.login_page'))
        if _is_admin(): return f(*args, **kwargs)
        if _plan_rank(current_user.plan) < _plan_rank('elite'):
            if _is_api_request():
                return jsonify({'error': 'Wolf Elite plan required.', 'redirect': url_for('pricing')}), 403
            flash('🐺 Upgrade to Wolf Elite ($97/mo) to unlock this feature.', 'warning')
            return redirect(url_for('pricing'))
        return f(*args, **kwargs)
    return decorated


def byakugan_required(f):
    """Byakugan All Seeing Eye ($197/mo) — Legends Lab, AI Future, all premium features."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            if _is_api_request():
                return jsonify({'error': 'Login required'}), 401
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('auth.login_page'))
        if _is_admin(): return f(*args, **kwargs)
        if _plan_rank(current_user.plan) < _plan_rank('byakugan'):
            if _is_api_request():
                return jsonify({'error': 'Byakugan All Seeing Eye plan required.', 'redirect': url_for('pricing')}), 403
            flash('👁️ Upgrade to Byakugan All Seeing Eye ($197/mo) to unlock this feature.', 'warning')
            return redirect(url_for('pricing'))
        return f(*args, **kwargs)
    return decorated
