# decorators.py — Wolf Elite Options Terminal
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
    """Higher number = more access. admin = full access."""
    return {'trial': 1, 'basic': 2, 'elite': 3, 'admin': 99}.get(plan, 0)


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
        if current_user.plan == 'expired':
            if _is_api_request():
                return jsonify({'error': 'Trial expired. Please upgrade.', 'redirect': url_for('pricing')}), 403
            flash('Your trial has expired. Please upgrade to continue.', 'warning')
            return redirect(url_for('pricing'))
        return f(*args, **kwargs)
    return decorated


def basic_required(f):
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
                return jsonify({'error': 'Basic plan or higher required.', 'redirect': url_for('pricing')}), 403
            flash('Upgrade to Basic ($29/mo) to unlock this feature.', 'warning')
            return redirect(url_for('pricing'))
        return f(*args, **kwargs)
    return decorated


def elite_required(f):
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
            flash('Upgrade to Wolf Elite ($150/mo) to unlock this feature.', 'warning')
            return redirect(url_for('pricing'))
        return f(*args, **kwargs)
    return decorated
