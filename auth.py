from flask import Blueprint, request, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User
from datetime import datetime
import re

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """Min 8 chars, at least one number."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number."
    return True, None


@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    
    # Validation
    email = data.get('email', '').strip().lower()
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not email or not username or not password:
        return jsonify({'error': 'All fields are required.'}), 400

    if not validate_email(email):
        return jsonify({'error': 'Invalid email address.'}), 400

    if len(username) < 3 or len(username) > 30:
        return jsonify({'error': 'Username must be 3-30 characters.'}), 400

    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return jsonify({'error': 'Username can only contain letters, numbers, and underscores.'}), 400

    valid, pw_error = validate_password(password)
    if not valid:
        return jsonify({'error': pw_error}), 400

    # Check duplicates
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'An account with this email already exists.'}), 409

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken.'}), 409

    # Create user
    user = User(email=email, username=username, password=password)
    db.session.add(user)
    db.session.commit()

    login_user(user, remember=True)
    user.last_login = datetime.utcnow()
    db.session.commit()

    return jsonify({
        'message': f'Welcome to JAYDAWOLFX, {user.username}! Your 20-day free trial has started. 🐺',
        'user': user.to_dict(),
        'trial_end': user.trial_end.isoformat(),
    }), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    identifier = data.get('identifier', '').strip()  # email or username
    password = data.get('password', '')
    remember = data.get('remember', False)

    if not identifier or not password:
        return jsonify({'error': 'Email/username and password are required.'}), 400

    # Find user by email or username
    user = User.query.filter(
        (User.email == identifier.lower()) | (User.username == identifier)
    ).first()

    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid credentials.'}), 401

    if not user.is_active:
        return jsonify({'error': 'This account has been deactivated.'}), 403

    login_user(user, remember=remember)
    user.last_login = datetime.utcnow()
    db.session.commit()

    return jsonify({
        'message': f'Welcome back, {user.username}! 🐺',
        'user': user.to_dict(),
    }), 200


@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully.'}), 200


@auth_bp.route('/me', methods=['GET'])
@login_required
def me():
    return jsonify({'user': current_user.to_dict()}), 200


@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    data = request.get_json()
    current_password = data.get('current_password', '')
    new_password = data.get('new_password', '')

    if not current_user.check_password(current_password):
        return jsonify({'error': 'Current password is incorrect.'}), 401

    valid, pw_error = validate_password(new_password)
    if not valid:
        return jsonify({'error': pw_error}), 400

    current_user.set_password(new_password)
    db.session.commit()

    return jsonify({'message': 'Password updated successfully.'}), 200


@auth_bp.route('/status', methods=['GET'])
@login_required
def subscription_status():
    user = current_user
    plan = user.effective_plan
    
    status = {
        'plan': plan,
        'username': user.username,
        'email': user.email,
    }

    if plan == 'trial':
        status['trial_days_remaining'] = user.trial_days_remaining
        status['trial_end'] = user.trial_end.isoformat()
        status['message'] = f'{user.trial_days_remaining} days remaining in your free trial.'
    elif plan == 'expired':
        status['message'] = 'Your trial has expired. Subscribe to continue trading. 🐺💰'
    elif plan == 'basic':
        status['message'] = 'Basic Plan — 5 analyses/day'
        status['subscription_end'] = user.subscription_end.isoformat() if user.subscription_end else None
    elif plan == 'pro':
        status['message'] = 'Pro Plan — Unlimited. You\'re a wolf. 🐺'
        status['subscription_end'] = user.subscription_end.isoformat() if user.subscription_end else None

    status['analyses_today'] = user.analyses_today
    status['daily_limit'] = user.daily_analysis_limit

    return jsonify(status), 200
