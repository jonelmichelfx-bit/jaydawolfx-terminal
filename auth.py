# auth.py — Wolf Elite Options Terminal
# encoding: utf-8

from flask import Blueprint, render_template, redirect, url_for, request, session, jsonify
from flask_login import login_user, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User
from datetime import datetime, timedelta

auth = Blueprint('auth', __name__)
auth_bp = auth


# ─────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────
@auth.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'GET':
        return render_template('auth.html')

    # POST — JSON from fetch()
    data       = request.get_json(silent=True) or {}
    identifier = data.get('identifier', '').strip().lower()
    password   = data.get('password', '')
    remember   = data.get('remember', True)

    user = User.query.filter_by(email=identifier).first()

    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid email or password.'}), 401

    # Check trial expiry
    if user.plan == 'trial':
        trial_end = user.created_at + timedelta(days=20)
        if datetime.utcnow() > trial_end:
            user.plan = 'expired'
            db.session.commit()
            return jsonify({'error': 'Your 20-day trial has expired. Please upgrade.'}), 403

    login_user(user, remember=remember)
    session.permanent = True

    return jsonify({'message': 'Welcome back, Wolf! Loading terminal...'}), 200


# ─────────────────────────────────────────────
# SIGNUP
# ─────────────────────────────────────────────
@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'GET':
        return render_template('auth.html')

    # POST — JSON from fetch()
    data     = request.get_json(silent=True) or {}
    email    = data.get('email', '').strip().lower()
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not email or not username or not password:
        return jsonify({'error': 'All fields are required.'}), 400

    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters.'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'An account with that email already exists.'}), 409

    new_user = User(
        email         = email,
        password_hash = generate_password_hash(password),
        plan          = 'trial',
        created_at    = datetime.utcnow()
    )
    db.session.add(new_user)
    db.session.commit()

    login_user(new_user, remember=True)
    session.permanent = True

    return jsonify({'message': 'Welcome to Wolf Elite! Your 20-day trial has started.'}), 200


# ─────────────────────────────────────────────
# LOGOUT  (GET + POST — both work)
# ─────────────────────────────────────────────
@auth.route('/logout', methods=['GET', 'POST'])
def logout():
    logout_user()
    session.clear()
    return redirect('/auth/login')
