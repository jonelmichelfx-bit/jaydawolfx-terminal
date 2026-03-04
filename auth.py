# auth.py — Wolf Elite Options Terminal
from flask import Blueprint, render_template, redirect, request, session, jsonify
from flask_login import login_user, logout_user, current_user
from models import db, User
from datetime import datetime

auth = Blueprint('auth', __name__)
auth_bp = auth


@auth.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect('/')

    if request.method == 'GET':
        return render_template('auth.html')

    data       = request.get_json(silent=True) or {}
    identifier = data.get('identifier', '').strip().lower()
    password   = data.get('password', '')
    remember   = data.get('remember', True)

    user = User.query.filter_by(email=identifier).first()

    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid email or password.'}), 401

    if user.plan == 'expired':
        return jsonify({'error': 'Your trial has expired. Please upgrade.'}), 403

    login_user(user, remember=remember)
    session.permanent = True

    return jsonify({'message': 'Welcome back, Wolf! Loading terminal...'}), 200


@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect('/')

    if request.method == 'GET':
        return render_template('auth.html')

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

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'That username is already taken.'}), 409

    new_user = User(email=email, username=username, password=password)
    db.session.add(new_user)
    db.session.commit()

    login_user(new_user, remember=True)
    session.permanent = True

    return jsonify({'message': 'Welcome to Wolf Elite! Your 20-day trial has started.'}), 200


@auth.route('/logout', methods=['GET', 'POST'])
def logout():
    logout_user()
    session.clear()
    return redirect('/login')
