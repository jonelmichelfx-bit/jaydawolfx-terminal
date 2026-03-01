# auth.py — Wolf Elite Options Terminal
# Blueprint: /auth — handles login, signup, logout
# encoding: utf-8

from flask import Blueprint, render_template, redirect, url_for, request, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User
from datetime import datetime, timedelta

auth = Blueprint('auth', __name__)


# ─────────────────────────────────────────────
# LOGIN PAGE  (GET)
# ─────────────────────────────────────────────
@auth.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password_hash, password):
            flash('Invalid email or password.', 'error')
            return redirect(url_for('auth.login_page'))

        # Check trial expiry
        if user.plan == 'trial':
            trial_end = user.created_at + timedelta(days=20)
            if datetime.utcnow() > trial_end:
                user.plan = 'expired'
                db.session.commit()
                flash('Your 20-day trial has expired. Please upgrade to continue.', 'warning')
                return redirect(url_for('auth.login_page'))

        login_user(user, remember=True)
        session.permanent = True

        next_page = request.args.get('next')
        return redirect(next_page or url_for('index'))

    return render_template('login.html')


# ─────────────────────────────────────────────
# SIGNUP PAGE  (GET + POST)
# ─────────────────────────────────────────────
@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email     = request.form.get('email', '').strip().lower()
        password  = request.form.get('password', '')
        password2 = request.form.get('password2', '')

        if not email or not password:
            flash('Email and password are required.', 'error')
            return redirect(url_for('auth.signup'))

        if password != password2:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('auth.signup'))

        existing = User.query.filter_by(email=email).first()
        if existing:
            flash('An account with that email already exists.', 'error')
            return redirect(url_for('auth.signup'))

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
        flash('Welcome to Wolf Elite! Your 20-day trial has started.', 'success')
        return redirect(url_for('index'))

    return render_template('signup.html')


# ─────────────────────────────────────────────
# LOGOUT  (GET + POST — both work)
# ─────────────────────────────────────────────
@auth.route('/logout', methods=['GET', 'POST'])
def logout():
    logout_user()
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login_page'))

auth_bp = auth
 
 