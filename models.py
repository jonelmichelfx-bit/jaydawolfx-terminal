from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    
    # Subscription
    plan = db.Column(db.String(20), default='trial')  # trial, basic, elite, expired
    trial_start = db.Column(db.DateTime, default=datetime.utcnow)
    trial_end = db.Column(db.DateTime)
    subscription_start = db.Column(db.DateTime)
    subscription_end = db.Column(db.DateTime)
    
    # Stripe
    stripe_customer_id = db.Column(db.String(100), unique=True, nullable=True)
    stripe_subscription_id = db.Column(db.String(100), unique=True, nullable=True)
    
    # Usage tracking
    analyses_today = db.Column(db.Integer, default=0)
    last_analysis_date = db.Column(db.Date, default=datetime.utcnow().date)
    
    # Meta
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    email_verified = db.Column(db.Boolean, default=False)
    
    # Relationships
    journal_entries = db.relationship('TradeJournal', backref='user', lazy=True)
    alerts = db.relationship('Alert', backref='user', lazy=True)

    def __init__(self, email, username, password):
        self.email = email.lower().strip()
        self.username = username.strip()
        self.password_hash = generate_password_hash(password)
        self.trial_start = datetime.utcnow()
        self.trial_end = datetime.utcnow() + timedelta(days=20)
        self.plan = 'trial'

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    @property
    def is_trial_active(self):
        return self.plan == 'trial' and datetime.utcnow() < self.trial_end

    @property
    def trial_days_remaining(self):
        if self.plan != 'trial':
            return 0
        delta = self.trial_end - datetime.utcnow()
        return max(0, delta.days)

    @property
    def has_active_subscription(self):
        if self.plan == 'trial' and self.is_trial_active:
            return True


    @property
    def effective_plan(self):
        """Returns the actual plan considering trial/expiry state."""
        if self.plan == 'trial':
            return 'trial' if self.is_trial_active else 'expired'

        return 'expired'

    @property
    def daily_analysis_limit(self):
        plan = self.effective_plan
        if plan == 'expired':
            return 0

        return 0

    def can_run_analysis(self):
        """Check if user can run another analysis today."""
        if not self.has_active_subscription:
            return False, "Your subscription has expired. Please upgrade to continue."
        
        today = datetime.utcnow().date()
        if self.last_analysis_date != today:
            self.analyses_today = 0
            self.last_analysis_date = today
            db.session.commit()

        limit = self.daily_analysis_limit
        if limit is None:
            return True, None
        if self.analyses_today >= limit:
            plan = self.effective_plan

            return False, f"Daily limit reached ({limit}/day). Your trial allows 5 analyses per day."
        return True, None

    def record_analysis(self):
        """Increment analysis counter."""
        today = datetime.utcnow().date()
        if self.last_analysis_date != today:
            self.analyses_today = 0
            self.last_analysis_date = today
        self.analyses_today += 1
        db.session.commit()

    def upgrade_to(self, plan, months=1):
        """Upgrade user to a paid plan."""
        now = datetime.utcnow()
        self.plan = plan
        self.subscription_start = now
        self.subscription_end = now + timedelta(days=30 * months)
        db.session.commit()

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'plan': self.effective_plan,
            'trial_days_remaining': self.trial_days_remaining,
            'analyses_today': self.analyses_today,
            'daily_limit': self.daily_analysis_limit,

        }

    def __repr__(self):
        return f'<User {self.username} [{self.effective_plan}]>'


class TradeJournal(db.Model):
    __tablename__ = 'trade_journal'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Trade info
    ticker = db.Column(db.String(20), nullable=False)
    strategy = db.Column(db.String(100))  # e.g., "Bull Call Spread", "Iron Condor"
    option_type = db.Column(db.String(10))  # call / put
    strike = db.Column(db.Float)
    expiration = db.Column(db.Date)
    contracts = db.Column(db.Integer, default=1)
    
    # Entry/Exit
    entry_price = db.Column(db.Float)
    exit_price = db.Column(db.Float)
    entry_date = db.Column(db.DateTime, default=datetime.utcnow)
    exit_date = db.Column(db.DateTime)
    
    # P&L
    realized_pnl = db.Column(db.Float)
    unrealized_pnl = db.Column(db.Float)
    
    # Notes
    thesis = db.Column(db.Text)
    notes = db.Column(db.Text)
    tags = db.Column(db.String(200))
    
    # Status
    status = db.Column(db.String(20), default='open')  # open, closed, expired
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'strategy': self.strategy,
            'option_type': self.option_type,
            'strike': self.strike,
            'expiration': self.expiration.isoformat() if self.expiration else None,
            'contracts': self.contracts,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'realized_pnl': self.realized_pnl,
            'thesis': self.thesis,
            'notes': self.notes,
            'tags': self.tags,
            'status': self.status,
        }


class Alert(db.Model):
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    ticker = db.Column(db.String(20), nullable=False)
    alert_type = db.Column(db.String(50))  # price_above, price_below, iv_spike, delta_cross
    threshold = db.Column(db.Float)
    message = db.Column(db.String(200))
    
    is_active = db.Column(db.Boolean, default=True)
    triggered_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'alert_type': self.alert_type,
            'threshold': self.threshold,
            'message': self.message,
            'is_active': self.is_active,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'created_at': self.created_at.isoformat(),
        }
