import stripe
import os
from flask import Blueprint, request, jsonify, redirect, url_for
from flask_login import login_required, current_user
from models import db, User
from datetime import datetime, timedelta

payments_bp = Blueprint('payments', __name__, url_prefix='/payments')

stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')

PRICE_IDS = {
    'basic': os.environ.get('STRIPE_BASIC_PRICE_ID'),
    'pro':   os.environ.get('STRIPE_PRO_PRICE_ID'),
}

PLAN_NAMES = {
    'basic': 'Basic Plan — $20/month',
    'pro':   'Pro Plan — $40/month',
}


@payments_bp.route('/create-checkout', methods=['POST'])
@login_required
def create_checkout():
    data = request.get_json()
    plan = data.get('plan')

    if plan not in PRICE_IDS:
        return jsonify({'error': 'Invalid plan.'}), 400

    price_id = PRICE_IDS[plan]
    if not price_id:
        return jsonify({'error': 'Plan not configured.'}), 500

    try:
        # Create or retrieve Stripe customer
        if current_user.stripe_customer_id:
            customer_id = current_user.stripe_customer_id
        else:
            customer = stripe.Customer.create(
                email=current_user.email,
                metadata={'user_id': current_user.id, 'username': current_user.username}
            )
            current_user.stripe_customer_id = customer.id
            db.session.commit()
            customer_id = customer.id

        # Create checkout session
        session = stripe.checkout.Session.create(
            customer=customer_id,
            line_items=[{'price': price_id, 'quantity': 1}],
            mode='subscription',
            success_url=url_for('payments.success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=url_for('payments.cancel', _external=True),
            metadata={'user_id': current_user.id, 'plan': plan}
        )

        return jsonify({'url': session.url}), 200

    except stripe.StripeError as e:
        return jsonify({'error': str(e)}), 500


@payments_bp.route('/success')
@login_required
def success():
    session_id = request.args.get('session_id')
    if session_id:
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            plan = session.metadata.get('plan', 'basic')
            current_user.upgrade_to(plan)
            current_user.stripe_subscription_id = session.subscription
            db.session.commit()
        except Exception:
            pass

    return redirect(url_for('index'))


@payments_bp.route('/cancel')
@login_required
def cancel():
    return redirect(url_for('pricing'))


@payments_bp.route('/webhook', methods=['POST'])
def webhook():
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')

    try:
        if webhook_secret:
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        else:
            event = stripe.Event.construct_from(
                request.get_json(), stripe.api_key
            )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Handle subscription events
    if event.type == 'invoice.payment_succeeded':
        invoice = event.data.object
        customer_id = invoice.customer
        user = User.query.filter_by(stripe_customer_id=customer_id).first()
        if user:
            # Extend subscription by 30 days
            user.subscription_end = datetime.utcnow() + timedelta(days=30)
            db.session.commit()

    elif event.type == 'customer.subscription.deleted':
        sub = event.data.object
        customer_id = sub.customer
        user = User.query.filter_by(stripe_customer_id=customer_id).first()
        if user:
            user.plan = 'expired'
            user.stripe_subscription_id = None
            db.session.commit()

    elif event.type == 'customer.subscription.updated':
        sub = event.data.object
        customer_id = sub.customer
        user = User.query.filter_by(stripe_customer_id=customer_id).first()
        if user and sub.status == 'active':
            user.subscription_end = datetime.utcnow() + timedelta(days=30)
            db.session.commit()

    return jsonify({'status': 'ok'}), 200


@payments_bp.route('/portal', methods=['POST'])
@login_required
def portal():
    """Sends user to Stripe billing portal to manage subscription."""
    if not current_user.stripe_customer_id:
        return jsonify({'error': 'No billing account found.'}), 400

    try:
        session = stripe.billing_portal.Session.create(
            customer=current_user.stripe_customer_id,
            return_url=url_for('index', _external=True),
        )
        return jsonify({'url': session.url}), 200
    except stripe.StripeError as e:
        return jsonify({'error': str(e)}), 500


@payments_bp.route('/status', methods=['GET'])
@login_required
def payment_status():
    return jsonify({
        'plan': current_user.effective_plan,
        'has_billing': bool(current_user.stripe_customer_id),
        'subscription_end': current_user.subscription_end.isoformat() if current_user.subscription_end else None,
    }), 200
