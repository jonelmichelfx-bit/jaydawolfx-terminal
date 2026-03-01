# forex.py — Wolf Elite Forex Terminal
# encoding: utf-8

from flask import Blueprint, render_template
from flask_login import login_required
from decorators import elite_required

forex_bp = Blueprint('forex', __name__)

@forex_bp.route('/forex')
@login_required
@elite_required
def forex():
    return render_template('forex.html')
