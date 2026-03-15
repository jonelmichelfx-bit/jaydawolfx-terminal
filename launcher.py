"""
JayDaWolfX Terminal — Desktop Launcher
=======================================
Drop this file next to your app.py and run:
    python launcher.py

Nothing in app.py, templates, blueprints, or any
existing file is changed. This is purely additive.
"""

import os
import sys

# ── Make sure the project folder is on the path ──────────────────────
# This is needed when PyInstaller bundles everything into a .exe
if getattr(sys, 'frozen', False):
    # Running as .exe — point to the bundle folder
    BASE_DIR = sys._MEIPASS
else:
    # Running as plain Python script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

# ── Load .env BEFORE importing app so all keys are available ─────────
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, '.env'))

# ── Import the existing Flask app — nothing inside it changes ─────────
from app import app

# ── Launch desktop window ─────────────────────────────────────────────
try:
    from flaskwebgui import FlaskUI

    print("=" * 55)
    print("  🐺 JayDaWolfX Terminal — Starting Desktop Mode")
    print("=" * 55)

    FlaskUI(
        app=app,
        server="flask",
        width=1440,
        height=900,
        port=5000,
        fullscreen=False,
    ).run()

except ImportError:
    # ── Fallback: flaskwebgui not installed — browser mode ───────────
    print("=" * 55)
    print("  flaskwebgui not found — running in browser mode")
    print("  Open your browser to:  http://localhost:5000")
    print("  To install desktop mode:  pip install flaskwebgui")
    print("=" * 55)
    app.run(debug=False, port=5000, use_reloader=False)
