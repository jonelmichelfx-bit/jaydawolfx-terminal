from app import app, db
from models import User

EMAIL    = "jonel.michelfx@gmail.com"   # <-- change this
USERNAME = "Jaydawolfx"        # <-- change this (your username)
PASSWORD = "JayceNailaPamela04$"  # <-- change this

with app.app_context():
    db.create_all()
    existing = User.query.filter_by(email=EMAIL).first()
    if existing:
        print(f"User {EMAIL} already exists — upgrading to admin...")
        existing.plan = "admin"
        existing.is_active = True
        db.session.commit()
        print("Done! Plan set to admin.")
    else:
        u = User(
            email=EMAIL,
            username=USERNAME,
            password=PASSWORD
        )
        u.plan = "admin"
        u.is_active = True
        db.session.add(u)
        db.session.commit()
        print(f"Admin user created: {EMAIL}")
