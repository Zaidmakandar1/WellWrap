#!/usr/bin/env python3
"""Simple script to create database and verify it works"""

from run_app import app, db, User, bcrypt
import os

print("🏥 Creating Healthcare Database...")

with app.app_context():
    # Create all tables
    db.create_all()
    print("✅ Database tables created")
    
    # Check database file location
    db_uri = app.config['SQLALCHEMY_DATABASE_URI']
    print(f"📁 Database URI: {db_uri}")
    
    if db_uri.startswith('sqlite:///'):
        db_path = db_uri.replace('sqlite:///', '')
        abs_path = os.path.abspath(db_path)
        print(f"📂 Database file path: {abs_path}")
        
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            print(f"✅ Database file exists! Size: {file_size} bytes")
        else:
            print("❌ Database file not found")
    
    # Check existing users
    user_count = User.query.count()
    print(f"👥 Users in database: {user_count}")
    
    if user_count > 0:
        users = User.query.all()
        for user in users:
            print(f"   - {user.username} ({user.first_name} {user.last_name})")
    
    print("🎉 Database setup completed!")
    print("💾 User data will persist between app restarts")
