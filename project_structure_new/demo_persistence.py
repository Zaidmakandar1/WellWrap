#!/usr/bin/env python3
"""
Demo: Real-Time User Persistence
This demonstrates that users register once and can login forever
"""

from run_app import app, db, User, bcrypt
from datetime import datetime
import getpass

def list_users():
    """List all users in the database"""
    print("\n👥 Current Users in Database:")
    print("=" * 40)
    
    with app.app_context():
        users = User.query.all()
        if not users:
            print("   No users found")
            return
        
        for i, user in enumerate(users, 1):
            print(f"{i}. Username: {user.username}")
            print(f"   Name: {user.first_name} {user.last_name}")
            print(f"   Email: {user.email}")
            print(f"   Created: {user.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if user.last_login:
                print(f"   Last Login: {user.last_login.strftime('%Y-%m-%d %H:%M:%S')}")
            print()

def create_user():
    """Create a new user"""
    print("\n➕ Register New User")
    print("=" * 30)
    
    username = input("Username: ").strip()
    if not username:
        print("❌ Username cannot be empty")
        return
    
    with app.app_context():
        # Check if username exists
        if User.query.filter_by(username=username).first():
            print(f"❌ Username '{username}' already exists!")
            return
        
        email = input("Email: ").strip()
        if User.query.filter_by(email=email).first():
            print(f"❌ Email '{email}' already registered!")
            return
        
        password = getpass.getpass("Password: ")
        first_name = input("First Name: ").strip()
        last_name = input("Last Name: ").strip()
        
        # Create user
        new_user = User(
            username=username,
            email=email,
            password_hash=bcrypt.generate_password_hash(password).decode('utf-8'),
            first_name=first_name,
            last_name=last_name,
            date_of_birth=datetime.now().date()
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        print(f"✅ User '{username}' registered successfully!")
        print("🔐 They can now login anytime at http://localhost:5000")

def test_login():
    """Test user login"""
    print("\n🔐 Test User Login")
    print("=" * 25)
    
    username = input("Username: ").strip()
    password = getpass.getpass("Password: ")
    
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            print(f"✅ Login successful!")
            print(f"   Welcome back, {user.first_name} {user.last_name}!")
            print(f"   Account created: {user.created_at.strftime('%Y-%m-%d')}")
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.session.commit()
            print(f"   Last login updated")
        else:
            print("❌ Invalid username or password")

def main():
    print("🏥 Medical Report Simplifier - User Persistence Demo")
    print("=" * 55)
    
    with app.app_context():
        db.create_all()
    
    while True:
        print("\nWhat would you like to do?")
        print("1. List all users")
        print("2. Register new user")
        print("3. Test login")
        print("4. Start web app")
        print("5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == '1':
            list_users()
        elif choice == '2':
            create_user()
        elif choice == '3':
            test_login()
        elif choice == '4':
            print("\n🚀 Starting web application...")
            print("📱 Visit http://localhost:5000 in your browser")
            print("🔐 Use any registered username/password to login")
            print("💾 All user data persists permanently!")
            import os
            os.system("python run_app.py")
            break
        elif choice == '5':
            print("\n👋 Goodbye! All user data remains saved.")
            print("💾 Users can login anytime in future sessions!")
            break
        else:
            print("❌ Invalid choice, please try again")

if __name__ == "__main__":
    main()
