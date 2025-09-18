#!/usr/bin/env python3
"""
Enhanced Flask app - Adding database and basic features back
"""

from flask import Flask, jsonify, render_template_string, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///healthcare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Simple User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@app.route('/')
def home():
    """Enhanced home page with database info"""
    try:
        user_count = User.query.count()
        return f"""
        <html>
        <head>
            <title>Medical Report Simplifier - Enhanced</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .btn {{ display: inline-block; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
                .btn:hover {{ background: #0056b3; }}
                .btn.success {{ background: #28a745; }}
                .btn.success:hover {{ background: #1e7e34; }}
                .status {{ padding: 15px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏥 WellWrap</h1>
                <div class="status">
                    <strong>✅ Enhanced Version Working!</strong><br>
                    📊 Database connected: {user_count} users registered
                </div>
                
                <h3>🧪 Test Features:</h3>
                <a href="/test" class="btn">Test JSON API</a>
                <a href="/register" class="btn success">Register New User</a>
                <a href="/users" class="btn">View Users</a>
                
                <h3>📋 Status:</h3>
                <ul>
                    <li>✅ Flask server working</li>
                    <li>✅ Database connected</li>
                    <li>✅ User model working</li>
                    <li>✅ No hanging issues</li>
                </ul>
                
                <p><em>This version includes database functionality!</em></p>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        return f"""
        <html>
        <body style="font-family: Arial; margin: 40px;">
            <h1>🏥 Medical Report Simplifier</h1>
            <div style="padding: 15px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">
                <strong>⚠️ Database Error:</strong> {str(e)}
            </div>
            <p><a href="/test">Test Basic API</a> (should still work)</p>
        </body>
        </html>
        """

@app.route('/test')
def test():
    """Test JSON endpoint with database info"""
    try:
        user_count = User.query.count()
        return jsonify({
            "status": "success",
            "message": "Enhanced Flask API is working!",
            "database": "connected",
            "users": user_count,
            "server": "enhanced_app.py",
            "features": ["database", "user_model", "bcrypt"]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "database": "error",
            "server": "enhanced_app.py"
        })

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Simple registration form"""
    if request.method == 'POST':
        try:
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            first_name = request.form['first_name']
            last_name = request.form['last_name']
            
            # Check if user exists
            if User.query.filter_by(username=username).first():
                return f"<h1>❌ Error</h1><p>Username already exists!</p><a href='/register'>Try again</a>"
            
            # Create new user
            password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
            new_user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                first_name=first_name,
                last_name=last_name
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            return f"""
            <html>
            <body style="font-family: Arial; margin: 40px;">
                <h1>✅ Registration Successful!</h1>
                <p>Welcome, {first_name} {last_name}!</p>
                <p>Username: {username}</p>
                <p><a href="/">← Back to Home</a></p>
            </body>
            </html>
            """
            
        except Exception as e:
            return f"<h1>❌ Registration Error</h1><p>{str(e)}</p><a href='/register'>Try again</a>"
    
    # GET request - show form
    return """
    <html>
    <head><title>Register</title></head>
    <body style="font-family: Arial; margin: 40px;">
        <h1>🔐 Register New User</h1>
        <form method="POST" style="max-width: 400px;">
            <p>
                <label>First Name:</label><br>
                <input type="text" name="first_name" required style="width: 100%; padding: 8px; margin: 5px 0;">
            </p>
            <p>
                <label>Last Name:</label><br>
                <input type="text" name="last_name" required style="width: 100%; padding: 8px; margin: 5px 0;">
            </p>
            <p>
                <label>Username:</label><br>
                <input type="text" name="username" required style="width: 100%; padding: 8px; margin: 5px 0;">
            </p>
            <p>
                <label>Email:</label><br>
                <input type="email" name="email" required style="width: 100%; padding: 8px; margin: 5px 0;">
            </p>
            <p>
                <label>Password:</label><br>
                <input type="password" name="password" required style="width: 100%; padding: 8px; margin: 5px 0;">
            </p>
            <p>
                <button type="submit" style="background: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 5px;">Register</button>
                <a href="/" style="margin-left: 10px;">Cancel</a>
            </p>
        </form>
    </body>
    </html>
    """

@app.route('/users')
def users():
    """View all users"""
    try:
        all_users = User.query.all()
        user_list = ""
        for user in all_users:
            user_list += f"""
            <tr>
                <td>{user.id}</td>
                <td>{user.username}</td>
                <td>{user.first_name} {user.last_name}</td>
                <td>{user.email}</td>
                <td>{user.created_at.strftime('%Y-%m-%d %H:%M')}</td>
            </tr>
            """
        
        return f"""
        <html>
        <head><title>Users</title></head>
        <body style="font-family: Arial; margin: 40px;">
            <h1>👥 Registered Users ({len(all_users)})</h1>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr style="background: #f8f9fa;">
                    <th style="padding: 10px;">ID</th>
                    <th style="padding: 10px;">Username</th>
                    <th style="padding: 10px;">Name</th>
                    <th style="padding: 10px;">Email</th>
                    <th style="padding: 10px;">Created</th>
                </tr>
                {user_list}
            </table>
            <p><a href="/">← Back to Home</a></p>
        </body>
        </html>
        """
    except Exception as e:
        return f"<h1>❌ Error</h1><p>{str(e)}</p><a href='/'>← Back to Home</a>"

def init_db():
    """Initialize database"""
    with app.app_context():
        db.create_all()
        print("✅ Database tables created")

if __name__ == '__main__':
    print("🚀 Starting Enhanced WellWrap")
    print("=" * 50)
    print("📂 Initializing database...")
    init_db()
    
    print("📍 Server will be available at:")
    print("   🌐 http://localhost:5000")
    print("   🧪 http://localhost:5000/test")
    print("   🔐 http://localhost:5000/register")
    print("   👥 http://localhost:5000/users")
    print()
    print("✅ Enhanced version with database support!")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)