#!/usr/bin/env python3
"""
Minimal Flask app to test basic functionality
"""

from flask import Flask, jsonify, render_template_string
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test-key'

@app.route('/')
def home():
    """Minimal home page"""
    return """
    <html>
    <head>
        <title>WellWrap - Working!</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .btn { display: inline-block; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }
            .btn:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 WellWrap</h1>
            <p><strong>✅ Flask app is working!</strong></p>
            <p>This is a minimal version to test basic functionality.</p>
            
            <h3>🧪 Test Links:</h3>
            <a href="/test" class="btn">Test JSON API</a>
            <a href="/register" class="btn">Register</a>
            <a href="/login" class="btn">Login</a>
            
            <h3>📋 Next Steps:</h3>
            <ol>
                <li>Verify this page loads quickly</li>
                <li>Test the JSON API endpoint</li>
                <li>If working, we'll gradually add features back</li>
            </ol>
            
            <p><em>If you see this page, the Flask server is working correctly!</em></p>
        </div>
    </body>
    </html>
    """

@app.route('/test')
def test():
    """Test JSON endpoint"""
    return jsonify({
        "status": "success",
        "message": "Flask API is working!",
        "server": "minimal_app.py",
        "working": True
    })

@app.route('/register')
def register():
    """Placeholder register page"""
    return """
    <html>
    <head><title>Register</title></head>
    <body style="font-family: Arial; margin: 40px;">
        <h1>🔐 Register</h1>
        <p>Registration functionality will be added back once basic app is working.</p>
        <p><a href="/">← Back to Home</a></p>
    </body>
    </html>
    """

@app.route('/login')
def login():
    """Placeholder login page"""
    return """
    <html>
    <head><title>Login</title></head>
    <body style="font-family: Arial; margin: 40px;">
        <h1>🔑 Login</h1>
        <p>Login functionality will be added back once basic app is working.</p>
        <p><a href="/">← Back to Home</a></p>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("🚀 Starting Minimal WellWrap")
    print("=" * 50)
    print("📍 Server will be available at:")
    print("   🌐 http://localhost:5000")
    print("   🧪 http://localhost:5000/test")
    print()
    print("✅ This version should load instantly!")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)