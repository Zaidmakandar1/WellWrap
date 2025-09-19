#!/usr/bin/env python3
"""
Production-ready startup script for WellWrap
Avoids the file watching issues of the development server
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def start_wellwrap():
    """Start WellWrap with stable configuration"""
    print("🏥 Starting WellWrap (Stable Mode)...")
    
    # Import the app
    from run_app import app, init_db
    
    # Initialize database
    print("📂 Initializing database...")
    init_db()
    
    # Configure for stability
    app.config['DEBUG'] = True
    app.config['TESTING'] = False
    
    print("📚 Web Application: http://localhost:5000")
    print("👤 Demo Login: demo_patient / demo123")
    print("📋 Upload medical reports and get AI analysis!")
    print("💾 All user data persists between sessions")
    print("🔧 Running in stable mode (no auto-reload)")
    print("Press Ctrl+C to stop")
    
    # Start the app without file watching
    try:
        app.run(
            debug=True,
            host='0.0.0.0', 
            port=5000,
            use_reloader=False,  # Disable auto-reload
            use_debugger=True,   # Keep debugger for error handling
            threaded=True        # Enable threading for better performance
        )
    except KeyboardInterrupt:
        print("\n👋 WellWrap stopped by user")
    except Exception as e:
        print(f"❌ Error starting WellWrap: {e}")
        sys.exit(1)

if __name__ == '__main__':
    start_wellwrap()