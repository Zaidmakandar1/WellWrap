#!/usr/bin/env python3
"""
Medical Report Simplifier - Development Startup Script
Starts both Flask backend and Streamlit ML interface simultaneously
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shared.utils.logger import setup_logger
from shared.utils.config import load_config

logger = setup_logger(__name__)
config = load_config()


class DevelopmentServer:
    """Manages development server startup and monitoring"""
    
    def __init__(self):
        self.backend_process = None
        self.streamlit_process = None
        self.backend_port = config.get('FLASK_PORT', 5000)
        self.streamlit_port = config.get('STREAMLIT_SERVER_PORT', 8501)
        self.backend_url = f"http://localhost:{self.backend_port}"
        self.streamlit_url = f"http://localhost:{self.streamlit_port}"
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        logger.info("🔍 Checking dependencies...")
        
        try:
            import flask
            import streamlit
            logger.info("✅ Core dependencies found")
            return True
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            logger.error("Please run: pip install -r requirements.txt")
            return False
    
    def setup_environment(self):
        """Setup environment variables and directories"""
        logger.info("🔧 Setting up environment...")
        
        # Create required directories
        directories = [
            project_root / "logs",
            project_root / "uploads",
            project_root / "data" / "processed",
            project_root / "ml" / "models" / "trained"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        
        # Set environment variables
        os.environ['FLASK_ENV'] = 'development'
        os.environ['FLASK_DEBUG'] = 'True'
        os.environ['PYTHONPATH'] = str(project_root)
        
        logger.info("✅ Environment setup completed")
    
    def start_backend(self):
        """Start Flask backend server"""
        logger.info("🚀 Starting Flask backend server...")
        
        backend_script = project_root / "backend" / "app.py"
        if not backend_script.exists():
            logger.error(f"❌ Backend script not found: {backend_script}")
            return False
        
        try:
            # Change to backend directory
            os.chdir(project_root / "backend")
            
            self.backend_process = subprocess.Popen([
                sys.executable, "app.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            logger.info(f"✅ Flask backend started on {self.backend_url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False
    
    def start_streamlit(self):
        """Start Streamlit ML interface"""
        logger.info("🤖 Starting Streamlit ML interface...")
        
        streamlit_script = project_root / "ml" / "streamlit_app" / "main.py"
        if not streamlit_script.exists():
            logger.error(f"❌ Streamlit script not found: {streamlit_script}")
            return False
        
        try:
            # Change to project root
            os.chdir(project_root)
            
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                str(streamlit_script),
                "--server.port", str(self.streamlit_port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            logger.info(f"✅ Streamlit interface started on {self.streamlit_url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Streamlit: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor running processes and handle errors"""
        while True:
            time.sleep(5)  # Check every 5 seconds
            
            # Check backend process
            if self.backend_process and self.backend_process.poll() is not None:
                logger.error("❌ Backend process stopped unexpectedly")
                self.show_process_output(self.backend_process, "Backend")
                break
            
            # Check Streamlit process
            if self.streamlit_process and self.streamlit_process.poll() is not None:
                logger.error("❌ Streamlit process stopped unexpectedly")
                self.show_process_output(self.streamlit_process, "Streamlit")
                break
    
    def show_process_output(self, process, name):
        """Show output from a process for debugging"""
        if process.stdout:
            stdout = process.stdout.read()
            if stdout:
                logger.info(f"📄 {name} stdout:\\n{stdout}")
        
        if process.stderr:
            stderr = process.stderr.read()
            if stderr:
                logger.error(f"📄 {name} stderr:\\n{stderr}")
    
    def open_browsers(self):
        """Open web browsers to the applications"""
        time.sleep(3)  # Wait for servers to start
        
        try:
            logger.info("🌐 Opening web browsers...")
            webbrowser.open(self.backend_url)
            time.sleep(1)
            webbrowser.open(self.streamlit_url)
        except Exception as e:
            logger.warning(f"⚠️ Could not open browsers automatically: {e}")
            logger.info(f"Please open manually:")
            logger.info(f"  📚 Backend: {self.backend_url}")
            logger.info(f"  🤖 ML Interface: {self.streamlit_url}")
    
    def cleanup(self):
        """Clean up processes on exit"""
        logger.info("🧹 Cleaning up processes...")
        
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            logger.info("✅ Backend process stopped")
        
        if self.streamlit_process:
            try:
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
            logger.info("✅ Streamlit process stopped")
    
    def run(self):
        """Main execution method"""
        try:
            # Pre-flight checks
            if not self.check_dependencies():
                return 1
            
            self.setup_environment()
            
            # Start servers
            logger.info("🚀 Starting Medical Report Simplifier Development Environment")
            logger.info("="*60)
            
            # Start backend in thread
            backend_thread = threading.Thread(target=self.start_backend)
            backend_thread.daemon = True
            backend_thread.start()
            
            # Wait a bit before starting Streamlit
            time.sleep(2)
            
            # Start Streamlit in thread
            streamlit_thread = threading.Thread(target=self.start_streamlit)
            streamlit_thread.daemon = True
            streamlit_thread.start()
            
            # Wait for both to start
            time.sleep(3)
            
            # Open browsers
            browser_thread = threading.Thread(target=self.open_browsers)
            browser_thread.daemon = True
            browser_thread.start()
            
            # Show status
            logger.info("="*60)
            logger.info("🎉 Medical Report Simplifier is running!")
            logger.info(f"📚 Flask Backend: {self.backend_url}")
            logger.info(f"🤖 ML Interface: {self.streamlit_url}")
            logger.info("="*60)
            logger.info("Press Ctrl+C to stop all services")
            logger.info("="*60)
            
            # Monitor processes
            self.monitor_processes()
            
        except KeyboardInterrupt:
            logger.info("\\n🛑 Shutting down servers...")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()
        
        return 0


def main():
    """Entry point for the development server"""
    server = DevelopmentServer()
    exit_code = server.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
