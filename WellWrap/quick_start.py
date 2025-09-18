#!/usr/bin/env python3
"""
Medical Report Simplifier - Quick Start Script
Get up and running with the new clean architecture immediately
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    print("🏥 Medical Report Simplifier - Clean Architecture")
    print("=" * 50)
    print()
    
    project_root = Path(__file__).parent
    
    print("🔍 Welcome to your newly organized Medical Report Simplifier!")
    print()
    print("📂 Your new project structure:")
    print("  📁 backend/     - Flask API and business logic")
    print("  📁 frontend/    - Web templates and static files") 
    print("  📁 ml/          - Machine learning models and services")
    print("  📁 data/        - Data storage and samples")
    print("  📁 shared/      - Common utilities and configuration")
    print("  📁 tests/       - Test suites")
    print("  📁 docs/        - Documentation")
    print("  📁 scripts/     - Setup and deployment scripts")
    print("  📁 config/      - Configuration files")
    print()
    
    print("🚀 Quick Start Options:")
    print()
    print("1. 📦 Install dependencies")
    print("2. 🔧 Setup development environment") 
    print("3. 🏃 Run the application")
    print("4. 📚 View documentation")
    print("5. 🧪 Run tests")
    print("6. 🔄 Migrate from old structure")
    print("7. ❓ Show help")
    print("0. 🚪 Exit")
    print()
    
    while True:
        choice = input("Choose an option (0-7): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
            
        elif choice == "1":
            install_dependencies(project_root)
            
        elif choice == "2":
            setup_environment(project_root)
            
        elif choice == "3":
            run_application(project_root)
            
        elif choice == "4":
            show_documentation(project_root)
            
        elif choice == "5":
            run_tests(project_root)
            
        elif choice == "6":
            migrate_structure(project_root)
            
        elif choice == "7":
            show_help()
            
        else:
            print("❌ Invalid choice. Please select 0-7.")
        
        print("\\n" + "-" * 50 + "\\n")


def install_dependencies(project_root: Path):
    """Install project dependencies"""
    print("📦 Installing dependencies...")
    
    requirements_file = project_root / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")


def setup_environment(project_root: Path):
    """Setup development environment"""
    print("🔧 Setting up development environment...")
    
    # Create necessary directories
    directories = [
        "logs",
        "uploads", 
        "data/processed",
        "ml/models/trained"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {directory}")
    
    # Copy .env.example to .env if it doesn't exist
    env_example = project_root / ".env.example"
    env_file = project_root / ".env"
    
    if env_example.exists() and not env_file.exists():
        env_file.write_text(env_example.read_text())
        print("📄 Created .env file from .env.example")
        print("⚠️  Please review and update .env with your settings")
    
    print("✅ Development environment setup complete!")


def run_application(project_root: Path):
    """Run the application"""
    print("🏃 Running Medical Report Simplifier...")
    print()
    print("Choose how to run:")
    print("1. 🌐 Full application (Flask + Streamlit)")
    print("2. 🔙 Backend only (Flask)")
    print("3. 🤖 ML interface only (Streamlit)")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        print("🚀 Starting full application...")
        print("📚 Backend will be at: http://localhost:5000")
        print("🤖 ML interface will be at: http://localhost:8501")
        print("\\nPress Ctrl+C to stop all services")
        
        # Check if start script exists
        start_script = project_root / "scripts" / "setup" / "start_development.py"
        if start_script.exists():
            try:
                subprocess.run([sys.executable, str(start_script)])
            except KeyboardInterrupt:
                print("\\n🛑 Application stopped")
        else:
            print("❌ Start script not found. Please run backend and ML interface manually.")
    
    elif choice == "2":
        print("🚀 Starting backend only...")
        backend_script = project_root / "backend" / "app.py"
        if backend_script.exists():
            try:
                os.chdir(project_root / "backend")
                subprocess.run([sys.executable, "app.py"])
            except KeyboardInterrupt:
                print("\\n🛑 Backend stopped")
        else:
            print("❌ Backend app.py not found!")
    
    elif choice == "3":
        print("🚀 Starting ML interface only...")
        ml_script = project_root / "ml" / "streamlit_app" / "main.py"
        if ml_script.exists():
            try:
                subprocess.run([
                    sys.executable, "-m", "streamlit", "run", str(ml_script)
                ])
            except KeyboardInterrupt:
                print("\\n🛑 ML interface stopped")
        else:
            print("❌ ML interface main.py not found!")
    
    else:
        print("❌ Invalid choice")


def show_documentation(project_root: Path):
    """Show documentation information"""
    print("📚 Documentation Overview:")
    print()
    
    docs_dir = project_root / "docs"
    if docs_dir.exists():
        print("📁 Documentation files available:")
        for doc_file in docs_dir.rglob("*.md"):
            print(f"  📄 {doc_file.relative_to(project_root)}")
    
    readme_file = project_root / "README.md"
    if readme_file.exists():
        print(f"\\n📖 Main documentation: {readme_file}")
        print("\\n📋 Key sections:")
        print("  🏗️  Project Structure")
        print("  🎯 Architecture Overview") 
        print("  🚀 Getting Started")
        print("  🛠️  Technology Stack")
        print("  📝 Contributing Guidelines")
    
    print("\\n🌐 Online resources:")
    print("  🔗 GitHub Repository")
    print("  📖 Documentation Site")
    print("  💬 Community Support")


def run_tests(project_root: Path):
    """Run project tests"""
    print("🧪 Running tests...")
    
    tests_dir = project_root / "tests"
    if not tests_dir.exists():
        print("❌ Tests directory not found!")
        return
    
    try:
        # Try to run pytest
        subprocess.run([
            sys.executable, "-m", "pytest", str(tests_dir), "-v"
        ], check=True)
        print("✅ All tests passed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Some tests failed: {e}")
    except FileNotFoundError:
        print("❌ pytest not installed. Install with: pip install pytest")


def migrate_structure(project_root: Path):
    """Migrate from old project structure"""
    print("🔄 Structure Migration")
    print()
    print("This will help you migrate from the old project structure")
    print("to the new clean architecture.")
    print()
    
    migration_script = project_root / "scripts" / "setup" / "migrate_to_clean_structure.py"
    
    if migration_script.exists():
        print("📁 Migration script found!")
        print("⚠️  This will create a backup of your current project")
        print("⚠️  and reorganize files according to clean architecture principles")
        print()
        
        response = input("Proceed with migration? (y/n): ").strip().lower()
        if response == 'y':
            try:
                subprocess.run([sys.executable, str(migration_script)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Migration failed: {e}")
        else:
            print("Migration cancelled.")
    else:
        print("❌ Migration script not found!")


def show_help():
    """Show detailed help information"""
    print("❓ Medical Report Simplifier - Help")
    print()
    print("🏗️  Architecture Overview:")
    print("   This project follows clean architecture principles with clear")
    print("   separation between backend, frontend, ML, and shared components.")
    print()
    print("📁 Key Directories:")
    print("   backend/   - Flask API, models, services, database")
    print("   frontend/  - HTML templates, CSS, JavaScript")
    print("   ml/        - Machine learning models and processing")
    print("   shared/    - Common utilities, configuration, constants")
    print("   tests/     - Unit, integration, and ML tests")
    print("   docs/      - Project documentation")
    print("   scripts/   - Setup and deployment automation")
    print("   config/    - Environment and deployment configuration")
    print()
    print("🚀 Getting Started:")
    print("   1. Install dependencies (option 1)")
    print("   2. Setup environment (option 2)")
    print("   3. Run the application (option 3)")
    print()
    print("🛠️  Development Workflow:")
    print("   1. Make changes to your code")
    print("   2. Run tests (option 5)")
    print("   3. Test the application (option 3)")
    print("   4. Update documentation as needed")
    print()
    print("📞 Support:")
    print("   - Check README.md for detailed documentation")
    print("   - Review docs/ directory for specific guides")
    print("   - Use make commands for automation (if available)")


if __name__ == "__main__":
    main()
