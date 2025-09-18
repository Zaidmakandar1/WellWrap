#!/usr/bin/env python3
"""
Medical Report Simplifier - Project Structure Migration Script
Migrates the existing project to the new clean architecture structure
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Migration mapping: (source_pattern, destination_path, file_pattern)
MIGRATION_MAP = [
    # Backend files
    ("app.py", "backend/", None),
    ("app_fixed.py", "backend/", None),
    ("run.py", "backend/", None),
    ("setup_database.py", "backend/database/", None),
    ("database_example.py", "backend/database/", None),
    ("backend/", "backend/", None),  # Keep existing backend structure
    
    # Frontend files
    ("frontend/", "frontend/", None),
    ("templates/", "frontend/templates/", None),
    
    # ML files
    ("advanced_medical_analyzer.py", "ml/streamlit_app/main.py", None),
    ("medical_ml_service.py", "ml/services/", None),
    ("ml/", "ml/", None),  # Keep existing ML structure
    
    # Data files
    ("medical_report_simplifier_package/data/", "data/samples/", None),
    ("data/", "data/", None),
    
    # Configuration files
    ("docker-compose.yml", "config/deployment/", None),
    ("Dockerfile", "config/deployment/", None),
    (".dockerignore", "config/deployment/", None),
    ("railway.toml", "config/deployment/", None),
    ("service.yaml", "config/deployment/", None),
    ("cloudbuild.yaml", "config/deployment/", None),
    ("config/", "config/", None),
    
    # Documentation
    ("README.md", "docs/", "README_old.md"),
    ("*_GUIDE.md", "docs/deployment/", None),
    ("WARP.md", "docs/", None),
    
    # Scripts
    ("start_application.py", "scripts/setup/", None),
    ("deploy-gcloud.ps1", "scripts/deployment/", None),
    ("deploy-gcloud.sh", "scripts/deployment/", None),
    ("integration_example.py", "scripts/", None),
    ("lambda_handler.py", "scripts/deployment/", None),
    
    # Requirements and setup
    ("requirements.txt", "docs/", "requirements_old.txt"),
    ("requirements-cloud.txt", "config/deployment/", None),
    ("runtime.txt", "config/deployment/", None),
    ("Procfile", "config/deployment/", None),
    
    # Package files
    ("medical_report_simplifier_package/src/", "ml/data_processing/", None),
    ("medical_report_simplifier_package/", "scripts/legacy/", None),
    
    # Instance and logs
    ("instance/", "data/", None),
    ("logs/", "logs/", None),
    ("uploads/", "uploads/", None),
]


class ProjectMigrator:
    """Handles migration of existing project to new structure"""
    
    def __init__(self, source_root: Path, target_root: Path):
        self.source_root = source_root
        self.target_root = target_root
        self.migration_log = []
        self.errors = []
    
    def migrate(self) -> bool:
        """Execute the migration process"""
        print("🚀 Starting Medical Report Simplifier structure migration...")
        print(f"📁 Source: {self.source_root}")
        print(f"📁 Target: {self.target_root}")
        print("=" * 60)
        
        try:
            # Create backup
            if not self.create_backup():
                return False
            
            # Copy sample data files first
            self.copy_sample_data()
            
            # Migrate files according to mapping
            for source_pattern, dest_path, new_name in MIGRATION_MAP:
                self.migrate_files(source_pattern, dest_path, new_name)
            
            # Create missing __init__.py files
            self.create_init_files()
            
            # Update import paths
            self.update_import_paths()
            
            # Generate migration report
            self.generate_report()
            
            print("\\n✅ Migration completed successfully!")
            print(f"📄 Migration log saved to: {self.target_root / 'migration_log.json'}")
            print("\\n🔧 Next steps:")
            print("1. Review the migrated files")
            print("2. Update any remaining import paths")
            print("3. Test the application: python backend/app.py")
            print("4. Run ML interface: streamlit run ml/streamlit_app/main.py")
            
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            self.errors.append(f"Migration failed: {e}")
            return False
    
    def create_backup(self) -> bool:
        """Create backup of existing project"""
        backup_dir = self.source_root.parent / f"{self.source_root.name}_backup"
        
        if backup_dir.exists():
            print(f"⚠️  Backup already exists: {backup_dir}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        try:
            print(f"📦 Creating backup at: {backup_dir}")
            shutil.copytree(self.source_root, backup_dir, ignore=shutil.ignore_patterns(
                '__pycache__', '*.pyc', '*.pyo', '.git', 'node_modules', 'venv', '.env'
            ))
            self.migration_log.append(f"Created backup at: {backup_dir}")
            return True
        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
            self.errors.append(f"Backup failed: {e}")
            return False
    
    def copy_sample_data(self):
        """Copy sample data files to the new structure"""
        sample_files = [
            "medical_report_simplifier_package/data/sample_cbc_report.txt",
            "medical_report_simplifier_package/data/sample_lipid_report.txt"
        ]
        
        samples_dir = self.target_root / "data" / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_file in sample_files:
            source_file = self.source_root / sample_file
            if source_file.exists():
                dest_file = samples_dir / source_file.name
                try:
                    shutil.copy2(source_file, dest_file)
                    print(f"📄 Copied sample: {source_file.name}")
                    self.migration_log.append(f"Copied: {sample_file} -> {dest_file}")
                except Exception as e:
                    print(f"⚠️  Failed to copy {sample_file}: {e}")
                    self.errors.append(f"Copy failed: {sample_file} - {e}")
    
    def migrate_files(self, source_pattern: str, dest_path: str, new_name: str = None):
        """Migrate files matching the pattern"""
        dest_dir = self.target_root / dest_path
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        if source_pattern.endswith('/'):
            # Directory migration
            source_dir = self.source_root / source_pattern.rstrip('/')
            if source_dir.exists() and source_dir.is_dir():
                self.copy_directory(source_dir, dest_dir, new_name)
        else:
            # File migration
            if '*' in source_pattern:
                # Pattern matching
                self.copy_pattern_files(source_pattern, dest_dir)
            else:
                # Single file
                source_file = self.source_root / source_pattern
                if source_file.exists():
                    dest_name = new_name or source_file.name
                    dest_file = dest_dir / dest_name
                    self.copy_file(source_file, dest_file)
    
    def copy_file(self, source: Path, dest: Path):
        """Copy a single file"""
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            print(f"📄 {source.name} -> {dest.relative_to(self.target_root)}")
            self.migration_log.append(f"Copied: {source} -> {dest}")
        except Exception as e:
            print(f"⚠️  Failed to copy {source.name}: {e}")
            self.errors.append(f"Copy failed: {source} -> {dest} - {e}")
    
    def copy_directory(self, source_dir: Path, dest_parent: Path, new_name: str = None):
        """Copy a directory"""
        dest_name = new_name or source_dir.name
        dest_dir = dest_parent / dest_name
        
        try:
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            
            shutil.copytree(source_dir, dest_dir, ignore=shutil.ignore_patterns(
                '__pycache__', '*.pyc', '*.pyo', '.git', 'node_modules'
            ))
            print(f"📁 {source_dir.name}/ -> {dest_dir.relative_to(self.target_root)}/")
            self.migration_log.append(f"Copied directory: {source_dir} -> {dest_dir}")
        except Exception as e:
            print(f"⚠️  Failed to copy directory {source_dir.name}: {e}")
            self.errors.append(f"Directory copy failed: {source_dir} -> {dest_dir} - {e}")
    
    def copy_pattern_files(self, pattern: str, dest_dir: Path):
        """Copy files matching a pattern"""
        from glob import glob
        
        files = glob(str(self.source_root / pattern))
        for file_path in files:
            source_file = Path(file_path)
            dest_file = dest_dir / source_file.name
            self.copy_file(source_file, dest_file)
    
    def create_init_files(self):
        """Create __init__.py files in Python packages"""
        python_dirs = [
            "backend",
            "backend/api",
            "backend/api/routes",
            "backend/api/middleware",
            "backend/models",
            "backend/database",
            "backend/services",
            "backend/utils",
            "ml",
            "ml/models",
            "ml/services",
            "ml/data_processing",
            "ml/training",
            "ml/inference",
            "shared",
            "shared/utils",
            "shared/constants",
            "shared/types",
        ]
        
        for dir_path in python_dirs:
            init_file = self.target_root / dir_path / "__init__.py"
            if not init_file.exists():
                try:
                    init_file.parent.mkdir(parents=True, exist_ok=True)
                    init_file.write_text(f'"""\\n{dir_path.replace("/", ".")} module\\n"""\\n')
                    self.migration_log.append(f"Created __init__.py: {init_file}")
                except Exception as e:
                    self.errors.append(f"Failed to create {init_file}: {e}")
    
    def update_import_paths(self):
        """Update import paths in Python files"""
        print("🔧 Updating import paths...")
        
        # Common import replacements
        replacements = {
            'from medical_report_simplifier_package.src': 'from ml.data_processing',
            'import medical_report_simplifier_package.src': 'import ml.data_processing',
            'from src.': 'from ml.data_processing.',
            'import src.': 'import ml.data_processing.',
        }
        
        python_files = list(self.target_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                modified = False
                
                for old_import, new_import in replacements.items():
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        modified = True
                
                if modified:
                    py_file.write_text(content, encoding='utf-8')
                    print(f"🔧 Updated imports in: {py_file.name}")
                    self.migration_log.append(f"Updated imports: {py_file}")
                    
            except Exception as e:
                self.errors.append(f"Failed to update imports in {py_file}: {e}")
    
    def generate_report(self):
        """Generate migration report"""
        report = {
            "migration_timestamp": str(pd.Timestamp.now()),
            "source_root": str(self.source_root),
            "target_root": str(self.target_root),
            "files_migrated": len(self.migration_log),
            "errors_count": len(self.errors),
            "migration_log": self.migration_log,
            "errors": self.errors,
            "next_steps": [
                "Review migrated files",
                "Update remaining import paths",
                "Test backend: python backend/app.py",
                "Test ML interface: streamlit run ml/streamlit_app/main.py",
                "Run tests: pytest tests/",
                "Update documentation"
            ]
        }
        
        report_file = self.target_root / "migration_log.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save migration report: {e}")


def main():
    """Main migration execution"""
    if len(sys.argv) > 1:
        source_root = Path(sys.argv[1])
    else:
        # Auto-detect source root
        current = Path(__file__).parent.parent.parent
        source_root = current.parent  # Go up to the original project root
    
    if not source_root.exists():
        print(f"❌ Source directory not found: {source_root}")
        sys.exit(1)
    
    # Target is the new structure directory
    target_root = Path(__file__).parent.parent.parent
    
    print(f"🔍 Source detected: {source_root}")
    print(f"🎯 Target: {target_root}")
    
    # Confirm migration
    response = input("\\nProceed with migration? (y/n): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        sys.exit(0)
    
    # Execute migration
    migrator = ProjectMigrator(source_root, target_root)
    success = migrator.migrate()
    
    if success:
        print("\\n🎉 Migration completed successfully!")
        sys.exit(0)
    else:
        print("\\n💥 Migration failed. Check the error log.")
        sys.exit(1)


if __name__ == "__main__":
    # Add pandas import for timestamp
    try:
        import pandas as pd
    except ImportError:
        import datetime
        pd = type('MockPandas', (), {'Timestamp': type('MockTimestamp', (), {'now': lambda: datetime.datetime.now()})})()
    
    main()
