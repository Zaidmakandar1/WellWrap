"""
Medical Report Simplifier - Configuration Management
Centralized configuration loading and management
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Default configuration values
DEFAULT_CONFIG = {
    # Application
    'APP_NAME': 'Medical Report Simplifier',
    'APP_VERSION': '1.0.0',
    'ENVIRONMENT': 'development',
    'DEBUG': True,
    'SECRET_KEY': 'dev-secret-key-change-in-production',
    
    # Flask
    'FLASK_HOST': '0.0.0.0',
    'FLASK_PORT': 5000,
    'FLASK_ENV': 'development',
    'FLASK_DEBUG': True,
    
    # Database
    'DATABASE_URL': 'sqlite:///healthcare.db',
    
    # Streamlit
    'STREAMLIT_SERVER_PORT': 8501,
    'STREAMLIT_SERVER_ADDRESS': 'localhost',
    
    # File Upload
    'UPLOAD_FOLDER': 'uploads/',
    'MAX_CONTENT_LENGTH': 16777216,  # 16MB
    'ALLOWED_EXTENSIONS': 'pdf,jpg,jpeg,png,txt,docx',
    
    # Security
    'SESSION_TIMEOUT': 3600,
    'BCRYPT_LOG_ROUNDS': 12,
    'JWT_ACCESS_TOKEN_EXPIRES': 3600,
    'JWT_REFRESH_TOKEN_EXPIRES': 604800,
    
    # ML Settings
    'ML_MODEL_PATH': 'ml/models/trained/',
    'ENABLE_GPU': False,
    'MODEL_CACHE_SIZE': 100,
    'BATCH_SIZE': 32,
    'MAX_SEQUENCE_LENGTH': 512,
    
    # Medical Analysis
    'ENABLE_DISEASE_DETECTION': True,
    'ENABLE_HEALTH_SCORING': True,
    'ENABLE_RISK_ASSESSMENT': True,
    'CONFIDENCE_THRESHOLD': 0.8,
    'MAX_REPORT_SIZE_MB': 10,
    
    # Logging
    'LOG_LEVEL': 'INFO',
    'LOG_FILE': 'logs/app.log',
    'LOG_MAX_SIZE': 10485760,  # 10MB
    'LOG_BACKUP_COUNT': 5,
    'ENABLE_ACCESS_LOGGING': True,
    
    # Cache
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300,
}


class ConfigManager:
    """Manages application configuration from multiple sources"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # Try to find project root
            current = Path(__file__).parent
            while current.parent != current:
                if (current / 'requirements.txt').exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path.cwd()
        
        self.project_root = project_root
        self.config = DEFAULT_CONFIG.copy()
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load configuration from all sources in order of precedence"""
        # 1. Load from config files
        self._load_yaml_config()
        
        # 2. Load from environment file
        self._load_env_file()
        
        # 3. Override with environment variables
        self._load_env_variables()
    
    def _load_yaml_config(self):
        """Load configuration from YAML files"""
        config_dir = self.project_root / 'config' / 'environments'
        
        # Load base config
        base_config_file = config_dir / 'base.yaml'
        if base_config_file.exists():
            with open(base_config_file, 'r') as f:
                base_config = yaml.safe_load(f)
                if base_config:
                    self.config.update(base_config)
        
        # Load environment-specific config
        env = os.getenv('ENVIRONMENT', 'development')
        env_config_file = config_dir / f'{env}.yaml'
        if env_config_file.exists():
            with open(env_config_file, 'r') as f:
                env_config = yaml.safe_load(f)
                if env_config:
                    self.config.update(env_config)
    
    def _load_env_file(self):
        """Load configuration from .env file"""
        env_files = [
            self.project_root / '.env.local',
            self.project_root / '.env',
        ]
        
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(env_file)
                break
    
    def _load_env_variables(self):
        """Load configuration from environment variables"""
        for key in self.config.keys():
            env_value = os.getenv(key)
            if env_value is not None:
                # Convert string values to appropriate types
                self.config[key] = self._convert_value(env_value, self.config[key])
    
    def _convert_value(self, value: str, default_value: Any) -> Any:
        """Convert string environment variable to appropriate type"""
        if isinstance(default_value, bool):
            return value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(default_value, int):
            try:
                return int(value)
            except ValueError:
                return default_value
        elif isinstance(default_value, float):
            try:
                return float(value)
            except ValueError:
                return default_value
        else:
            return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            'DATABASE_URL': self.get('DATABASE_URL'),
            'SQLALCHEMY_DATABASE_URI': self.get('DATABASE_URL'),
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'SQLALCHEMY_ECHO': self.get('DEBUG', False),
        }
    
    def get_flask_config(self) -> Dict[str, Any]:
        """Get Flask configuration"""
        return {
            'SECRET_KEY': self.get('SECRET_KEY'),
            'DEBUG': self.get('DEBUG'),
            'TESTING': self.get('ENVIRONMENT') == 'testing',
            'MAX_CONTENT_LENGTH': self.get('MAX_CONTENT_LENGTH'),
            'UPLOAD_FOLDER': self.get('UPLOAD_FOLDER'),
            'SESSION_PERMANENT': False,
            'PERMANENT_SESSION_LIFETIME': self.get('SESSION_TIMEOUT'),
        }
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get ML configuration"""
        return {
            'MODEL_PATH': self.project_root / self.get('ML_MODEL_PATH'),
            'ENABLE_GPU': self.get('ENABLE_GPU'),
            'MODEL_CACHE_SIZE': self.get('MODEL_CACHE_SIZE'),
            'BATCH_SIZE': self.get('BATCH_SIZE'),
            'MAX_SEQUENCE_LENGTH': self.get('MAX_SEQUENCE_LENGTH'),
            'CONFIDENCE_THRESHOLD': self.get('CONFIDENCE_THRESHOLD'),
        }
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit configuration"""
        return {
            'server.port': self.get('STREAMLIT_SERVER_PORT'),
            'server.address': self.get('STREAMLIT_SERVER_ADDRESS'),
            'server.headless': True,
            'browser.gatherUsageStats': False,
            'theme.primaryColor': self.get('STREAMLIT_THEME_PRIMARY_COLOR', '#FF6B35'),
            'theme.backgroundColor': self.get('STREAMLIT_THEME_BACKGROUND_COLOR', '#FFFFFF'),
            'theme.secondaryBackgroundColor': self.get('STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR', '#F0F2F6'),
        }
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.get('ENVIRONMENT') == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.get('ENVIRONMENT') == 'production'
    
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.get('ENVIRONMENT') == 'testing'
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        errors = []
        
        # Validate required fields
        required_fields = ['SECRET_KEY', 'DATABASE_URL']
        for field in required_fields:
            if not self.get(field):
                errors.append(f"Missing required configuration: {field}")
        
        # Validate SECRET_KEY in production
        if self.is_production() and self.get('SECRET_KEY') == DEFAULT_CONFIG['SECRET_KEY']:
            errors.append("SECRET_KEY must be changed in production")
        
        # Validate ports
        for port_key in ['FLASK_PORT', 'STREAMLIT_SERVER_PORT']:
            port = self.get(port_key)
            if not isinstance(port, int) or port < 1 or port > 65535:
                errors.append(f"Invalid port configuration: {port_key}={port}")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True


# Global configuration instance
_config_manager = None


def load_config(project_root: Optional[Path] = None) -> ConfigManager:
    """Load and return the global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(project_root)
    return _config_manager


def get_config() -> ConfigManager:
    """Get the global configuration manager"""
    if _config_manager is None:
        return load_config()
    return _config_manager


# Convenience functions
def get(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return get_config().get(key, default)


def get_database_url() -> str:
    """Get database URL"""
    return get('DATABASE_URL')


def get_secret_key() -> str:
    """Get secret key"""
    return get('SECRET_KEY')


def is_development() -> bool:
    """Check if running in development mode"""
    return get_config().is_development()


def is_production() -> bool:
    """Check if running in production mode"""
    return get_config().is_production()


def is_testing() -> bool:
    """Check if running in testing mode"""
    return get_config().is_testing()
