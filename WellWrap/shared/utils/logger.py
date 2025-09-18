"""
Medical Report Simplifier - Logging Configuration
Centralized logging setup and utilities
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Import structlog if available for structured logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\\033[36m',      # Cyan
        'INFO': '\\033[32m',       # Green
        'WARNING': '\\033[33m',    # Yellow
        'ERROR': '\\033[31m',      # Red
        'CRITICAL': '\\033[35m',   # Magenta
        'RESET': '\\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        # Add emoji for different log levels
        emoji_map = {
            'DEBUG': '🐛',
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '💥'
        }
        
        original_levelname = record.levelname.replace(
            self.COLORS.get(record.levelname.split('\\033')[0], ''), ''
        ).replace(self.COLORS['RESET'], '')
        
        emoji = emoji_map.get(original_levelname, '')
        record.emoji = emoji
        
        return super().format(record)


class MedicalReportLogger:
    """Enhanced logger for Medical Report Simplifier application"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration"""
        # Try to find project root for log file path
        current = Path(__file__).parent
        while current.parent != current:
            if (current / 'requirements.txt').exists():
                project_root = current
                break
            current = current.parent
        else:
            project_root = Path.cwd()
        
        return {
            'level': os.getenv('LOG_LEVEL', 'INFO').upper(),
            'log_file': project_root / os.getenv('LOG_FILE', 'logs/app.log'),
            'max_file_size': int(os.getenv('LOG_MAX_SIZE', 10 * 1024 * 1024)),  # 10MB
            'backup_count': int(os.getenv('LOG_BACKUP_COUNT', 5)),
            'enable_console': True,
            'enable_file': True,
            'console_format': '%(emoji)s %(levelname)s [%(name)s] %(message)s',
            'file_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
            'date_format': '%Y-%m-%d %H:%M:%S',
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup and configure the logger"""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config['level']))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.config['enable_console']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(
                fmt=self.config['console_format'],
                datefmt=self.config['date_format']
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, self.config['level']))
            logger.addHandler(console_handler)
        
        # File handler
        if self.config['enable_file']:
            # Ensure log directory exists
            log_file = Path(self.config['log_file'])
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=self.config['max_file_size'],
                backupCount=self.config['backup_count'],
                encoding='utf-8'
            )
            
            file_formatter = logging.Formatter(
                fmt=self.config['file_format'],
                datefmt=self.config['date_format']
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(getattr(logging, self.config['level']))
            logger.addHandler(file_handler)
        
        # Prevent duplicate logs from parent loggers
        logger.propagate = False
        
        return logger
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, extra=kwargs)
    
    def log_medical_event(self, event_type: str, user_id: Optional[str] = None, 
                         report_id: Optional[str] = None, details: Optional[Dict] = None):
        """Log medical-specific events with structured data"""
        event_data = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'report_id': report_id,
            'details': details or {}
        }
        
        self.info(f"Medical Event: {event_type}", **event_data)
    
    def log_security_event(self, event_type: str, user_id: Optional[str] = None,
                          ip_address: Optional[str] = None, details: Optional[Dict] = None):
        """Log security-related events"""
        event_data = {
            'security_event': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'ip_address': ip_address,
            'details': details or {}
        }
        
        self.warning(f"Security Event: {event_type}", **event_data)
    
    def log_performance(self, operation: str, duration: float, 
                       details: Optional[Dict] = None):
        """Log performance metrics"""
        perf_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details or {}
        }
        
        level = 'WARNING' if duration > 5.0 else 'INFO'
        getattr(self, level.lower())(f"Performance: {operation} took {duration:.2f}s", **perf_data)


class StructuredLogger:
    """Structured logger using structlog if available"""
    
    def __init__(self, name: str):
        if not STRUCTLOG_AVAILABLE:
            raise ImportError("structlog is not available. Use MedicalReportLogger instead.")
        
        self.name = name
        self._setup_structlog()
        self.logger = structlog.get_logger(name)
    
    def _setup_structlog(self):
        """Setup structlog configuration"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def bind(self, **kwargs):
        """Bind context to logger"""
        return self.logger.bind(**kwargs)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, **kwargs)


# Global logger instances
_loggers: Dict[str, MedicalReportLogger] = {}


def setup_logger(name: str, config: Optional[Dict[str, Any]] = None) -> MedicalReportLogger:
    """Setup and return a logger instance"""
    if name not in _loggers:
        _loggers[name] = MedicalReportLogger(name, config)
    return _loggers[name]


def get_logger(name: str) -> MedicalReportLogger:
    """Get an existing logger or create a new one"""
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


def setup_structured_logger(name: str) -> StructuredLogger:
    """Setup structured logger if structlog is available"""
    if not STRUCTLOG_AVAILABLE:
        raise ImportError(
            "structlog is not available. Install with: pip install structlog"
        )
    return StructuredLogger(name)


# Context managers for logging
class LogExecutionTime:
    """Context manager to log execution time of operations"""
    
    def __init__(self, logger: MedicalReportLogger, operation: str, 
                 log_level: str = 'INFO', **context):
        self.logger = logger
        self.operation = operation
        self.log_level = log_level.upper()
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation}", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is not None:
            self.logger.error(
                f"Failed {self.operation} after {duration:.2f}s: {exc_val}",
                duration=duration,
                error_type=exc_type.__name__,
                **self.context
            )
        else:
            log_method = getattr(self.logger, self.log_level.lower())
            log_method(
                f"Completed {self.operation} in {duration:.2f}s",
                duration=duration,
                **self.context
            )


# Convenience function to create execution time context
def log_execution_time(logger: MedicalReportLogger, operation: str, **context):
    """Create LogExecutionTime context manager"""
    return LogExecutionTime(logger, operation, **context)
