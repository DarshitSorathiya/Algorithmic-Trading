"""
Logging configuration for trading strategy system
"""
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime


class TradingLogger:
    """Configure and manage logging"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = None
    
    def setup_logger(self, name: str = 'trading_strategy',
                    level: str = 'INFO',
                    log_file: str = 'trading_strategy.log',
                    console_logging: bool = True,
                    file_logging: bool = True,
                    max_bytes: int = 10485760,  # 10MB
                    backup_count: int = 5) -> logging.Logger:
        """
        Setup logging configuration
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Log file path
            console_logging: Enable console output
            file_logging: Enable file output
            max_bytes: Max file size before rotation
            backup_count: Number of backup files to keep
            
        Returns:
            Configured logger
        """
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # Console handler
        if console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if file_logging:
            # Create logs directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
        
        return self.logger
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        if self.logger is None:
            return self.setup_logger()
        return self.logger


# Convenience functions
def get_logger(name: str = 'trading_strategy') -> logging.Logger:
    """
    Get or create logger
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    trading_logger = TradingLogger()
    logger = trading_logger.get_logger()
    
    # If getting a child logger
    if name != 'trading_strategy':
        return logger.getChild(name)
    
    return logger


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls
    
    Args:
        logger: Logger instance
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_performance(logger: logging.Logger):
    """
    Decorator to log function performance
    
    Args:
        logger: Logger instance
    """
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} took {elapsed:.2f} seconds")
            return result
        return wrapper
    return decorator
