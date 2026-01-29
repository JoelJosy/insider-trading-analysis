"""
Logger Module for Insider Trading Analysis.

Provides centralized logging configuration with file and console output.
Uses rotating file handler to manage log file sizes.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up and return a configured logger instance.

    Args:
        name: Name of the logger (typically __name__).
        log_file: Path to the log file. If None, only console logging is enabled.
        level: Logging level (default: INFO).
        log_format: Custom log format string. Uses default if None.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of backup log files to keep.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger(__name__, "logs/app.log")
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Name of the logger (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class that provides logging capability to any class.

    Example:
        >>> class MyClass(LoggerMixin):
        ...     def do_something(self):
        ...         self.logger.info("Doing something")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Return a logger named after the class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
