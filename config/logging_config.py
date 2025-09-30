import sys
import logging
from pathlib import Path
from loguru import logger
from typing import Optional

class LoggerSetup:
    """Industrial-grade logging configuration"""

    def __init__(self,
                 log_dir: Path = Path("logs"),
                 log_level: str = "INFO",
                 log_file: str = "crm_analysis.log"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_level = log_level
        self.log_file = log_file

        # Remove default logger
        logger.remove()

        # Configure logger
        self._setup_logger()

    def _setup_logger(self):
        """Setup Loguru logger with custom configuration"""

        # Console handler with color
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=self.log_level,
            filter=lambda record: record["level"].no < 40  # INFO and below
        )

        # Console handler for errors
        logger.add(
            sys.stderr,
            format="<red>{time:YYYY-MM-DD HH:mm:ss}</red> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level="ERROR"
        )

        # File handler for all logs
        logger.add(
            self.log_dir / self.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            enqueue=True  # Thread-safe
        )

        # File handler for errors only
        logger.add(
            self.log_dir / "errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation="50 MB",
            retention="60 days",
            backtrace=True,
            diagnose=True
        )

        # Performance log
        logger.add(
            self.log_dir / "performance.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | PERF | {message}",
            filter=lambda record: "performance" in record["extra"],
            rotation="50 MB"
        )

    @staticmethod
    def get_logger(name: str = None):
        """Get logger instance"""
        if name:
            return logger.bind(name=name)
        return logger


# Initialize logger
log_setup = LoggerSetup()
logger = log_setup.get_logger()