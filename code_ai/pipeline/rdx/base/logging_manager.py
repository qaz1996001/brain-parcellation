#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified logging configuration for all pipelines

Author: Architecture Redesign Team
Created: 2025-10-20
"""
import logging
import sys
import time
from pathlib import Path
from typing import Optional


class LoggingManager:
    """Unified logging configuration with daily log files

    Features:
        - Daily log file rotation (YYYYMMDD.log)
        - Console and file output
        - Configurable log levels
        - Consistent formatting across pipelines

    Example:
        >>> logger = LoggingManager.setup_logger(
        ...     log_dir=Path("./logs"),
        ...     pipeline_name="aneurysm_pipeline",
        ...     verbose=False
        ... )
        >>> logger.info("Pipeline started")
        >>> logger.error("Processing failed", exc_info=True)
    """

    # Standard log format
    DEFAULT_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    # Verbose format with more details
    VERBOSE_FORMAT = '%(asctime)s [%(name)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s'

    @staticmethod
    def setup_logger(
        log_dir: Path,
        pipeline_name: str,
        verbose: bool = False,
        console_output: bool = True,
        log_level: int = logging.INFO
    ) -> logging.Logger:
        """Setup logger with daily log files and optional console output

        Args:
            log_dir: Directory for log files
            pipeline_name: Pipeline identifier for logger name
            verbose: Enable verbose logging with additional details
            console_output: Enable console output (default: True)
            log_level: Logging level (default: logging.INFO)

        Returns:
            Configured logger instance

        Note:
            - Log files are named by date: YYYYMMDD.log
            - Multiple calls with same pipeline_name return same logger
            - Logger is configured with both file and console handlers
        """
        # Create log directory if it doesn't exist
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Get or create logger
        logger = logging.getLogger(pipeline_name)

        # Prevent duplicate handlers if logger already configured
        if logger.handlers:
            return logger

        logger.setLevel(log_level)

        # Choose format based on verbose flag
        log_format = LoggingManager.VERBOSE_FORMAT if verbose else LoggingManager.DEFAULT_FORMAT
        formatter = logging.Formatter(
            fmt=log_format,
            datefmt=LoggingManager.DEFAULT_DATE_FORMAT
        )

        # Create daily log file
        log_file = LoggingManager._get_daily_log_file(log_dir)

        # File handler
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler (optional)
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Prevent propagation to root logger
        logger.propagate = False

        logger.info(f"Logger initialized for {pipeline_name}")
        logger.info(f"Log file: {log_file}")

        return logger

    @staticmethod
    def _get_daily_log_file(log_dir: Path) -> Path:
        """Get log file path for current date

        Args:
            log_dir: Directory for log files

        Returns:
            Path to log file: {log_dir}/YYYYMMDD.log

        Note:
            Creates empty file if it doesn't exist
        """
        # Get current date
        localt = time.localtime(time.time())
        date_str = f"{localt.tm_year}{localt.tm_mon:02d}{localt.tm_mday:02d}"

        log_file = log_dir / f"{date_str}.log"

        # Create empty file if it doesn't exist
        if not log_file.exists():
            log_file.touch()

        return log_file

    @staticmethod
    def get_logger(pipeline_name: str) -> logging.Logger:
        """Get existing logger by name

        Args:
            pipeline_name: Pipeline identifier

        Returns:
            Existing logger or creates basic logger if not found

        Note:
            If logger doesn't exist, creates basic console-only logger
            Use setup_logger() for full configuration
        """
        logger = logging.getLogger(pipeline_name)

        # If logger has no handlers, add basic console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                fmt=LoggingManager.DEFAULT_FORMAT,
                datefmt=LoggingManager.DEFAULT_DATE_FORMAT
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)

        return logger

    @staticmethod
    def set_log_level(logger: logging.Logger, level: int) -> None:
        """Change log level for logger and all handlers

        Args:
            logger: Logger instance to modify
            level: New log level (e.g., logging.DEBUG, logging.WARNING)

        Example:
            >>> logger = LoggingManager.get_logger("pipeline")
            >>> LoggingManager.set_log_level(logger, logging.DEBUG)
        """
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

    @staticmethod
    def add_file_handler(
        logger: logging.Logger,
        log_file: Path,
        level: int = logging.INFO,
        verbose: bool = False
    ) -> None:
        """Add additional file handler to existing logger

        Args:
            logger: Logger instance to modify
            log_file: Path to log file
            level: Log level for this handler
            verbose: Use verbose format

        Example:
            >>> logger = LoggingManager.get_logger("pipeline")
            >>> LoggingManager.add_file_handler(
            ...     logger, Path("./debug.log"), logging.DEBUG, verbose=True
            ... )
        """
        # Create parent directory if needed
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Choose format
        log_format = LoggingManager.VERBOSE_FORMAT if verbose else LoggingManager.DEFAULT_FORMAT
        formatter = logging.Formatter(
            fmt=log_format,
            datefmt=LoggingManager.DEFAULT_DATE_FORMAT
        )

        # Create and add handler
        handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    @staticmethod
    def cleanup_old_logs(log_dir: Path, keep_days: int = 30) -> int:
        """Remove log files older than specified days

        Args:
            log_dir: Directory containing log files
            keep_days: Number of days to keep logs (default: 30)

        Returns:
            Number of log files deleted

        Example:
            >>> deleted = LoggingManager.cleanup_old_logs(Path("./logs"), keep_days=7)
            >>> print(f"Deleted {deleted} old log files")
        """
        import time
        from datetime import datetime, timedelta

        log_dir = Path(log_dir)
        if not log_dir.exists():
            return 0

        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        deleted_count = 0

        for log_file in log_dir.glob("*.log"):
            if log_file.is_file():
                file_mtime = log_file.stat().st_mtime
                if file_mtime < cutoff_time:
                    try:
                        log_file.unlink()
                        deleted_count += 1
                    except Exception:
                        # Skip if unable to delete
                        pass

        return deleted_count


class PipelineLogger:
    """Context manager for pipeline logging with automatic cleanup

    Example:
        >>> with PipelineLogger("aneurysm_pipeline", Path("./logs")) as logger:
        ...     logger.info("Processing started")
        ...     # Pipeline operations
        ...     logger.info("Processing completed")
    """

    def __init__(
        self,
        pipeline_name: str,
        log_dir: Path,
        verbose: bool = False,
        cleanup_days: Optional[int] = None
    ):
        """Initialize PipelineLogger context manager

        Args:
            pipeline_name: Pipeline identifier
            log_dir: Directory for log files
            verbose: Enable verbose logging
            cleanup_days: Auto-cleanup logs older than N days (None to disable)
        """
        self.pipeline_name = pipeline_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.cleanup_days = cleanup_days
        self.logger: Optional[logging.Logger] = None

    def __enter__(self) -> logging.Logger:
        """Enter context - setup logger"""
        # Cleanup old logs if requested
        if self.cleanup_days is not None:
            deleted = LoggingManager.cleanup_old_logs(self.log_dir, self.cleanup_days)
            if deleted > 0:
                # Use basic logger for cleanup message
                temp_logger = logging.getLogger(self.pipeline_name)
                temp_logger.info(f"Cleaned up {deleted} old log files")

        # Setup logger
        self.logger = LoggingManager.setup_logger(
            log_dir=self.log_dir,
            pipeline_name=self.pipeline_name,
            verbose=self.verbose
        )

        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - log any exceptions"""
        if self.logger and exc_type is not None:
            self.logger.error(
                f"Pipeline failed with {exc_type.__name__}: {exc_val}",
                exc_info=True
            )
        return False  # Don't suppress exceptions
