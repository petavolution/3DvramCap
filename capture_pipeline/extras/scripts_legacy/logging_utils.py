#!/usr/bin/env python3
"""
Logging Utilities for Capture Pipeline
=======================================
Provides structured logging with file rotation and JSON output support.

Usage:
    from logging_utils import setup_logging, get_logger

    setup_logging(log_dir="logs", level="INFO")
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"capture": "scene_01.rdc"})

Features:
    - Structured format with timestamps
    - Console and file handlers
    - Rotating file handler (10MB max, 5 backups)
    - JSON logging for machine parsing
    - Context injection (process ID, script name)

Author: Capture Pipeline
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


# Default log format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# JSON format for structured logging
JSON_FORMAT_KEYS = ["timestamp", "level", "logger", "message", "pid", "extra"]


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter for machine-readable output.

    Output format:
        {"timestamp": "...", "level": "INFO", "logger": "...", "message": "...", "extra": {...}}
    """

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pid": os.getpid(),
        }

        # Add extra fields if present
        extra = {}
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__ and key not in ["message", "asctime"]:
                # Skip internal attributes
                if not key.startswith("_"):
                    extra[key] = value

        if extra:
            log_data["extra"] = extra

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ColorFormatter(logging.Formatter):
    """
    Colored console formatter for better readability.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record):
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class PipelineLogger:
    """
    Enhanced logger with context support for pipeline operations.

    Usage:
        logger = PipelineLogger("extractor")
        with logger.context(capture="scene_01.rdc"):
            logger.info("Starting extraction")
    """

    def __init__(self, name, base_logger=None):
        self.name = name
        self._logger = base_logger or logging.getLogger(name)
        self._context = {}

    def _log(self, level, msg, *args, **kwargs):
        extra = kwargs.pop("extra", {})
        extra.update(self._context)
        kwargs["extra"] = extra
        getattr(self._logger, level)(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log("debug", msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log("info", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log("warning", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log("error", msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._log("critical", msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        kwargs["exc_info"] = True
        self._log("error", msg, *args, **kwargs)

    def set_context(self, **kwargs):
        """Set persistent context fields."""
        self._context.update(kwargs)

    def clear_context(self):
        """Clear all context fields."""
        self._context = {}

    class _ContextManager:
        def __init__(self, logger, **kwargs):
            self.logger = logger
            self.kwargs = kwargs
            self.old_context = {}

        def __enter__(self):
            self.old_context = self.logger._context.copy()
            self.logger._context.update(self.kwargs)
            return self.logger

        def __exit__(self, *args):
            self.logger._context = self.old_context

    def context(self, **kwargs):
        """Context manager for temporary context fields."""
        return self._ContextManager(self, **kwargs)


def setup_logging(
    log_dir="logs",
    level="INFO",
    console=True,
    file=True,
    json_file=True,
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5,
    use_colors=True
):
    """
    Configure logging for the capture pipeline.

    Args:
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Enable console output
        file: Enable rotating file output
        json_file: Enable JSON file output
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep
        use_colors: Enable colored console output

    Returns:
        Root logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColorFormatter(
            fmt=DEFAULT_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
            use_colors=use_colors
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Rotating file handler (human-readable)
    if file:
        timestamp = datetime.now().strftime("%Y%m%d")
        file_path = log_path / f"pipeline_{timestamp}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt=DEFAULT_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # JSON file handler (machine-readable)
    if json_file:
        timestamp = datetime.now().strftime("%Y%m%d")
        json_path = log_path / f"pipeline_{timestamp}.json.log"
        json_handler = logging.handlers.RotatingFileHandler(
            json_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(json_handler)

    return root_logger


def get_logger(name):
    """
    Get a PipelineLogger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        PipelineLogger instance
    """
    return PipelineLogger(name)


class ProgressLogger:
    """
    Progress tracking logger for batch operations.

    Usage:
        progress = ProgressLogger("meshes", total=100)
        for mesh in meshes:
            process(mesh)
            progress.update()
        progress.finish()
    """

    def __init__(self, name, total, logger=None, log_interval=10):
        self.name = name
        self.total = total
        self.current = 0
        self.logger = logger or get_logger("progress")
        self.log_interval = log_interval
        self.start_time = datetime.now()

        self.logger.info(
            f"Starting {name}: {total} items",
            extra={"operation": name, "total": total}
        )

    def update(self, count=1, message=None):
        """Update progress counter."""
        self.current += count

        if self.current % self.log_interval == 0 or self.current == self.total:
            percent = (self.current / self.total) * 100 if self.total > 0 else 100
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0

            msg = message or f"{self.name}: {self.current}/{self.total} ({percent:.1f}%)"
            self.logger.info(
                msg,
                extra={
                    "operation": self.name,
                    "current": self.current,
                    "total": self.total,
                    "percent": percent,
                    "rate": rate
                }
            )

    def finish(self, message=None):
        """Mark operation as complete."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.total / elapsed if elapsed > 0 else 0

        msg = message or f"Completed {self.name}: {self.total} items in {elapsed:.2f}s"
        self.logger.info(
            msg,
            extra={
                "operation": self.name,
                "total": self.total,
                "elapsed_seconds": elapsed,
                "rate": rate,
                "status": "complete"
            }
        )


class StepLogger:
    """
    Pipeline step logger with timing and status tracking.

    Usage:
        step = StepLogger("extraction", logger)
        step.start()
        # ... do work ...
        step.success("Extracted 150 meshes")
        # or
        step.failure("Failed to load capture file")
    """

    def __init__(self, step_name, logger=None):
        self.step_name = step_name
        self.logger = logger or get_logger("pipeline")
        self.start_time = None
        self.status = "pending"

    def start(self, message=None):
        """Mark step as started."""
        self.start_time = datetime.now()
        self.status = "running"

        msg = message or f"Step '{self.step_name}' started"
        self.logger.info(
            msg,
            extra={"step": self.step_name, "status": "started"}
        )
        return self

    def success(self, message=None, **extra):
        """Mark step as successful."""
        elapsed = self._elapsed()
        self.status = "success"

        msg = message or f"Step '{self.step_name}' completed"
        extra.update({
            "step": self.step_name,
            "status": "success",
            "elapsed_seconds": elapsed
        })
        self.logger.info(msg, extra=extra)

    def failure(self, message=None, exception=None, **extra):
        """Mark step as failed."""
        elapsed = self._elapsed()
        self.status = "failed"

        msg = message or f"Step '{self.step_name}' failed"
        extra.update({
            "step": self.step_name,
            "status": "failed",
            "elapsed_seconds": elapsed
        })

        if exception:
            self.logger.exception(msg, extra=extra)
        else:
            self.logger.error(msg, extra=extra)

    def warning(self, message, **extra):
        """Log a warning within the step."""
        extra.update({"step": self.step_name})
        self.logger.warning(message, extra=extra)

    def _elapsed(self):
        """Get elapsed time since start."""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.failure(str(exc_val), exception=exc_val)
        else:
            self.success()
        return False  # Don't suppress exceptions


# Convenience function for quick setup
def quick_setup(verbose=False):
    """
    Quick logging setup with sensible defaults.

    Args:
        verbose: If True, use DEBUG level; otherwise INFO
    """
    level = "DEBUG" if verbose else "INFO"
    return setup_logging(level=level)


if __name__ == "__main__":
    # Demo usage
    setup_logging(level="DEBUG")

    logger = get_logger("demo")
    logger.info("Pipeline logging system initialized")

    # Context example
    with logger.context(capture="scene_01.rdc", step="extraction"):
        logger.info("Starting extraction process")
        logger.debug("Loading capture file")
        logger.warning("Large file detected", extra={"size_mb": 512})

    # Progress example
    progress = ProgressLogger("meshes", total=50, log_interval=10)
    for i in range(50):
        progress.update()
    progress.finish()

    # Step example
    with StepLogger("validation") as step:
        step.warning("Minor issue detected")

    print("\n[OK] Logging demo complete!")
