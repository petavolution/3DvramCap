#!/usr/bin/env python3
"""
Structured Logging System
=========================
JSON-formatted logging for monitoring, debugging, and audit trails.

Features:
    - JSON structured output for log aggregation
    - Context-aware logging (task ID, session ID)
    - Progress tracking for long operations
    - Performance timing
    - Log rotation and archival
    - Integration with monitoring systems

Usage:
    from core_logging import get_logger, TaskLogger

    # Basic logging
    logger = get_logger("extraction")
    logger.info("Starting extraction", capture="scene.rdc")

    # Task-scoped logging
    with TaskLogger("process_capture", capture_id="123") as task:
        task.info("Processing started")
        task.progress(50, "Halfway done")
        task.info("Processing complete")

Author: Capture Pipeline
"""

import json
import logging
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import threading


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

    def __init__(self, include_timestamp: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }

        if self.include_timestamp:
            log_data['timestamp'] = datetime.utcnow().isoformat() + 'Z'

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno',
                          'pathname', 'filename', 'module', 'lineno',
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'message', 'exc_info', 'exc_text', 'stack_info'):
                log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable format with color support."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime('%H:%M:%S')
        level = record.levelname

        if self.use_color:
            color = self.COLORS.get(level, '')
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        # Build message
        msg = f"[{timestamp}] {level_str} {record.name}: {record.getMessage()}"

        # Add extra context
        extras = []
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno',
                          'pathname', 'filename', 'module', 'lineno',
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'message', 'exc_info', 'exc_text', 'stack_info'):
                extras.append(f"{key}={value}")

        if extras:
            msg += f" ({', '.join(extras)})"

        # Add exception
        if record.exc_info:
            msg += '\n' + ''.join(traceback.format_exception(*record.exc_info))

        return msg


class ProgressHandler(logging.Handler):
    """Handler for progress bar updates."""

    def __init__(self):
        super().__init__()
        self.progress_bars: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        if hasattr(record, 'progress'):
            task_id = getattr(record, 'task_id', 'default')
            progress = record.progress
            total = getattr(record, 'total', 100)
            message = record.getMessage()

            with self._lock:
                self._update_progress(task_id, progress, total, message)

    def _update_progress(self, task_id: str, progress: int, total: int, message: str):
        """Update progress display."""
        width = 40
        filled = int(width * progress / total)
        bar = '=' * filled + '-' * (width - filled)
        percent = progress * 100 / total

        # Clear line and print progress
        sys.stdout.write(f'\r[{bar}] {percent:5.1f}% {message}')
        sys.stdout.flush()

        if progress >= total:
            sys.stdout.write('\n')


@dataclass
class LogContext:
    """Context for structured logging."""
    session_id: str = ""
    task_id: str = ""
    capture_id: str = ""
    mesh_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.session_id:
            result['session_id'] = self.session_id
        if self.task_id:
            result['task_id'] = self.task_id
        if self.capture_id:
            result['capture_id'] = self.capture_id
        if self.mesh_id:
            result['mesh_id'] = self.mesh_id
        result.update(self.extra)
        return result


class ContextLogger:
    """Logger with automatic context injection."""

    def __init__(self, logger: logging.Logger, context: LogContext = None):
        self._logger = logger
        self._context = context or LogContext()

    def _log(self, level: int, msg: str, **kwargs):
        extra = self._context.to_dict()
        extra.update(kwargs)
        self._logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, exc_info: bool = False, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)
        if exc_info:
            self._logger.exception(msg, extra=self._context.to_dict())

    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)

    def progress(self, current: int, message: str = "", total: int = 100):
        """Log progress update."""
        extra = self._context.to_dict()
        extra['progress'] = current
        extra['total'] = total
        self._logger.info(message, extra=extra)

    def with_context(self, **kwargs) -> 'ContextLogger':
        """Create new logger with additional context."""
        new_context = LogContext(
            session_id=self._context.session_id,
            task_id=self._context.task_id,
            capture_id=kwargs.get('capture_id', self._context.capture_id),
            mesh_id=kwargs.get('mesh_id', self._context.mesh_id),
            extra={**self._context.extra, **kwargs}
        )
        return ContextLogger(self._logger, new_context)


class TaskLogger(ContextLogger):
    """Context manager for task-scoped logging with timing."""

    def __init__(self, task_name: str, logger: logging.Logger = None, **context):
        if logger is None:
            logger = get_logger(task_name)

        ctx = LogContext(
            task_id=str(uuid.uuid4())[:8],
            **{k: v for k, v in context.items() if k in LogContext.__dataclass_fields__}
        )
        ctx.extra = {k: v for k, v in context.items() if k not in LogContext.__dataclass_fields__}

        super().__init__(logger, ctx)
        self.task_name = task_name
        self.start_time: float = 0
        self.end_time: float = 0
        self.success: bool = False
        self.error: Optional[str] = None

    def __enter__(self) -> 'TaskLogger':
        self.start_time = time.time()
        self.info(f"Task started: {self.task_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type is None:
            self.success = True
            self.info(f"Task completed: {self.task_name}",
                     duration_seconds=round(duration, 2))
        else:
            self.success = False
            self.error = str(exc_val)
            self.error(f"Task failed: {self.task_name}",
                      duration_seconds=round(duration, 2),
                      error=str(exc_val))

        return False  # Don't suppress exceptions

    @property
    def duration(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class LogManager:
    """Central log management."""

    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _handlers: List[logging.Handler] = []
    _session_id: str = ""
    _log_dir: Path = Path("logs")
    _format: str = "json"
    _level: int = logging.INFO

    @classmethod
    def initialize(cls,
                  log_dir: str = "logs",
                  level: str = "INFO",
                  format: str = "json",
                  console: bool = True,
                  file: bool = True):
        """Initialize the logging system."""
        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        cls._format = format
        cls._level = getattr(logging, level.upper(), logging.INFO)
        cls._session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

        # Clear existing handlers
        for handler in cls._handlers:
            handler.close()
        cls._handlers = []

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            if format == "json":
                console_handler.setFormatter(JsonFormatter())
            else:
                console_handler.setFormatter(TextFormatter())
            console_handler.setLevel(cls._level)
            cls._handlers.append(console_handler)

        # File handler
        if file:
            log_file = cls._log_dir / f"pipeline_{cls._session_id}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JsonFormatter())
            file_handler.setLevel(cls._level)
            cls._handlers.append(file_handler)

        # Progress handler (text mode only)
        if format == "text" and console:
            cls._handlers.append(ProgressHandler())

        # Configure root logger
        root = logging.getLogger()
        root.setLevel(cls._level)
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in cls._handlers:
            root.addHandler(handler)

    @classmethod
    def get_logger(cls, name: str) -> ContextLogger:
        """Get or create a logger."""
        if name not in cls._loggers:
            logger = logging.getLogger(f"pipeline.{name}")
            logger.setLevel(cls._level)
            cls._loggers[name] = logger

        context = LogContext(session_id=cls._session_id)
        return ContextLogger(cls._loggers[name], context)

    @classmethod
    def get_session_id(cls) -> str:
        return cls._session_id

    @classmethod
    def get_log_file(cls) -> Path:
        return cls._log_dir / f"pipeline_{cls._session_id}.log"


def get_logger(name: str) -> ContextLogger:
    """Get a logger by name."""
    return LogManager.get_logger(name)


def init_logging(log_dir: str = "logs",
                level: str = "INFO",
                format: str = "text",
                console: bool = True,
                file: bool = True):
    """Initialize the logging system."""
    LogManager.initialize(log_dir, level, format, console, file)


def log_function(logger_name: str = None):
    """Decorator to log function entry/exit."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = logger_name or func.__module__
            logger = get_logger(name)

            logger.debug(f"Entering {func.__name__}",
                        args=str(args)[:100],
                        kwargs=str(kwargs)[:100])

            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                logger.debug(f"Exiting {func.__name__}",
                            duration_ms=round(duration * 1000, 1))
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"Exception in {func.__name__}: {e}",
                            duration_ms=round(duration * 1000, 1))
                raise

        return wrapper
    return decorator


@contextmanager
def log_operation(name: str, logger: ContextLogger = None, **context):
    """Context manager for logging an operation."""
    if logger is None:
        logger = get_logger("operation")

    task = TaskLogger(name, logger._logger, **context)
    with task:
        yield task


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def record(self, name: str, value: float):
        """Record a metric value."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = self.metrics.get(name, [])
        if not values:
            return {}

        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'sum': sum(values)
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics}

    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self.metrics.clear()


# Global metrics collector
_metrics = MetricsCollector()


def record_metric(name: str, value: float):
    """Record a metric value."""
    _metrics.record(name, value)


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics


if __name__ == "__main__":
    # Demo
    init_logging(format="text", level="DEBUG")

    logger = get_logger("demo")
    logger.info("Starting demo", version="1.0")

    with TaskLogger("process_data", capture_id="test123") as task:
        task.info("Processing step 1")
        time.sleep(0.1)
        task.progress(33, "Step 1 complete")

        task.info("Processing step 2")
        time.sleep(0.1)
        task.progress(66, "Step 2 complete")

        task.info("Processing step 3")
        time.sleep(0.1)
        task.progress(100, "All steps complete")

    record_metric("processing_time", 0.3)
    record_metric("mesh_count", 150)

    print("\nMetrics:")
    print(json.dumps(_metrics.get_all_stats(), indent=2))
