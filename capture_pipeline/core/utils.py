#!/usr/bin/env python3
"""
Core Utilities - Logging and Error Handling
============================================
Unified logging, error handling, and common utilities for the pipeline.

Features:
    - Dual output: console + debug-log.txt
    - Structured error messages
    - Performance timing
    - Safe file operations

Usage:
    from core.utils import logger, log_error, timed, safe_open

    logger.info("Processing started")
    logger.debug("Detailed info")
    log_error("Something failed", exception=e)
"""

import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union


# =============================================================================
# Logging Configuration
# =============================================================================

# Find project root (where pipeline.py lives)
def _find_project_root() -> Path:
    """Find project root by looking for pipeline.py or config.yaml."""
    current = Path(__file__).parent.parent
    if (current / 'pipeline.py').exists() or (current / 'config.yaml').exists():
        return current
    return Path.cwd()

PROJECT_ROOT = _find_project_root()
LOG_FILE = PROJECT_ROOT / 'debug-log.txt'

# Custom formatter for console (concise)
class ConsoleFormatter(logging.Formatter):
    """Concise console output."""
    FORMATS = {
        logging.DEBUG: "\033[90m[DEBUG]\033[0m %(message)s",
        logging.INFO: "[INFO] %(message)s",
        logging.WARNING: "\033[33m[WARN]\033[0m %(message)s",
        logging.ERROR: "\033[31m[ERROR]\033[0m %(message)s",
        logging.CRITICAL: "\033[31;1m[CRITICAL]\033[0m %(message)s",
    }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(fmt)
        return formatter.format(record)


# Custom formatter for file (detailed)
class FileFormatter(logging.Formatter):
    """Detailed file output with timestamps."""
    def format(self, record):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname.ljust(8)

        # Include module info for debugging
        module = record.module if hasattr(record, 'module') else 'unknown'
        lineno = record.lineno if hasattr(record, 'lineno') else 0

        msg = f"{timestamp} | {level} | {module}:{lineno} | {record.getMessage()}"

        # Include exception info if present
        if record.exc_info:
            msg += '\n' + ''.join(traceback.format_exception(*record.exc_info))

        return msg


def setup_logger(name: str = 'pipeline', level: str = 'DEBUG') -> logging.Logger:
    """
    Setup logger with console and file handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    # Console handler (INFO and above)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(ConsoleFormatter())
    logger.addHandler(console)

    # File handler (all levels)
    try:
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)
    except (OSError, IOError) as e:
        logger.warning(f"Could not create log file: {e}")

    return logger


# Global logger instance
logger = setup_logger('pipeline')


def log_error(message: str, exception: Exception = None,
              context: Dict[str, Any] = None) -> None:
    """
    Log an error with optional exception and context.

    Args:
        message: Error message
        exception: Optional exception object
        context: Optional dict with additional context
    """
    full_msg = message

    if context:
        ctx_str = ', '.join(f"{k}={v}" for k, v in context.items())
        full_msg += f" [{ctx_str}]"

    if exception:
        logger.error(full_msg, exc_info=True)
    else:
        logger.error(full_msg)


def log_section(title: str) -> None:
    """Log a section header for visual separation."""
    separator = '=' * 60
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


# =============================================================================
# Performance Timing
# =============================================================================

@contextmanager
def timed(operation: str):
    """
    Context manager for timing operations.

    Usage:
        with timed("Processing meshes"):
            process_meshes()
    """
    start = time.perf_counter()
    logger.debug(f"Starting: {operation}")

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"Completed: {operation} ({elapsed:.2f}s)")


def timed_function(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    return wrapper


# =============================================================================
# Safe File Operations
# =============================================================================

def safe_mkdir(path: Union[str, Path]) -> Path:
    """
    Safely create directory, logging any errors.

    Returns:
        Path object, or None if failed
    """
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except (OSError, IOError) as e:
        log_error(f"Failed to create directory: {path}", exception=e)
        return None


def safe_read_json(path: Union[str, Path]) -> Optional[Dict]:
    """
    Safely read JSON file, returning None on failure.
    """
    path = Path(path)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.debug(f"File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        log_error(f"Invalid JSON: {path}", exception=e)
        return None
    except (OSError, IOError) as e:
        log_error(f"Failed to read: {path}", exception=e)
        return None


def safe_write_json(path: Union[str, Path], data: Dict,
                    indent: int = 2) -> bool:
    """
    Safely write JSON file with atomic write.

    Returns:
        True if successful
    """
    path = Path(path)
    temp_path = path.with_suffix('.tmp')

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        # Atomic rename
        temp_path.replace(path)
        logger.debug(f"Wrote: {path}")
        return True

    except (OSError, IOError, TypeError) as e:
        log_error(f"Failed to write: {path}", exception=e)
        if temp_path.exists():
            temp_path.unlink()
        return False


def safe_copy(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Safely copy file with error handling.
    """
    import shutil

    src, dst = Path(src), Path(dst)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.debug(f"Copied: {src} -> {dst}")
        return True
    except (OSError, IOError, shutil.Error) as e:
        log_error(f"Failed to copy: {src} -> {dst}", exception=e)
        return False


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_path(path: Union[str, Path], must_exist: bool = True,
                  file_type: str = None) -> Optional[Path]:
    """
    Validate a path, optionally checking existence and type.

    Args:
        path: Path to validate
        must_exist: If True, path must exist
        file_type: Optional extension to check (e.g., '.obj', '.glb')

    Returns:
        Path object if valid, None otherwise
    """
    if not path:
        logger.error("Path is empty or None")
        return None

    path = Path(path)

    if must_exist and not path.exists():
        logger.error(f"Path does not exist: {path}")
        return None

    if file_type and path.suffix.lower() != file_type.lower():
        logger.warning(f"Expected {file_type}, got {path.suffix}: {path}")

    return path


def validate_directory(path: Union[str, Path],
                       create: bool = False) -> Optional[Path]:
    """
    Validate a directory path.

    Args:
        path: Directory path
        create: If True, create if doesn't exist

    Returns:
        Path object if valid/created, None otherwise
    """
    if not path:
        logger.error("Directory path is empty or None")
        return None

    path = Path(path)

    if path.exists():
        if not path.is_dir():
            logger.error(f"Path is not a directory: {path}")
            return None
        return path

    if create:
        return safe_mkdir(path)

    logger.error(f"Directory does not exist: {path}")
    return None


# =============================================================================
# Error Classes
# =============================================================================

class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class ExtractionError(PipelineError):
    """Error during RenderDoc extraction."""
    pass


class ProcessingError(PipelineError):
    """Error during mesh processing."""
    pass


class ExportError(PipelineError):
    """Error during export."""
    pass


class ValidationError(PipelineError):
    """Error during validation."""
    pass


# =============================================================================
# Session Management
# =============================================================================

def start_session(name: str = None) -> str:
    """
    Start a new logging session with separator.

    Returns:
        Session ID (timestamp)
    """
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Write session header to log file
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write('=' * 80 + '\n')
        f.write(f"SESSION: {session_id}")
        if name:
            f.write(f" - {name}")
        f.write('\n')
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write('=' * 80 + '\n')

    logger.info(f"Session started: {session_id}")
    return session_id


def end_session(session_id: str, success: bool = True) -> None:
    """End a logging session."""
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"Session {session_id}: {status}")

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\nSession {session_id} ended: {status}\n")
        f.write('-' * 80 + '\n')


# =============================================================================
# Initialization
# =============================================================================

def init_logging(verbose: bool = False) -> None:
    """
    Initialize logging for a new run.

    Args:
        verbose: Enable DEBUG output to console
    """
    if verbose:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)

    logger.debug(f"Logging initialized. Log file: {LOG_FILE}")


# Auto-initialize on import
init_logging()
