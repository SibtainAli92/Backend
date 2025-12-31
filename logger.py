"""
Structured logging for Book RAG Agent.

Provides consistent, parseable logging throughout the application
with support for different log levels and structured output.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps
import traceback


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }

        # Add source location
        log_data["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName
        }

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m"
    }

    ICONS = {
        "DEBUG": "[D]",
        "INFO": "[I]",
        "WARNING": "[W]",
        "ERROR": "[E]",
        "CRITICAL": "[!]"
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        icon = self.ICONS.get(record.levelname, "")

        timestamp = datetime.utcnow().strftime("%H:%M:%S")

        # Base message
        msg = f"{color}[{timestamp}] {icon} {record.levelname:8}{reset} | {record.name}: {record.getMessage()}"

        # Add extra data if present
        if hasattr(record, "extra_data") and record.extra_data:
            data_str = ", ".join(f"{k}={v}" for k, v in record.extra_data.items())
            msg += f" | {data_str}"

        # Add exception if present
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"

        return msg


class AppLogger:
    """Application logger with structured logging support."""

    _instances: Dict[str, 'AppLogger'] = {}

    def __init__(self, name: str, level: str = None):
        """
        Initialize logger.

        Args:
            name: Logger name (usually module name)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.name = name
        self.level = level or os.getenv("LOG_LEVEL", "INFO")
        self.logger = logging.getLogger(name)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup logging handlers."""
        if self.logger.handlers:
            return  # Already configured

        self.logger.setLevel(getattr(logging, self.level.upper(), logging.INFO))

        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ConsoleFormatter())
        self.logger.addHandler(console_handler)

        # File handler (structured JSON) - optional
        log_file = os.getenv("LOG_FILE")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def _log(self, level: int, message: str, extra: Dict[str, Any] = None, exc_info: bool = False) -> None:
        """Internal logging method with extra data support."""
        record = self.logger.makeRecord(
            self.name, level, "", 0, message, (), None
        )
        if extra:
            record.extra_data = extra
        self.logger.handle(record)

        if exc_info:
            self.logger.exception(message)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, kwargs if kwargs else None)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, kwargs if kwargs else None)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, kwargs if kwargs else None)

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        if exc_info:
            kwargs["traceback"] = traceback.format_exc()
        self._log(logging.ERROR, message, kwargs if kwargs else None)

    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log critical message."""
        if exc_info:
            kwargs["traceback"] = traceback.format_exc()
        self._log(logging.CRITICAL, message, kwargs if kwargs else None)

    def request(self, method: str, path: str, status: int, duration_ms: float, **kwargs) -> None:
        """Log HTTP request."""
        self.info(
            f"{method} {path} -> {status}",
            method=method,
            path=path,
            status=status,
            duration_ms=round(duration_ms, 2),
            **kwargs
        )

    def tool_call(self, tool: str, action: str, success: bool, **kwargs) -> None:
        """Log tool/agent action."""
        level = logging.INFO if success else logging.WARNING
        status = "SUCCESS" if success else "FAILED"
        self._log(
            level,
            f"Tool [{tool}] {action}: {status}",
            {"tool": tool, "action": action, "success": success, **kwargs}
        )

    def qdrant_query(self, collection: str, results_count: int, duration_ms: float, **kwargs) -> None:
        """Log Qdrant query."""
        self.info(
            f"Qdrant query on '{collection}': {results_count} results",
            collection=collection,
            results=results_count,
            duration_ms=round(duration_ms, 2),
            **kwargs
        )

    def llm_call(self, model: str, prompt_tokens: int = None, response_tokens: int = None, **kwargs) -> None:
        """Log LLM API call."""
        self.info(
            f"LLM call to {model}",
            model=model,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            **kwargs
        )


def get_logger(name: str) -> AppLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        AppLogger instance
    """
    if name not in AppLogger._instances:
        AppLogger._instances[name] = AppLogger(name)
    return AppLogger._instances[name]


def log_function_call(logger: AppLogger = None):
    """
    Decorator to log function entry/exit.

    Args:
        logger: Optional logger instance
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            func_name = func.__name__
            logger.debug(f"Entering {func_name}", args_count=len(args), kwargs_keys=list(kwargs.keys()))

            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func_name}", success=True)
                return result
            except Exception as e:
                logger.error(f"Exception in {func_name}: {str(e)}", exc_info=True)
                raise

        return wrapper
    return decorator


def log_async_function_call(logger: AppLogger = None):
    """
    Decorator to log async function entry/exit.

    Args:
        logger: Optional logger instance
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            func_name = func.__name__
            logger.debug(f"Entering {func_name}", args_count=len(args), kwargs_keys=list(kwargs.keys()))

            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Exiting {func_name}", success=True)
                return result
            except Exception as e:
                logger.error(f"Exception in {func_name}: {str(e)}", exc_info=True)
                raise

        return wrapper
    return decorator
