"""Custom logging configuration example for Sifaka.

This example shows various logging configurations for different use cases:
- Development debugging
- Production with structured logs
- Log aggregation (ELK stack, CloudWatch, etc.)
- Performance profiling
"""

import logging
import logging.config
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import traceback

from pythonjsonlogger import jsonlogger

from sifaka import improve_sync
from sifaka.core.middleware import BaseMiddleware
from sifaka.core.models import SifakaResult


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional context."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        # Add timestamp
        log_record["timestamp"] = datetime.utcnow().isoformat()

        # Add log level
        log_record["level"] = record.levelname

        # Add logger name
        log_record["logger"] = record.name

        # Add any extra fields
        if hasattr(record, "extra_fields"):
            log_record.update(record.extra_fields)


class LoggingMiddleware(BaseMiddleware):
    """Middleware to add structured logging to Sifaka operations."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize logging middleware.

        Args:
            logger: Logger instance (uses 'sifaka' logger if None)
        """
        self.logger = logger or logging.getLogger("sifaka")

    async def pre_improve(self, text: str, **kwargs) -> Dict[str, Any]:
        """Log improvement start."""
        self.logger.info(
            "Starting text improvement",
            extra={
                "extra_fields": {
                    "event": "improvement_start",
                    "text_length": len(text),
                    "critic": kwargs.get("critics", ["reflexion"])[0],
                    "max_iterations": kwargs.get("max_iterations", 3),
                    "model": kwargs.get("model", "default"),
                }
            },
        )
        return {"start_time": datetime.utcnow()}

    async def post_improve(
        self,
        result: SifakaResult,
        context: Dict[str, Any],
        error: Optional[Exception] = None,
    ):
        """Log improvement completion."""
        duration = (datetime.utcnow() - context["start_time"]).total_seconds()

        if error:
            self.logger.error(
                f"Improvement failed: {str(error)}",
                extra={
                    "extra_fields": {
                        "event": "improvement_failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                        "duration_seconds": duration,
                    }
                },
            )
        else:
            self.logger.info(
                "Improvement completed successfully",
                extra={
                    "extra_fields": {
                        "event": "improvement_completed",
                        "iterations": result.iterations,
                        "total_tokens": result.total_tokens,
                        "total_cost": result.total_cost,
                        "duration_seconds": duration,
                        "text_length_original": len(result.history[0].text),
                        "text_length_final": len(result.final_text),
                        "improvement_ratio": len(result.final_text)
                        / len(result.history[0].text),
                    }
                },
            )

    async def on_critique(self, critique: str, iteration: int, **kwargs):
        """Log critique generation."""
        self.logger.debug(
            f"Generated critique for iteration {iteration}",
            extra={
                "extra_fields": {
                    "event": "critique_generated",
                    "iteration": iteration,
                    "critique_length": len(critique),
                    "critic": kwargs.get("critic", "unknown"),
                }
            },
        )

    async def on_generation(self, text: str, iteration: int, **kwargs):
        """Log text generation."""
        self.logger.debug(
            f"Generated improved text for iteration {iteration}",
            extra={
                "extra_fields": {
                    "event": "text_generated",
                    "iteration": iteration,
                    "text_length": len(text),
                    "critic": kwargs.get("critic", "unknown"),
                }
            },
        )


# Logging configuration examples
LOGGING_CONFIGS = {
    # Development configuration with colored console output
    "development": {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": "colorlog.ColoredFormatter",
                "format": "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "log_colors": {
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "colored",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "colored",
                "filename": "sifaka_debug.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "sifaka": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False,
            }
        },
        "root": {"level": "INFO", "handlers": ["console"]},
    },
    # Production configuration with JSON structured logs
    "production": {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "logging_configuration.StructuredFormatter",
                "format": "%(timestamp)s %(level)s %(name)s %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": "logs/sifaka.log",
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "json",
                "filename": "logs/sifaka_errors.log",
                "maxBytes": 52428800,  # 50MB
                "backupCount": 10,
            },
        },
        "loggers": {
            "sifaka": {
                "level": "INFO",
                "handlers": ["console", "file", "error_file"],
                "propagate": False,
            },
            "sifaka.performance": {
                "level": "INFO",
                "handlers": ["file"],
                "propagate": False,
            },
        },
        "root": {"level": "WARNING", "handlers": ["console"]},
    },
    # ELK Stack configuration
    "elk": {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "logstash": {
                "()": "logstash_formatter.LogstashFormatterV1",
                "metadata": {
                    "service": "sifaka",
                    "environment": "production",
                    "version": "0.0.7",
                },
            }
        },
        "handlers": {
            "logstash": {
                "class": "logstash_async.handler.AsynchronousLogstashHandler",
                "host": "localhost",
                "port": 5959,
                "transport": "logstash_async.transport.TcpTransport",
                "formatter": "logstash",
                "database_path": "logstash.db",
            },
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "sifaka": {
                "level": "INFO",
                "handlers": ["logstash", "console"],
                "propagate": False,
            }
        },
        "root": {"level": "WARNING", "handlers": ["console"]},
    },
    # AWS CloudWatch configuration
    "cloudwatch": {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "aws": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "cloudwatch": {
                "class": "watchtower.CloudWatchLogHandler",
                "level": "INFO",
                "formatter": "aws",
                "log_group": "/aws/lambda/sifaka",
                "stream_name": "{machine_name}/{program_name}/{logger_name}/{process_id}",
                "send_interval": 10,
                "max_batch_size": 10,
                "max_batch_count": 10,
            },
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "aws",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "sifaka": {
                "level": "INFO",
                "handlers": ["cloudwatch", "console"],
                "propagate": False,
            }
        },
        "root": {"level": "WARNING", "handlers": ["console"]},
    },
}


def setup_logging(config_name: str = "development") -> logging.Logger:
    """Set up logging configuration.

    Args:
        config_name: Name of configuration to use

    Returns:
        Configured logger for Sifaka
    """
    config = LOGGING_CONFIGS.get(config_name)
    if not config:
        raise ValueError(f"Unknown logging config: {config_name}")

    # Create logs directory if needed
    Path("logs").mkdir(exist_ok=True)

    # Apply configuration
    logging.config.dictConfig(config)

    # Return sifaka logger
    return logging.getLogger("sifaka")


# Performance logging utilities
class PerformanceLogger:
    """Logger for performance metrics."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("sifaka.performance")

    def log_operation(self, operation: str, duration: float, metadata: Dict[str, Any]):
        """Log a performance metric."""
        self.logger.info(
            f"Performance: {operation}",
            extra={
                "extra_fields": {
                    "metric_type": "performance",
                    "operation": operation,
                    "duration_ms": duration * 1000,
                    **metadata,
                }
            },
        )

    def log_memory(self, operation: str, memory_mb: float):
        """Log memory usage."""
        self.logger.info(
            f"Memory usage: {operation}",
            extra={
                "extra_fields": {
                    "metric_type": "memory",
                    "operation": operation,
                    "memory_mb": memory_mb,
                }
            },
        )


# Custom log filters
class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from logs."""

    def __init__(self, patterns: list):
        super().__init__()
        self.patterns = patterns

    def filter(self, record):
        # Redact sensitive data in message
        message = record.getMessage()
        for pattern in self.patterns:
            message = message.replace(pattern, "[REDACTED]")
        record.msg = message

        # Redact in extra fields if present
        if hasattr(record, "extra_fields"):
            for key, value in record.extra_fields.items():
                if isinstance(value, str):
                    for pattern in self.patterns:
                        record.extra_fields[key] = value.replace(pattern, "[REDACTED]")

        return True


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sifaka logging example")
    parser.add_argument(
        "--config",
        choices=["development", "production", "elk", "cloudwatch"],
        default="development",
        help="Logging configuration to use",
    )
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.config)
    logger.info(f"Using {args.config} logging configuration")

    # Create logging middleware
    log_middleware = LoggingMiddleware(logger)

    # Add sensitive data filter
    sensitive_filter = SensitiveDataFilter(["api-key-123", "secret-token"])
    for handler in logger.handlers:
        handler.addFilter(sensitive_filter)

    # Example 1: Simple improvement with logging
    text = "Machine learning is important. api-key-123 should be hidden."

    try:
        result = improve_sync(
            text, critics=["reflexion"], max_iterations=2, middleware=[log_middleware]
        )

        logger.info(
            "Improvement successful",
            extra={
                "extra_fields": {
                    "user_id": "user123",
                    "session_id": "session456",
                    "improvement_quality": "high",
                }
            },
        )

    except Exception:
        logger.exception("Failed to improve text")

    # Example 2: Performance logging
    perf_logger = PerformanceLogger()

    import time

    start = time.time()
    # Simulate some operation
    time.sleep(0.1)
    duration = time.time() - start

    perf_logger.log_operation(
        "text_analysis", duration, {"text_length": 100, "complexity": "medium"}
    )

    # Example 3: Structured logging for analytics
    logger.info(
        "User action",
        extra={
            "extra_fields": {
                "event_type": "user_action",
                "action": "improve_text",
                "user_id": "user123",
                "feature": "reflexion_critic",
                "success": True,
                "response_time_ms": 1234,
                "tokens_used": 500,
                "cost_usd": 0.05,
            }
        },
    )

    # Example 4: Log aggregation query examples
    print(f"\n--- Log Query Examples for {args.config} ---")

    if args.config == "elk":
        print("Elasticsearch queries:")
        print('  Error rate: {"query": {"match": {"level": "ERROR"}}}')
        print('  Slow operations: {"query": {"range": {"duration_ms": {"gte": 5000}}}}')
        print('  By user: {"query": {"match": {"user_id": "user123"}}}')

    elif args.config == "cloudwatch":
        print("CloudWatch Insights queries:")
        print(
            "  Error count: fields @timestamp, @message | filter level = 'ERROR' | stats count()"
        )
        print(
            "  Avg duration: fields duration_ms | stats avg(duration_ms) by operation"
        )
        print("  User activity: fields user_id, action | filter user_id = 'user123'")

    print(f"\nLogging configured for {args.config} environment")
    print(
        f"Check logs in: {logger.handlers[0].__dict__.get('baseFilename', 'console')}"
    )
