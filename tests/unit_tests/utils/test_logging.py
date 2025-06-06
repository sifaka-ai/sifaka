"""Comprehensive unit tests for logging utilities.

This module tests the logging infrastructure:
- Logger configuration and setup
- Log level management
- Structured logging with metadata
- Performance logging
- Error handling in logging

Tests cover:
- Basic logging functionality
- Configuration management
- Structured logging features
- Performance characteristics
- Error scenarios
"""

import pytest
import logging
import json
from unittest.mock import Mock, patch
from io import StringIO

from sifaka.utils.logging import (
    get_logger,
    setup_logging,
    PerformanceLogger,
    SifakaFormatter,
    configure_for_development,
    configure_for_production,
)


class TestGetLogger:
    """Test the get_logger function."""

    def test_get_logger_basic(self):
        """Test basic logger creation."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "sifaka.test_module"

    def test_get_logger_with_class(self):
        """Test logger creation with class name."""

        class TestClass:
            pass

        logger = get_logger(TestClass)

        assert isinstance(logger, logging.Logger)
        assert "TestClass" in logger.name

    def test_get_logger_caching(self):
        """Test that loggers are cached properly."""
        logger1 = get_logger("test_cache")
        logger2 = get_logger("test_cache")

        # Should return the same logger instance
        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Test that different names create different loggers."""
        logger1 = get_logger("test_name1")
        logger2 = get_logger("test_name2")

        assert logger1 is not logger2
        assert logger1.name != logger2.name


class TestSetupLogging:
    """Test the setup_logging function."""

    def test_setup_logging_default(self):
        """Test logging configuration with default settings."""
        setup_logging()

        # Should set up basic logging configuration
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.INFO

    def test_setup_logging_debug_level(self):
        """Test logging configuration with debug level."""
        setup_logging(level="DEBUG")

        sifaka_logger = logging.getLogger("sifaka")
        assert sifaka_logger.level <= logging.DEBUG

    def test_setup_logging_warning_level(self):
        """Test logging configuration with warning level."""
        setup_logging(level="WARNING")

        sifaka_logger = logging.getLogger("sifaka")
        assert sifaka_logger.level <= logging.WARNING

    def test_setup_logging_with_file(self):
        """Test logging configuration with file output."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name

        try:
            setup_logging(log_file=log_file)

            # Test that logging to file works
            logger = get_logger("test_file")
            logger.info("Test message")

            # Check that file was created and contains log
            assert os.path.exists(log_file)
            with open(log_file, "r") as f:
                content = f.read()
                assert "Test message" in content

        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_setup_logging_structured(self):
        """Test logging configuration with structured output."""
        setup_logging(format_type="structured")

        logger = get_logger("test_structured")
        logger.info("Test message", extra={"key": "value"})

        # Test passes if no exception is raised


class TestSifakaFormatter:
    """Test the SifakaFormatter class."""

    def test_formatter_creation_structured(self):
        """Test creating a structured formatter."""
        formatter = SifakaFormatter("structured")
        assert formatter.format_type == "structured"

    def test_formatter_creation_simple(self):
        """Test creating a simple formatter."""
        formatter = SifakaFormatter("simple")
        assert formatter.format_type == "simple"

    def test_format_record_simple(self):
        """Test formatting a log record with simple format."""
        formatter = SifakaFormatter("simple")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted
        assert "INFO" in formatted


class TestPerformanceLogger:
    """Test the PerformanceLogger class."""

    @pytest.fixture
    def perf_logger(self):
        """Create a performance logger for testing."""
        base_logger = logging.getLogger("test_performance")
        return PerformanceLogger(base_logger)

    def test_performance_logger_creation(self, perf_logger):
        """Test performance logger creation."""
        assert perf_logger.logger.name == "test_performance"
        assert hasattr(perf_logger, "performance_timer")

    def test_performance_timer_context_manager(self, perf_logger):
        """Test timing operations with context manager."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        perf_logger.logger.addHandler(handler)
        perf_logger.logger.setLevel(logging.INFO)

        import time

        with perf_logger.performance_timer("test_operation"):
            time.sleep(0.01)  # Simulate work

        output = log_stream.getvalue()
        assert "test_operation" in output
        assert "Completed" in output
        # Should log execution time
        assert "duration_seconds" in output

    def test_log_thought_event(self, perf_logger):
        """Test logging thought events."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        perf_logger.logger.addHandler(handler)
        perf_logger.logger.setLevel(logging.INFO)

        perf_logger.log_thought_event("generation_start", "thought-123", iteration=1, model="gpt-4")

        output = log_stream.getvalue()
        assert "generation_start" in output
        assert "thought-123" in output

    def test_log_model_call(self, perf_logger):
        """Test logging model calls."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        perf_logger.logger.addHandler(handler)
        perf_logger.logger.setLevel(logging.INFO)

        perf_logger.log_model_call(
            model="gpt-4", operation="generation", duration=1.5, tokens_used=150, cost=0.002
        )

        output = log_stream.getvalue()
        assert "gpt-4" in output
        assert "generation" in output


class TestConfigureFunctions:
    """Test the configure_for_development and configure_for_production functions."""

    def test_configure_for_development(self):
        """Test development configuration."""
        configure_for_development()

        # Should set up logging without errors
        logger = get_logger("dev_test")
        logger.info("Development test message")

    def test_configure_for_production(self):
        """Test production configuration."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            configure_for_production(temp_dir)

            # Should set up logging without errors
            logger = get_logger("prod_test")
            logger.info("Production test message")


class TestLoggingIntegration:
    """Test logging integration with Sifaka components."""

    def test_thought_logging(self):
        """Test logging thought operations."""
        from sifaka.core.thought import SifakaThought

        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)

        # Configure logging
        logger = get_logger("thought_test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Create and log thought operations
        thought = SifakaThought(prompt="Test logging")

        logger.info(
            "Thought created",
            extra={
                "thought_id": thought.id,
                "prompt_length": len(thought.prompt),
                "iteration": thought.iteration,
            },
        )

        output = log_stream.getvalue()
        assert "Thought created" in output
        assert thought.id in output

    def test_validation_logging(self):
        """Test logging validation operations."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)

        logger = get_logger("validation_test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Simulate validation logging
        validation_result = {
            "validator": "length_validator",
            "passed": True,
            "score": 0.85,
            "duration_ms": 25.0,
        }

        logger.info("Validation completed", extra=validation_result)

        output = log_stream.getvalue()
        assert "Validation completed" in output
        assert "length_validator" in output

    def test_critique_logging(self):
        """Test logging critique operations."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)

        logger = get_logger("critique_test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Simulate critique logging
        critique_result = {
            "critic": "constitutional_critic",
            "needs_improvement": True,
            "confidence": 0.8,
            "suggestions_count": 3,
            "duration_ms": 1250.0,
        }

        logger.info("Critique completed", extra=critique_result)

        output = log_stream.getvalue()
        assert "Critique completed" in output
        assert "constitutional_critic" in output

    def test_error_logging(self):
        """Test error logging with context."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)

        logger = get_logger("error_test")
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        try:
            # Simulate an error
            raise ValueError("Test validation error")
        except Exception as e:
            logger.error(
                "Operation failed",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "operation": "test_operation",
                    "context": {"input": "test_data"},
                },
                exc_info=True,
            )

        output = log_stream.getvalue()
        assert "Operation failed" in output
        assert "ValueError" in output
        assert "test_operation" in output


class TestLoggingPerformance:
    """Test logging performance characteristics."""

    def test_logging_overhead(self):
        """Test that logging doesn't add significant overhead."""
        logger = get_logger("performance_test")

        import time

        # Test without logging
        start_time = time.time()
        for i in range(1000):
            pass  # No-op
        no_log_time = time.time() - start_time

        # Test with logging (but disabled)
        logger.setLevel(logging.CRITICAL)  # Disable most logging
        start_time = time.time()
        for i in range(1000):
            logger.debug("Debug message %d", i)
        disabled_log_time = time.time() - start_time

        # Logging overhead should be minimal when disabled
        overhead = disabled_log_time - no_log_time
        assert overhead < 0.1  # Less than 100ms overhead for 1000 calls

    def test_structured_logging_performance(self):
        """Test structured logging performance."""
        logger = get_logger("perf_test")
        logger.logger.setLevel(logging.CRITICAL)  # Disable output

        import time

        start_time = time.time()
        for i in range(100):
            logger.info("Test message %d", i, extra={"iteration": i, "data": "test_data"})
        end_time = time.time()

        # Should complete quickly even with structured data
        duration = end_time - start_time
        assert duration < 1.0  # Less than 1 second for 100 calls
