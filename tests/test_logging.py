"""
Tests for the logging utilities.

This module contains tests for the logging utilities in the Sifaka framework.
"""

import logging
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from sifaka.utils.logging import configure_logging, get_logger


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def test_configure_with_default_level(self) -> None:
        """Test configuring logging with the default level (INFO)."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            configure_logging()
            
            mock_get_logger.assert_called_once_with("sifaka")
            mock_logger.setLevel.assert_called_once_with(logging.INFO)
            assert mock_logger.addHandler.call_count == 1

    def test_configure_with_custom_level_int(self) -> None:
        """Test configuring logging with a custom level as integer."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            configure_logging(level=logging.DEBUG)
            
            mock_get_logger.assert_called_once_with("sifaka")
            mock_logger.setLevel.assert_called_once_with(logging.DEBUG)
            assert mock_logger.addHandler.call_count == 1

    def test_configure_with_custom_level_str(self) -> None:
        """Test configuring logging with a custom level as string."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            configure_logging(level="DEBUG")
            
            mock_get_logger.assert_called_once_with("sifaka")
            mock_logger.setLevel.assert_called_once_with(logging.DEBUG)
            assert mock_logger.addHandler.call_count == 1

    def test_configure_with_invalid_level_str(self) -> None:
        """Test configuring logging with an invalid level string."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            configure_logging(level="INVALID")
            
            mock_get_logger.assert_called_once_with("sifaka")
            # Should default to INFO
            mock_logger.setLevel.assert_called_once_with(logging.INFO)
            assert mock_logger.addHandler.call_count == 1

    def test_configure_with_log_file(self) -> None:
        """Test configuring logging with a log file."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as temp_file:
            log_file = temp_file.name
            
            try:
                with patch("logging.getLogger") as mock_get_logger:
                    mock_logger = MagicMock()
                    mock_get_logger.return_value = mock_logger
                    
                    configure_logging(log_file=log_file)
                    
                    mock_get_logger.assert_called_once_with("sifaka")
                    mock_logger.setLevel.assert_called_once_with(logging.INFO)
                    # Should add both console and file handlers
                    assert mock_logger.addHandler.call_count == 2
            finally:
                # Clean up
                if os.path.exists(log_file):
                    os.unlink(log_file)

    def test_configure_with_real_logger(self) -> None:
        """Test configuring logging with a real logger."""
        # Get the root logger and save its handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        original_level = root_logger.level
        
        try:
            # Remove all handlers to start clean
            for handler in root_logger.handlers:
                root_logger.removeHandler(handler)
            
            # Configure logging
            configure_logging(level=logging.DEBUG)
            
            # Get the sifaka logger
            logger = logging.getLogger("sifaka")
            
            # Check that the logger is configured correctly
            assert logger.level == logging.DEBUG
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0], logging.StreamHandler)
            
            # Test logging a message
            with patch.object(logger.handlers[0], "emit") as mock_emit:
                logger.debug("Test debug message")
                mock_emit.assert_called_once()
        finally:
            # Restore the original handlers and level
            root_logger.setLevel(original_level)
            for handler in root_logger.handlers:
                root_logger.removeHandler(handler)
            for handler in original_handlers:
                root_logger.addHandler(handler)


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_get_logger(self) -> None:
        """Test getting a logger for a specific module."""
        with patch("logging.getLogger") as mock_get_logger:
            get_logger("test")
            mock_get_logger.assert_called_once_with("sifaka.test")

    def test_get_logger_with_real_logger(self) -> None:
        """Test getting a real logger."""
        logger = get_logger("test")
        assert logger.name == "sifaka.test"

    def test_logger_hierarchy(self) -> None:
        """Test that loggers form a proper hierarchy."""
        # Configure the root logger
        with patch("logging.getLogger") as mock_get_logger:
            mock_root_logger = MagicMock()
            mock_get_logger.return_value = mock_root_logger
            
            configure_logging(level=logging.DEBUG)
            
            mock_get_logger.assert_called_once_with("sifaka")
        
        # Now test with real loggers
        root_logger = logging.getLogger("sifaka")
        test_logger = get_logger("test")
        
        # The test logger should inherit from the root logger
        assert test_logger.parent == root_logger
