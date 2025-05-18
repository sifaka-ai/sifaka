"""
Tests for the retry module.

This module contains tests for the retry functionality in the Sifaka framework.
"""

import pytest
import time
from typing import Any, List, Optional, Type, Union
from unittest.mock import MagicMock, patch

from sifaka.retry.base import RetryStrategy, retry
from sifaka.retry.strategies import ExponentialBackoff, FixedInterval, LinearBackoff
from sifaka.errors import RetryError


class TestRetryStrategy:
    """Tests for the RetryStrategy abstract base class."""

    def test_abstract_class(self) -> None:
        """Test that RetryStrategy is an abstract class."""
        with pytest.raises(TypeError):
            RetryStrategy()  # Should raise TypeError because it's abstract


class TestExponentialBackoff:
    """Tests for the ExponentialBackoff strategy."""

    def test_init_with_defaults(self) -> None:
        """Test initializing an ExponentialBackoff with default parameters."""
        strategy = ExponentialBackoff()
        assert strategy.base_delay == 1.0
        assert strategy.max_delay == 60.0
        assert strategy.jitter is True
        assert strategy.retryable_exceptions == [Exception]

    def test_init_with_custom_parameters(self) -> None:
        """Test initializing an ExponentialBackoff with custom parameters."""
        strategy = ExponentialBackoff(
            base_delay=0.5,
            max_delay=30.0,
            jitter=False,
            retryable_exceptions=ValueError,
        )
        assert strategy.base_delay == 0.5
        assert strategy.max_delay == 30.0
        assert strategy.jitter is False
        assert strategy.retryable_exceptions == [ValueError]

    def test_get_delay_without_jitter(self) -> None:
        """Test get_delay without jitter."""
        strategy = ExponentialBackoff(base_delay=1.0, max_delay=60.0, jitter=False)

        # First attempt (1-based)
        assert strategy.get_delay(1) == 1.0

        # Second attempt
        assert strategy.get_delay(2) == 2.0

        # Third attempt
        assert strategy.get_delay(3) == 4.0

        # Fourth attempt
        assert strategy.get_delay(4) == 8.0

    def test_get_delay_with_max_delay(self) -> None:
        """Test get_delay with max_delay."""
        strategy = ExponentialBackoff(base_delay=1.0, max_delay=10.0, jitter=False)

        # Sixth attempt (2^5 = 32, but max is 10)
        assert strategy.get_delay(6) == 10.0

    def test_get_delay_with_jitter(self) -> None:
        """Test get_delay with jitter."""
        strategy = ExponentialBackoff(base_delay=1.0, max_delay=60.0, jitter=True)

        # With jitter, the delay should be between 0.5x and 1.5x the base delay
        delay = strategy.get_delay(1)
        assert 0.5 <= delay <= 1.5

    def test_should_retry_with_retryable_exception(self) -> None:
        """Test should_retry with a retryable exception."""
        strategy = ExponentialBackoff(retryable_exceptions=ValueError)
        assert strategy.should_retry(1, ValueError("Test error")) is True

    def test_should_retry_with_non_retryable_exception(self) -> None:
        """Test should_retry with a non-retryable exception."""
        strategy = ExponentialBackoff(retryable_exceptions=ValueError)
        assert strategy.should_retry(1, TypeError("Test error")) is False


class TestFixedInterval:
    """Tests for the FixedInterval strategy."""

    def test_init_with_defaults(self) -> None:
        """Test initializing a FixedInterval with default parameters."""
        strategy = FixedInterval()
        assert strategy.delay == 1.0
        assert strategy.jitter is True
        assert strategy.retryable_exceptions == [Exception]

    def test_init_with_custom_parameters(self) -> None:
        """Test initializing a FixedInterval with custom parameters."""
        strategy = FixedInterval(
            delay=2.0,
            jitter=False,
            retryable_exceptions=[ValueError, TypeError],
        )
        assert strategy.delay == 2.0
        assert strategy.jitter is False
        assert strategy.retryable_exceptions == [ValueError, TypeError]

    def test_get_delay_without_jitter(self) -> None:
        """Test get_delay without jitter."""
        strategy = FixedInterval(delay=2.0, jitter=False)

        # All attempts should have the same delay
        assert strategy.get_delay(1) == 2.0
        assert strategy.get_delay(2) == 2.0
        assert strategy.get_delay(3) == 2.0

    def test_get_delay_with_jitter(self) -> None:
        """Test get_delay with jitter."""
        strategy = FixedInterval(delay=2.0, jitter=True)

        # With jitter, the delay should be between 1.0 and 3.0
        delay = strategy.get_delay(1)
        assert 1.0 <= delay <= 3.0

    def test_should_retry_with_multiple_exceptions(self) -> None:
        """Test should_retry with multiple retryable exceptions."""
        strategy = FixedInterval(retryable_exceptions=[ValueError, TypeError])
        assert strategy.should_retry(1, ValueError("Test error")) is True
        assert strategy.should_retry(1, TypeError("Test error")) is True
        assert strategy.should_retry(1, KeyError("Test error")) is False


class TestLinearBackoff:
    """Tests for the LinearBackoff strategy."""

    def test_init_with_defaults(self) -> None:
        """Test initializing a LinearBackoff with default parameters."""
        strategy = LinearBackoff()
        assert strategy.base_delay == 1.0
        assert strategy.increment == 1.0
        assert strategy.max_delay == 60.0
        assert strategy.jitter is True
        assert strategy.retryable_exceptions == [Exception]

    def test_init_with_custom_parameters(self) -> None:
        """Test initializing a LinearBackoff with custom parameters."""
        strategy = LinearBackoff(
            base_delay=0.5,
            increment=0.5,
            max_delay=10.0,
            jitter=False,
            retryable_exceptions=ValueError,
        )
        assert strategy.base_delay == 0.5
        assert strategy.increment == 0.5
        assert strategy.max_delay == 10.0
        assert strategy.jitter is False
        assert strategy.retryable_exceptions == [ValueError]

    def test_get_delay_without_jitter(self) -> None:
        """Test get_delay without jitter."""
        strategy = LinearBackoff(base_delay=1.0, increment=1.0, max_delay=10.0, jitter=False)

        # First attempt (1-based)
        assert strategy.get_delay(1) == 1.0

        # Second attempt
        assert strategy.get_delay(2) == 2.0

        # Third attempt
        assert strategy.get_delay(3) == 3.0

        # Fourth attempt
        assert strategy.get_delay(4) == 4.0

    def test_get_delay_with_max_delay(self) -> None:
        """Test get_delay with max_delay."""
        strategy = LinearBackoff(base_delay=1.0, increment=1.0, max_delay=5.0, jitter=False)

        # Sixth attempt (1 + 5 = 6, but max is 5)
        assert strategy.get_delay(6) == 5.0

    def test_get_delay_with_jitter(self) -> None:
        """Test get_delay with jitter."""
        strategy = LinearBackoff(base_delay=1.0, increment=1.0, max_delay=10.0, jitter=True)

        # With jitter, the delay should be between 0.5x and 1.5x the calculated delay
        delay = strategy.get_delay(2)  # Base 1.0 + increment 1.0 = 2.0
        assert 1.0 <= delay <= 3.0


class TestRetryDecorator:
    """Tests for the retry decorator."""

    def test_successful_execution(self) -> None:
        """Test successful execution without retries."""
        strategy = MagicMock(spec=RetryStrategy)

        @retry(strategy=strategy)
        def test_function():
            return "success"

        result = test_function()
        assert result == "success"
        strategy.get_delay.assert_not_called()
        strategy.should_retry.assert_not_called()

    def test_retry_on_exception(self) -> None:
        """Test retrying on exception."""
        strategy = MagicMock(spec=RetryStrategy)
        strategy.should_retry.return_value = True
        strategy.get_delay.return_value = 0.0  # No delay for testing

        mock_function = MagicMock(side_effect=[ValueError("Test error"), "success"])

        @retry(strategy=strategy, max_attempts=3)
        def test_function():
            return mock_function()

        result = test_function()
        assert result == "success"
        assert mock_function.call_count == 2
        strategy.should_retry.assert_called_once()
        strategy.get_delay.assert_called_once_with(1)

    def test_max_attempts_reached(self) -> None:
        """Test reaching maximum retry attempts."""
        strategy = MagicMock(spec=RetryStrategy)
        strategy.should_retry.return_value = True
        strategy.get_delay.return_value = 0.0  # No delay for testing

        error = ValueError("Test error")
        mock_function = MagicMock(side_effect=[error, error, error])

        @retry(strategy=strategy, max_attempts=3)
        def test_function():
            return mock_function()

        with pytest.raises(RetryError) as excinfo:
            test_function()

        assert "Failed after 3 attempts" in str(excinfo.value)
        assert mock_function.call_count == 3
        assert strategy.should_retry.call_count == 2
        assert strategy.get_delay.call_count == 2

    def test_specific_exceptions(self) -> None:
        """Test retrying only on specific exceptions."""
        strategy = MagicMock(spec=RetryStrategy)
        strategy.should_retry.return_value = True
        strategy.get_delay.return_value = 0.0  # No delay for testing

        value_error = ValueError("Value error")
        type_error = TypeError("Type error")
        mock_function = MagicMock(side_effect=[value_error, type_error, "success"])

        @retry(strategy=strategy, exceptions=ValueError, max_attempts=3)
        def test_function():
            return mock_function()

        # Should retry on ValueError but not on TypeError
        with pytest.raises(TypeError) as excinfo:
            test_function()

        assert str(excinfo.value) == "Type error"
        assert mock_function.call_count == 2
        strategy.should_retry.assert_called_once()
        strategy.get_delay.assert_called_once()

    def test_on_retry_callback(self) -> None:
        """Test the on_retry callback."""
        strategy = MagicMock(spec=RetryStrategy)
        strategy.should_retry.return_value = True
        strategy.get_delay.return_value = 0.0  # No delay for testing

        error = ValueError("Test error")
        mock_function = MagicMock(side_effect=[error, "success"])
        on_retry_mock = MagicMock()

        @retry(strategy=strategy, max_attempts=3, on_retry=on_retry_mock)
        def test_function():
            return mock_function()

        result = test_function()
        assert result == "success"
        on_retry_mock.assert_called_once_with(1, error)

    def test_should_not_retry(self) -> None:
        """Test not retrying when strategy says not to."""
        strategy = MagicMock(spec=RetryStrategy)
        strategy.should_retry.return_value = False

        error = ValueError("Test error")
        mock_function = MagicMock(side_effect=[error, "success"])

        @retry(strategy=strategy, max_attempts=3)
        def test_function():
            return mock_function()

        with pytest.raises(RetryError) as excinfo:
            test_function()

        # The actual error message may vary, but it should contain the original error
        assert "Test error" in str(excinfo.value)
        assert mock_function.call_count == 1
        strategy.should_retry.assert_called_once()
        strategy.get_delay.assert_not_called()
