"""Tests for enhanced error recovery features in Sifaka."""

import pytest
import time
from unittest.mock import Mock, patch

from sifaka.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerError,
)
from sifaka.utils.retry import (
    RetryConfig,
    RetryManager,
    BackoffStrategy,
    RetryError,
    retry_with_backoff,
)
from sifaka.utils.fallback import (
    FallbackChain,
    FallbackConfig,
    FallbackStrategy,
    FallbackError,
)
from sifaka.utils.error_handling import (
    CircuitBreakerError as ErrorHandlingCircuitBreakerError,
    RetryExhaustedError,
    FallbackError as ErrorHandlingFallbackError,
    create_actionable_suggestions,
    enhance_error_message,
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test-service", config)

        assert breaker.state == CircuitBreakerState.CLOSED

        # Successful calls should keep it closed
        with breaker.protect_call():
            pass

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.stats.successful_requests == 1

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test-service", config)

        # Cause failures
        for i in range(2):
            with pytest.raises(ValueError):
                with breaker.protect_call():
                    raise ValueError(f"Test failure {i}")

        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.stats.failed_requests == 2

    def test_circuit_breaker_rejects_when_open(self):
        """Test circuit breaker rejects calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=10.0)
        breaker = CircuitBreaker("test-service", config)

        # Cause failure to open circuit
        with pytest.raises(ValueError):
            with breaker.protect_call():
                raise ValueError("Test failure")

        assert breaker.state == CircuitBreakerState.OPEN

        # Next call should be rejected
        with pytest.raises(CircuitBreakerError):
            with breaker.protect_call():
                pass

        assert breaker.stats.rejected_requests == 1

    def test_circuit_breaker_half_open_transition(self):
        """Test circuit breaker transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = CircuitBreaker("test-service", config)

        # Open the circuit
        with pytest.raises(ValueError):
            with breaker.protect_call():
                raise ValueError("Test failure")

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(0.2)

        # Next call should transition to half-open
        with breaker.protect_call():
            pass

        assert breaker.state == CircuitBreakerState.CLOSED  # Should close on success


class TestRetryManager:
    """Test retry mechanism functionality."""

    def test_retry_success_on_first_attempt(self):
        """Test successful execution on first attempt."""
        config = RetryConfig(max_attempts=3)
        manager = RetryManager(config)

        def success_func():
            return "success"

        result = manager.execute(success_func)
        assert result == "success"
        assert manager.stats.total_attempts == 1
        assert manager.stats.successful_attempts == 1

    def test_retry_success_after_failures(self):
        """Test successful execution after some failures."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        manager = RetryManager(config)

        attempt_count = 0

        def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError(f"Attempt {attempt_count} failed")
            return "success"

        result = manager.execute(flaky_func)
        assert result == "success"
        assert manager.stats.total_attempts == 3
        assert manager.stats.successful_attempts == 1
        assert manager.stats.failed_attempts == 2

    def test_retry_exhausted(self):
        """Test retry exhaustion when all attempts fail."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        manager = RetryManager(config)

        def always_fail():
            raise ConnectionError("Always fails")

        with pytest.raises(RetryError) as exc_info:
            manager.execute(always_fail)

        assert manager.stats.total_attempts == 2
        assert manager.stats.failed_attempts == 2
        assert isinstance(exc_info.value.original_exception, ConnectionError)

    def test_retry_backoff_strategies(self):
        """Test different backoff strategies."""
        strategies = [
            BackoffStrategy.EXPONENTIAL,
            BackoffStrategy.LINEAR,
            BackoffStrategy.FIXED,
            BackoffStrategy.FIBONACCI,
        ]

        for strategy in strategies:
            config = RetryConfig(
                max_attempts=3,
                base_delay=0.1,
                backoff_strategy=strategy,
                jitter=False,  # Disable jitter for predictable testing
            )
            manager = RetryManager(config)

            # Test delay calculation
            delay1 = manager._calculate_delay(0)
            delay2 = manager._calculate_delay(1)
            delay3 = manager._calculate_delay(2)

            assert delay1 >= 0
            assert delay2 >= 0
            assert delay3 >= 0

            if strategy == BackoffStrategy.EXPONENTIAL:
                assert delay2 > delay1
                assert delay3 > delay2
            elif strategy == BackoffStrategy.FIXED:
                assert delay1 == delay2 == delay3

    def test_retry_decorator(self):
        """Test retry decorator functionality."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)

        attempt_count = 0

        @retry_with_backoff(config)
        def decorated_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = decorated_func()
        assert result == "success"
        assert attempt_count == 2


class TestFallbackChain:
    """Test fallback chain functionality."""

    def test_fallback_primary_success(self):
        """Test successful execution with primary service."""
        config = FallbackConfig()
        chain = FallbackChain("test-service", config)

        def primary_func(query):
            return f"primary: {query}"

        chain.add_primary(primary_func)

        result = chain.execute("test")
        assert result == "primary: test"
        assert chain.stats.successful_calls == 1

    def test_fallback_to_secondary(self):
        """Test fallback to secondary service when primary fails."""
        config = FallbackConfig()
        chain = FallbackChain("test-service", config)

        def primary_func(query):
            raise ConnectionError("Primary failed")

        def fallback_func(query):
            return f"fallback: {query}"

        chain.add_primary(primary_func)
        chain.add_fallback(fallback_func, priority=1)

        result = chain.execute("test")
        assert result == "fallback: test"
        assert chain.stats.successful_calls == 1

    def test_fallback_all_fail(self):
        """Test when all fallback options fail."""
        config = FallbackConfig()
        chain = FallbackChain("test-service", config)

        def primary_func(query):
            raise ConnectionError("Primary failed")

        def fallback_func(query):
            raise ConnectionError("Fallback failed")

        chain.add_primary(primary_func)
        chain.add_fallback(fallback_func, priority=1)

        with pytest.raises(FallbackError) as exc_info:
            chain.execute("test")

        assert "Primary failed" in str(exc_info.value.last_exception) or "Fallback failed" in str(
            exc_info.value.last_exception
        )
        assert len(exc_info.value.attempted_options) == 2

    def test_fallback_strategies(self):
        """Test different fallback strategies."""
        strategies = [
            FallbackStrategy.SEQUENTIAL,
            FallbackStrategy.PRIORITY,
            FallbackStrategy.ROUND_ROBIN,
            FallbackStrategy.FASTEST_FIRST,
        ]

        for strategy in strategies:
            config = FallbackConfig(strategy=strategy)
            chain = FallbackChain("test-service", config)

            def primary_func(query):
                return f"primary: {query}"

            def fallback1_func(query):
                return f"fallback1: {query}"

            def fallback2_func(query):
                return f"fallback2: {query}"

            chain.add_primary(primary_func)
            chain.add_fallback(fallback1_func, priority=1)
            chain.add_fallback(fallback2_func, priority=2)

            # Test execution order
            execution_order = chain._get_execution_order()
            assert len(execution_order) >= 1
            assert execution_order[0].name == "primary"


class TestErrorHandling:
    """Test enhanced error handling utilities."""

    def test_create_actionable_suggestions(self):
        """Test creation of actionable suggestions."""
        # Test connection error
        conn_error = ConnectionError("Connection failed")
        suggestions = create_actionable_suggestions(conn_error)
        assert any("network" in s.lower() for s in suggestions)

        # Test timeout error
        timeout_error = TimeoutError("Request timed out")
        suggestions = create_actionable_suggestions(timeout_error)
        assert any("timeout" in s.lower() for s in suggestions)

        # Test with component context
        suggestions = create_actionable_suggestions(conn_error, component="model")
        assert any("model" in s.lower() for s in suggestions)

    def test_enhance_error_message(self):
        """Test error message enhancement."""
        # Test with connection error to get suggestions
        error = ConnectionError("Connection failed")

        # Test basic enhancement
        enhanced = enhance_error_message(error)
        assert "Connection failed" in enhanced
        assert "Suggestions:" in enhanced  # Connection errors should get suggestions

        # Test with component and operation
        enhanced = enhance_error_message(error, component="TestComponent", operation="test_op")
        assert "[TestComponent]" in enhanced
        assert "(during test_op)" in enhanced
        assert "Suggestions:" in enhanced

    def test_error_context_managers(self):
        """Test specialized error context managers."""
        from sifaka.utils.error_handling import (
            circuit_breaker_context,
            retry_context,
            fallback_context,
        )

        # Test circuit breaker context
        with pytest.raises(ErrorHandlingCircuitBreakerError):
            with circuit_breaker_context("test-service"):
                raise ConnectionError("Service down")

        # Test retry context
        with pytest.raises(RetryExhaustedError):
            with retry_context(max_attempts=3):
                raise ConnectionError("Retry failed")

        # Test fallback context
        with pytest.raises(ErrorHandlingFallbackError):
            with fallback_context("test-service", attempted_fallbacks=["fallback1"]):
                raise ConnectionError("All fallbacks failed")


class TestIntegration:
    """Test integration of error recovery components."""

    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker combined with retry mechanism."""
        cb_config = CircuitBreakerConfig(failure_threshold=5)  # Higher threshold to allow retries
        breaker = CircuitBreaker("test-service", cb_config)

        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)

        attempt_count = 0

        @retry_with_backoff(retry_config)
        def protected_func():
            nonlocal attempt_count
            attempt_count += 1
            with breaker.protect_call():
                if attempt_count < 3:  # Fail first 2 attempts, succeed on 3rd
                    raise ConnectionError(f"Attempt {attempt_count} failed")
                return "success"

        # This should eventually succeed after retries
        result = protected_func()
        assert result == "success"
        assert attempt_count == 3

    def test_fallback_with_circuit_breaker(self):
        """Test fallback chain with circuit breaker protection."""
        config = FallbackConfig()
        chain = FallbackChain("test-service", config)

        cb_config = CircuitBreakerConfig(failure_threshold=1)
        primary_breaker = CircuitBreaker("primary", cb_config)

        def protected_primary(query):
            with primary_breaker.protect_call():
                raise ConnectionError("Primary always fails")

        def reliable_fallback(query):
            return f"fallback: {query}"

        chain.add_primary(protected_primary)
        chain.add_fallback(reliable_fallback, priority=1)

        # First call should fail primary and use fallback
        result = chain.execute("test")
        assert result == "fallback: test"

        # Circuit should be open now, so next call should go directly to fallback
        result = chain.execute("test2")
        assert result == "fallback: test2"


if __name__ == "__main__":
    pytest.main([__file__])
