"""Circuit breaker pattern implementation for Sifaka.

This module provides circuit breaker functionality to handle failures in external
services gracefully. Circuit breakers prevent cascading failures by temporarily
stopping calls to failing services and allowing them time to recover.

The circuit breaker has three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests are rejected immediately
- HALF_OPEN: Testing if service has recovered

Example:
    ```python
    from sifaka.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

    # Create circuit breaker
    config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=30,
        expected_exception=ConnectionError
    )
    breaker = CircuitBreaker("redis-service", config)

    # Use with function
    @breaker.protect
    def call_redis():
        # Your Redis call here
        pass

    # Or use as context manager
    with breaker:
        # Your external service call here
        pass
    ```
"""

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union
import logging

from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 30.0  # Seconds to wait before trying again
    expected_exception: Type[Exception] = Exception  # Exception type to catch
    success_threshold: int = 1  # Successes needed to close from half-open
    timeout: float = 10.0  # Timeout for individual calls

    # Advanced configuration
    failure_rate_threshold: float = 0.5  # Failure rate to trigger opening
    minimum_requests: int = 10  # Minimum requests before considering failure rate
    sliding_window_size: int = 100  # Size of sliding window for failure tracking


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: List[str] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, service_name: str, state: CircuitBreakerState, stats: CircuitBreakerStats):
        self.service_name = service_name
        self.state = state
        self.stats = stats

        message = (
            f"Circuit breaker for '{service_name}' is {state.value}. "
            f"Failure rate: {stats.failure_rate:.1%}, "
            f"Total requests: {stats.total_requests}, "
            f"Failed: {stats.failed_requests}"
        )
        super().__init__(message)


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""

    def __init__(self, service_name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.

        Args:
            service_name: Name of the service being protected.
            config: Circuit breaker configuration.
        """
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.stats = CircuitBreakerStats()
        self.last_state_change = time.time()
        self._lock = asyncio.Lock() if asyncio.iscoroutinefunction else None  # type: ignore[truthy-function]

        # Sliding window for failure tracking
        self._failure_window: List[bool] = []  # True for failure, False for success

        logger.info(f"Initialized circuit breaker for {service_name}")

    def _record_success(self) -> None:
        """Record a successful call."""
        self.stats.successful_requests += 1
        self.stats.total_requests += 1
        self.stats.last_success_time = time.time()

        # Add to sliding window
        self._failure_window.append(False)
        if len(self._failure_window) > self.config.sliding_window_size:
            self._failure_window.pop(0)

        # Check if we should close from half-open
        if self.state == CircuitBreakerState.HALF_OPEN:
            consecutive_successes = 0
            for i in range(len(self._failure_window) - 1, -1, -1):
                if not self._failure_window[i]:
                    consecutive_successes += 1
                else:
                    break

            if consecutive_successes >= self.config.success_threshold:
                self._transition_to_closed()

    def _record_failure(self) -> None:
        """Record a failed call."""
        self.stats.failed_requests += 1
        self.stats.total_requests += 1
        self.stats.last_failure_time = time.time()

        # Add to sliding window
        self._failure_window.append(True)
        if len(self._failure_window) > self.config.sliding_window_size:
            self._failure_window.pop(0)

        # Check if we should open the circuit
        if self.state == CircuitBreakerState.CLOSED:
            should_open = False

            # Check failure threshold
            if self.stats.failed_requests >= self.config.failure_threshold:
                should_open = True

            # Check failure rate if we have enough requests
            if (
                self.stats.total_requests >= self.config.minimum_requests
                and self.stats.failure_rate >= self.config.failure_rate_threshold
            ):
                should_open = True

            if should_open:
                self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        self.state = CircuitBreakerState.OPEN
        self.last_state_change = time.time()
        self.stats.state_changes.append(f"OPEN at {time.time()}")
        logger.warning(f"Circuit breaker for {self.service_name} opened due to failures")

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.last_state_change = time.time()
        self.stats.state_changes.append(f"HALF_OPEN at {time.time()}")
        logger.info(f"Circuit breaker for {self.service_name} transitioning to half-open")

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.last_state_change = time.time()
        self.stats.state_changes.append(f"CLOSED at {time.time()}")
        logger.info(f"Circuit breaker for {self.service_name} closed - service recovered")

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_state_change >= self.config.recovery_timeout:
                self._transition_to_half_open()
                return True
            return False

        if self.state == CircuitBreakerState.HALF_OPEN:
            return True

        return False  # type: ignore[unreachable]

    @contextmanager
    def protect_call(self) -> Any:
        """Context manager for protecting calls."""
        if not self._should_allow_request():
            self.stats.rejected_requests += 1
            raise CircuitBreakerError(self.service_name, self.state, self.stats)

        try:
            yield
            self._record_success()
        except self.config.expected_exception as e:
            self._record_failure()
            raise
        except Exception as e:
            # Unexpected exception - still record as failure but re-raise
            self._record_failure()
            raise

    def __enter__(self) -> Any:
        """Enter context manager."""
        return self.protect_call().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        """Exit context manager."""
        return self.protect_call().__exit__(exc_type, exc_val, exc_tb)

    def protect(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator for protecting functions."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.protect_call():
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.protect_call():
                return await func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    def get_stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return self.stats

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitBreakerState.CLOSED
        self.stats = CircuitBreakerStats()
        self.last_state_change = time.time()
        self._failure_window.clear()
        logger.info(f"Circuit breaker for {self.service_name} reset")


# Global registry for circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    service_name: str, config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create a circuit breaker for a service.

    Args:
        service_name: Name of the service.
        config: Optional configuration for new circuit breakers.

    Returns:
        Circuit breaker instance.
    """
    if service_name not in _circuit_breakers:
        _circuit_breakers[service_name] = CircuitBreaker(service_name, config)
    return _circuit_breakers[service_name]


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers."""
    for breaker in _circuit_breakers.values():
        breaker.reset()
    logger.info("All circuit breakers reset")
