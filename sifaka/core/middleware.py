"""Middleware system for adding cross-cutting functionality to Sifaka.

This module provides a flexible middleware pipeline that allows you to
add functionality like logging, metrics, caching, and rate limiting
without modifying core Sifaka code.

## Middleware Pattern:

Middleware components form a chain where each component can:
1. Process the request before passing it on
2. Modify the request or context
3. Handle the response after processing
4. Short-circuit the chain if needed

## Built-in Middleware:

- **LoggingMiddleware**: Logs all improvement operations
- **MetricsMiddleware**: Collects performance metrics
- **CachingMiddleware**: Caches improvement results
- **RateLimitingMiddleware**: Limits request rate
- **TimeoutMiddleware**: Enforces time limits

## Usage:

    >>> from sifaka import improve, MiddlewarePipeline
    >>> from sifaka.core.middleware import LoggingMiddleware, MetricsMiddleware
    >>>
    >>> # Create middleware pipeline
    >>> pipeline = MiddlewarePipeline([
    ...     LoggingMiddleware(log_level="DEBUG"),
    ...     MetricsMiddleware(),
    ... ])
    >>>
    >>> # Use with improve()
    >>> result = await improve(
    ...     "text to improve",
    ...     middleware=pipeline
    ... )
    >>>
    >>> # Access metrics
    >>> metrics = pipeline.get_middleware(MetricsMiddleware)
    >>> print(metrics.get_metrics())

## Custom Middleware:

    >>> class MyMiddleware(Middleware):
    ...     async def process(self, text, next_handler, context):
    ...         # Pre-processing
    ...         print(f"Processing: {text[:50]}...")
    ...
    ...         # Call next in chain
    ...         result = await next_handler(text)
    ...
    ...         # Post-processing
    ...         print(f"Completed with {result.iteration} iterations")
    ...
    ...         return result
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, cast

from .models import SifakaResult
from .type_defs import MiddlewareContext

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """Abstract base class for all middleware components.

    Middleware allows you to intercept and modify the improvement process
    without changing core Sifaka logic. Each middleware can inspect or
    modify the request, add to the context, and process the response.

    The middleware pattern enables:
    - Logging and debugging
    - Performance monitoring
    - Caching and optimization
    - Security and rate limiting
    - Request/response transformation

    Example:
        >>> class TimingMiddleware(Middleware):
        ...     async def process(self, text, next_handler, context):
        ...         start = time.time()
        ...         result = await next_handler(text)
        ...         elapsed = time.time() - start
        ...         print(f"Processing took {elapsed:.2f} seconds")
        ...         return result
    """

    @abstractmethod
    async def process(
        self, text: str, next_handler: Callable[[str], Any], context: MiddlewareContext
    ) -> SifakaResult:
        """Process the request through this middleware.

        This is the main method that each middleware must implement. It
        receives the request, can perform pre-processing, must call the
        next handler, and can perform post-processing.

        Args:
            text: The input text to be improved. Middleware can modify
                this before passing it to the next handler.
            next_handler: Async callable representing the next middleware
                in the chain or the final improve handler. Must be called
                to continue processing.
            context: Mutable dictionary shared between all middleware in
                the pipeline. Used to pass data between middleware components.
                Common keys include 'critics', 'validators', 'config'.

        Returns:
            SifakaResult from the improvement process. Middleware can
            inspect or modify this before returning.

        Raises:
            Any exception from the improvement process. Middleware can
            catch and handle exceptions or let them propagate.

        Example:
            >>> async def process(self, text, next_handler, context):
            ...     # Pre-processing
            ...     context['start_time'] = time.time()
            ...
            ...     # Must call next handler
            ...     result = await next_handler(text)
            ...
            ...     # Post-processing
            ...     context['duration'] = time.time() - context['start_time']
            ...
            ...     return result
        """


class LoggingMiddleware(Middleware):
    """Middleware that logs all improvement operations.

    Provides detailed logging of the improvement process including:
    - Input text (truncated)
    - Configuration (critics, validators)
    - Processing time
    - Results (iterations, confidence, success)
    - Errors with full context

    Useful for debugging, monitoring, and understanding how
    improvements are performed.

    Example:
        >>> # Basic usage
        >>> middleware = LoggingMiddleware()
        >>>
        >>> # With custom log level
        >>> middleware = LoggingMiddleware(log_level="DEBUG")
        >>>
        >>> # In pipeline
        >>> pipeline = MiddlewarePipeline([LoggingMiddleware()])
        >>> result = await improve(text, middleware=pipeline)
    """

    def __init__(self, log_level: str = "INFO"):
        """Initialize logging middleware with specified level.

        Args:
            log_level: Logging level as string. Valid values are:
                - "DEBUG": Detailed information for debugging
                - "INFO": General informational messages (default)
                - "WARNING": Warning messages
                - "ERROR": Error messages only
                Case-insensitive.
        """
        self.log_level = getattr(logging, log_level.upper())

    async def process(
        self, text: str, next_handler: Callable[[str], Any], context: MiddlewareContext
    ) -> SifakaResult:
        """Log the improvement process."""
        start_time = time.time()

        logger.log(self.log_level, f"Starting improvement for text: {text[:100]}...")
        validators = context.get("validators", [])
        validator_count = len(validators) if validators is not None else 0
        logger.log(
            self.log_level,
            f"Context: critics={context.get('critics', [])}, "
            f"validators={validator_count}",
        )

        try:
            result = await next_handler(text)

            elapsed = time.time() - start_time
            logger.log(self.log_level, f"Improvement completed in {elapsed:.2f}s")
            # Get latest confidence from critiques
            latest_confidence = 0.0
            for critique in result.critiques:
                if critique.confidence is not None:
                    latest_confidence = critique.confidence
            logger.log(
                self.log_level,
                f"Result: iterations={result.iteration}, "
                f"confidence={latest_confidence:.2f}, "
                f"improved={not result.needs_improvement}",
            )

            return cast(SifakaResult, result)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Improvement failed after {elapsed:.2f}s: {type(e).__name__}: {e!s}"
            )
            raise


class MetricsMiddleware(Middleware):
    """Middleware that collects detailed metrics about improvements.

    Tracks comprehensive metrics including:
    - Total requests and success rate
    - Processing time statistics
    - Iteration counts and patterns
    - Confidence score trends
    - Token usage and costs
    - Error rates and types

    Metrics are accumulated across all requests and can be retrieved
    for analysis or monitoring dashboards.

    Example:
        >>> metrics_mw = MetricsMiddleware()
        >>> pipeline = MiddlewarePipeline([metrics_mw])
        >>>
        >>> # Process some requests
        >>> for text in texts:
        ...     await improve(text, middleware=pipeline)
        >>>
        >>> # Get metrics
        >>> stats = metrics_mw.get_metrics()
        >>> print(f"Average time: {stats['average_time']:.2f}s")
        >>> print(f"Total tokens: {stats['tokens_used']}")
    """

    def __init__(self) -> None:
        """Initialize metrics collection with zero counters.

        Creates a metrics dictionary that tracks:
        - total_requests: Number of improve() calls
        - total_iterations: Sum of all iterations across requests
        - total_time: Cumulative processing time
        - average_confidence: Running average of final confidence
        - errors: Count of failed requests
        - llm_calls: Total LLM API calls made
        - tokens_used: Total tokens consumed
        """
        self.metrics = {
            "total_requests": 0,
            "total_iterations": 0,
            "total_time": 0.0,
            "average_confidence": 0.0,
            "errors": 0,
            "llm_calls": 0,
            "tokens_used": 0,
        }

    async def process(
        self, text: str, next_handler: Callable[[str], Any], context: MiddlewareContext
    ) -> SifakaResult:
        """Collect metrics about the improvement."""
        start_time = time.time()
        self.metrics["total_requests"] += 1

        try:
            # Track LLM calls via context
            initial_llm_calls = context.get("llm_calls", 0)

            result = await next_handler(text)

            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["total_time"] += elapsed
            self.metrics["total_iterations"] += result.iteration

            # Update average confidence from critiques
            total = self.metrics["total_requests"]
            old_avg = self.metrics["average_confidence"]
            # Get latest confidence from critiques
            latest_confidence = 0.0
            for critique in result.critiques:
                if critique.confidence is not None:
                    latest_confidence = critique.confidence
            self.metrics["average_confidence"] = (
                old_avg * (total - 1) + latest_confidence
            ) / total

            # Track LLM calls
            final_llm_calls = context.get("llm_calls", 0)
            if isinstance(final_llm_calls, (int, float)) and isinstance(
                initial_llm_calls, (int, float)
            ):
                self.metrics["llm_calls"] += int(final_llm_calls - initial_llm_calls)

            # Track tokens
            for gen in result.generations:
                self.metrics["tokens_used"] += gen.tokens_used

            return cast(SifakaResult, result)

        except Exception:
            self.metrics["errors"] += 1
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = self.metrics.copy()

        # Calculate averages
        if metrics["total_requests"] > 0:
            metrics["avg_time_per_request"] = (
                metrics["total_time"] / metrics["total_requests"]
            )
            metrics["avg_iterations_per_request"] = (
                metrics["total_iterations"] / metrics["total_requests"]
            )
            metrics["avg_llm_calls_per_request"] = (
                metrics["llm_calls"] / metrics["total_requests"]
            )

        return metrics


class CachingMiddleware(Middleware):
    """Caches improvement results for identical inputs."""

    def __init__(self, max_size: int = 100):
        """Initialize caching middleware.

        Args:
            max_size: Maximum number of cached results
        """
        self.cache: Dict[str, SifakaResult] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _get_cache_key(self, text: str, context: MiddlewareContext) -> str:
        """Generate cache key from text and context."""
        critics_list = context.get("critics", [])
        critics = ",".join(sorted(str(c) for c in critics_list))
        validators_list = context.get("validators", [])
        validators = len(validators_list) if validators_list is not None else 0
        config_key = (
            f"{context.get('model', 'default')}_{context.get('temperature', 0.7)}"
        )

        return f"{hash(text)}_{critics}_{validators}_{config_key}"

    async def process(
        self, text: str, next_handler: Callable[[str], Any], context: MiddlewareContext
    ) -> SifakaResult:
        """Check cache before processing."""
        cache_key = self._get_cache_key(text, context)

        # Check cache
        if cache_key in self.cache:
            self.hits += 1
            logger.debug(f"Cache hit for key: {cache_key}")
            return self.cache[cache_key]

        # Cache miss
        self.misses += 1
        result = cast(SifakaResult, await next_handler(text))

        # Store in cache
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = result

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
        }


class RateLimitingMiddleware(Middleware):
    """Rate limits improvement requests."""

    def __init__(self, max_requests_per_minute: int = 60):
        """Initialize rate limiting.

        Args:
            max_requests_per_minute: Maximum requests allowed per minute
        """
        self.max_requests = max_requests_per_minute
        self.requests: List[float] = []

    async def process(
        self, text: str, next_handler: Callable[[str], Any], context: MiddlewareContext
    ) -> SifakaResult:
        """Check rate limit before processing."""
        now = time.time()

        # Remove old requests
        self.requests = [t for t in self.requests if now - t < 60]

        # Check rate limit
        if len(self.requests) >= self.max_requests:
            wait_time = 60 - (now - self.requests[0])
            raise RuntimeError(
                f"Rate limit exceeded. Try again in {wait_time:.1f} seconds."
            )

        # Add current request
        self.requests.append(now)

        return cast(SifakaResult, await next_handler(text))


class MiddlewarePipeline:
    """Manages the middleware pipeline."""

    def __init__(self) -> None:
        """Initialize empty pipeline."""
        self.middleware: List[Middleware] = []

    def add(self, middleware: Middleware) -> "MiddlewarePipeline":
        """Add middleware to the pipeline.

        Args:
            middleware: Middleware instance to add

        Returns:
            Self for chaining
        """
        self.middleware.append(middleware)
        return self

    async def execute(
        self,
        text: str,
        final_handler: Callable[[str], Any],
        context: Optional[MiddlewareContext] = None,
    ) -> SifakaResult:
        """Execute the middleware pipeline.

        Args:
            text: Input text
            final_handler: The actual improvement function
            context: Shared context between middleware

        Returns:
            SifakaResult from the pipeline
        """
        if context is None:
            context = {}

        # Build the chain
        async def chain(index: int, current_text: str) -> SifakaResult:
            if index >= len(self.middleware):
                # End of middleware chain, call final handler
                return cast(SifakaResult, await final_handler(current_text))

            # Call current middleware
            current = self.middleware[index]
            return await current.process(
                current_text, lambda t: chain(index + 1, t), context
            )

        return await chain(0, text)


@asynccontextmanager
async def monitor(
    include_logging: bool = True, include_metrics: bool = True, log_level: str = "INFO"
) -> AsyncIterator[Dict[str, Any]]:
    """Context manager for monitoring improvements.

    Args:
        include_logging: Whether to include logging middleware
        include_metrics: Whether to include metrics middleware
        log_level: Logging level

    Yields:
        Dictionary with pipeline and metrics
    """
    pipeline = MiddlewarePipeline()
    metrics_middleware = None

    if include_logging:
        pipeline.add(LoggingMiddleware(log_level))

    if include_metrics:
        metrics_middleware = MetricsMiddleware()
        pipeline.add(metrics_middleware)

    data = {"pipeline": pipeline, "metrics": metrics_middleware}

    yield data

    # After completion, could log final metrics
    if metrics_middleware:
        final_metrics = metrics_middleware.get_metrics()
        logger.info(f"Session metrics: {final_metrics}")
