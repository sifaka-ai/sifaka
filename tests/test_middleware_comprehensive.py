"""Comprehensive tests for middleware pipeline functionality.

This test suite covers:
- Middleware pipeline construction and execution
- Built-in middleware components (Logging, Metrics, Caching, RateLimit)
- Custom middleware implementation
- Error handling and recovery
- Middleware chaining and order
- Performance characteristics
"""

import pytest
import asyncio
import time
from unittest.mock import patch
from typing import Dict, Any, Callable, Awaitable

from sifaka.core.models import SifakaResult
from sifaka.core.middleware import (
    MiddlewarePipeline,
    LoggingMiddleware,
    MetricsMiddleware,
    CachingMiddleware,
    RateLimitingMiddleware,
    Middleware,
)
from sifaka.core.exceptions import SifakaError


@pytest.fixture
def sample_result():
    """Create a sample SifakaResult for testing."""
    return SifakaResult(
        original_text="Original test text",
        final_text="Improved test text",
        iteration=1,
        generations=[],
        critiques=[],
        validations=[],
        processing_time=1.0,
    )


@pytest.fixture
def sample_handler():
    """Create a sample async handler function."""

    async def handler(text: str) -> SifakaResult:
        return SifakaResult(
            original_text=text,
            final_text=f"Processed: {text}",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=0.5,
        )

    return handler


class TestMiddlewarePipeline:
    """Test the core MiddlewarePipeline functionality."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = MiddlewarePipeline()
        assert len(pipeline.middlewares) == 0

    def test_add_middleware(self):
        """Test adding middleware to pipeline."""
        pipeline = MiddlewarePipeline()
        middleware = LoggingMiddleware()

        pipeline.add(middleware)
        assert len(pipeline.middlewares) == 1
        assert pipeline.middlewares[0] == middleware

    def test_add_multiple_middleware(self):
        """Test adding multiple middleware components."""
        pipeline = MiddlewarePipeline()

        logging_mw = LoggingMiddleware()
        metrics_mw = MetricsMiddleware()
        caching_mw = CachingMiddleware()

        pipeline.add(logging_mw)
        pipeline.add(metrics_mw)
        pipeline.add(caching_mw)

        assert len(pipeline.middlewares) == 3
        assert pipeline.middlewares[0] == logging_mw
        assert pipeline.middlewares[1] == metrics_mw
        assert pipeline.middlewares[2] == caching_mw

    @pytest.mark.asyncio
    async def test_empty_pipeline_execution(self, sample_handler):
        """Test executing empty pipeline."""
        pipeline = MiddlewarePipeline()
        context = {"test": "value"}

        result = await pipeline.execute("test text", sample_handler, context)

        assert isinstance(result, SifakaResult)
        assert result.original_text == "test text"
        assert result.final_text == "Processed: test text"

    @pytest.mark.asyncio
    async def test_single_middleware_execution(self, sample_handler):
        """Test executing pipeline with single middleware."""
        pipeline = MiddlewarePipeline()

        # Create a simple test middleware
        class TestMiddleware(Middleware):
            def __init__(self):
                self.called = False
                self.context_received = None

            async def process(
                self,
                text: str,
                next_handler: Callable[[str], Awaitable[SifakaResult]],
                context: Dict[str, Any],
            ) -> SifakaResult:
                self.called = True
                self.context_received = context.copy()
                # Modify text before passing to next handler
                modified_text = f"[MIDDLEWARE] {text}"
                return await next_handler(modified_text)

        test_middleware = TestMiddleware()
        pipeline.add(test_middleware)

        context = {"middleware_test": True}
        result = await pipeline.execute("original text", sample_handler, context)

        assert test_middleware.called
        assert test_middleware.context_received["middleware_test"] is True
        assert result.original_text == "[MIDDLEWARE] original text"

    @pytest.mark.asyncio
    async def test_middleware_chaining(self, sample_handler):
        """Test middleware chaining and execution order."""
        pipeline = MiddlewarePipeline()

        execution_order = []

        class OrderedMiddleware(Middleware):
            def __init__(self, name: str):
                self.name = name

            async def process(
                self,
                text: str,
                next_handler: Callable[[str], Awaitable[SifakaResult]],
                context: Dict[str, Any],
            ) -> SifakaResult:
                execution_order.append(f"{self.name}_start")
                result = await next_handler(f"[{self.name}] {text}")
                execution_order.append(f"{self.name}_end")
                return result

        # Add middleware in specific order
        pipeline.add(OrderedMiddleware("FIRST"))
        pipeline.add(OrderedMiddleware("SECOND"))
        pipeline.add(OrderedMiddleware("THIRD"))

        result = await pipeline.execute("test", sample_handler, {})

        # Check execution order (should be nested)
        expected_order = [
            "FIRST_start",
            "SECOND_start",
            "THIRD_start",
            "THIRD_end",
            "SECOND_end",
            "FIRST_end",
        ]
        assert execution_order == expected_order

        # Check text transformation
        assert result.original_text == "[THIRD] [SECOND] [FIRST] test"

    @pytest.mark.asyncio
    async def test_middleware_error_handling(self, sample_handler):
        """Test error handling in middleware pipeline."""
        pipeline = MiddlewarePipeline()

        class ErrorMiddleware(Middleware):
            async def process(
                self,
                text: str,
                next_handler: Callable[[str], Awaitable[SifakaResult]],
                context: Dict[str, Any],
            ) -> SifakaResult:
                raise ValueError("Middleware error")

        pipeline.add(ErrorMiddleware())

        with pytest.raises(ValueError, match="Middleware error"):
            await pipeline.execute("test", sample_handler, {})

    @pytest.mark.asyncio
    async def test_handler_error_propagation(self):
        """Test that handler errors propagate through middleware."""
        pipeline = MiddlewarePipeline()

        class LoggingMiddleware(Middleware):
            def __init__(self):
                self.error_caught = False

            async def process(
                self,
                text: str,
                next_handler: Callable[[str], Awaitable[SifakaResult]],
                context: Dict[str, Any],
            ) -> SifakaResult:
                try:
                    return await next_handler(text)
                except Exception:
                    self.error_caught = True
                    raise

        async def error_handler(text: str) -> SifakaResult:
            raise RuntimeError("Handler error")

        logging_mw = LoggingMiddleware()
        pipeline.add(logging_mw)

        with pytest.raises(RuntimeError, match="Handler error"):
            await pipeline.execute("test", error_handler, {})

        assert logging_mw.error_caught


class TestLoggingMiddleware:
    """Test LoggingMiddleware functionality."""

    def test_logging_middleware_initialization(self):
        """Test LoggingMiddleware initialization."""
        middleware = LoggingMiddleware()
        assert middleware.logger is not None

    def test_logging_middleware_custom_logger(self):
        """Test LoggingMiddleware with custom logger."""
        import logging

        custom_logger = logging.getLogger("test_logger")
        middleware = LoggingMiddleware(logger=custom_logger)
        assert middleware.logger == custom_logger

    @pytest.mark.asyncio
    async def test_logging_middleware_execution(self, sample_handler):
        """Test LoggingMiddleware execution and logging."""
        with patch("logging.Logger.info") as mock_info:
            middleware = LoggingMiddleware()

            async def next_handler(text: str) -> SifakaResult:
                return await sample_handler(text)

            context = {"test_context": "value"}
            result = await middleware.process("test text", next_handler, context)

            # Check that logging occurred
            assert mock_info.call_count >= 2  # Start and end logs

            # Check result is passed through
            assert isinstance(result, SifakaResult)
            assert result.original_text == "test text"

    @pytest.mark.asyncio
    async def test_logging_middleware_error_logging(self):
        """Test LoggingMiddleware error logging."""
        with patch("logging.Logger.error") as mock_error:
            middleware = LoggingMiddleware()

            async def error_handler(text: str) -> SifakaResult:
                raise ValueError("Test error")

            with pytest.raises(ValueError):
                await middleware.process("test", error_handler, {})

            # Check that error was logged
            mock_error.assert_called_once()


class TestMetricsMiddleware:
    """Test MetricsMiddleware functionality."""

    def test_metrics_middleware_initialization(self):
        """Test MetricsMiddleware initialization."""
        middleware = MetricsMiddleware()
        assert hasattr(middleware, "metrics")
        assert isinstance(middleware.metrics, dict)

    @pytest.mark.asyncio
    async def test_metrics_collection(self, sample_handler):
        """Test metrics collection during execution."""
        middleware = MetricsMiddleware()

        await middleware.process("test text", sample_handler, {})

        # Check that metrics were collected
        metrics = middleware.get_metrics()
        assert "total_requests" in metrics
        assert "total_processing_time" in metrics
        assert "average_processing_time" in metrics
        assert metrics["total_requests"] == 1
        assert metrics["total_processing_time"] > 0

    @pytest.mark.asyncio
    async def test_metrics_multiple_requests(self, sample_handler):
        """Test metrics with multiple requests."""
        middleware = MetricsMiddleware()

        # Process multiple requests
        for i in range(5):
            await middleware.process(f"test {i}", sample_handler, {})

        metrics = middleware.get_metrics()
        assert metrics["total_requests"] == 5
        assert metrics["total_processing_time"] > 0
        assert metrics["average_processing_time"] > 0

    @pytest.mark.asyncio
    async def test_metrics_error_tracking(self):
        """Test error tracking in metrics."""
        middleware = MetricsMiddleware()

        async def error_handler(text: str) -> SifakaResult:
            raise ValueError("Test error")

        # Process some successful and some failed requests
        await middleware.process("success", sample_handler, {})

        with pytest.raises(ValueError):
            await middleware.process("error", error_handler, {})

        metrics = middleware.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["total_errors"] == 1
        assert metrics["error_rate"] == 0.5

    def test_metrics_reset(self, sample_handler):
        """Test metrics reset functionality."""
        middleware = MetricsMiddleware()

        # Add some metrics
        asyncio.run(middleware.process("test", sample_handler, {}))

        metrics_before = middleware.get_metrics()
        assert metrics_before["total_requests"] == 1

        # Reset metrics
        middleware.reset_metrics()

        metrics_after = middleware.get_metrics()
        assert metrics_after["total_requests"] == 0
        assert metrics_after["total_processing_time"] == 0


class TestCachingMiddleware:
    """Test CachingMiddleware functionality."""

    def test_caching_middleware_initialization(self):
        """Test CachingMiddleware initialization."""
        middleware = CachingMiddleware()
        assert hasattr(middleware, "cache")
        assert middleware.max_size == 100  # Default size

    def test_caching_middleware_custom_size(self):
        """Test CachingMiddleware with custom cache size."""
        middleware = CachingMiddleware(max_size=50)
        assert middleware.max_size == 50

    @pytest.mark.asyncio
    async def test_cache_hit_and_miss(self, sample_handler):
        """Test cache hit and miss scenarios."""
        middleware = CachingMiddleware()

        call_count = 0

        async def counting_handler(text: str) -> SifakaResult:
            nonlocal call_count
            call_count += 1
            return await sample_handler(text)

        # First call - cache miss
        result1 = await middleware.process("test text", counting_handler, {})
        assert call_count == 1

        # Second call with same text - cache hit
        result2 = await middleware.process("test text", counting_handler, {})
        assert call_count == 1  # Handler not called again

        # Results should be equivalent
        assert result1.original_text == result2.original_text
        assert result1.final_text == result2.final_text

    @pytest.mark.asyncio
    async def test_cache_different_inputs(self, sample_handler):
        """Test caching with different inputs."""
        middleware = CachingMiddleware()

        call_count = 0

        async def counting_handler(text: str) -> SifakaResult:
            nonlocal call_count
            call_count += 1
            return await sample_handler(text)

        # Different inputs should not hit cache
        await middleware.process("text 1", counting_handler, {})
        await middleware.process("text 2", counting_handler, {})
        await middleware.process("text 1", counting_handler, {})  # Should hit cache

        assert call_count == 2  # Only 2 unique inputs

    @pytest.mark.asyncio
    async def test_cache_size_limit(self, sample_handler):
        """Test cache size limiting."""
        middleware = CachingMiddleware(max_size=2)

        # Fill cache beyond limit
        await middleware.process("text 1", sample_handler, {})
        await middleware.process("text 2", sample_handler, {})
        await middleware.process("text 3", sample_handler, {})  # Should evict oldest

        # Cache should not exceed max size
        assert len(middleware.cache) <= 2

    def test_cache_key_generation(self):
        """Test cache key generation."""
        middleware = CachingMiddleware()

        key1 = middleware._generate_cache_key("test", {"a": 1})
        key2 = middleware._generate_cache_key("test", {"a": 1})
        key3 = middleware._generate_cache_key("test", {"a": 2})

        assert key1 == key2  # Same input should generate same key
        assert key1 != key3  # Different context should generate different key

    def test_cache_clear(self, sample_handler):
        """Test cache clearing."""
        middleware = CachingMiddleware()

        # Add items to cache
        asyncio.run(middleware.process("test 1", sample_handler, {}))
        asyncio.run(middleware.process("test 2", sample_handler, {}))

        assert len(middleware.cache) == 2

        # Clear cache
        middleware.clear_cache()
        assert len(middleware.cache) == 0


class TestRateLimitingMiddleware:
    """Test RateLimitingMiddleware functionality."""

    def test_rate_limiting_middleware_initialization(self):
        """Test RateLimitingMiddleware initialization."""
        middleware = RateLimitingMiddleware(max_requests=10, time_window=60)
        assert middleware.max_requests == 10
        assert middleware.time_window == 60

    @pytest.mark.asyncio
    async def test_rate_limiting_within_limit(self, sample_handler):
        """Test requests within rate limit."""
        middleware = RateLimitingMiddleware(max_requests=5, time_window=60)

        # Make requests within limit
        for i in range(3):
            result = await middleware.process(f"test {i}", sample_handler, {})
            assert isinstance(result, SifakaResult)

    @pytest.mark.asyncio
    async def test_rate_limiting_exceeded(self, sample_handler):
        """Test rate limiting when limit is exceeded."""
        middleware = RateLimitingMiddleware(max_requests=2, time_window=60)

        # Make requests up to limit
        await middleware.process("test 1", sample_handler, {})
        await middleware.process("test 2", sample_handler, {})

        # Next request should be rate limited
        with pytest.raises(SifakaError, match="Rate limit exceeded"):
            await middleware.process("test 3", sample_handler, {})

    @pytest.mark.asyncio
    async def test_rate_limiting_time_window_reset(self, sample_handler):
        """Test rate limiting time window reset."""
        middleware = RateLimitingMiddleware(
            max_requests=1, time_window=0.1
        )  # 100ms window

        # Make first request
        await middleware.process("test 1", sample_handler, {})

        # Wait for time window to pass
        await asyncio.sleep(0.2)

        # Should be able to make another request
        result = await middleware.process("test 2", sample_handler, {})
        assert isinstance(result, SifakaResult)

    def test_rate_limiting_request_tracking(self):
        """Test request tracking in rate limiter."""
        middleware = RateLimitingMiddleware(max_requests=5, time_window=60)

        # Check initial state
        assert len(middleware.request_times) == 0

        # Simulate requests
        current_time = time.time()
        for i in range(3):
            middleware.request_times.append(current_time + i)

        # Check tracking
        assert len(middleware.request_times) == 3


class TestCustomMiddleware:
    """Test custom middleware implementation."""

    @pytest.mark.asyncio
    async def test_custom_middleware_implementation(self, sample_handler):
        """Test implementing custom middleware."""

        class CustomMiddleware(Middleware):
            def __init__(self):
                self.processed_texts = []

            async def process(
                self,
                text: str,
                next_handler: Callable[[str], Awaitable[SifakaResult]],
                context: Dict[str, Any],
            ) -> SifakaResult:
                # Custom processing logic
                self.processed_texts.append(text)

                # Modify context
                context["custom_middleware"] = True

                # Transform text
                transformed_text = text.upper()

                # Call next handler
                result = await next_handler(transformed_text)

                # Post-process result
                result.final_text = f"CUSTOM: {result.final_text}"

                return result

        custom_mw = CustomMiddleware()
        pipeline = MiddlewarePipeline()
        pipeline.add(custom_mw)

        context = {}
        result = await pipeline.execute("test text", sample_handler, context)

        # Check custom middleware effects
        assert "test text" in custom_mw.processed_texts
        assert context["custom_middleware"] is True
        assert result.original_text == "TEST TEXT"
        assert result.final_text.startswith("CUSTOM:")

    @pytest.mark.asyncio
    async def test_middleware_context_modification(self, sample_handler):
        """Test middleware modifying context for downstream middleware."""

        class ContextModifierMiddleware(Middleware):
            async def process(
                self,
                text: str,
                next_handler: Callable[[str], Awaitable[SifakaResult]],
                context: Dict[str, Any],
            ) -> SifakaResult:
                context["modified_by"] = "first_middleware"
                context["counter"] = context.get("counter", 0) + 1
                return await next_handler(text)

        class ContextReaderMiddleware(Middleware):
            def __init__(self):
                self.context_received = None

            async def process(
                self,
                text: str,
                next_handler: Callable[[str], Awaitable[SifakaResult]],
                context: Dict[str, Any],
            ) -> SifakaResult:
                self.context_received = context.copy()
                context["counter"] = context.get("counter", 0) + 1
                return await next_handler(text)

        pipeline = MiddlewarePipeline()
        modifier_mw = ContextModifierMiddleware()
        reader_mw = ContextReaderMiddleware()

        pipeline.add(modifier_mw)
        pipeline.add(reader_mw)

        context = {"initial": True}
        await pipeline.execute("test", sample_handler, context)

        # Check context modifications
        assert reader_mw.context_received["modified_by"] == "first_middleware"
        assert reader_mw.context_received["counter"] == 1
        assert context["counter"] == 2  # Modified by both middleware


class TestMiddlewareIntegration:
    """Test middleware integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_middleware_stack(self, sample_handler):
        """Test complete middleware stack integration."""
        pipeline = MiddlewarePipeline()

        # Add all built-in middleware
        logging_mw = LoggingMiddleware()
        metrics_mw = MetricsMiddleware()
        caching_mw = CachingMiddleware(max_size=10)
        rate_limit_mw = RateLimitingMiddleware(max_requests=100, time_window=60)

        pipeline.add(logging_mw)
        pipeline.add(metrics_mw)
        pipeline.add(caching_mw)
        pipeline.add(rate_limit_mw)

        # Execute with full stack
        context = {"integration_test": True}
        result = await pipeline.execute("integration test", sample_handler, context)

        # Verify result
        assert isinstance(result, SifakaResult)
        assert result.original_text == "integration test"

        # Verify metrics were collected
        metrics = metrics_mw.get_metrics()
        assert metrics["total_requests"] == 1

        # Verify caching works
        await pipeline.execute("integration test", sample_handler, context)
        assert (
            metrics_mw.get_metrics()["total_requests"] == 2
        )  # Cache hit still counts as request

    @pytest.mark.asyncio
    async def test_middleware_performance_impact(self, sample_handler):
        """Test performance impact of middleware stack."""
        # Test without middleware
        start_time = time.time()
        result1 = await sample_handler("performance test")
        no_middleware_time = time.time() - start_time

        # Test with middleware stack
        pipeline = MiddlewarePipeline()
        pipeline.add(LoggingMiddleware())
        pipeline.add(MetricsMiddleware())
        pipeline.add(CachingMiddleware())

        start_time = time.time()
        result2 = await pipeline.execute("performance test", sample_handler, {})
        with_middleware_time = time.time() - start_time

        # Middleware should not add significant overhead
        overhead_ratio = with_middleware_time / no_middleware_time
        assert overhead_ratio < 5.0  # Less than 5x overhead

        # Results should be equivalent
        assert result1.original_text == result2.original_text
