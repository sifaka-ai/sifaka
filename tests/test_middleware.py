"""Tests for middleware system."""

import logging
import time
from unittest.mock import Mock

import pytest

from sifaka.core.middleware import (
    CachingMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    MiddlewarePipeline,
    RateLimitingMiddleware,
    monitor,
)
from sifaka.core.models import CritiqueResult, Generation, SifakaResult


class TestMiddleware:
    """Test base Middleware class."""

    def test_abstract_base_class(self):
        """Test that Middleware is abstract."""
        with pytest.raises(TypeError):
            Middleware()  # type: ignore


class TestLoggingMiddleware:
    """Test LoggingMiddleware class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        result = SifakaResult(original_text="Original", final_text="Final", iteration=2)
        # Add a critique with confidence
        critique = CritiqueResult(
            critic="test_critic",
            feedback="Test feedback",
            suggestions=["Improve this"],
            confidence=0.85,
        )
        result.critiques.append(critique)
        return result

    def test_initialization_default(self):
        """Test default initialization."""
        middleware = LoggingMiddleware()
        assert middleware.log_level == logging.INFO

    def test_initialization_custom(self):
        """Test custom log level."""
        middleware = LoggingMiddleware(log_level="DEBUG")
        assert middleware.log_level == logging.DEBUG

    @pytest.mark.asyncio
    async def test_successful_processing(self, sample_result, caplog):
        """Test successful request processing."""
        middleware = LoggingMiddleware(log_level="INFO")

        async def next_handler(text):
            return sample_result

        context = {"critics": ["test_critic"], "validators": [Mock(), Mock()]}

        with caplog.at_level(logging.INFO):
            result = await middleware.process("Test text", next_handler, context)

        assert result == sample_result
        assert "Starting improvement for text" in caplog.text
        assert "Context: critics=['test_critic']" in caplog.text
        assert "validators=2" in caplog.text
        assert "Improvement completed" in caplog.text
        assert "iterations=2" in caplog.text
        assert "confidence=0.85" in caplog.text

    @pytest.mark.asyncio
    async def test_long_text_truncation(self, sample_result, caplog):
        """Test that long text is truncated in logs."""
        middleware = LoggingMiddleware()
        long_text = "x" * 200

        async def next_handler(text):
            return sample_result

        with caplog.at_level(logging.INFO):
            await middleware.process(long_text, next_handler, {})

        assert "x" * 100 + "..." in caplog.text
        assert "x" * 200 not in caplog.text

    @pytest.mark.asyncio
    async def test_exception_handling(self, caplog):
        """Test exception logging."""
        middleware = LoggingMiddleware()

        async def failing_handler(text):
            raise ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                await middleware.process("Test", failing_handler, {})

        assert "Improvement failed" in caplog.text
        assert "ValueError: Test error" in caplog.text

    @pytest.mark.asyncio
    async def test_no_confidence_in_critiques(self):
        """Test handling when critiques have no confidence."""
        middleware = LoggingMiddleware()
        result = SifakaResult(original_text="Original", final_text="Final")
        # Add critique without confidence
        result.critiques.append(
            CritiqueResult(critic="test_critic", feedback="Test", suggestions=["Test"])
        )

        async def next_handler(text):
            return result

        # Should not raise error
        await middleware.process("Test", next_handler, {})


class TestMetricsMiddleware:
    """Test MetricsMiddleware class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample result with metrics."""
        result = SifakaResult(original_text="Original", final_text="Final", iteration=3)
        # Add critique with confidence
        critique = CritiqueResult(
            critic="test_critic",
            feedback="Test feedback",
            suggestions=["Test"],
            confidence=0.75,
        )
        result.critiques.append(critique)
        # Add generation with token count
        generation = Generation(
            text="Generated text", model="test-model", iteration=1, tokens_used=100
        )
        result.generations.append(generation)
        return result

    def test_initialization(self):
        """Test metrics initialization."""
        middleware = MetricsMiddleware()
        assert middleware.metrics["total_requests"] == 0
        assert middleware.metrics["total_iterations"] == 0
        assert middleware.metrics["errors"] == 0

    @pytest.mark.asyncio
    async def test_metrics_collection(self, sample_result):
        """Test metrics are collected correctly."""
        middleware = MetricsMiddleware()

        async def next_handler(text):
            return sample_result

        context = {"llm_calls": 2}
        await middleware.process("Test", next_handler, context)

        assert middleware.metrics["total_requests"] == 1
        assert middleware.metrics["total_iterations"] == 3
        assert middleware.metrics["tokens_used"] == 100
        assert middleware.metrics["average_confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_multiple_requests(self, sample_result):
        """Test metrics across multiple requests."""
        middleware = MetricsMiddleware()

        async def next_handler(text):
            return sample_result

        # Process 3 requests
        for _ in range(3):
            await middleware.process("Test", next_handler, {})

        assert middleware.metrics["total_requests"] == 3
        assert middleware.metrics["total_iterations"] == 9  # 3 * 3
        assert middleware.metrics["tokens_used"] == 300  # 3 * 100

    @pytest.mark.asyncio
    async def test_error_tracking(self, sample_result):
        """Test error counting."""
        middleware = MetricsMiddleware()

        async def failing_handler(text):
            raise RuntimeError("Test error")

        # Process one successful, one failed
        async def success_handler(text):
            return sample_result

        await middleware.process("Test", success_handler, {})

        with pytest.raises(RuntimeError):
            await middleware.process("Test", failing_handler, {})

        assert middleware.metrics["total_requests"] == 2
        assert middleware.metrics["errors"] == 1

    @pytest.mark.asyncio
    async def test_llm_call_tracking(self, sample_result):
        """Test LLM call tracking via context."""
        middleware = MetricsMiddleware()

        async def next_handler(text):
            # Simulate LLM calls
            context["llm_calls"] = 5
            return sample_result

        context = {"llm_calls": 2}
        await middleware.process("Test", next_handler, context)

        assert middleware.metrics["llm_calls"] == 3  # 5 - 2

    def test_get_metrics(self):
        """Test metrics calculation."""
        middleware = MetricsMiddleware()
        middleware.metrics = {
            "total_requests": 10,
            "total_iterations": 30,
            "total_time": 100.0,
            "llm_calls": 50,
            "average_confidence": 0.8,
            "errors": 2,
            "tokens_used": 1000,
        }

        metrics = middleware.get_metrics()
        assert metrics["avg_time_per_request"] == 10.0
        assert metrics["avg_iterations_per_request"] == 3.0
        assert metrics["avg_llm_calls_per_request"] == 5.0

    def test_get_metrics_no_requests(self):
        """Test metrics with no requests."""
        middleware = MetricsMiddleware()
        metrics = middleware.get_metrics()
        # Should not have average fields
        assert "avg_time_per_request" not in metrics


class TestCachingMiddleware:
    """Test CachingMiddleware class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample result."""
        return SifakaResult(original_text="Original", final_text="Final")

    def test_initialization(self):
        """Test initialization."""
        middleware = CachingMiddleware(max_size=50)
        assert middleware.max_size == 50
        assert len(middleware.cache) == 0
        assert middleware.hits == 0
        assert middleware.misses == 0

    def test_cache_key_generation(self):
        """Test cache key generation."""
        middleware = CachingMiddleware()

        context1 = {
            "critics": ["critic1", "critic2"],
            "validators": [Mock(), Mock()],
            "model": "gpt-4",
            "temperature": 0.5,
        }

        key1 = middleware._get_cache_key("Test text", context1)

        # Same inputs should generate same key
        key2 = middleware._get_cache_key("Test text", context1)
        assert key1 == key2

        # Different text should generate different key
        key3 = middleware._get_cache_key("Different text", context1)
        assert key1 != key3

        # Different context should generate different key
        context2 = context1.copy()
        context2["temperature"] = 0.7
        key4 = middleware._get_cache_key("Test text", context2)
        assert key1 != key4

    @pytest.mark.asyncio
    async def test_cache_miss(self, sample_result):
        """Test cache miss."""
        middleware = CachingMiddleware()

        async def next_handler(text):
            return sample_result

        result = await middleware.process("Test", next_handler, {})

        assert result == sample_result
        assert middleware.misses == 1
        assert middleware.hits == 0
        assert len(middleware.cache) == 1

    @pytest.mark.asyncio
    async def test_cache_hit(self, sample_result):
        """Test cache hit."""
        middleware = CachingMiddleware()

        call_count = 0

        async def counting_handler(text):
            nonlocal call_count
            call_count += 1
            return sample_result

        context = {"critics": ["test"]}

        # First call - miss
        result1 = await middleware.process("Test", counting_handler, context)
        assert call_count == 1
        assert middleware.misses == 1

        # Second call - hit
        result2 = await middleware.process("Test", counting_handler, context)
        assert call_count == 1  # Handler not called again
        assert middleware.hits == 1
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_eviction(self, sample_result):
        """Test cache eviction when full."""
        middleware = CachingMiddleware(max_size=2)

        async def next_handler(text):
            return sample_result

        # Fill cache
        await middleware.process("Text1", next_handler, {})
        await middleware.process("Text2", next_handler, {})
        assert len(middleware.cache) == 2

        # Add third item - should evict first
        await middleware.process("Text3", next_handler, {})
        assert len(middleware.cache) == 2

        # First item should be evicted
        # The cache should only contain the last 2 items added
        # We can't check specific keys due to hash, but can check count
        assert len(middleware.cache) == 2

    def test_get_stats(self):
        """Test cache statistics."""
        middleware = CachingMiddleware()
        middleware.hits = 10
        middleware.misses = 5
        middleware.cache = {"key1": Mock(), "key2": Mock()}

        stats = middleware.get_stats()
        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["hit_rate"] == 10 / 15
        assert stats["size"] == 2
        assert stats["max_size"] == 100

    def test_get_stats_no_requests(self):
        """Test stats with no requests."""
        middleware = CachingMiddleware()
        stats = middleware.get_stats()
        assert stats["hit_rate"] == 0.0


class TestRateLimitingMiddleware:
    """Test RateLimitingMiddleware class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample result."""
        return SifakaResult(original_text="Original", final_text="Final")

    def test_initialization(self):
        """Test initialization."""
        middleware = RateLimitingMiddleware(max_requests_per_minute=30)
        assert middleware.max_requests == 30
        assert len(middleware.requests) == 0

    @pytest.mark.asyncio
    async def test_under_limit(self, sample_result):
        """Test requests under rate limit."""
        middleware = RateLimitingMiddleware(max_requests_per_minute=10)

        async def next_handler(text):
            return sample_result

        # Make 5 requests
        for _ in range(5):
            result = await middleware.process("Test", next_handler, {})
            assert result == sample_result

        assert len(middleware.requests) == 5

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, sample_result):
        """Test rate limit exceeded."""
        middleware = RateLimitingMiddleware(max_requests_per_minute=2)

        async def next_handler(text):
            return sample_result

        # Make 2 requests - should succeed
        await middleware.process("Test", next_handler, {})
        await middleware.process("Test", next_handler, {})

        # Third request should fail
        with pytest.raises(RuntimeError) as exc_info:
            await middleware.process("Test", next_handler, {})

        assert "Rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_old_requests_cleanup(self, sample_result):
        """Test that old requests are cleaned up."""
        middleware = RateLimitingMiddleware()

        async def next_handler(text):
            return sample_result

        # Add old requests (more than 60 seconds ago)
        old_time = time.time() - 61
        middleware.requests = [old_time, old_time]

        # Make new request - old ones should be cleaned
        await middleware.process("Test", next_handler, {})

        assert len(middleware.requests) == 1
        assert middleware.requests[0] > old_time


class TestMiddlewarePipeline:
    """Test MiddlewarePipeline class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample result."""
        return SifakaResult(original_text="Original", final_text="Final")

    def test_initialization(self):
        """Test initialization."""
        pipeline = MiddlewarePipeline()
        assert len(pipeline.middleware) == 0

    def test_add_middleware(self):
        """Test adding middleware."""
        pipeline = MiddlewarePipeline()
        m1 = LoggingMiddleware()
        m2 = MetricsMiddleware()

        # Test chaining
        result = pipeline.add(m1).add(m2)
        assert result is pipeline
        assert len(pipeline.middleware) == 2
        assert pipeline.middleware[0] is m1
        assert pipeline.middleware[1] is m2

    @pytest.mark.asyncio
    async def test_empty_pipeline(self, sample_result):
        """Test empty pipeline."""
        pipeline = MiddlewarePipeline()

        async def final_handler(text):
            return sample_result

        result = await pipeline.execute("Test", final_handler)
        assert result == sample_result

    @pytest.mark.asyncio
    async def test_single_middleware(self, sample_result):
        """Test pipeline with single middleware."""
        pipeline = MiddlewarePipeline()

        # Track execution
        executed = []

        class TrackingMiddleware(Middleware):
            async def process(self, text, next_handler, context):
                executed.append("middleware")
                return await next_handler(text)

        pipeline.add(TrackingMiddleware())

        async def final_handler(text):
            executed.append("final")
            return sample_result

        result = await pipeline.execute("Test", final_handler)

        assert result == sample_result
        assert executed == ["middleware", "final"]

    @pytest.mark.asyncio
    async def test_multiple_middleware_order(self, sample_result):
        """Test middleware execution order."""
        pipeline = MiddlewarePipeline()

        executed = []

        class OrderMiddleware(Middleware):
            def __init__(self, name):
                self.name = name

            async def process(self, text, next_handler, context):
                executed.append(f"{self.name}_before")
                result = await next_handler(text)
                executed.append(f"{self.name}_after")
                return result

        pipeline.add(OrderMiddleware("first"))
        pipeline.add(OrderMiddleware("second"))

        async def final_handler(text):
            executed.append("final")
            return sample_result

        await pipeline.execute("Test", final_handler)

        assert executed == [
            "first_before",
            "second_before",
            "final",
            "second_after",
            "first_after",
        ]

    @pytest.mark.asyncio
    async def test_context_sharing(self, sample_result):
        """Test context sharing between middleware."""
        pipeline = MiddlewarePipeline()

        class ContextMiddleware1(Middleware):
            async def process(self, text, next_handler, context):
                context["value1"] = "from_middleware1"
                return await next_handler(text)

        class ContextMiddleware2(Middleware):
            async def process(self, text, next_handler, context):
                assert context["value1"] == "from_middleware1"
                context["value2"] = "from_middleware2"
                return await next_handler(text)

        pipeline.add(ContextMiddleware1())
        pipeline.add(ContextMiddleware2())

        context = {}

        async def final_handler(text):
            assert context["value1"] == "from_middleware1"
            assert context["value2"] == "from_middleware2"
            return sample_result

        await pipeline.execute("Test", final_handler, context)

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Test that exceptions propagate through pipeline."""
        pipeline = MiddlewarePipeline()
        pipeline.add(LoggingMiddleware())

        async def failing_handler(text):
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await pipeline.execute("Test", failing_handler)


class TestMonitorContextManager:
    """Test the monitor context manager."""

    @pytest.mark.asyncio
    async def test_basic_usage(self):
        """Test basic monitor usage."""
        async with monitor() as mon:
            assert "pipeline" in mon
            assert "metrics" in mon
            assert isinstance(mon["pipeline"], MiddlewarePipeline)
            assert isinstance(mon["metrics"], MetricsMiddleware)

    @pytest.mark.asyncio
    async def test_without_metrics(self):
        """Test monitor without metrics."""
        async with monitor(include_metrics=False) as mon:
            assert mon["metrics"] is None
            assert len(mon["pipeline"].middleware) == 1  # Only logging

    @pytest.mark.asyncio
    async def test_without_logging(self):
        """Test monitor without logging."""
        async with monitor(include_logging=False) as mon:
            assert len(mon["pipeline"].middleware) == 1  # Only metrics

    @pytest.mark.asyncio
    async def test_custom_log_level(self):
        """Test custom log level."""
        async with monitor(log_level="DEBUG") as mon:
            logging_middleware = mon["pipeline"].middleware[0]
            assert isinstance(logging_middleware, LoggingMiddleware)
            assert logging_middleware.log_level == logging.DEBUG

    @pytest.mark.asyncio
    async def test_final_metrics_logging(self, caplog):
        """Test that final metrics are logged."""
        with caplog.at_level(logging.INFO):
            async with monitor() as mon:
                # Simulate some metrics
                mon["metrics"].metrics["total_requests"] = 5
                mon["metrics"].metrics["total_time"] = 10.0

        # Check that metrics were logged after context exit
        assert "Session metrics:" in caplog.text
