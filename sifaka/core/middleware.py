"""Middleware system for Sifaka to enable cross-cutting concerns."""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Any, Dict, cast, AsyncIterator
import time
import logging
from contextlib import asynccontextmanager

from .models import SifakaResult
from .interfaces import Validator, Critic


logger = logging.getLogger(__name__)


class Middleware(ABC):
    """Base class for middleware components."""
    
    @abstractmethod
    async def process(
        self, 
        text: str, 
        next_handler: Callable[[str], Any],
        context: Dict[str, Any]
    ) -> SifakaResult:
        """Process the request and call the next handler.
        
        Args:
            text: Input text
            next_handler: Next middleware or final handler
            context: Shared context between middleware
            
        Returns:
            SifakaResult from the pipeline
        """
        pass


class LoggingMiddleware(Middleware):
    """Logs improvement requests and results."""
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize logging middleware.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_level = getattr(logging, log_level.upper())
    
    async def process(
        self, 
        text: str, 
        next_handler: Callable[[str], Any],
        context: Dict[str, Any]
    ) -> SifakaResult:
        """Log the improvement process."""
        start_time = time.time()
        
        logger.log(self.log_level, f"Starting improvement for text: {text[:100]}...")
        logger.log(self.log_level, f"Context: critics={context.get('critics', [])}, "
                                  f"validators={len(context.get('validators', []))}")
        
        try:
            result = await next_handler(text)
            
            elapsed = time.time() - start_time
            logger.log(self.log_level, f"Improvement completed in {elapsed:.2f}s")
            # Get latest confidence from critiques
            latest_confidence = 0.0
            for critique in result.critiques:
                if critique.confidence is not None:
                    latest_confidence = critique.confidence
            logger.log(self.log_level, f"Result: iterations={result.iteration}, "
                                      f"confidence={latest_confidence:.2f}, "
                                      f"improved={not result.needs_improvement}")
            
            return cast(SifakaResult, result)
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Improvement failed after {elapsed:.2f}s: {type(e).__name__}: {str(e)}")
            raise


class MetricsMiddleware(Middleware):
    """Collects metrics about the improvement process."""
    
    def __init__(self) -> None:
        """Initialize metrics collection."""
        self.metrics = {
            "total_requests": 0,
            "total_iterations": 0,
            "total_time": 0.0,
            "average_confidence": 0.0,
            "errors": 0,
            "llm_calls": 0,
            "tokens_used": 0
        }
    
    async def process(
        self, 
        text: str, 
        next_handler: Callable[[str], Any],
        context: Dict[str, Any]
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
                (old_avg * (total - 1) + latest_confidence) / total
            )
            
            # Track LLM calls
            final_llm_calls = context.get("llm_calls", 0)
            self.metrics["llm_calls"] += (final_llm_calls - initial_llm_calls)
            
            # Track tokens
            for gen in result.generations:
                self.metrics["tokens_used"] += gen.tokens_used
            
            return cast(SifakaResult, result)
            
        except Exception as e:
            self.metrics["errors"] += 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = self.metrics.copy()
        
        # Calculate averages
        if metrics["total_requests"] > 0:
            metrics["avg_time_per_request"] = metrics["total_time"] / metrics["total_requests"]
            metrics["avg_iterations_per_request"] = metrics["total_iterations"] / metrics["total_requests"]
            metrics["avg_llm_calls_per_request"] = metrics["llm_calls"] / metrics["total_requests"]
        
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
    
    def _get_cache_key(self, text: str, context: Dict[str, Any]) -> str:
        """Generate cache key from text and context."""
        critics = ",".join(sorted(context.get("critics", [])))
        validators = len(context.get("validators", []))
        config_key = f"{context.get('model', 'default')}_{context.get('temperature', 0.7)}"
        
        return f"{hash(text)}_{critics}_{validators}_{config_key}"
    
    async def process(
        self, 
        text: str, 
        next_handler: Callable[[str], Any],
        context: Dict[str, Any]
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
            "max_size": self.max_size
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
        self, 
        text: str, 
        next_handler: Callable[[str], Any],
        context: Dict[str, Any]
    ) -> SifakaResult:
        """Check rate limit before processing."""
        now = time.time()
        
        # Remove old requests
        self.requests = [t for t in self.requests if now - t < 60]
        
        # Check rate limit
        if len(self.requests) >= self.max_requests:
            wait_time = 60 - (now - self.requests[0])
            raise RuntimeError(f"Rate limit exceeded. Try again in {wait_time:.1f} seconds.")
        
        # Add current request
        self.requests.append(now)
        
        return cast(SifakaResult, await next_handler(text))


class MiddlewarePipeline:
    """Manages the middleware pipeline."""
    
    def __init__(self) -> None:
        """Initialize empty pipeline."""
        self.middleware: List[Middleware] = []
    
    def add(self, middleware: Middleware) -> 'MiddlewarePipeline':
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
        context: Optional[Dict[str, Any]] = None
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
        async def chain(index: int) -> SifakaResult:
            if index >= len(self.middleware):
                # End of middleware chain, call final handler
                return cast(SifakaResult, await final_handler(text))
            
            # Call current middleware
            current = self.middleware[index]
            return await current.process(
                text,
                lambda t: chain(index + 1),
                context
            )
        
        return await chain(0)


@asynccontextmanager
async def monitor(
    include_logging: bool = True,
    include_metrics: bool = True,
    log_level: str = "INFO"
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
    
    data = {
        "pipeline": pipeline,
        "metrics": metrics_middleware
    }
    
    yield data
    
    # After completion, could log final metrics
    if metrics_middleware:
        final_metrics = metrics_middleware.get_metrics()
        logger.info(f"Session metrics: {final_metrics}")