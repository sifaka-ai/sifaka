"""Resilient retriever wrapper with enhanced error recovery for Sifaka.

This module provides a resilient retriever wrapper that implements:
- Circuit breaker pattern for external database calls
- Retry mechanisms with exponential backoff
- Fallback to alternative retrievers
- Graceful degradation strategies

Example:
    ```python
    from sifaka.retrievers.resilient import ResilientRetriever
    from sifaka.retrievers.redis import RedisRetriever
    from sifaka.retrievers.simple import InMemoryRetriever
    from sifaka.utils.circuit_breaker import CircuitBreakerConfig
    from sifaka.utils.retry import RetryConfig
    from sifaka.utils.fallback import FallbackConfig

    # Create primary and fallback retrievers
    primary_retriever = RedisRetriever(host="localhost", port=6379)
    fallback_retriever = InMemoryRetriever()

    # Configure resilience
    resilient_retriever = ResilientRetriever(
        primary_retriever=primary_retriever,
        fallback_retrievers=[fallback_retriever],
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3),
        retry_config=RetryConfig(max_attempts=3),
        fallback_config=FallbackConfig(max_fallbacks=2)
    )

    # Use like any other retriever
    results = resilient_retriever.retrieve("search query")
    ```
"""

import time
from typing import Any, List, Optional

from sifaka.core.interfaces import Retriever
from sifaka.core.thought import Document, Thought
from sifaka.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError
from sifaka.utils.retry import RetryConfig, RetryManager, RetryError
from sifaka.utils.fallback import FallbackChain, FallbackConfig, FallbackError
from sifaka.utils.error_handling import (
    RetrieverError,
    ServiceUnavailableError,
    DegradedServiceError,
    enhance_error_message,
    log_error,
)
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class ResilientRetriever:
    """Resilient retriever wrapper with enhanced error recovery.

    This class wraps one or more retrievers and provides:
    - Circuit breaker protection for database calls
    - Automatic retry with exponential backoff
    - Fallback to alternative retrievers
    - Health monitoring and recovery

    The wrapper implements the Retriever protocol and can be used as a drop-in
    replacement for any retriever implementation.
    """

    def __init__(
        self,
        primary_retriever: Retriever,
        fallback_retrievers: Optional[List[Retriever]] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        fallback_config: Optional[FallbackConfig] = None,
        service_name: Optional[str] = None,
    ):
        """Initialize resilient retriever wrapper.

        Args:
            primary_retriever: Primary retriever to use.
            fallback_retrievers: Optional list of fallback retrievers.
            circuit_breaker_config: Circuit breaker configuration.
            retry_config: Retry configuration.
            fallback_config: Fallback configuration.
            service_name: Optional service name for logging.
        """
        self.primary_retriever = primary_retriever
        self.fallback_retrievers = fallback_retrievers or []
        self.service_name = service_name or f"retriever-{id(self)}"

        # Initialize circuit breaker
        cb_config = circuit_breaker_config or CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=30.0, expected_exception=Exception
        )
        self.circuit_breaker = CircuitBreaker(f"{self.service_name}-primary", cb_config)

        # Initialize retry manager
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3, base_delay=1.0, max_delay=30.0, backoff_factor=2.0, jitter=True
        )

        # Initialize fallback chain
        fb_config = fallback_config or FallbackConfig(
            max_fallbacks=len(self.fallback_retrievers),
            fallback_timeout=30.0,
            track_performance=True,
        )
        self.fallback_chain = FallbackChain(self.service_name, fb_config)

        # Set up fallback chain
        self._setup_fallback_chain()

        # Health monitoring
        self.last_health_check = time.time()
        self.health_check_interval = 300.0  # 5 minutes
        self.is_degraded = False

        logger.info(f"Initialized resilient retriever wrapper: {self.service_name}")

    def _setup_fallback_chain(self) -> None:
        """Set up the fallback chain with primary and fallback retrievers."""

        # Add primary retriever with circuit breaker protection
        def protected_primary_retrieve(*args, **kwargs):
            with self.circuit_breaker.protect_call():
                return self.primary_retriever.retrieve(*args, **kwargs)

        self.fallback_chain.add_primary(protected_primary_retrieve, "primary")

        # Add fallback retrievers
        for i, retriever in enumerate(self.fallback_retrievers):

            def fallback_retrieve(*args, retriever=retriever, **kwargs):
                return retriever.retrieve(*args, **kwargs)

            self.fallback_chain.add_fallback(
                fallback_retrieve, priority=i + 1, name=f"fallback-{i}"
            )

    def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """Execute function with retry logic.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            RetrieverError: If all retries fail.
        """
        retry_manager = RetryManager(self.retry_config)

        try:
            return retry_manager.execute(func, *args, **kwargs)
        except RetryError as e:
            # Convert to RetrieverError with enhanced message
            enhanced_message = enhance_error_message(
                e.original_exception, component="ResilientRetriever", operation="retrieval"
            )

            raise RetrieverError(
                message=enhanced_message,
                component="ResilientRetriever",
                operation="retrieval",
                suggestions=[
                    "Check if the retriever service is available",
                    "Verify database connection and credentials",
                    "Consider using a different retriever",
                    "Check network connectivity",
                ],
                metadata={
                    "service_name": self.service_name,
                    "retry_attempts": e.stats.total_attempts,
                    "total_delay": e.stats.total_delay,
                    "is_degraded": self.is_degraded,
                },
            ) from e

    def _check_health(self) -> bool:
        """Check health of the retriever service.

        Returns:
            True if healthy, False otherwise.
        """
        current_time = time.time()

        # Only check if enough time has passed
        if current_time - self.last_health_check < self.health_check_interval:
            return not self.is_degraded

        try:
            # Simple health check - try a basic retrieve operation
            self.primary_retriever.retrieve("health check")
            self.is_degraded = False
            self.last_health_check = current_time
            return True

        except Exception as e:
            logger.warning(f"Health check failed for {self.service_name}: {e}")
            self.is_degraded = True
            self.last_health_check = current_time
            return False

    def retrieve(self, query: str) -> List[str]:
        """Retrieve documents using the resilient retriever chain.

        Args:
            query: The search query.
            max_results: Optional maximum number of results.

        Returns:
            List of retrieved document texts.

        Raises:
            RetrieverError: If all retrievers fail.
        """
        # Check health status
        is_healthy = self._check_health()
        if not is_healthy:
            logger.warning(f"Retriever service {self.service_name} is in degraded mode")

        # Prepare retrieval function
        def retrieve_func():
            return self.fallback_chain.execute(query)

        try:
            # Execute with retry logic
            result = self._execute_with_retry(retrieve_func)

            # Log successful retrieval
            if self.is_degraded:
                logger.info(f"Successful retrieval in degraded mode: {self.service_name}")

            return result

        except FallbackError as e:
            # All fallbacks failed
            enhanced_message = enhance_error_message(
                e.last_exception, component="ResilientRetriever", operation="retrieval"
            )

            raise RetrieverError(
                message=f"All retriever fallbacks failed: {enhanced_message}",
                component="ResilientRetriever",
                operation="retrieval",
                suggestions=[
                    "Check if all retriever services are available",
                    "Verify database connections and credentials",
                    "Consider adding more fallback retrievers",
                    "Check service status pages for outages",
                ],
                metadata={
                    "service_name": self.service_name,
                    "attempted_fallbacks": e.attempted_options,
                    "is_degraded": self.is_degraded,
                    "fallback_stats": self.fallback_chain.get_stats().__dict__,
                },
            ) from e

        except CircuitBreakerError as e:
            # Circuit breaker is open
            raise ServiceUnavailableError(
                message=f"Retriever service {self.service_name} is temporarily unavailable",
                component="ResilientRetriever",
                operation="retrieval",
                suggestions=[
                    "Wait for the service to recover automatically",
                    "Check service health and logs",
                    "Use alternative retrievers if available",
                    "Contact service provider if issue persists",
                ],
                metadata={
                    "service_name": self.service_name,
                    "circuit_breaker_state": e.state.value,
                    "circuit_breaker_stats": e.stats.__dict__,
                },
            ) from e

    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        """Retrieve documents for a thought using the resilient retriever chain.

        Args:
            thought: The thought to retrieve context for.
            is_pre_generation: Whether this is pre-generation context.

        Returns:
            Thought with added context.

        Raises:
            RetrieverError: If all retrievers fail.
        """
        # Check health status
        is_healthy = self._check_health()
        if not is_healthy:
            logger.warning(f"Retriever service {self.service_name} is in degraded mode")

        try:
            # Try primary retriever first
            if hasattr(self.primary_retriever, "retrieve_for_thought"):
                with self.circuit_breaker.protect_call():
                    return self.primary_retriever.retrieve_for_thought(thought, is_pre_generation)
            else:
                # Fallback to retrieve and convert
                with self.circuit_breaker.protect_call():
                    texts = self.primary_retriever.retrieve(thought.prompt)
                    documents = [
                        Document(text=text, metadata={"source": "primary", "index": i})
                        for i, text in enumerate(texts)
                    ]
                    if is_pre_generation:
                        return thought.add_pre_generation_context(documents)
                    else:
                        return thought.add_post_generation_context(documents)

        except (CircuitBreakerError, Exception) as e:
            # Try fallback retrievers
            for i, retriever in enumerate(self.fallback_retrievers):
                try:
                    if hasattr(retriever, "retrieve_for_thought"):
                        result = retriever.retrieve_for_thought(thought, is_pre_generation)
                        logger.info(f"Fallback retriever {i} succeeded for thought retrieval")
                        return result
                    else:
                        # Fallback to retrieve and convert
                        texts = retriever.retrieve(thought.prompt)
                        documents = [
                            Document(text=text, metadata={"source": f"fallback-{i}", "index": j})
                            for j, text in enumerate(texts)
                        ]
                        if is_pre_generation:
                            result = thought.add_pre_generation_context(documents)
                        else:
                            result = thought.add_post_generation_context(documents)
                        logger.info(f"Fallback retriever {i} succeeded for thought retrieval")
                        return result

                except Exception as fallback_error:
                    logger.warning(f"Fallback retriever {i} failed: {fallback_error}")
                    continue

            # All failed
            enhanced_message = enhance_error_message(
                e, component="ResilientRetriever", operation="thought_retrieval"
            )

            raise RetrieverError(
                message=f"All retrievers failed for thought retrieval: {enhanced_message}",
                component="ResilientRetriever",
                operation="thought_retrieval",
                suggestions=[
                    "Check if retriever services are available",
                    "Verify the thought prompt is valid",
                    "Consider using simpler retrieval methods",
                    "Check database connectivity",
                ],
                metadata={
                    "service_name": self.service_name,
                    "thought_id": thought.id,
                    "is_pre_generation": is_pre_generation,
                    "is_degraded": self.is_degraded,
                },
            ) from e

    def get_health_status(self) -> dict:
        """Get current health status and statistics.

        Returns:
            Dictionary with health status and statistics.
        """
        return {
            "service_name": self.service_name,
            "is_healthy": not self.is_degraded,
            "is_degraded": self.is_degraded,
            "last_health_check": self.last_health_check,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_stats": self.circuit_breaker.get_stats().__dict__,
            "fallback_stats": self.fallback_chain.get_stats().__dict__,
            "fallback_health": self.fallback_chain.health_check(),
        }

    def reset_health(self) -> None:
        """Reset health status and statistics."""
        self.is_degraded = False
        self.last_health_check = time.time()
        self.circuit_breaker.reset()
        self.fallback_chain.reset_stats()
        logger.info(f"Reset health status for {self.service_name}")


def create_resilient_retriever_chain(
    primary_retriever: Retriever, fallback_retrievers: List[Retriever], **kwargs
) -> ResilientRetriever:
    """Create a resilient retriever chain.

    Args:
        primary_retriever: Primary retriever instance.
        fallback_retrievers: List of fallback retriever instances.
        **kwargs: Additional arguments for ResilientRetriever.

    Returns:
        Configured resilient retriever.
    """
    return ResilientRetriever(
        primary_retriever=primary_retriever, fallback_retrievers=fallback_retrievers, **kwargs
    )
