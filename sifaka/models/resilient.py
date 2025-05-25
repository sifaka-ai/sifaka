"""Resilient model wrapper with enhanced error recovery for Sifaka.

This module provides a resilient model wrapper that implements:
- Circuit breaker pattern for external API calls
- Retry mechanisms with exponential backoff
- Fallback to alternative models
- Graceful degradation strategies

Example:
    ```python
    from sifaka.models.resilient import ResilientModel
    from sifaka.models.base import create_model
    from sifaka.utils.circuit_breaker import CircuitBreakerConfig
    from sifaka.utils.retry import RetryConfig
    from sifaka.utils.fallback import FallbackConfig
    
    # Create primary and fallback models
    primary_model = create_model("openai:gpt-4")
    fallback_model = create_model("anthropic:claude-3-sonnet")
    
    # Configure resilience
    resilient_model = ResilientModel(
        primary_model=primary_model,
        fallback_models=[fallback_model],
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3),
        retry_config=RetryConfig(max_attempts=3),
        fallback_config=FallbackConfig(max_fallbacks=2)
    )
    
    # Use like any other model
    response = resilient_model.generate("Write about AI")
    ```
"""

import time
from typing import Any, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError
from sifaka.utils.retry import RetryConfig, RetryManager, RetryError
from sifaka.utils.fallback import FallbackChain, FallbackConfig, FallbackError
from sifaka.utils.error_handling import (
    ModelError,
    ServiceUnavailableError,
    DegradedServiceError,
    enhance_error_message,
    log_error,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin

# Configure logger
logger = get_logger(__name__)


class ResilientModel(ContextAwareMixin):
    """Resilient model wrapper with enhanced error recovery.
    
    This class wraps one or more models and provides:
    - Circuit breaker protection for API calls
    - Automatic retry with exponential backoff
    - Fallback to alternative models
    - Health monitoring and recovery
    
    The wrapper implements the Model protocol and can be used as a drop-in
    replacement for any model implementation.
    """
    
    def __init__(
        self,
        primary_model: Model,
        fallback_models: Optional[List[Model]] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        fallback_config: Optional[FallbackConfig] = None,
        service_name: Optional[str] = None,
    ):
        """Initialize resilient model wrapper.
        
        Args:
            primary_model: Primary model to use.
            fallback_models: Optional list of fallback models.
            circuit_breaker_config: Circuit breaker configuration.
            retry_config: Retry configuration.
            fallback_config: Fallback configuration.
            service_name: Optional service name for logging.
        """
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.service_name = service_name or f"model-{id(self)}"
        
        # Initialize circuit breaker
        cb_config = circuit_breaker_config or CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=Exception
        )
        self.circuit_breaker = CircuitBreaker(f"{self.service_name}-primary", cb_config)
        
        # Initialize retry manager
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            backoff_factor=2.0,
            jitter=True
        )
        
        # Initialize fallback chain
        fb_config = fallback_config or FallbackConfig(
            max_fallbacks=len(self.fallback_models),
            fallback_timeout=30.0,
            track_performance=True
        )
        self.fallback_chain = FallbackChain(self.service_name, fb_config)
        
        # Set up fallback chain
        self._setup_fallback_chain()
        
        # Health monitoring
        self.last_health_check = time.time()
        self.health_check_interval = 300.0  # 5 minutes
        self.is_degraded = False
        
        logger.info(f"Initialized resilient model wrapper: {self.service_name}")
    
    def _setup_fallback_chain(self) -> None:
        """Set up the fallback chain with primary and fallback models."""
        # Add primary model with circuit breaker protection
        def protected_primary_generate(*args, **kwargs):
            with self.circuit_breaker.protect_call():
                return self.primary_model.generate(*args, **kwargs)
        
        self.fallback_chain.add_primary(protected_primary_generate, "primary")
        
        # Add fallback models
        for i, model in enumerate(self.fallback_models):
            def fallback_generate(*args, model=model, **kwargs):
                return model.generate(*args, **kwargs)
            
            self.fallback_chain.add_fallback(
                fallback_generate,
                priority=i + 1,
                name=f"fallback-{i}"
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
            ModelError: If all retries fail.
        """
        retry_manager = RetryManager(self.retry_config)
        
        try:
            return retry_manager.execute(func, *args, **kwargs)
        except RetryError as e:
            # Convert to ModelError with enhanced message
            enhanced_message = enhance_error_message(
                e.original_exception,
                component="ResilientModel",
                operation="generation"
            )
            
            raise ModelError(
                message=enhanced_message,
                component="ResilientModel",
                operation="generation",
                suggestions=[
                    "Check if the model service is available",
                    "Verify API credentials and quotas",
                    "Consider using a different model",
                    "Check network connectivity"
                ],
                metadata={
                    "service_name": self.service_name,
                    "retry_attempts": e.stats.total_attempts,
                    "total_delay": e.stats.total_delay,
                    "is_degraded": self.is_degraded
                }
            ) from e
    
    def _check_health(self) -> bool:
        """Check health of the model service.
        
        Returns:
            True if healthy, False otherwise.
        """
        current_time = time.time()
        
        # Only check if enough time has passed
        if current_time - self.last_health_check < self.health_check_interval:
            return not self.is_degraded
        
        try:
            # Simple health check - try to count tokens
            self.primary_model.count_tokens("health check")
            self.is_degraded = False
            self.last_health_check = current_time
            return True
        
        except Exception as e:
            logger.warning(f"Health check failed for {self.service_name}: {e}")
            self.is_degraded = True
            self.last_health_check = current_time
            return False
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using the resilient model chain.
        
        Args:
            prompt: The input prompt.
            system_message: Optional system message.
            temperature: Optional temperature parameter.
            max_tokens: Optional max tokens parameter.
            **kwargs: Additional model parameters.
        
        Returns:
            Generated text.
        
        Raises:
            ModelError: If all models fail.
        """
        # Check health status
        is_healthy = self._check_health()
        if not is_healthy:
            logger.warning(f"Model service {self.service_name} is in degraded mode")
        
        # Prepare generation function
        def generate_func():
            return self.fallback_chain.execute(
                prompt,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        
        try:
            # Execute with retry logic
            result = self._execute_with_retry(generate_func)
            
            # Log successful generation
            if self.is_degraded:
                logger.info(f"Successful generation in degraded mode: {self.service_name}")
            
            return result
        
        except FallbackError as e:
            # All fallbacks failed
            enhanced_message = enhance_error_message(
                e.last_exception,
                component="ResilientModel",
                operation="generation"
            )
            
            raise ModelError(
                message=f"All model fallbacks failed: {enhanced_message}",
                component="ResilientModel",
                operation="generation",
                suggestions=[
                    "Check if all model services are available",
                    "Verify API credentials for all models",
                    "Consider adding more fallback models",
                    "Check service status pages for outages"
                ],
                metadata={
                    "service_name": self.service_name,
                    "attempted_fallbacks": e.attempted_options,
                    "is_degraded": self.is_degraded,
                    "fallback_stats": self.fallback_chain.get_stats().__dict__
                }
            ) from e
        
        except CircuitBreakerError as e:
            # Circuit breaker is open
            raise ServiceUnavailableError(
                message=f"Model service {self.service_name} is temporarily unavailable",
                component="ResilientModel",
                operation="generation",
                suggestions=[
                    "Wait for the service to recover automatically",
                    "Check service health and logs",
                    "Use alternative models if available",
                    "Contact service provider if issue persists"
                ],
                metadata={
                    "service_name": self.service_name,
                    "circuit_breaker_state": e.state.value,
                    "circuit_breaker_stats": e.stats.__dict__
                }
            ) from e
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the primary model.
        
        Args:
            text: The text to count tokens in.
        
        Returns:
            The number of tokens.
        
        Raises:
            ModelError: If token counting fails.
        """
        try:
            # Try primary model first
            return self.primary_model.count_tokens(text)
        
        except Exception as e:
            # Try fallback models for token counting
            for i, model in enumerate(self.fallback_models):
                try:
                    result = model.count_tokens(text)
                    logger.debug(f"Token counting fallback {i} succeeded for {self.service_name}")
                    return result
                except Exception:
                    continue
            
            # All failed
            enhanced_message = enhance_error_message(
                e,
                component="ResilientModel",
                operation="token_counting"
            )
            
            raise ModelError(
                message=f"Token counting failed: {enhanced_message}",
                component="ResilientModel",
                operation="token_counting",
                suggestions=[
                    "Check if the model service is available",
                    "Verify the text format is supported",
                    "Try with a smaller text sample"
                ],
                metadata={
                    "service_name": self.service_name,
                    "text_length": len(text),
                    "is_degraded": self.is_degraded
                }
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
            "fallback_health": self.fallback_chain.health_check()
        }
    
    def reset_health(self) -> None:
        """Reset health status and statistics."""
        self.is_degraded = False
        self.last_health_check = time.time()
        self.circuit_breaker.reset()
        self.fallback_chain.reset_stats()
        logger.info(f"Reset health status for {self.service_name}")


def create_resilient_model(
    primary_model_spec: str,
    fallback_model_specs: Optional[List[str]] = None,
    **kwargs
) -> ResilientModel:
    """Create a resilient model from model specifications.
    
    Args:
        primary_model_spec: Primary model specification (e.g., "openai:gpt-4").
        fallback_model_specs: List of fallback model specifications.
        **kwargs: Additional arguments for ResilientModel.
    
    Returns:
        Configured resilient model.
    """
    from sifaka.models.base import create_model
    
    # Create primary model
    primary_model = create_model(primary_model_spec)
    
    # Create fallback models
    fallback_models = []
    if fallback_model_specs:
        for spec in fallback_model_specs:
            try:
                fallback_model = create_model(spec)
                fallback_models.append(fallback_model)
            except Exception as e:
                logger.warning(f"Failed to create fallback model {spec}: {e}")
    
    return ResilientModel(
        primary_model=primary_model,
        fallback_models=fallback_models,
        **kwargs
    )
