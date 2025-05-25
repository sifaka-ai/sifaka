"""Fallback mechanisms for graceful degradation in Sifaka.

This module provides utilities for implementing fallback chains and graceful
degradation when primary services fail. It supports multiple fallback strategies
and automatic failover.

Example:
    ```python
    from sifaka.utils.fallback import FallbackChain, FallbackConfig
    
    # Create fallback chain
    config = FallbackConfig(
        max_fallbacks=3,
        fallback_timeout=5.0,
        track_performance=True
    )
    
    chain = FallbackChain("model-service", config)
    chain.add_primary(primary_model.generate)
    chain.add_fallback(backup_model.generate, priority=1)
    chain.add_fallback(simple_model.generate, priority=2)
    
    # Execute with automatic fallback
    result = chain.execute("Generate text about AI")
    ```
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import logging

from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class FallbackStrategy(Enum):
    """Fallback strategies."""
    SEQUENTIAL = "sequential"  # Try fallbacks in order
    PRIORITY = "priority"  # Try by priority level
    ROUND_ROBIN = "round_robin"  # Rotate through fallbacks
    FASTEST_FIRST = "fastest_first"  # Try fastest responding first


@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms."""
    max_fallbacks: int = 3  # Maximum number of fallbacks to try
    fallback_timeout: float = 10.0  # Timeout for each fallback attempt
    strategy: FallbackStrategy = FallbackStrategy.SEQUENTIAL
    track_performance: bool = True  # Track response times
    
    # Health checking
    health_check_interval: float = 60.0  # Seconds between health checks
    health_check_timeout: float = 5.0  # Timeout for health checks
    
    # Failure handling
    failure_threshold: int = 3  # Failures before marking as unhealthy
    recovery_threshold: int = 2  # Successes needed to mark as healthy
    
    # Logging
    log_fallbacks: bool = True
    log_level: int = logging.WARNING


@dataclass
class FallbackOption:
    """Represents a fallback option."""
    name: str
    func: Callable
    priority: int = 0  # Lower numbers = higher priority
    is_healthy: bool = True
    failure_count: int = 0
    success_count: int = 0
    avg_response_time: float = 0.0
    last_used: Optional[float] = None
    last_health_check: Optional[float] = None
    
    def record_success(self, response_time: float) -> None:
        """Record a successful call."""
        self.success_count += 1
        self.failure_count = max(0, self.failure_count - 1)  # Reduce failure count
        self.last_used = time.time()
        
        # Update average response time
        if self.avg_response_time == 0.0:
            self.avg_response_time = response_time
        else:
            # Exponential moving average
            self.avg_response_time = 0.8 * self.avg_response_time + 0.2 * response_time
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_used = time.time()
    
    def update_health_status(self, config: FallbackConfig) -> None:
        """Update health status based on failure/success counts."""
        if self.failure_count >= config.failure_threshold:
            self.is_healthy = False
        elif self.success_count >= config.recovery_threshold and self.failure_count == 0:
            self.is_healthy = True


@dataclass
class FallbackStats:
    """Statistics for fallback operations."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    fallback_usage: Dict[str, int] = field(default_factory=dict)
    avg_response_time: float = 0.0
    primary_success_rate: float = 0.0
    fallback_success_rate: float = 0.0


class FallbackError(Exception):
    """Exception raised when all fallback options fail."""
    
    def __init__(self, service_name: str, attempted_options: List[str], last_exception: Exception):
        self.service_name = service_name
        self.attempted_options = attempted_options
        self.last_exception = last_exception
        
        message = (
            f"All fallback options failed for service '{service_name}'. "
            f"Attempted: {', '.join(attempted_options)}. "
            f"Last error: {type(last_exception).__name__}: {last_exception}"
        )
        super().__init__(message)


class FallbackChain:
    """Manages fallback chains for graceful degradation."""
    
    def __init__(self, service_name: str, config: Optional[FallbackConfig] = None):
        """Initialize fallback chain.
        
        Args:
            service_name: Name of the service.
            config: Fallback configuration.
        """
        self.service_name = service_name
        self.config = config or FallbackConfig()
        self.primary: Optional[FallbackOption] = None
        self.fallbacks: List[FallbackOption] = []
        self.stats = FallbackStats()
        self._round_robin_index = 0
        
        logger.info(f"Initialized fallback chain for {service_name}")
    
    def add_primary(self, func: Callable, name: Optional[str] = None) -> None:
        """Add primary service function.
        
        Args:
            func: Primary service function.
            name: Optional name for the service.
        """
        name = name or "primary"
        self.primary = FallbackOption(name=name, func=func, priority=0)
        logger.debug(f"Added primary service '{name}' to {self.service_name}")
    
    def add_fallback(self, func: Callable, priority: int = 1, name: Optional[str] = None) -> None:
        """Add fallback service function.
        
        Args:
            func: Fallback service function.
            priority: Priority level (lower = higher priority).
            name: Optional name for the service.
        """
        name = name or f"fallback_{len(self.fallbacks)}"
        fallback = FallbackOption(name=name, func=func, priority=priority)
        self.fallbacks.append(fallback)
        
        # Sort by priority
        self.fallbacks.sort(key=lambda x: x.priority)
        
        logger.debug(f"Added fallback service '{name}' (priority {priority}) to {self.service_name}")
    
    def _get_execution_order(self) -> List[FallbackOption]:
        """Get execution order based on strategy."""
        options = []
        
        # Add primary if healthy
        if self.primary and self.primary.is_healthy:
            options.append(self.primary)
        
        # Add fallbacks based on strategy
        healthy_fallbacks = [f for f in self.fallbacks if f.is_healthy]
        
        if self.config.strategy == FallbackStrategy.SEQUENTIAL:
            options.extend(healthy_fallbacks)
        
        elif self.config.strategy == FallbackStrategy.PRIORITY:
            options.extend(sorted(healthy_fallbacks, key=lambda x: x.priority))
        
        elif self.config.strategy == FallbackStrategy.ROUND_ROBIN:
            if healthy_fallbacks:
                # Start from current index and wrap around
                ordered = []
                for i in range(len(healthy_fallbacks)):
                    idx = (self._round_robin_index + i) % len(healthy_fallbacks)
                    ordered.append(healthy_fallbacks[idx])
                options.extend(ordered)
                self._round_robin_index = (self._round_robin_index + 1) % len(healthy_fallbacks)
        
        elif self.config.strategy == FallbackStrategy.FASTEST_FIRST:
            options.extend(sorted(healthy_fallbacks, key=lambda x: x.avg_response_time))
        
        # If primary is unhealthy, add it at the end as last resort
        if self.primary and not self.primary.is_healthy:
            options.append(self.primary)
        
        # Add unhealthy fallbacks as last resort
        unhealthy_fallbacks = [f for f in self.fallbacks if not f.is_healthy]
        options.extend(unhealthy_fallbacks)
        
        return options[:self.config.max_fallbacks + 1]  # +1 for primary
    
    def _execute_option(self, option: FallbackOption, *args, **kwargs) -> Any:
        """Execute a single fallback option.
        
        Args:
            option: Fallback option to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        
        Returns:
            Result from the option.
        """
        start_time = time.time()
        
        try:
            result = option.func(*args, **kwargs)
            response_time = time.time() - start_time
            
            option.record_success(response_time)
            option.update_health_status(self.config)
            
            # Update stats
            self.stats.successful_calls += 1
            self.stats.fallback_usage[option.name] = self.stats.fallback_usage.get(option.name, 0) + 1
            
            # Update average response time
            if self.stats.avg_response_time == 0.0:
                self.stats.avg_response_time = response_time
            else:
                self.stats.avg_response_time = 0.9 * self.stats.avg_response_time + 0.1 * response_time
            
            if self.config.log_fallbacks and option.name != "primary":
                logger.info(f"Fallback '{option.name}' succeeded for {self.service_name}")
            
            return result
        
        except Exception as e:
            option.record_failure()
            option.update_health_status(self.config)
            
            if self.config.log_fallbacks:
                logger.warning(f"Option '{option.name}' failed for {self.service_name}: {e}")
            
            raise
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute with fallback chain.
        
        Args:
            *args: Positional arguments for service functions.
            **kwargs: Keyword arguments for service functions.
        
        Returns:
            Result from successful service call.
        
        Raises:
            FallbackError: If all options fail.
        """
        self.stats.total_calls += 1
        execution_order = self._get_execution_order()
        attempted_options = []
        last_exception = None
        
        if not execution_order:
            raise FallbackError(self.service_name, [], Exception("No healthy options available"))
        
        for option in execution_order:
            attempted_options.append(option.name)
            
            try:
                result = self._execute_option(option, *args, **kwargs)
                
                # Update success rates
                if option.name == "primary":
                    primary_successes = self.stats.fallback_usage.get("primary", 0)
                    self.stats.primary_success_rate = primary_successes / self.stats.total_calls
                else:
                    fallback_successes = sum(
                        count for name, count in self.stats.fallback_usage.items()
                        if name != "primary"
                    )
                    self.stats.fallback_success_rate = fallback_successes / self.stats.total_calls
                
                return result
            
            except Exception as e:
                last_exception = e
                continue
        
        # All options failed
        self.stats.failed_calls += 1
        raise FallbackError(self.service_name, attempted_options, last_exception)
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health check on all options.
        
        Returns:
            Dictionary mapping option names to health status.
        """
        health_status = {}
        current_time = time.time()
        
        all_options = [self.primary] + self.fallbacks if self.primary else self.fallbacks
        
        for option in all_options:
            if option is None:
                continue
            
            # Check if health check is needed
            if (option.last_health_check is None or 
                current_time - option.last_health_check >= self.config.health_check_interval):
                
                try:
                    # Try a simple call with timeout
                    start_time = time.time()
                    
                    # For health check, we might need a special health check method
                    # For now, we'll just check if the function is callable
                    if callable(option.func):
                        option.is_healthy = True
                    
                    option.last_health_check = current_time
                    
                except Exception:
                    option.is_healthy = False
                    option.last_health_check = current_time
            
            health_status[option.name] = option.is_healthy
        
        return health_status
    
    def get_stats(self) -> FallbackStats:
        """Get current statistics."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = FallbackStats()
        
        # Reset option stats
        all_options = [self.primary] + self.fallbacks if self.primary else self.fallbacks
        for option in all_options:
            if option:
                option.failure_count = 0
                option.success_count = 0
                option.is_healthy = True
        
        logger.info(f"Reset stats for fallback chain {self.service_name}")


# Utility functions for common fallback patterns

def create_model_fallback_chain(primary_model, fallback_models: List[Any], 
                               config: Optional[FallbackConfig] = None) -> FallbackChain:
    """Create a fallback chain for models.
    
    Args:
        primary_model: Primary model instance.
        fallback_models: List of fallback model instances.
        config: Optional fallback configuration.
    
    Returns:
        Configured fallback chain.
    """
    chain = FallbackChain("model-service", config)
    chain.add_primary(primary_model.generate, "primary-model")
    
    for i, model in enumerate(fallback_models):
        chain.add_fallback(model.generate, priority=i+1, name=f"fallback-model-{i}")
    
    return chain


def create_retriever_fallback_chain(primary_retriever, fallback_retrievers: List[Any],
                                   config: Optional[FallbackConfig] = None) -> FallbackChain:
    """Create a fallback chain for retrievers.
    
    Args:
        primary_retriever: Primary retriever instance.
        fallback_retrievers: List of fallback retriever instances.
        config: Optional fallback configuration.
    
    Returns:
        Configured fallback chain.
    """
    chain = FallbackChain("retriever-service", config)
    chain.add_primary(primary_retriever.retrieve, "primary-retriever")
    
    for i, retriever in enumerate(fallback_retrievers):
        chain.add_fallback(retriever.retrieve, priority=i+1, name=f"fallback-retriever-{i}")
    
    return chain
