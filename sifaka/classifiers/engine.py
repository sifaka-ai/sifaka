"""
Classifier Engine Module

This module provides the core classification engine for the Sifaka classifiers system.
It coordinates the flow between components, handles caching, and manages state.

## Components
1. **Engine**: Core classification engine that coordinates the flow

## Usage Examples
```python
from sifaka.classifiers.v2.engine import Engine
from sifaka.classifiers.v2.config import ClassifierConfig
from sifaka.classifiers.v2.state import StateTracker

# Create engine
engine = Engine(
    state_tracker=StateTracker(),
    config=ClassifierConfig(cache_enabled=True, cache_size=100)
)

# Classify text
result = engine.classify(
    text="This is a friendly message.",
    implementation=implementation
)

# Access result
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```
"""

from typing import Any, Dict, List, Optional
import time
import logging

from .interfaces import ClassifierImplementation
from .state import StateTracker
from .result import ClassificationResult
from .config import ClassifierConfig
from .errors import (
    ClassifierError, ImplementationError, safely_execute
)

# Configure logger
logger = logging.getLogger(__name__)


class Engine:
    """Core classification engine for the Sifaka classifiers system."""
    
    def __init__(
        self,
        state_tracker: StateTracker,
        config: Optional[ClassifierConfig] = None,
    ):
        """
        Initialize the engine.
        
        Args:
            state_tracker: State tracker for state management
            config: Engine configuration
        """
        self._state_tracker = state_tracker
        self._config = config or ClassifierConfig()
        
        # Initialize state
        self._state_tracker.update("config", self._config)
        self._state_tracker.update("initialized", True)
        self._state_tracker.update("execution_count", 0)
        self._state_tracker.update("result_cache", {})
        
        # Set metadata
        self._state_tracker.set_metadata("component_type", "engine")
        self._state_tracker.set_metadata("creation_time", time.time())
    
    def classify(
        self,
        text: str,
        implementation: ClassifierImplementation,
    ) -> ClassificationResult:
        """
        Classify the given text.
        
        Args:
            text: The text to classify
            implementation: The classifier implementation to use
            
        Returns:
            The classification result
            
        Raises:
            ClassifierError: If classification fails
            ImplementationError: If implementation fails
        """
        # Track execution count
        execution_count = self._state_tracker.get("execution_count", 0)
        self._state_tracker.update("execution_count", execution_count + 1)
        
        # Check cache if enabled
        if self._config.cache_enabled:
            cache = self._state_tracker.get("result_cache", {})
            cache_key = f"{text}_{implementation.__class__.__name__}"
            if cache_key in cache:
                self._state_tracker.set_metadata("cache_hit", True)
                return cache[cache_key]
        
        # Record start time
        start_time = time.time()
        
        try:
            # Create execution snapshot
            snapshot_id = self._state_tracker.create_snapshot(f"execution_{execution_count}")
            
            # Store components in state
            self._state_tracker.update("implementation", implementation)
            self._state_tracker.update("text", text)
            
            # Classify text
            result = self._classify_text(text, implementation)
            
            # Cache result if caching is enabled
            if self._config.cache_enabled:
                cache = self._state_tracker.get("result_cache", {})
                cache_size = self._config.cache_size
                cache_key = f"{text}_{implementation.__class__.__name__}"
                
                # If cache is full, remove oldest entry
                if len(cache) >= cache_size:
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]
                
                # Add result to cache
                cache[cache_key] = result
                self._state_tracker.update("result_cache", cache)
            
            return result
        
        except Exception as e:
            # Track error
            error_count = self._state_tracker.get_metadata("error_count", 0)
            self._state_tracker.set_metadata("error_count", error_count + 1)
            self._state_tracker.set_metadata("last_error", str(e))
            self._state_tracker.set_metadata("last_error_time", time.time())
            
            # Log error
            logger.error(f"Engine execution error: {str(e)}")
            
            # Raise as classifier error
            raise ClassifierError(f"Engine execution failed: {str(e)}")
        
        finally:
            # Record execution time
            end_time = time.time()
            execution_time = end_time - start_time
            self._state_tracker.set_metadata("last_execution_time", execution_time)
            
            # Update average execution time
            avg_time = self._state_tracker.get_metadata("avg_execution_time", 0)
            count = execution_count + 1
            new_avg = (avg_time * (count - 1) + execution_time) / count
            self._state_tracker.set_metadata("avg_execution_time", new_avg)
            
            # Update max execution time
            max_time = self._state_tracker.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_tracker.set_metadata("max_execution_time", execution_time)
    
    def _classify_text(
        self,
        text: str,
        implementation: ClassifierImplementation,
    ) -> ClassificationResult:
        """
        Classify text using the implementation.
        
        Args:
            text: The text to classify
            implementation: The classifier implementation to use
            
        Returns:
            The classification result
            
        Raises:
            ImplementationError: If implementation fails
        """
        def classify_operation():
            return implementation.classify(text)
        
        result = safely_execute(
            operation=classify_operation,
            component_name="implementation",
            component_type="ClassifierImplementation",
            error_class=ImplementationError,
        )
        
        # Check if result meets minimum confidence threshold
        if result.confidence < self._config.min_confidence:
            # Add issue and suggestion
            result = result.with_issues([
                f"Confidence ({result.confidence:.2f}) below threshold ({self._config.min_confidence:.2f})"
            ])
            result = result.with_suggestions([
                "Consider using a different classifier or improving the input text"
            ])
        
        # Track statistics
        stats = self._state_tracker.get_metadata("label_stats", {})
        stats[result.label] = stats.get(result.label, 0) + 1
        self._state_tracker.set_metadata("label_stats", stats)
        
        return result
    
    async def classify_async(
        self,
        text: str,
        implementation: ClassifierImplementation,
    ) -> ClassificationResult:
        """
        Classify the given text asynchronously.
        
        Args:
            text: The text to classify
            implementation: The classifier implementation to use
            
        Returns:
            The classification result
            
        Raises:
            ClassifierError: If classification fails
            ImplementationError: If implementation fails
        """
        # Check if async is enabled
        if not self._config.async_enabled:
            raise ClassifierError("Async execution is not enabled in the configuration")
        
        # Track execution count
        execution_count = self._state_tracker.get("execution_count", 0)
        self._state_tracker.update("execution_count", execution_count + 1)
        
        # Check cache if enabled
        if self._config.cache_enabled:
            cache = self._state_tracker.get("result_cache", {})
            cache_key = f"{text}_{implementation.__class__.__name__}"
            if cache_key in cache:
                self._state_tracker.set_metadata("cache_hit", True)
                return cache[cache_key]
        
        # Record start time
        start_time = time.time()
        
        try:
            # Create execution snapshot
            snapshot_id = self._state_tracker.create_snapshot(f"execution_{execution_count}")
            
            # Store components in state
            self._state_tracker.update("implementation", implementation)
            self._state_tracker.update("text", text)
            
            # Classify text asynchronously
            result = await implementation.classify_async(text)
            
            # Cache result if caching is enabled
            if self._config.cache_enabled:
                cache = self._state_tracker.get("result_cache", {})
                cache_size = self._config.cache_size
                cache_key = f"{text}_{implementation.__class__.__name__}"
                
                # If cache is full, remove oldest entry
                if len(cache) >= cache_size:
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]
                
                # Add result to cache
                cache[cache_key] = result
                self._state_tracker.update("result_cache", cache)
            
            return result
        
        except Exception as e:
            # Track error
            error_count = self._state_tracker.get_metadata("error_count", 0)
            self._state_tracker.set_metadata("error_count", error_count + 1)
            self._state_tracker.set_metadata("last_error", str(e))
            self._state_tracker.set_metadata("last_error_time", time.time())
            
            # Log error
            logger.error(f"Engine execution error: {str(e)}")
            
            # Raise as classifier error
            raise ClassifierError(f"Engine execution failed: {str(e)}")
        
        finally:
            # Record execution time
            end_time = time.time()
            execution_time = end_time - start_time
            self._state_tracker.set_metadata("last_execution_time", execution_time)
            
            # Update average execution time
            avg_time = self._state_tracker.get_metadata("avg_execution_time", 0)
            count = execution_count + 1
            new_avg = (avg_time * (count - 1) + execution_time) / count
            self._state_tracker.set_metadata("avg_execution_time", new_avg)
            
            # Update max execution time
            max_time = self._state_tracker.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_tracker.set_metadata("max_execution_time", execution_time)
