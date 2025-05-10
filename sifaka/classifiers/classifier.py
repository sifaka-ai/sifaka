"""
Classifier Module

This module provides the main Classifier class for the Sifaka classifiers system.
It serves as the primary user-facing interface for classification.

## Components
1. **Classifier**: Main user-facing class for classification

## Usage Examples
```python
from sifaka.classifiers.v2 import Classifier
from sifaka.classifiers.v2.implementations.content import ToxicityClassifier

# Create classifier implementation
implementation = ToxicityClassifier()

# Create classifier
classifier = Classifier(
    implementation=implementation,
    name="toxicity_classifier",
    description="Detects toxic content in text"
)

# Classify text
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")

# Classify batch of texts
results = classifier.classify_batch([
    "This is a friendly message.",
    "This is a toxic message!",
    "This is another friendly message."
])
for i, result in enumerate(results):
    print(f"Text {i+1}: {result.label} ({result.confidence:.2f})")
```
"""

from typing import Any, Dict, List, Optional
import time
import logging
import asyncio

from .interfaces import ClassifierImplementation
from .engine import Engine
from .state import StateTracker
from .result import ClassificationResult
from .config import ClassifierConfig
from .errors import ClassifierError

# Configure logger
logger = logging.getLogger(__name__)


class Classifier:
    """Main user-facing class for classification."""
    
    def __init__(
        self,
        implementation: ClassifierImplementation,
        config: Optional[ClassifierConfig] = None,
        name: str = "classifier",
        description: str = "Sifaka classifier for text classification",
    ):
        """
        Initialize the classifier.
        
        Args:
            implementation: The classifier implementation to use
            config: Classifier configuration
            name: Classifier name
            description: Classifier description
        """
        self._name = name
        self._description = description
        self._implementation = implementation
        self._config = config or ClassifierConfig()
        
        # Create state tracker
        self._state_tracker = StateTracker()
        
        # Create engine
        self._engine = Engine(
            state_tracker=self._state_tracker,
            config=self._config,
        )
        
        # Initialize state
        self._state_tracker.update("name", name)
        self._state_tracker.update("description", description)
        self._state_tracker.update("implementation", implementation)
        self._state_tracker.update("config", self._config)
        self._state_tracker.update("initialized", True)
        self._state_tracker.update("execution_count", 0)
        self._state_tracker.update("result_cache", {})
        
        # Set metadata
        self._state_tracker.set_metadata("component_type", "classifier")
        self._state_tracker.set_metadata("creation_time", time.time())
    
    @property
    def name(self) -> str:
        """Get classifier name."""
        return self._name
    
    @property
    def description(self) -> str:
        """Get classifier description."""
        return self._description
    
    @property
    def config(self) -> ClassifierConfig:
        """Get classifier configuration."""
        return self._config
    
    def update_config(self, config: ClassifierConfig) -> None:
        """
        Update classifier configuration.
        
        Args:
            config: New classifier configuration
        """
        self._config = config
        self._state_tracker.update("config", config)
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify the given text.
        
        Args:
            text: The text to classify
            
        Returns:
            The classification result
            
        Raises:
            ClassifierError: If classification fails
        """
        try:
            # Track execution count
            execution_count = self._state_tracker.get("execution_count", 0)
            self._state_tracker.update("execution_count", execution_count + 1)
            
            # Record start time
            start_time = time.time()
            self._state_tracker.set_metadata("execution_start_time", start_time)
            
            # Run engine
            result = self._engine.classify(
                text=text,
                implementation=self._implementation,
            )
            
            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Update statistics
            self._update_statistics(execution_time, success=True)
            
            return result
        
        except Exception as e:
            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Update statistics
            self._update_statistics(execution_time, success=False, error=e)
            
            # Raise as classifier error
            if isinstance(e, ClassifierError):
                raise e
            raise ClassifierError(f"Classification failed: {str(e)}")
    
    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.
        
        Args:
            texts: The texts to classify
            
        Returns:
            List of classification results
            
        Raises:
            ClassifierError: If classification fails
        """
        results = []
        for text in texts:
            try:
                result = self.classify(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify text: {str(e)}")
                # Create fallback result
                results.append(ClassificationResult(
                    label=self._config.params.get("fallback_label", "unknown"),
                    confidence=0.0,
                    metadata={"error": str(e), "error_type": type(e).__name__},
                    issues=[f"Classification failed: {str(e)}"],
                    suggestions=["Try with a different classifier or improve the input text"],
                ))
        return results
    
    async def classify_async(self, text: str) -> ClassificationResult:
        """
        Classify the given text asynchronously.
        
        Args:
            text: The text to classify
            
        Returns:
            The classification result
            
        Raises:
            ClassifierError: If classification fails
        """
        # Check if async is enabled
        if not self._config.async_enabled:
            raise ClassifierError("Async execution is not enabled in the configuration")
        
        try:
            # Track execution count
            execution_count = self._state_tracker.get("execution_count", 0)
            self._state_tracker.update("execution_count", execution_count + 1)
            
            # Record start time
            start_time = time.time()
            self._state_tracker.set_metadata("execution_start_time", start_time)
            
            # Run engine
            result = await self._engine.classify_async(
                text=text,
                implementation=self._implementation,
            )
            
            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Update statistics
            self._update_statistics(execution_time, success=True)
            
            return result
        
        except Exception as e:
            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Update statistics
            self._update_statistics(execution_time, success=False, error=e)
            
            # Raise as classifier error
            if isinstance(e, ClassifierError):
                raise e
            raise ClassifierError(f"Classification failed: {str(e)}")
    
    async def classify_batch_async(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts asynchronously.
        
        Args:
            texts: The texts to classify
            
        Returns:
            List of classification results
            
        Raises:
            ClassifierError: If classification fails
        """
        # Check if async is enabled
        if not self._config.async_enabled:
            raise ClassifierError("Async execution is not enabled in the configuration")
        
        # Create tasks for all texts
        tasks = [self.classify_async(text) for text in texts]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Failed to classify text: {str(result)}")
                # Create fallback result
                processed_results.append(ClassificationResult(
                    label=self._config.params.get("fallback_label", "unknown"),
                    confidence=0.0,
                    metadata={"error": str(result), "error_type": type(result).__name__},
                    issues=[f"Classification failed: {str(result)}"],
                    suggestions=["Try with a different classifier or improve the input text"],
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _update_statistics(
        self,
        execution_time: float,
        success: bool,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Update classifier statistics.
        
        Args:
            execution_time: Execution time in seconds
            success: Whether execution was successful
            error: Optional error that occurred
        """
        # Update success/failure count
        if success:
            success_count = self._state_tracker.get_metadata("success_count", 0)
            self._state_tracker.set_metadata("success_count", success_count + 1)
        else:
            failure_count = self._state_tracker.get_metadata("failure_count", 0)
            self._state_tracker.set_metadata("failure_count", failure_count + 1)
        
        # Update execution time statistics
        self._state_tracker.set_metadata("last_execution_time", execution_time)
        
        avg_time = self._state_tracker.get_metadata("avg_execution_time", 0)
        count = self._state_tracker.get("execution_count", 0)
        new_avg = (avg_time * (count - 1) + execution_time) / count
        self._state_tracker.set_metadata("avg_execution_time", new_avg)
        
        max_time = self._state_tracker.get_metadata("max_execution_time", 0)
        if execution_time > max_time:
            self._state_tracker.set_metadata("max_execution_time", execution_time)
        
        # Update error statistics if applicable
        if error:
            error_count = self._state_tracker.get_metadata("error_count", 0)
            self._state_tracker.set_metadata("error_count", error_count + 1)
            self._state_tracker.set_metadata("last_error", str(error))
            self._state_tracker.set_metadata("last_error_time", time.time())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier statistics.
        
        Returns:
            Dictionary with classifier statistics
        """
        return {
            "name": self._name,
            "execution_count": self._state_tracker.get("execution_count", 0),
            "success_count": self._state_tracker.get_metadata("success_count", 0),
            "failure_count": self._state_tracker.get_metadata("failure_count", 0),
            "error_count": self._state_tracker.get_metadata("error_count", 0),
            "avg_execution_time": self._state_tracker.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state_tracker.get_metadata("max_execution_time", 0),
            "last_execution_time": self._state_tracker.get_metadata("last_execution_time", 0),
            "last_error": self._state_tracker.get_metadata("last_error", None),
            "last_error_time": self._state_tracker.get_metadata("last_error_time", None),
            "cache_size": len(self._state_tracker.get("result_cache", {})),
            "label_stats": self._state_tracker.get_metadata("label_stats", {}),
        }
    
    def clear_cache(self) -> None:
        """Clear the classifier result cache."""
        self._state_tracker.update("result_cache", {})
        logger.debug("Classifier cache cleared")
    
    def reset_state(self) -> None:
        """Reset classifier state."""
        self._state_tracker.reset()
        
        # Re-initialize state
        self._state_tracker.update("name", self._name)
        self._state_tracker.update("description", self._description)
        self._state_tracker.update("implementation", self._implementation)
        self._state_tracker.update("config", self._config)
        self._state_tracker.update("initialized", True)
        self._state_tracker.update("execution_count", 0)
        self._state_tracker.update("result_cache", {})
        
        logger.debug("Classifier state reset")
