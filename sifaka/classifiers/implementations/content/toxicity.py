"""
Toxicity classifier using the Detoxify model.

This module provides a toxicity classifier that uses the Detoxify model to detect
toxic content in text. It categorizes text into various toxicity categories including
toxic, severe_toxic, obscene, threat, insult, identity_hate, or non_toxic.

## Architecture

ToxicityClassifier follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods (inherited)
2. **Caching Layer**: _classify_impl() handles caching (inherited)
3. **Core Logic**: _classify_impl_uncached() implements toxicity detection
4. **State Management**: Uses StateManager for internal state

## Lifecycle

1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract thresholds from config.params
   - Set up default values

2. **Warm-up**: Load Detoxify resources
   - Load Detoxify model when needed
   - Initialize only once
   - Handle initialization errors gracefully

3. **Classification**: Process input text
   - Validate input text
   - Apply Detoxify toxicity detection
   - Convert scores to standardized format
   - Handle empty text and edge cases

4. **Result Creation**: Return standardized results
   - Map toxicity scores to labels
   - Convert scores to confidence values
   - Include detailed scores in metadata

## Usage Examples

```python
from sifaka.classifiers.implementations.content.toxicity import create_toxicity_classifier

# Create a toxicity classifier with default settings
classifier = create_toxicity_classifier()

# Classify text
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

# Create a classifier with custom thresholds
custom_classifier = create_toxicity_classifier(
    general_threshold=0.3,     # Lower threshold to be more sensitive
    severe_toxic_threshold=0.5,  # Lower threshold for severe toxicity
    threat_threshold=0.5,        # Lower threshold for threats
    cache_size=100               # Enable caching
)

# Batch classify multiple texts
texts = [
    "This is a friendly message.",
    "I hate you and wish you would die.",
    "This product is okay but could be better."
]
results = custom_classifier.batch_classify(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")
```
"""

import importlib
from typing import (
    Any,
    Dict,
    List,
    Optional,
    ClassVar,
    TypeVar,
    Type,
    Tuple,
    # Union,
)
from typing_extensions import TypeGuard
from pydantic import ConfigDict

from sifaka.classifiers.base import BaseClassifier
from sifaka.classifiers.models import ClassificationResult
from sifaka.classifiers.config import ClassifierConfig
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_classifier_state
from sifaka.classifiers.implementations.content.toxicity_model import ToxicityModel
from sifaka.utils.config import extract_classifier_config_params

logger = get_logger(__name__)

# Type variable for the classifier
T = TypeVar("T", bound="ToxicityClassifier")


class ToxicityClassifier(BaseClassifier[str, str]):
    """
    A lightweight toxicity classifier using the Detoxify model.

    This classifier uses the Detoxify model to detect toxic content in text.
    It provides a fast, local alternative to API-based toxicity detection and
    can identify various forms of toxic content including severe toxicity,
    obscenity, threats, insults, and identity-based hate.

    ## Architecture

    ToxicityClassifier follows the standard Sifaka classifier architecture:
    1. **Public API**: classify() and batch_classify() methods (inherited)
    2. **Caching Layer**: _classify_impl() handles caching (inherited)
    3. **Core Logic**: _classify_impl_uncached() implements toxicity detection
    4. **State Management**: Uses StateManager for internal state

    ## Lifecycle

    1. **Initialization**: Set up configuration and parameters
       - Initialize with name, description, and config
       - Extract thresholds from config.params
       - Set up default values

    2. **Warm-up**: Load Detoxify resources
       - Load Detoxify model when needed
       - Initialize only once
       - Handle initialization errors gracefully

    3. **Classification**: Process input text
       - Validate input text
       - Apply Detoxify toxicity detection
       - Convert scores to standardized format
       - Handle empty text and edge cases

    4. **Result Creation**: Return standardized results
       - Map toxicity scores to labels
       - Convert scores to confidence values
       - Include detailed scores in metadata

    ## Examples

    ```python
    from sifaka.classifiers.implementations.content.toxicity import create_toxicity_classifier

    # Create a toxicity classifier with default settings
    classifier = create_toxicity_classifier()

    # Classify text
    result = classifier.classify("This is a friendly message.")
    print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

    # Access detailed scores
    for category, score in result.metadata["all_scores"].items():
        print(f"{category}: {score:.4f}")
    ```

    Requires the 'toxicity' extra to be installed:
    pip install sifaka[toxicity]
    """

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
        "non_toxic",
    ]
    DEFAULT_COST: ClassVar[int] = 2  # Moderate cost for local ML model

    # Default thresholds
    DEFAULT_SEVERE_TOXIC_THRESHOLD: ClassVar[float] = 0.7
    DEFAULT_THREAT_THRESHOLD: ClassVar[float] = 0.7
    DEFAULT_GENERAL_THRESHOLD: ClassVar[float] = 0.5

    # Properties for backward compatibility with tests
    @property
    def _initialized(self) -> bool:
        """Get initialization status from state manager."""
        return self._state_manager.get("initialized", False)

    @property
    def _model(self) -> Any:
        """Get model from state manager."""
        return self._state_manager.get("model")

    @property
    def _model_name(self) -> str:
        """Get model name from config params."""
        return self.config.params.get("model_name", "original")

    def __init__(
        self,
        name: str = "toxicity_classifier",
        description: str = "Detects toxic content using Detoxify",
        config: Optional[ClassifierConfig[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the toxicity classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Set up default configuration if none provided
        if config is None:
            # Extract thresholds from kwargs
            params = kwargs.pop("params", {})
            params.update(
                {
                    "general_threshold": kwargs.pop(
                        "general_threshold", self.DEFAULT_GENERAL_THRESHOLD
                    ),
                    "severe_toxic_threshold": kwargs.pop(
                        "severe_toxic_threshold", self.DEFAULT_SEVERE_TOXIC_THRESHOLD
                    ),
                    "threat_threshold": kwargs.pop(
                        "threat_threshold", self.DEFAULT_THREAT_THRESHOLD
                    ),
                    "model_name": kwargs.pop("model_name", "original"),
                }
            )

            # Create config
            config = ClassifierConfig[str](
                labels=self.DEFAULT_LABELS, cost=self.DEFAULT_COST, params=params, **kwargs
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

    def _validate_model(self, model: Any) -> TypeGuard[ToxicityModel]:
        """
        Validate that a model implements the required protocol.

        Args:
            model: The model to validate

        Returns:
            True if the model is valid

        Raises:
            ValueError: If the model doesn't implement the ToxicityModel protocol
        """
        return self.validate_component(model, ToxicityModel, "Model")

    def _load_detoxify(self) -> ToxicityModel:
        """
        Load the Detoxify package and model.

        Returns:
            Initialized Detoxify model

        Raises:
            ImportError: If the Detoxify package is not installed
            RuntimeError: If Detoxify initialization fails
        """
        try:
            detoxify_module = importlib.import_module("detoxify")
            # Get model name from config params
            model_name = self.config.params.get("model_name", "original")
            model = detoxify_module.Detoxify(model_type=model_name)
            self._validate_model(model)
            return model
        except ImportError:
            raise ImportError(
                "Detoxify package is required for ToxicityClassifier. "
                "Install it with: pip install sifaka[toxicity]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Detoxify: {e}")

    def warm_up(self) -> None:
        """
        Initialize the model if needed.

        This method loads the Detoxify model if it hasn't been loaded yet.
        It is called automatically when needed but can also be called
        explicitly to pre-initialize resources.

        Raises:
            RuntimeError: If model initialization fails
        """
        # Check if already initialized
        if not self._state_manager.get("initialized", False):
            try:
                # Load model
                model = self._load_detoxify()
                self._state_manager.update("model", model)

                # Store thresholds in state cache
                cache = self._state_manager.get("cache", {})
                cache["thresholds"] = {
                    "general_threshold": self.config.params.get(
                        "general_threshold", self.DEFAULT_GENERAL_THRESHOLD
                    ),
                    "severe_toxic_threshold": self.config.params.get(
                        "severe_toxic_threshold", self.DEFAULT_SEVERE_TOXIC_THRESHOLD
                    ),
                    "threat_threshold": self.config.params.get(
                        "threat_threshold", self.DEFAULT_THREAT_THRESHOLD
                    ),
                }
                self._state_manager.update("cache", cache)

                # Store model name in state cache
                cache = self._state_manager.get("cache", {})
                cache["model_name"] = self.config.params.get("model_name", "original")
                self._state_manager.update("cache", cache)

                # Mark as initialized
                self._state_manager.update("initialized", True)
            except Exception as e:
                logger.error("Failed to initialize toxicity model: %s", e)
                self._state_manager.update("error", f"Failed to initialize toxicity model: {e}")
                raise RuntimeError(f"Failed to initialize toxicity model: {e}") from e

    def _get_thresholds(self) -> Dict[str, float]:
        """
        Get thresholds for toxicity classification.

        Returns:
            Dictionary of threshold values for different toxicity categories
        """
        # Get thresholds from state cache or use defaults
        if "thresholds" in self._state_manager.get("cache", {}):
            return self._state_manager.get("cache", {})["thresholds"]

        # If not in cache, use defaults from config
        return {
            "general_threshold": self.config.params.get(
                "general_threshold", self.DEFAULT_GENERAL_THRESHOLD
            ),
            "severe_toxic_threshold": self.config.params.get(
                "severe_toxic_threshold", self.DEFAULT_SEVERE_TOXIC_THRESHOLD
            ),
            "threat_threshold": self.config.params.get(
                "threat_threshold", self.DEFAULT_THREAT_THRESHOLD
            ),
        }

    def _get_toxicity_label(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Get toxicity label and confidence based on scores.

        This method analyzes the toxicity scores from the model and determines
        the most appropriate label and confidence level based on configured thresholds.

        Args:
            scores: Dictionary of toxicity scores from the model

        Returns:
            Tuple of (toxicity_label, confidence)
        """
        # Get thresholds
        thresholds = self._get_thresholds()

        # Check for severe toxicity or threats first (high priority categories)
        if scores.get("severe_toxic", 0) >= thresholds["severe_toxic_threshold"]:
            return "severe_toxic", scores["severe_toxic"]
        elif scores.get("threat", 0) >= thresholds["threat_threshold"]:
            return "threat", scores["threat"]

        # Get highest toxicity score and category
        max_category = max(scores.items(), key=lambda x: x[1])
        label, confidence = max_category

        # Only return toxic label if it meets the general threshold
        if confidence >= thresholds["general_threshold"]:
            return label, confidence

        # Special case: If all toxicity scores are extremely low (e.g., < 0.01),
        # this is clearly non-toxic content, so return high confidence for non_toxic
        max_toxicity_score = max(scores.values()) if scores else 0.0
        if max_toxicity_score < 0.01:
            # If content is extremely non-toxic, return high confidence (0.95)
            return "non_toxic", 0.95

        # Default case: content is non-toxic but with lower confidence
        return "non_toxic", confidence

    def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
        """
        Implement toxicity classification logic without caching.

        This method contains the core toxicity detection logic using the Detoxify model.
        It is called by the caching layer when a cache miss occurs.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with toxicity scores
        """
        # Ensure resources are initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Handle empty or whitespace-only text
        from sifaka.utils.text import handle_empty_text_for_classifier

        empty_result = handle_empty_text_for_classifier(
            text,
            metadata={
                "all_scores": {
                    "toxic": 0.0,
                    "severe_toxic": 0.0,
                    "obscene": 0.0,
                    "threat": 0.0,
                    "insult": 0.0,
                    "identity_hate": 0.0,
                },
            },
        )
        if empty_result:
            return empty_result

        try:
            # Get toxicity scores from Detoxify
            model = self._state_manager.get("model")
            scores = model.predict(text)
            scores = {k: float(v) for k, v in scores.items()}

            # Determine toxicity label and confidence
            label, confidence = self._get_toxicity_label(scores)

            # Create the result
            result = ClassificationResult[str](
                label=label,
                confidence=confidence,
                metadata={"all_scores": scores},
            )

            # Track statistics in state
            stats = self._state_manager.get("statistics", {})
            stats[label] = stats.get(label, 0) + 1
            self._state_manager.update("statistics", stats)

            return result
        except Exception as e:
            # Log the error and return a fallback result
            logger.error("Failed to classify text: %s", e)
            self._state_manager.update("error", f"Failed to classify text: {e}")

            # Track errors in state
            error_info = {"error": str(e), "type": type(e).__name__}
            errors = self._state_manager.get("errors", [])
            errors.append(error_info)
            self._state_manager.update("errors", errors)

            return ClassificationResult[str](
                label="unknown",
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "reason": "classification_error",
                },
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult[str]]:
        """
        Classify multiple texts efficiently.

        This method provides an optimized implementation for classifying multiple
        texts at once, which can be more efficient than calling classify() multiple
        times, especially with the Detoxify model.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults with toxicity scores
        """
        # Validate input
        self.validate_batch_input(texts)

        # Ensure resources are initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Process empty texts
        from sifaka.utils.text import is_empty_text, handle_empty_text_for_classifier

        results = []
        non_empty_texts = []
        non_empty_indices = []

        # First handle empty texts and collect non-empty ones
        for i, text in enumerate(texts):
            if is_empty_text(text):
                results.append(
                    handle_empty_text_for_classifier(
                        text,
                        metadata={
                            "all_scores": {
                                "toxic": 0.0,
                                "severe_toxic": 0.0,
                                "obscene": 0.0,
                                "threat": 0.0,
                                "insult": 0.0,
                                "identity_hate": 0.0,
                            },
                        },
                    )
                )
            else:
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        # If all texts were empty, return early
        if not non_empty_texts:
            return results

        try:
            # Get batch predictions for non-empty texts
            model = self._state_manager.get("model")
            batch_scores = model.predict(non_empty_texts)
            non_empty_results = []

            # Process each non-empty text
            for i in range(len(non_empty_texts)):
                scores = {k: float(v[i]) for k, v in batch_scores.items()}
                label, confidence = self._get_toxicity_label(scores)

                non_empty_results.append(
                    ClassificationResult[str](
                        label=label,
                        confidence=confidence,
                        metadata={"all_scores": scores},
                    )
                )

            # Merge results in the original order
            final_results = [None] * len(texts)
            for i, result in zip(non_empty_indices, non_empty_results):
                final_results[i] = result

            # Fill in the empty text results
            for i, result in enumerate(results):
                if final_results[i] is None:
                    final_results[i] = result

            return final_results
        except Exception as e:
            # Log the error and return fallback results
            logger.error("Failed to batch classify texts: %s", e)
            self._state_manager.update("error", f"Failed to batch classify texts: {e}")

            # Track errors in state
            error_info = {"error": str(e), "type": type(e).__name__, "batch_size": len(texts)}
            errors = self._state_manager.get("errors", [])
            errors.append(error_info)
            self._state_manager.update("errors", errors)

            # Create error results for non-empty texts
            error_results = [
                ClassificationResult[str](
                    label="unknown",
                    confidence=0.0,
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "reason": "batch_classification_error",
                    },
                )
                for _ in non_empty_texts
            ]

            # Merge error results in the original order
            final_results = [None] * len(texts)
            for i, result in zip(non_empty_indices, error_results):
                final_results[i] = result

            # Fill in the empty text results
            for i, result in enumerate(results):
                if final_results[i] is None:
                    final_results[i] = result

            return final_results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier usage statistics.

        This method provides access to statistics collected during classifier operation,
        including classification counts by label, error counts, cache information, and model details.

        Returns:
            Dictionary containing statistics
        """
        stats = {
            # Classification counts by label
            "classifications": self._state_manager.get("statistics", {}),
            # Number of errors encountered
            "error_count": len(self._state_manager.get("errors", [])),
            # Cache information
            "cache_enabled": self.config.cache_size > 0,
            "cache_size": self.config.cache_size,
            # State initialization status
            "initialized": self._state_manager.get("initialized", False),
            # Model information
            "model_name": self._state_manager.get("cache", {}).get("model_name", self._model_name),
        }

        # Add thresholds information
        thresholds = self._get_thresholds()
        stats["thresholds"] = thresholds

        # Add cache hit ratio if caching is enabled
        if hasattr(self, "_result_cache"):
            stats["cache_entries"] = len(self._result_cache)

        return stats

    def clear_cache(self) -> None:
        """
        Clear any cached data in the classifier.

        This method clears both the result cache and resets statistics in the state
        but preserves the model and initialization status.
        """
        # Clear classification result cache
        if hasattr(self, "_result_cache"):
            self._result_cache.clear()

        # Reset statistics
        self._state_manager.update("statistics", {})

        # Reset errors list but keep model and initialized status
        self._state_manager.update("errors", [])

        # Keep the model, thresholds, and model_name in cache
        cache = self._state_manager.get("cache", {})
        preserved_cache = {k: v for k, v in cache.items() if k in ("thresholds", "model_name")}
        self._state_manager.update("cache", preserved_cache)

    @classmethod
    def create(
        cls: Type[T],
        name: str = "toxicity_classifier",
        description: str = "Detects toxic content using Detoxify",
        labels: Optional[List[str]] = None,
        cache_size: int = 0,
        min_confidence: float = 0.0,
        cost: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> T:
        """
        Factory method to create a toxicity classifier.

        Args:
            name: Name of the classifier
            description: Description of what the classifier does
            labels: Optional list of labels (defaults to predefined labels)
            cache_size: Size of the classification cache (0 for no caching)
            min_confidence: Minimum confidence threshold for results
            cost: Computational cost metric (defaults to class default)
            params: Optional dictionary of additional parameters
            **kwargs: Additional configuration parameters

        Returns:
            Configured ToxicityClassifier instance
        """
        from sifaka.utils.config import extract_classifier_config_params

        # Set up default params with thresholds
        default_params = {
            "general_threshold": cls.DEFAULT_GENERAL_THRESHOLD,
            "severe_toxic_threshold": cls.DEFAULT_SEVERE_TOXIC_THRESHOLD,
            "threat_threshold": cls.DEFAULT_THREAT_THRESHOLD,
            "model_name": "original",
        }

        # Extract and merge configuration parameters
        config_dict = extract_classifier_config_params(
            labels=labels if labels else cls.DEFAULT_LABELS,
            cache_size=cache_size,
            min_confidence=min_confidence,
            cost=cost if cost is not None else cls.DEFAULT_COST,
            provided_params=params,
            default_params=default_params,
            **kwargs,
        )

        # Create config with merged parameters
        config = ClassifierConfig[str](**config_dict)

        # Create and return the classifier instance
        return cls(name=name, description=description, config=config)

    @classmethod
    def create_with_custom_model(
        cls: Type[T],
        model: ToxicityModel,
        name: str = "custom_toxicity_classifier",
        description: str = "Custom toxicity model",
        **kwargs: Any,
    ) -> T:
        """
        Factory method to create a classifier with a custom model.

        This method creates a new instance of the classifier with a custom
        toxicity model implementation, which can be useful for testing
        or specialized toxicity detection needs.

        Args:
            model: Custom toxicity model implementation
            name: Name of the classifier
            description: Description of the classifier
            **kwargs: Additional configuration parameters

        Returns:
            Configured ToxicityClassifier instance

        Raises:
            ValueError: If the model doesn't implement the ToxicityModel protocol
        """
        # Validate model first
        if not isinstance(model, ToxicityModel):
            raise ValueError(f"Model must implement ToxicityModel protocol, got {type(model)}")

        # Create instance with extracted parameters and kwargs
        instance = cls(
            name=name,
            description=description,
            **kwargs,
        )

        # Set the model and mark as initialized
        instance._state_manager.update("model", model)
        instance._state_manager.update("initialized", True)

        return instance


def create_toxicity_classifier(
    model_name: str = "original",
    name: str = "toxicity_classifier",
    description: str = "Detects toxic content using Detoxify",
    general_threshold: float = 0.5,
    severe_toxic_threshold: float = 0.7,
    threat_threshold: float = 0.7,
    cache_size: int = 0,
    min_confidence: float = 0.0,
    cost: int = 2,
    **kwargs: Any,
) -> ToxicityClassifier:
    """
    Factory function to create a toxicity classifier.

    This function provides a simpler interface for creating a toxicity classifier
    with the specified parameters, handling the creation of the ClassifierConfig
    object and setting up the classifier with the appropriate parameters.

    Args:
        model_name: Name of the Detoxify model to use ('original', 'unbiased', etc.)
        name: Name of the classifier
        description: Description of the classifier
        general_threshold: General toxicity threshold (0-1)
        severe_toxic_threshold: Severe toxicity threshold (0-1)
        threat_threshold: Threat threshold (0-1)
        cache_size: Size of the classification cache (0 to disable)
        min_confidence: Minimum confidence for classification
        cost: Computational cost of this classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured ToxicityClassifier instance

    Examples:
        ```python
        from sifaka.classifiers.implementations.content.toxicity import create_toxicity_classifier

        # Create a toxicity classifier with default settings
        classifier = create_toxicity_classifier()

        # Create a classifier with custom thresholds
        custom_classifier = create_toxicity_classifier(
            general_threshold=0.3,     # Lower threshold to be more sensitive
            severe_toxic_threshold=0.5,  # Lower threshold for severe toxicity
            threat_threshold=0.5,        # Lower threshold for threats
            cache_size=100               # Enable caching
        )
        ```
    """
    # Prepare params
    params: Dict[str, Any] = kwargs.pop("params", {})
    params.update(
        {
            "model_name": model_name,
            "general_threshold": general_threshold,
            "severe_toxic_threshold": severe_toxic_threshold,
            "threat_threshold": threat_threshold,
        }
    )

    # Create and return classifier using the class factory method
    return ToxicityClassifier.create(
        name=name,
        description=description,
        cache_size=cache_size,
        min_confidence=min_confidence,
        cost=cost,
        params=params,
        **kwargs,
    )
