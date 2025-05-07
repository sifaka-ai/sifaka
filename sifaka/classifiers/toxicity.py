"""
Toxicity classifier using the Detoxify model.

This module provides a toxicity classifier that uses the Detoxify model to detect
toxic content in text. It categorizes text into various toxicity categories including
toxic, severe_toxic, obscene, threat, insult, identity_hate, or non_toxic.

## Architecture

ToxicityClassifier follows the composition over inheritance pattern:
1. **Classifier**: Provides the public API and handles caching
2. **Implementation**: Contains the core classification logic
3. **Factory Function**: Creates a classifier with the toxicity implementation

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
from sifaka.classifiers.toxicity import create_toxicity_classifier

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
    Tuple,
)
from typing_extensions import TypeGuard

from sifaka.classifiers.base import (
    Classifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.state import ClassifierState
from sifaka.classifiers.toxicity_model import ToxicityModel

logger = get_logger(__name__)


class ToxicityClassifierImplementation:
    """
    Implementation of toxicity classification logic using the Detoxify model.

    This implementation uses the Detoxify model to detect toxic content in text.
    It provides a fast, local alternative to API-based toxicity detection and
    can identify various forms of toxic content including severe toxicity,
    obscenity, threats, insults, and identity-based hate.

    ## Architecture

    ToxicityClassifierImplementation follows the composition pattern:
    1. **Core Logic**: classify_impl() implements toxicity detection
    2. **State Management**: Uses ClassifierState for internal state
    3. **Resource Management**: Loads and manages Detoxify model

    ## Lifecycle

    1. **Initialization**: Set up configuration and parameters
       - Initialize with config
       - Extract thresholds from config.params
       - Set up default values

    2. **Warm-up**: Load Detoxify resources
       - Load Detoxify model when needed
       - Initialize only once
       - Handle initialization errors gracefully

    3. **Classification**: Process input text
       - Apply Detoxify toxicity detection
       - Convert scores to standardized format
       - Handle empty text and edge cases

    4. **Result Creation**: Return standardized results
       - Map toxicity scores to labels
       - Convert scores to confidence values
       - Include detailed scores in metadata
    """

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

    def __init__(
        self,
        config: ClassifierConfig,
    ) -> None:
        """
        Initialize the toxicity classifier implementation.

        Args:
            config: Configuration for the classifier
        """
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False
        self._state.cache = {}

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
        if not isinstance(model, ToxicityModel):
            raise ValueError(f"Model must implement ToxicityModel protocol, got {type(model)}")
        return True

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

    def warm_up_impl(self) -> None:
        """
        Initialize the model if needed.

        This method loads the Detoxify model if it hasn't been loaded yet.
        It is called automatically when needed but can also be called
        explicitly to pre-initialize resources.

        Raises:
            RuntimeError: If model initialization fails
        """
        # Check if already initialized
        if not self._state.initialized:
            try:
                # Load model
                self._state.model = self._load_detoxify()

                # Store thresholds in state cache
                self._state.cache["thresholds"] = {
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

                # Store model name in state cache
                self._state.cache["model_name"] = self.config.params.get("model_name", "original")

                # Mark as initialized
                self._state.initialized = True
            except Exception as e:
                logger.error("Failed to initialize toxicity model: %s", e)
                self._state.error = f"Failed to initialize toxicity model: {e}"
                raise RuntimeError(f"Failed to initialize toxicity model: {e}") from e

    def _get_thresholds(self) -> Dict[str, float]:
        """
        Get thresholds for toxicity classification.

        Returns:
            Dictionary of threshold values for different toxicity categories
        """
        # Get thresholds from state cache or use defaults
        if "thresholds" in self._state.cache:
            return self._state.cache["thresholds"]

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

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """
        Implement toxicity classification logic.

        This method contains the core toxicity detection logic using the Detoxify model.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with toxicity scores
        """
        # Ensure resources are initialized
        if not self._state.initialized:
            self.warm_up_impl()

        # Handle empty or whitespace-only text
        if not text.strip():
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={
                    "reason": "empty_input",
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

        try:
            # Get toxicity scores from Detoxify
            scores = self._state.model.predict(text)
            scores = {k: float(v) for k, v in scores.items()}

            # Determine toxicity label and confidence
            label, confidence = self._get_toxicity_label(scores)

            # Return result with detailed metadata
            return ClassificationResult[str](
                label=label,
                confidence=confidence,
                metadata={"all_scores": scores},
            )
        except Exception as e:
            # Log the error and return a fallback result
            logger.error("Failed to classify text: %s", e)
            self._state.error = f"Failed to classify text: {e}"
            return ClassificationResult[str](
                label="unknown",
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "reason": "classification_error",
                },
            )

    def batch_classify_impl(self, texts: List[str]) -> List[ClassificationResult[str]]:
        """
        Implement batch toxicity classification logic.

        This method provides an optimized implementation for classifying multiple
        texts at once, which can be more efficient than calling classify_impl() multiple
        times, especially with the Detoxify model.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults with toxicity scores
        """
        # Ensure resources are initialized
        if not self._state.initialized:
            self.warm_up_impl()

        # Process empty texts
        results = []
        non_empty_texts = []
        non_empty_indices = []

        # First handle empty texts and collect non-empty ones
        for i, text in enumerate(texts):
            if not text.strip():
                results.append(
                    ClassificationResult(
                        label="unknown",
                        confidence=0.0,
                        metadata={
                            "reason": "empty_input",
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
            batch_scores = self._state.model.predict(non_empty_texts)
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
            self._state.error = f"Failed to batch classify texts: {e}"

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
) -> Classifier[str, str]:
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
        Configured Classifier instance with ToxicityClassifierImplementation

    Examples:
        ```python
        from sifaka.classifiers.toxicity import create_toxicity_classifier

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
    # Define default labels
    DEFAULT_LABELS = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
        "non_toxic",
    ]

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

    # Create config
    config = ClassifierConfig(
        labels=kwargs.pop("labels", DEFAULT_LABELS),
        cache_size=cache_size,
        min_confidence=min_confidence,
        cost=cost,
        params=params,
    )

    # Create implementation
    implementation = ToxicityClassifierImplementation(config)

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )


def create_toxicity_classifier_with_custom_model(
    model: ToxicityModel,
    name: str = "custom_toxicity_classifier",
    description: str = "Custom toxicity model",
    cache_size: int = 0,
    min_confidence: float = 0.0,
    cost: int = 2,
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Factory function to create a toxicity classifier with a custom model.

    This function creates a classifier with a custom toxicity model implementation,
    which can be useful for testing or specialized toxicity detection needs.

    Args:
        model: Custom toxicity model implementation
        name: Name of the classifier
        description: Description of the classifier
        cache_size: Size of the classification cache (0 to disable)
        min_confidence: Minimum confidence for classification
        cost: Computational cost of this classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured Classifier instance with ToxicityClassifierImplementation

    Raises:
        ValueError: If the model doesn't implement the ToxicityModel protocol
    """
    # Validate model first
    if not isinstance(model, ToxicityModel):
        raise ValueError(f"Model must implement ToxicityModel protocol, got {type(model)}")

    # Define default labels
    DEFAULT_LABELS = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
        "non_toxic",
    ]

    # Prepare params
    params: Dict[str, Any] = kwargs.pop("params", {})

    # Create config
    config = ClassifierConfig(
        labels=kwargs.pop("labels", DEFAULT_LABELS),
        cache_size=cache_size,
        min_confidence=min_confidence,
        cost=cost,
        params=params,
    )

    # Create implementation
    implementation = ToxicityClassifierImplementation(config)

    # Set the model directly in the implementation's state
    implementation._state.model = model
    implementation._state.initialized = True

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )
