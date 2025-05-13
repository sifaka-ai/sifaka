"""
Spam classifier using scikit-learn's Naive Bayes.

This module provides a spam classifier that uses scikit-learn's Naive Bayes model
to detect spam content in text. It categorizes text as either 'ham', 'spam', or
'unknown' based on the trained model's prediction.

## Overview
The SpamClassifier is a specialized classifier that leverages scikit-learn's
Naive Bayes algorithm to detect spam content in text. It provides a fast,
trainable approach to spam detection with support for custom datasets,
model saving/loading, and detailed prediction probabilities.

## Architecture
SpamClassifier follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods (inherited)
2. **Caching Layer**: _classify_impl() handles caching (inherited)
3. **Core Logic**: _classify_impl_uncached() implements spam detection
4. **State Management**: Uses StateManager for internal state
5. **Model Management**: Support for training, saving, and loading models
6. **Vectorization**: TF-IDF vectorization with configurable features

## Lifecycle
1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract model path and features from config and config and config and config and config and config and config and config and config and config and config.params
   - Set up default values

2. **Warm-up**: Load model and dependencies
   - Load scikit-learn dependencies when needed
   - Initialize only once
   - Handle initialization errors gracefully

3. **Classification**: Process input text
   - Validate input text
   - Apply spam detection
   - Calculate confidence based on prediction probabilities
   - Handle empty text and edge cases

4. **Result Creation**: Return standardized results
   - Map prediction to labels
   - Include prediction probabilities in metadata
   - Handle errors gracefully

## Usage Examples
```python
from sifaka.classifiers.implementations.content.spam import create_spam_classifier

# Create a spam classifier with default settings
classifier = create_spam_classifier()

# Classify text
result = classifier.classify("Buy now! Limited time offer!") if classifier else ""
print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

# Create a classifier with a pre-trained model
custom_classifier = create_spam_classifier(
    model_path="/path/to/model.pkl",
    cache_size=100
)

# Train a custom classifier
from sifaka.classifiers.implementations.content.spam import SpamClassifier
texts = ["Buy now!", "Hello, how are you?", "Claim your prize!"]
labels = ["spam", "ham", "spam"]
trained_classifier = (SpamClassifier and SpamClassifier.create_pretrained(texts, labels)

# Save the trained model
trained_classifier._save_model("/path/to/save/model.pkl") if trained_classifier else ""
```

## Error Handling
The classifier provides robust error handling:
- ImportError: When scikit-learn is not installed
- RuntimeError: When model initialization or loading fails
- ValueError: When training with invalid data
- Graceful handling of empty or invalid inputs
- Fallback to "unknown" with zero confidence for edge cases

## Configuration
Key configuration options include:
- model_path: Path to a pre-trained model file
- max_features: Maximum number of features for the vectorizer
- use_bigrams: Whether to use bigrams in addition to unigrams
- cache_size: Size of the classification cache (0 to disable)
"""

import importlib
import os
import pickle
from typing import ClassVar, List, Optional, Dict, Any

from pydantic import PrivateAttr
from sifaka.classifiers.classifier import Classifier
from sifaka.core.results import ClassificationResult
from sifaka.utils.config.classifiers import ClassifierConfig
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_classifier_state

logger = get_logger(__name__)


class SpamClassifier(Classifier):
    """
    A spam classifier using Naive Bayes from scikit-learn.

    This classifier detects spam content in text using a trained Naive Bayes model.
    It provides a fast, local alternative to API-based spam detection and
    can be trained on custom datasets.

    ## Architecture
    SpamClassifier follows a component-based architecture:
    - Extends the base Classifier class for consistent interface
    - Uses scikit-learn's Naive Bayes for spam detection
    - Implements TF-IDF vectorization for text feature extraction
    - Provides detailed prediction probabilities in result metadata
    - Uses StateManager for efficient state tracking and caching
    - Supports both synchronous and batch classification
    - Includes model training, saving, and loading capabilities

    ## Lifecycle
    1. **Initialization**: Set up configuration and parameters
       - Initialize with name, description, and config
       - Extract model path and features from config and config and config and config and config and config and config and config and config and config and config.params
       - Set up default values and constants

    2. **Warm-up**: Load model and dependencies
       - Load scikit-learn dependencies when needed (lazy initialization)
       - Initialize only once and cache for reuse
       - Handle initialization errors gracefully with clear messages
       - Load pre-trained model if available or create a new one

    3. **Classification**: Process input text
       - Validate input text and handle edge cases
       - Apply TF-IDF vectorization and Naive Bayes prediction
       - Calculate confidence based on prediction probabilities
       - Handle errors with detailed error information

    4. **Training**: Support for model training
       - Fit on labeled training data
       - Save trained model to disk
       - Support for custom feature configurations

    ## Examples
    ```python
    from sifaka.classifiers.implementations.content.spam import SpamClassifier

    # Create and train a classifier
    texts = ["Buy now!", "Hello friend", "Claim your prize!"]
    labels = ["spam", "ham", "spam"]
    classifier = (SpamClassifier and SpamClassifier.create_pretrained(texts, labels)

    # Classify new text
    result = classifier.classify("Limited time offer!") if classifier else ""
    print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

    # Access prediction probabilities
    probs = result.metadata["probabilities"]
    print(f"Spam probability: {probs['spam']:.2f}")
    print(f"Ham probability: {probs['ham']:.2f}")
    ```

    ## Configuration Options
    - model_path: Path to a pre-trained model file
    - max_features: Maximum number of features for the vectorizer
    - use_bigrams: Whether to use bigrams in addition to unigrams
    - cache_size: Size of the classification cache (0 to disable)
    - labels: Available classification labels (ham/spam)

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = ["ham", "spam"]
    DEFAULT_COST: ClassVar[int] = 1.5  # Slightly higher cost for ML-based model

    # State management
    _state_manager = PrivateAttr(default_factory=create_classifier_state)

    def __init__(
        self,
        name: str = "spam_classifier",
        description: str = "Detects spam content in text",
        config: Optional[Optional[ClassifierConfig[str]]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the spam classifier.

        This method sets up the classifier with the provided name, description,
        and configuration. If no configuration is provided, it creates a default
        configuration with sensible defaults for spam detection.

        Args:
            name: The name of the classifier for identification and logging
            description: Human-readable description of the classifier's purpose
            config: Optional classifier configuration with settings like model path,
                   feature configuration, cache size, and labels
            **kwargs: Additional configuration parameters that will be extracted
                     and added to the config and config and config and config and config and config and config and config and config and config and config.params dictionary
        """
        # Create config if not provided
        if config is None:
            # Extract model path and features from kwargs
            model_params = {
                "model_path": kwargs.pop("model_path", None) if kwargs else None,
                "max_features": kwargs.pop("max_features", 1000) if kwargs else 1000,
                "use_bigrams": kwargs.pop("use_bigrams", True) if kwargs else True,
            }

            # Create config with remaining kwargs
            config = ClassifierConfig[str](
                labels=self.DEFAULT_LABELS, cost=self.DEFAULT_COST, params=model_params, **kwargs
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

        # Try to load model if path is provided
        model_path = (
            self.config.params.get("model_path") if hasattr(self.config, "params") else None
        )
        if model_path and os.path.exists(model_path):
            self.warm_up()

    def _load_dependencies(self) -> bool:
        """
        Load scikit-learn dependencies.

        This method dynamically imports the necessary scikit-learn modules
        required for spam detection. It handles import errors gracefully
        with clear installation instructions.

        Returns:
            True if dependencies were successfully loaded

        Raises:
            ImportError: If scikit-learn is not installed, with instructions
                        on how to install it
            RuntimeError: If module loading fails for other reasons
        """
        try:
            # Import necessary scikit-learn modules
            cache = self._state_manager.get("cache", {})
            cache["sklearn_feature_extraction_text"] = importlib.import_module(
                "sklearn.feature_extraction.text"
            )
            cache["sklearn_naive_bayes"] = importlib.import_module("sklearn.naive_bayes")
            cache["sklearn_pipeline"] = importlib.import_module("sklearn.pipeline")
            self._state_manager.update("cache", cache)

            # Mark dependencies as loaded
            self._state_manager.update("dependencies_loaded", True)
            return True
        except ImportError:
            raise ImportError(
                "scikit-learn is required for SpamClassifier. "
                "Install it with: pip install scikit-learn"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scikit-learn modules: {e}")

    def warm_up(self) -> None:
        """
        Initialize the model if needed.

        This method loads the necessary dependencies and initializes the
        spam detection model. It either loads a pre-trained model from disk
        if a model_path is provided, or creates a new model with default
        parameters. The method ensures initialization happens only once.

        The initialization process includes:
        1. Loading scikit-learn dependencies
        2. Loading a pre-trained model or creating a new one
        3. Setting up the TF-IDF vectorizer and Naive Bayes classifier
        4. Creating the classification pipeline
        5. Marking the model as initialized

        Raises:
            ImportError: If required packages are not installed
            RuntimeError: If model initialization or loading fails
        """
        if not self._state_manager.get("initialized", False):
            # Load dependencies
            self._load_dependencies()

            model_path = (
                self.config.params.get("model_path") if hasattr(self.config, "params") else None
            )
            if model_path and os.path.exists(model_path):
                self._load_model(model_path)
            else:
                # Get parameters from config
                max_features = (
                    self.config.params.get("max_features", 1000)
                    if hasattr(self.config, "params")
                    else 1000
                )
                use_bigrams = (
                    self.config.params.get("use_bigrams", True)
                    if hasattr(self.config, "params")
                    else True
                )

                # Get scikit-learn modules from state
                cache = self._state_manager.get("cache", {})
                sklearn_feature_extraction_text = cache["sklearn_feature_extraction_text"]
                sklearn_naive_bayes = cache["sklearn_naive_bayes"]
                sklearn_pipeline = cache["sklearn_pipeline"]

                # Create TF-IDF vectorizer
                ngram_range = (1, 2) if use_bigrams else (1, 1)
                vectorizer = sklearn_feature_extraction_text.TfidfVectorizer(
                    max_features=max_features,
                    stop_words="english",
                    ngram_range=ngram_range,
                )

                # Create Naive Bayes model
                model = sklearn_naive_bayes.MultinomialNB()

                # Create pipeline
                pipeline = sklearn_pipeline.Pipeline(
                    [
                        ("vectorizer", vectorizer),
                        ("classifier", model),
                    ]
                )

                # Store in state
                self._state_manager.update("vectorizer", vectorizer)
                self._state_manager.update("model", model)
                self._state_manager.update("pipeline", pipeline)

            # Mark as initialized
            self._state_manager.update("initialized", True)

    def _save_model(self, path: str) -> None:
        """
        Save the model to a file.

        This method serializes the trained model pipeline to a pickle file
        at the specified path. This allows the model to be reloaded later
        without retraining.

        Args:
            path: File path where the model should be saved

        Raises:
            RuntimeError: If the model has not been initialized
            IOError: If the file cannot be written
        """
        try:
            with open(path, "wb") as f:
                pickle.dump(self._state_manager.get("pipeline"), f)
            if logger:
                logger.info(f"Model saved to {path}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to save model: {e}")

    def _load_model(self, path: str) -> None:
        """
        Load the model from a file.

        This method deserializes a previously saved model from the specified
        path and initializes the classifier with the loaded components. It
        loads the pipeline and extracts the vectorizer and classifier model.

        Args:
            path: File path from which to load the model

        Raises:
            IOError: If the file cannot be read
            RuntimeError: If the loaded data is invalid or incompatible
        """
        try:
            with open(path, "rb") as f:
                pipeline = pickle.load(f)
                self._state_manager.update("pipeline", pipeline)
                # Extract vectorizer and model from pipeline
                self._state_manager.update("vectorizer", pipeline.named_steps["vectorizer"])
                self._state_manager.update("model", pipeline.named_steps["classifier"])
            if logger:
                logger.info(f"Model loaded from {path}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def fit(self, texts: List[str], labels: List[str]) -> "SpamClassifier":
        """
        Fit the spam classifier on a corpus of texts.

        This method trains the spam classifier using the provided texts and
        their corresponding labels. It creates and trains a TF-IDF vectorizer
        and Naive Bayes classifier pipeline on the training data.

        Args:
            texts: List of texts to fit the model on, which should contain
                  examples of both spam and non-spam (ham) content
            labels: List of labels ("ham" or "spam") corresponding to each text

        Returns:
            self: The fitted classifier, to allow method chaining

        Raises:
            ValueError: If the number of texts and labels don't match
            ValueError: If labels contain values not in the classifier's label set
            RuntimeError: If training fails
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

        if not all(label in self.config.labels for label in labels):
            raise ValueError(f"Labels must be one of {self.config.labels}")

        # Convert string labels to integers for scikit-learn
        label_indices = [self.config.labels.index(label) for label in labels]

        # Ensure model is initialized
        self.warm_up()

        # Fit the pipeline
        pipeline = self._state_manager.get("pipeline")
        pipeline.fit(texts, label_indices)
        self._state_manager.update("pipeline", pipeline)

        # Save the model if path is provided
        model_path = self.config.params.get("model_path")
        if model_path:
            self._save_model(model_path)

        return self

    def _classify_impl_uncached(self, text: str) -> ClassificationResult[Any, str]:
        """
        Implement spam classification logic without caching.

        This method contains the core spam detection logic using the trained
        Naive Bayes model. It is called by the caching layer when a cache miss
        occurs. The method handles the entire classification process, from text
        validation to result creation, including error handling and statistics tracking.

        Args:
            text: The text to classify for spam content

        Returns:
            ClassificationResult with prediction details, containing:
            - label: The classification label (ham, spam, or unknown)
            - confidence: A confidence score between 0.0 and 1.0
            - metadata: Detailed prediction information including probabilities
                       for each class

        Raises:
            RuntimeError: If the model is not initialized
        """
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        try:
            # Get prediction probabilities
            pipeline = self._state_manager.get("pipeline")
            proba = pipeline.predict_proba([text])[0]

            # Get predicted label index and confidence
            label_idx = proba.argmax()
            confidence = float(proba[label_idx])

            # Get label from index
            label = self.config.labels[label_idx]

            result = ClassificationResult[str, Any](
                label=label,
                confidence=confidence,
                metadata={
                    "probabilities": {l: float(p) for l, p in zip(self.config.labels, proba)}
                },
            )

            # Track statistics
            stats = self._state_manager.get("statistics", {})
            stats[label] = stats.get(label, 0) + 1
            self._state_manager.update("statistics", stats)

            return result
        except Exception as e:
            logger.error(f"Failed to classify text: {e}")

            # Track errors in state
            error_info = {"error": str(e), "type": type(e).__name__}
            errors = self._state_manager.get("errors", [])
            errors.append(error_info)
            self._state_manager.update("errors", errors)

            return ClassificationResult[str, Any](
                label="unknown",
                confidence=0.0,
                metadata={"error": str(e), "reason": "classification_error"},
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult[Any, str]]:
        """
        Classify multiple texts efficiently.

        This method efficiently classifies multiple texts in a single batch
        operation, which is more efficient than calling classify() multiple
        times. It applies the classification pipeline to all texts at once,
        then processes the results individually to create standardized
        ClassificationResult objects.

        Args:
            texts: List of texts to classify for spam content

        Returns:
            List of ClassificationResults, one for each input text, with
            classification labels, confidence scores, and detailed metadata
            including prediction probabilities

        Raises:
            RuntimeError: If the model is not initialized
            ValueError: If an empty list is provided
        """
        self.validate_batch_input(texts)

        if not self._state_manager.get("initialized", False):
            self.warm_up()

        try:
            # Predict probabilities for all texts
            pipeline = self._state_manager.get("pipeline")
            probas = pipeline.predict_proba(texts)

            results = []
            for proba in probas:
                label_idx = proba.argmax()
                confidence = float(proba[label_idx])
                label = self.config.labels[label_idx]

                results.append(
                    ClassificationResult[str, Any](
                        label=label,
                        confidence=confidence,
                        metadata={
                            "probabilities": {
                                l: float(p) for l, p in zip(self.config.labels, proba)
                            }
                        },
                    )
                )

                # Track statistics
                stats = self._state_manager.get("statistics", {})
                stats[label] = stats.get(label, 0) + 1
                self._state_manager.update("statistics", stats)

            return results
        except Exception as e:
            logger.error(f"Failed to batch classify texts: {e}")

            # Track errors in state
            error_info = {"error": str(e), "type": type(e).__name__, "batch_size": len(texts)}
            errors = self._state_manager.get("errors", [])
            errors.append(error_info)
            self._state_manager.update("errors", errors)

            return [
                ClassificationResult[str, Any](
                    label="unknown",
                    confidence=0.0,
                    metadata={"error": str(e), "reason": "batch_classification_error"},
                )
                for _ in texts
            ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier usage statistics.

        This method provides access to statistics collected during classifier operation,
        including classification counts by label, error counts, and cache information.
        It aggregates data from the state manager to provide a comprehensive view of the
        classifier's performance and usage patterns.

        Returns:
            Dictionary containing statistics including:
            - classifications: Counts of classifications by label (ham/spam)
            - error_count: Number of errors encountered
            - cache_enabled: Whether caching is enabled
            - cache_size: Maximum cache size
            - initialized: Whether the model has been initialized
            - model_path: Path to the loaded model file (if any)
            - max_features: Maximum number of features used in the vectorizer
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
            "model_path": self.config.params.get("model_path"),
            "max_features": self.config.params.get("max_features", 1000),
        }

        # Add cache entries count if caching is enabled
        if hasattr(self.config, "cache_enabled") and self.config.cache_enabled:
            stats["cache_entries"] = len(self._state_manager.get("result_cache", {}))

        return stats

    def clear_cache(self) -> None:
        """
        Clear any cached data in the classifier.

        This method clears the result cache and resets statistics in the state
        but preserves the model and initialization status. This is useful for
        freeing up memory or resetting usage statistics while keeping the trained
        model intact.

        The method specifically:
        - Clears the classification result cache
        - Resets classification statistics
        - Clears error logs
        - Preserves the trained model and pipeline
        """
        # Clear result cache using standardized state management
        self._state_manager.update("result_cache", {})

        # Reset statistics
        self._state_manager.update("statistics", {})

        # Reset errors list but keep model and initialized status
        self._state_manager.update("errors", [])

    @classmethod
    def create_pretrained(
        cls,
        texts: List[str],
        labels: List[str],
        name: str = "pretrained_spam_classifier",
        description: str = "Pre-trained spam classifier",
        **kwargs,
    ) -> "SpamClassifier":
        """
        Create and train a spam classifier in one step.

        This convenience method creates a new spam classifier instance and
        immediately trains it on the provided data. It combines the instantiation
        and training steps into a single operation for ease of use.

        Args:
            texts: List of texts to train on, which should contain examples
                  of both spam and non-spam (ham) content
            labels: List of labels ("ham" or "spam") corresponding to each text
            name: Name of the classifier for identification and logging
            description: Human-readable description of the classifier's purpose
            **kwargs: Additional configuration parameters including:
                     - model_path: Path to save the trained model
                     - max_features: Maximum features for the vectorizer
                     - use_bigrams: Whether to use bigrams in addition to unigrams
                     - cache_size: Size of the classification cache

        Returns:
            Trained SpamClassifier ready for immediate use

        Raises:
            ValueError: If the number of texts and labels don't match
            ValueError: If labels contain values not in the classifier's label set
            RuntimeError: If training fails
        """
        # Create instance with provided kwargs
        classifier = cls(
            name=name,
            description=description,
            **kwargs,
        )

        # Train the classifier and return it
        return classifier.fit(texts, labels)


def create_spam_classifier(
    name: str = "spam_classifier",
    description: str = "Detects spam content in text",
    model_path: Optional[Optional[str]] = None,
    max_features: int = 1000,
    use_bigrams: bool = True,
    cache_size: int = 0,
    cost: int = 1.5,
    **kwargs,
) -> SpamClassifier:
    """
    Factory function to create a spam classifier.

    This function provides a simpler interface for creating a spam classifier
    with the specified parameters, handling the creation of the ClassifierConfig
    object and setting up the classifier with the appropriate parameters.

    ## Architecture
    The factory function follows a standardized pattern:
    1. Extract and prepare parameters for configuration
    2. Create a configuration dictionary with standardized structure
    3. Pass the configuration to the classifier's constructor
    4. Return the fully configured classifier instance

    ## Examples
    ```python
    # Create with default settings
    classifier = create_spam_classifier()

    # Create with a pre-trained model
    model_classifier = create_spam_classifier(
        model_path="/path/to/model.pkl",
        cache_size=100
    )

    # Create with custom feature settings
    custom_classifier = create_spam_classifier(
        max_features=2000,
        use_bigrams=False,
        name="custom_spam_classifier"
    )
    ```

    Args:
        name: Name of the classifier for identification and logging
        description: Human-readable description of the classifier's purpose
        model_path: Path to a saved model file (optional)
        max_features: Maximum number of features to use in vectorizer
        use_bigrams: Whether to use bigrams in addition to unigrams
        cache_size: Size of the classification cache (0 to disable)
        cost: Computational cost metric for resource allocation decisions
        **kwargs: Additional configuration parameters to pass to the classifier

    Returns:
        Configured SpamClassifier instance ready for immediate use
    """
    # Prepare params
    params = kwargs.pop("params", {})
    params.update(
        {
            "model_path": model_path,
            "max_features": max_features,
            "use_bigrams": use_bigrams,
        }
    )

    # Create config
    config = ClassifierConfig[str](
        labels=SpamClassifier.DEFAULT_LABELS,
        cache_size=cache_size,
        cost=cost,
        params=params,
    )

    # Create classifier
    return SpamClassifier(
        name=name,
        description=description,
        config=config,
        **kwargs,
    )
