"""
Bias detection classifiers for Sifaka.

This module provides classifiers that detect various types of bias in text,
including gender bias, racial bias, and other forms of discriminatory language.

## Overview
The BiasDetector is a specialized classifier that leverages machine learning models
to detect various forms of bias in text, including gender bias, racial bias, and other
forms of stereotyping. It provides a standardized interface for bias detection with
configurable thresholds and detailed metadata.

## Architecture
BiasDetector follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods (inherited)
2. **Caching Layer**: _classify_impl() handles caching (inherited)
3. **Core Logic**: _classify_impl_uncached() implements bias detection
4. **State Management**: Uses StateManager for internal state
5. **Thresholds**: Configurable threshold for bias detection
6. **Model Loading**: On-demand loading of the bias detection model

## Lifecycle
1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract thresholds from config and config and config and config and config and config and config and config and config and config and config and config and config and config and config and config.params
   - Set up default values

2. **Warm-up**: Load model resources
   - Load model when needed
   - Initialize only once
   - Handle initialization errors gracefully

3. **Classification**: Process input text
   - Validate input text
   - Apply bias detection
   - Convert scores to standardized format
   - Handle empty text and edge cases

4. **Result Creation**: Return standardized results
   - Map scores to bias labels
   - Convert scores to confidence values
   - Include detailed scores in metadata

## Usage Examples
```python
from sifaka.classifiers.implementations.content.bias import BiasDetector

# Create a bias detector with default settings
detector = BiasDetector()

# Classify text
result = (detector and detector.classify("Men are better at math than women.")
print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

# Create a detector with custom threshold
custom_detector = BiasDetector(
    config=ClassifierConfig(
        min_confidence=0.3,  # More sensitive bias detection
        cache_size=100       # Enable caching
    )
)

# Batch classify multiple texts
texts = [
    "Men are better at math than women.",
    "People of all genders can excel at mathematics.",
    "This product is excellent."
]
results = (custom_detector and custom_detector.batch_classify(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

# Access detailed bias information
result = (detector and detector.classify("Women are too emotional to be leaders.")
print(f"Bias type: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Features: {result.(metadata and metadata.get('features', {})}")
```

## Error Handling
The classifier provides robust error handling:
- ImportError: When required model packages are not installed
- RuntimeError: When model initialization fails
- Graceful handling of empty or invalid inputs
- Fallback to "neutral" with low confidence for edge cases

## Configuration
Key configuration options include:
- min_confidence: Threshold for bias detection (default: 0.7)
- model_path: Path to a pre-trained model file
- max_features: Maximum number of features for the vectorizer
- cache_size: Size of the classification cache (0 to disable)
"""

import os
import re
import pickle
import importlib
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from pydantic import ConfigDict, PrivateAttr

from sifaka.classifiers.classifier import Classifier
from sifaka.core.results import ClassificationResult
from sifaka.utils.config.classifiers import ClassifierConfig
from sifaka.utils.logging import get_logger
from sifaka.utils.config.classifiers import extract_classifier_config_params

logger = get_logger(__name__)

# Default bias types to detect
DEFAULT_BIAS_TYPES: List[str] = [
    "gender",
    "racial",
    "political",
    "age",
    "socioeconomic",
    "religious",
    "cultural",
    "educational",
    "geographical",
    "neutral",
]


class BiasDetector(Classifier):
    """
    A bias detector using Support Vector Machines from scikit-learn.

    This classifier identifies various forms of bias in text, including:
    - Gender bias
    - Racial bias
    - Political bias
    - Age bias
    - Socioeconomic bias
    - Religious bias
    - Cultural bias
    - Educational bias
    - Geographical bias

    It also includes a "neutral" category for unbiased text.

    ## Architecture
    BiasDetector follows a component-based architecture:
    - Extends the base Classifier class for consistent interface
    - Uses scikit-learn's SVM models for bias detection
    - Implements configurable thresholds for bias categories
    - Provides detailed bias information in result metadata
    - Uses StateManager for efficient state tracking and caching
    - Supports both synchronous and batch classification
    - Includes model training, saving, and loading capabilities

    ## Lifecycle
    1. **Initialization**: Set up configuration and parameters
       - Initialize with name, description, and config
       - Extract thresholds from config and config and config and config and config and config and config and config and config and config and config and config and config and config and config and config.params
       - Set up default values and constants

    2. **Warm-up**: Load model resources
       - Load scikit-learn dependencies when needed (lazy initialization)
       - Initialize only once and cache for reuse
       - Handle initialization errors gracefully with clear messages

    3. **Classification**: Process input text
       - Validate input text and handle edge cases
       - Apply bias detection algorithms
       - Convert scores to standardized format
       - Apply thresholds to determine bias categories
       - Extract and include bias-specific features

    4. **Result Creation**: Return standardized results
       - Map bias scores to appropriate labels
       - Convert scores to confidence values
       - Include detailed bias information in metadata for transparency
       - Track statistics for monitoring and debugging

    ## Examples
    ```python
    from sifaka.classifiers.implementations.content.bias import BiasDetector
    from sifaka.utils.config and config and config and config.classifiers import ClassifierConfig

    # Create a bias detector with default settings
    detector = BiasDetector()

    # Classify text
    result = (detector and detector.classify("Men are better at math than women.")
    print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

    # Train a custom bias detector
    texts = ["Men are better at math", "Women are too emotional", "This is neutral"]
    labels = ["gender", "gender", "neutral"]
    custom_detector = BiasDetector()
    (custom_detector and custom_detector.fit(texts, labels)

    # Save the trained model
    (custom_detector and custom_detector._save_model("bias_model.pkl")

    # Load a pre-trained model
    loaded_detector = BiasDetector(
        config=ClassifierConfig(
            params={"model_path": "bias_model.pkl"}
        )
    )
    ```

    ## Configuration Options
    - min_confidence: Threshold for bias detection (default: 0.7)
    - model_path: Path to a pre-trained model file
    - max_features: Maximum number of features for the vectorizer
    - random_state: Random seed for reproducibility
    - bias_keywords: Dictionary of bias types and associated keywords

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class constants
    DEFAULT_COST: ClassVar[float] = 2.5

    # State management
    _state_manager = PrivateAttr(default_factory=lambda: {})

    # Default keywords for each bias category to enhance detection
    DEFAULT_BIAS_KEYWORDS: ClassVar[Dict[str, List[str]]] = {
        "gender": ["man", "woman", "male", "female", "gender", "sex", "sexual"],
        "racial": [
            "race",
            "ethnic",
            "minority",
            "white",
            "black",
            "asian",
            "hispanic",
            "indigenous",
        ],
        "political": [
            "conservative",
            "liberal",
            "right-wing",
            "left-wing",
            "republican",
            "democrat",
        ],
        "age": ["young", "old", "elderly", "teen", "millennial", "boomer", "generation"],
        "socioeconomic": ["rich", "poor", "wealthy", "poverty", "class", "income", "privilege"],
        "religious": ["christian", "muslim", "jewish", "hindu", "atheist", "faith", "belief"],
        "cultural": ["western", "eastern", "traditional", "modern", "heritage", "customs"],
        "educational": [
            "educated",
            "uneducated",
            "academic",
            "intellectual",
            "school",
            "college",
        ],
        "geographical": ["urban", "rural", "city", "country", "coastal", "inland", "regional"],
    }

    def __init__(
        self,
        name: str = "bias_detector",
        description: str = "Detects various forms of bias in text",
        config: Optional[Optional[ClassifierConfig]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the bias detector.

        This method sets up the classifier with the provided name, description,
        and configuration. If no configuration is provided, it creates a default
        configuration with sensible defaults for bias detection.

        Args:
            name: The name of the classifier for identification and logging
            description: Human-readable description of the classifier's purpose
            config: Optional classifier configuration with settings like thresholds,
                   cache size, and labels
            **kwargs: Additional configuration parameters that will be extracted
                     and added to the config and config and config and config and config and config and config and config and config and config and config and config and config and config and config and config.params dictionary
        """
        # Create default config if not provided
        if config is None:
            params = kwargs and kwargs.pop("params", {})

            # Add default bias types and keywords if not provided
            if "bias_types" not in params:
                params["bias_types"] = DEFAULT_BIAS_TYPES
            if "bias_keywords" not in params:
                params["bias_keywords"] = self.DEFAULT_BIAS_KEYWORDS

            config = ClassifierConfig(
                labels=(params and params.get("bias_types", DEFAULT_BIAS_TYPES)),
                cost=self.DEFAULT_COST,
                min_confidence=(params and params.get("min_confidence", 0.7)),
                params=params,
            )

        super().__init__(name=name, description=description, config=config)

    def _load_dependencies(self) -> Dict[str, Any]:
        """
        Load scikit-learn dependencies.

        This method dynamically imports the necessary scikit-learn modules
        required for bias detection. It handles import errors gracefully
        with clear installation instructions.

        Returns:
            Dictionary mapping module names to imported module objects

        Raises:
            ImportError: If scikit-learn is not installed, with instructions
                        on how to install it
            RuntimeError: If module loading fails for other reasons
        """
        try:
            # Import necessary scikit-learn modules
            sklearn_modules = {
                "feature_extraction_text": (
                    importlib and importlib.import_module("sklearn.feature_extraction.text")
                ),
                "svm": (importlib and importlib.import_module("sklearn.svm")),
                "pipeline": (importlib and importlib.import_module("sklearn.pipeline")),
                "calibration": (importlib and importlib.import_module("sklearn.calibration")),
            }
            return sklearn_modules
        except ImportError:
            raise ImportError(
                "scikit-learn is required for BiasDetector. "
                "Install it with: pip install scikit-learn"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scikit-learn modules: {e}")

    def warm_up(self) -> None:
        """
        Initialize the model if needed.

        This method loads the necessary dependencies and initializes the
        bias detection model. It either loads a pre-trained model from disk
        if a model_path is provided, or creates a new model with default
        parameters. The method ensures initialization happens only once.

        The initialization process includes:
        1. Loading scikit-learn dependencies
        2. Creating or loading vectorizers and models
        3. Setting up the classification pipeline
        4. Marking the model as initialized

        Raises:
            ImportError: If required packages are not installed
            RuntimeError: If model initialization fails
        """
        # Check if already initialized
        if self._state_manager and self._state_manager.get("initialized", False):
            return

        # Load dependencies
        sklearn = self._load_dependencies()
        self._state_manager.update("dependencies_loaded", True)

        # Get configuration from params
        model_path = (
            self.config.params.get("model_path") if hasattr(self.config, "params") else None
        )
        max_features = (
            self.config.params.get("max_features", 3000) if hasattr(self.config, "params") else 3000
        )
        random_state = (
            self.config.params.get("random_state", 42) if hasattr(self.config, "params") else 42
        )

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # Create TF-IDF vectorizer with custom analyzer to catch bias keywords
            vectorizer = sklearn["feature_extraction_text"].TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
                ngram_range=(1, 2),
            )
            self._state_manager.update("vectorizer", vectorizer)

            # Create SVM model with probability estimates
            svm = sklearn["svm"].LinearSVC(
                random_state=random_state,
                class_weight="balanced",
                max_iter=10000,
            )

            # Calibrate the SVM to get probability estimates
            model = sklearn["calibration"].CalibratedClassifierCV(
                estimator=svm,
                cv=3,
            )
            self._state_manager.update("model", model)

            # Create pipeline
            pipeline = sklearn["pipeline"].Pipeline(
                [
                    ("vectorizer", vectorizer),
                    ("classifier", model),
                ]
            )
            self._state_manager.update("pipeline", pipeline)

        # Mark as initialized
        self._state_manager.update("initialized", True)

    def _extract_bias_features(self, text: str) -> Dict[str, float]:
        """
        Extract bias-related features from text.

        This method analyzes the input text for bias-related keywords and
        calculates feature scores for each bias category. It uses the
        bias_keywords dictionary from the configuration to identify
        relevant keywords for each bias type.

        Args:
            text: The text to analyze for bias features

        Returns:
            Dictionary mapping bias feature names to their normalized scores
        """
        features = {}
        # Get bias keywords from params or use default empty dict
        bias_keywords = (
            self.config.params.get("bias_keywords", {}) if hasattr(self.config, "params") else {}
        )
        for bias_type, keywords in bias_keywords.items():
            count = sum(1 for kw in keywords if re.search(rf"\b{kw}\b", text.lower()))
            features[f"bias_{bias_type}"] = count / max(len(keywords), 1)  # Avoid division by zero
        return features

    def _save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        This method serializes the trained model components (vectorizer,
        classifier, pipeline) and configuration parameters to a pickle file
        at the specified path. This allows the model to be reloaded later
        without retraining.

        Args:
            path: File path where the model should be saved

        Raises:
            RuntimeError: If the model has not been initialized
            IOError: If the file cannot be written
        """
        # Check if initialized
        if not self._state_manager or not self._state_manager.get("initialized", False):
            raise RuntimeError("Model not initialized")

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": (
                        self._state_manager.get("vectorizer") if self._state_manager else None
                    ),
                    "model": self._state_manager.get("model") if self._state_manager else None,
                    "pipeline": (
                        self._state_manager.get("pipeline") if self._state_manager else None
                    ),
                    "config_params": self.config.params if self.config else None,
                    "feature_names": (
                        self._state_manager.get("feature_names") if self._state_manager else None
                    ),
                },
                f,
            )

    def _load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        This method deserializes a previously saved model from the specified
        path and initializes the classifier with the loaded components. It
        loads the vectorizer, classifier model, pipeline, and configuration
        parameters.

        Args:
            path: File path from which to load the model

        Raises:
            IOError: If the file cannot be read
            RuntimeError: If the loaded data is invalid or incompatible
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
            if self._state_manager:
                self._state_manager.update("vectorizer", data["vectorizer"])
                self._state_manager.update("model", data["model"])
                self._state_manager.update("pipeline", data["pipeline"])

            # Load feature names if available
            if "feature_names" in data and self._state_manager:
                self._state_manager.update("feature_names", data["feature_names"])

            # Update config params if available in the saved model
            if "config_params" in data and self.config:
                # Create a new config with the loaded params
                self.config = ClassifierConfig(
                    labels=self.config.labels if self.config else [],
                    cost=self.config.cost if self.config else 0.0,
                    min_confidence=self.config.min_confidence if self.config else 0.7,
                    params=data["config_params"],
                )

            # Mark as initialized
            if self._state_manager:
                self._state_manager.update("initialized", True)

    def fit(self, texts: List[str], labels: List[str]) -> "BiasDetector":
        """
        Train the bias detector on labeled examples.

        This method trains the bias detector using the provided texts and
        their corresponding bias labels. It creates and trains a TF-IDF
        vectorizer and SVM classifier pipeline on the training data.

        Args:
            texts: List of text examples for training
            labels: List of corresponding bias labels for each text

        Returns:
            Self, to allow method chaining

        Raises:
            ValueError: If empty training data is provided
            RuntimeError: If training fails
        """
        if not texts or not labels:
            raise ValueError("Empty training data")

        # Load scikit-learn dependencies
        sklearn = self._load_dependencies()
        if self._state_manager:
            self._state_manager.update("dependencies_loaded", True)

        # Get configuration from params
        max_features = 3000
        random_state = 42
        if self.config and hasattr(self.config, "params"):
            max_features = self.config.params.get("max_features", 3000)
            random_state = self.config.params.get("random_state", 42)

        # Create vectorizer
        vectorizer = sklearn["feature_extraction_text"].TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
        )
        if self._state_manager:
            self._state_manager.update("vectorizer", vectorizer)

        # Create and train SVM classifier
        model = sklearn["svm"].SVC(
            kernel="linear",
            probability=True,
            random_state=random_state,
        )
        if self._state_manager:
            self._state_manager.update("model", model)

        # Create pipeline
        pipeline = sklearn["pipeline"].Pipeline(
            [
                ("vectorizer", vectorizer),
                ("classifier", model),
            ]
        )

        # Fit pipeline
        pipeline.fit(texts, labels)
        if self._state_manager:
            self._state_manager.update("pipeline", pipeline)
            self._state_manager.update("initialized", True)

        # Extract feature explanations
        self._extract_explanations()

        # Save model if path specified
        model_path = None
        if self.config and hasattr(self.config, "params"):
            model_path = self.config.params.get("model_path")
        if model_path:
            self._save_model(model_path)

        return self

    def _extract_explanations(self) -> None:
        """
        Extract feature coefficients for explanations.

        This method extracts the feature coefficients from the trained SVM model
        to provide explanations for bias classifications. It identifies the most
        important positive and negative features for each bias category and
        stores them in the state manager for later use in result metadata.

        The explanations help users understand why a particular text was
        classified as having a specific type of bias by showing which words
        or phrases contributed most to the classification decision.

        Raises:
            RuntimeError: If feature extraction fails
        """
        try:
            if not self._state_manager:
                return

            model = self._state_manager.get("model")
            if not model or not hasattr(model, "base_estimator"):
                return

            # Get feature names
            vectorizer = self._state_manager.get("vectorizer")
            if not vectorizer:
                return

            feature_names = vectorizer.get_feature_names_out()

            # Store feature names in state
            self._state_manager.update(
                "feature_names",
                {
                    "names": feature_names,
                    "count": len(feature_names),
                },
            )

            # Extract coefficients from the SVM
            if hasattr(model.base_estimator, "coef_"):
                coefficients = model.base_estimator.coef_

                # For each class, extract the top features
                explanations = {}
                # Use labels from config instead of bias_types
                if self.config and hasattr(self.config, "labels"):
                    for i, bias_type in enumerate(self.config.labels):
                        # Skip if this is beyond the number of classes in the model
                        if i >= coefficients.shape[0]:
                            continue

                        # Get the coefficients for this class
                        class_coef = coefficients[i]

                        # Get the top positive and negative features
                        top_pos_indices = class_coef.argsort()[-20:][::-1]
                        top_neg_indices = class_coef.argsort()[:20]

                        explanations[bias_type] = {
                            "positive": {
                                feature_names[j]: float(class_coef[j]) for j in top_pos_indices
                            },
                            "negative": {
                                feature_names[j]: float(class_coef[j]) for j in top_neg_indices
                            },
                        }

                # Store explanations in state cache
                cache = self._state_manager.get("cache", {})
                cache["explanations"] = explanations
                self._state_manager.update("cache", cache)
        except Exception as e:
            logger.warning(f"Could not extract explanations: {e}")
            if self._state_manager:
                self._state_manager.update("error", f"Failed to extract explanations: {e}")

                # Track errors in state
                error_info = {
                    "error": str(e),
                    "type": type(e).__name__,
                    "context": "extract_explanations",
                }
                errors = self._state_manager.get("errors", [])
                errors.append(error_info)
                self._state_manager.update("errors", errors)

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement classification logic.

        This method contains the core bias detection logic. It ensures the
        model is initialized, applies the classification pipeline to the
        input text, and creates a standardized result with confidence scores
        and detailed metadata.

        Args:
            text: The text to classify for bias

        Returns:
            ClassificationResult with bias label, confidence score, and metadata

        Raises:
            RuntimeError: If the model is not initialized
        """
        # Check if initialized
        if not self._state_manager or not self._state_manager.get("initialized", False):
            self.warm_up()

        # Check if still not initialized
        if not self._state_manager or not self._state_manager.get("initialized", False):
            raise RuntimeError("Model not initialized")

        # Get prediction and probability
        pipeline = self._state_manager.get("pipeline") if self._state_manager else None
        if not pipeline:
            raise RuntimeError("Pipeline not initialized")

        label = pipeline.predict([text])[0]
        probs = pipeline.predict_proba([text])[0]
        confidence = max(probs)

        # Extract bias features
        bias_features = self._extract_bias_features(text)

        # Get explanations from cache
        explanations = {}
        if self._state_manager:
            explanations = (
                self._state_manager.get("cache", {}).get("explanations", {}).get(label, {})
            )

        result = ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "features": bias_features,
                "explanations": explanations,
            },
        )

        # Track statistics
        if self._state_manager:
            stats = self._state_manager.get("statistics", {})
            stats[label] = stats.get(label, 0) + 1
            self._state_manager.update("statistics", stats)

        return result

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts in batch.

        This method efficiently classifies multiple texts in a single batch
        operation, which is more efficient than calling classify() multiple
        times. It applies the classification pipeline to all texts at once,
        then processes the results individually to create standardized
        ClassificationResult objects.

        Args:
            texts: List of texts to classify for bias

        Returns:
            List of ClassificationResults, one for each input text, with
            bias labels, confidence scores, and detailed metadata

        Raises:
            RuntimeError: If the model is not initialized
            ValueError: If an empty list is provided
        """
        # Check if initialized
        if not self._state_manager or not self._state_manager.get("initialized", False):
            self.warm_up()

        # Check if still not initialized
        pipeline = self._state_manager.get("pipeline") if self._state_manager else None
        if not pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Get model predictions for all texts
        probas = pipeline.predict_proba(texts)

        results = []
        for i, text in enumerate(texts):
            # Extract bias features
            bias_features = self._extract_bias_features(text)

            # Get probabilities for this text
            proba = probas[i]

            # Create dictionary of probabilities for each label
            all_probs = {}
            if self.config and hasattr(self.config, "labels"):
                all_probs = {self.config.labels[j]: float(prob) for j, prob in enumerate(proba)}

            # Get the most likely bias type
            label = max(all_probs.items(), key=lambda x: x[1])[0] if all_probs else "unknown"
            confidence = all_probs.get(label, 0.0)

            # Get min_confidence from config
            min_confidence = 0.7
            if self.config and hasattr(self.config, "min_confidence"):
                min_confidence = self.config.min_confidence

            # Get explanations from cache
            explanations = {}
            if self._state_manager:
                explanations = (
                    self._state_manager.get("cache", {}).get("explanations", {}).get(label, {})
                )

            metadata = {
                "probabilities": all_probs,
                "threshold": min_confidence,
                "is_confident": confidence >= min_confidence,
                "bias_features": bias_features,
                "explanations": explanations,
            }

            results.append(
                ClassificationResult(
                    label=label,
                    confidence=confidence,
                    metadata=metadata,
                )
            )

            # Track statistics for this label
            if self._state_manager:
                stats = self._state_manager.get("statistics", {})
                stats[label] = stats.get(label, 0) + 1
                self._state_manager.update("statistics", stats)

        return results

    def get_bias_explanation(self, bias_type: str, text: str) -> Dict[str, Any]:
        """
        Get detailed explanation about why a text was classified with a particular bias.

        This method provides a detailed explanation of why a specific text was
        classified as having a particular type of bias. It extracts and analyzes
        the features that contributed most to the classification decision and
        returns them in a structured format for user understanding.

        Args:
            bias_type: The bias type to explain (e.g., "gender", "racial")
            text: The text that was classified and needs explanation

        Returns:
            Dictionary with explanation details including:
            - top_features: Words or phrases that contributed most to the classification
            - feature_scores: Numerical scores for each feature
            - bias_keywords: Keywords associated with this bias type
            - overall_score: Overall bias score for this category
        """
        if (
            not self.config
            or not hasattr(self.config, "labels")
            or bias_type not in self.config.labels
        ):
            valid_labels = (
                self.config.labels if self.config and hasattr(self.config, "labels") else []
            )
            raise ValueError(f"Invalid bias type: {bias_type}. Must be one of {valid_labels}")

        # Get the classification result without explanation to avoid recursion
        result = self._classify_impl(text)

        # Extract the features that contributed to this bias type
        explanation = {}
        if self._state_manager:
            explanations = self._state_manager.get("cache", {}).get("explanations", {})
            if explanations and bias_type in explanations:
                explanation = explanations[bias_type]

        # Get bias-specific features
        bias_features = self._extract_bias_features(text)

        # Get the probability for this bias type
        bias_probability = 0.0
        if hasattr(result, "metadata") and result.metadata:
            bias_probability = result.metadata.get("probabilities", {}).get(bias_type, 0.0)

        # Get bias keywords
        bias_keywords = []
        if self.config and hasattr(self.config, "params"):
            bias_keywords = self.config.params.get("bias_keywords", {}).get(bias_type, [])

        # Create the explanation
        return {
            "bias_type": bias_type,
            "probability": bias_probability,
            "confidence": result.confidence,
            "is_primary_bias": bias_type == result.label,
            "contributing_features": explanation.get("positive", {}) if explanation else {},
            "countering_features": explanation.get("negative", {}) if explanation else {},
            "bias_specific_features": bias_features,
            "examples": bias_keywords,
        }

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
            "classifications": (
                self._state_manager.get("statistics", {}) if self._state_manager else {}
            ),
            # Number of errors encountered
            "error_count": len(self._state_manager.get("errors", [])) if self._state_manager else 0,
            # Cache information
            "cache_enabled": (
                self.config.cache_size > 0
                if self.config and hasattr(self.config, "cache_size")
                else False
            ),
            "cache_size": (
                self.config.cache_size if self.config and hasattr(self.config, "cache_size") else 0
            ),
            # State initialization status
            "initialized": (
                self._state_manager.get("initialized", False) if self._state_manager else False
            ),
            # Model information
            "model_path": (
                self.config.params.get("model_path")
                if self.config and hasattr(self.config, "params")
                else None
            ),
            "max_features": (
                self.config.params.get("max_features", 3000)
                if self.config and hasattr(self.config, "params")
                else 3000
            ),
            "random_state": (
                self.config.params.get("random_state", 42)
                if self.config and hasattr(self.config, "params")
                else 42
            ),
        }

        # Add feature information if available
        if self._state_manager:
            feature_names = self._state_manager.get("feature_names")
            if feature_names:
                stats["feature_count"] = feature_names.get("count", 0)

        # Add cache hit ratio if caching is enabled
        if hasattr(self, "_result_cache"):
            stats["cache_entries"] = len(self._result_cache)

        return stats

    def clear_cache(self) -> None:
        """
        Clear any cached data in the classifier.

        This method clears both the result cache and resets statistics in the state
        but preserves the model, vectorizer, pipeline and initialization status.
        """
        # Clear classification result cache
        if hasattr(self, "_result_cache"):
            self._result_cache.clear()

        # Reset statistics if state manager exists
        if self._state_manager:
            self._state_manager.update("statistics", {})

            # Reset errors list but keep model components and initialized status
            self._state_manager.update("errors", [])

            # Clear explanation cache but keep model, vectorizer, and pipeline
            cache = self._state_manager.get("cache", {})
            if cache:
                cache.pop("explanations", None)
                self._state_manager.update("cache", cache)

    @classmethod
    def create_pretrained(
        cls,
        texts: List[str],
        labels: List[str],
        name: str = "pretrained_bias_detector",
        description: str = "Pre-trained bias detector",
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> "BiasDetector":
        """
        Create and train a bias detector in one step.

        Args:
            texts: List of texts to train on
            labels: List of bias type labels
            name: Name of the classifier
            description: Description of the classifier
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters

        Returns:
            Trained BiasDetector
        """
        # Create default config if not provided
        if config is None:
            params = kwargs.pop("params", {}) if kwargs else {}

            # Add default bias types and keywords if not provided
            if "bias_types" not in params:
                params["bias_types"] = DEFAULT_BIAS_TYPES
            if "bias_keywords" not in params:
                params["bias_keywords"] = cls.DEFAULT_BIAS_KEYWORDS

            config = ClassifierConfig(
                labels=params.get("bias_types", DEFAULT_BIAS_TYPES),
                cost=cls.DEFAULT_COST,
                min_confidence=params.get("min_confidence", 0.7),
                params=params,
            )

        # Create instance with provided configuration
        classifier = cls(name=name, description=description, config=config, **kwargs)

        # Train the classifier and return it
        return classifier.fit(texts, labels)


def create_bias_detector(
    name: str = "bias_detector",
    description: str = "Detects various forms of bias in text",
    bias_types: Optional[List[str]] = None,
    bias_keywords: Optional[Dict[str, List[str]]] = None,
    min_confidence: float = 0.7,
    max_features: int = 3000,
    random_state: int = 42,
    cache_size: int = 100,
    cost: float = BiasDetector.DEFAULT_COST,
    config: Optional[Union[Dict[str, Any], ClassifierConfig]] = None,
    **kwargs: Any,
) -> BiasDetector:
    """
    Factory function to create a bias detector.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        bias_types: List of bias types to detect (defaults to DEFAULT_BIAS_TYPES)
        bias_keywords: Dictionary mapping bias types to keywords (defaults to DEFAULT_BIAS_KEYWORDS)
        min_confidence: Minimum confidence threshold
        max_features: Maximum number of features for the vectorizer
        random_state: Random state for reproducibility
        cache_size: Size of the cache for memoization
        cost: Cost of running the classifier
        config: Optional classifier configuration
        **kwargs: Additional configuration parameters

    Returns:
        Configured BiasDetector instance
    """
    bias_types = bias_types or DEFAULT_BIAS_TYPES
    bias_keywords = bias_keywords or BiasDetector.DEFAULT_BIAS_KEYWORDS

    # Set up default params
    default_params = {
        "bias_types": bias_types,
        "bias_keywords": bias_keywords,
        "max_features": max_features,
        "random_state": random_state,
        "min_confidence": min_confidence,
    }

    # Extract and merge configuration parameters
    provided_params = kwargs.pop("params", {}) if kwargs else {}

    config_dict = extract_classifier_config_params(
        labels=bias_types,
        cache_size=cache_size,
        min_confidence=min_confidence,
        cost=cost,
        provided_params=provided_params,
        default_params=default_params,
        **kwargs,
    )

    # Create config with merged parameters
    final_config = ClassifierConfig(**config_dict)

    # Create and return classifier
    return BiasDetector(
        name=name,
        description=description,
        config=final_config,
    )
