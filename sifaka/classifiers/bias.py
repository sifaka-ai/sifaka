"""
Bias detection classifiers for Sifaka.

This module provides classifiers that detect various types of bias in text,
including gender bias, racial bias, and other forms of discriminatory language.

## Architecture

BiasDetector follows the composition over inheritance pattern:
1. **Classifier**: Provides the public API and handles caching
2. **Implementation**: Contains the core classification logic
3. **Factory Function**: Creates a classifier with the bias detection implementation

## Lifecycle

1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract bias types and keywords from config.params
   - Set up default values

2. **Warm-up**: Load scikit-learn resources
   - Load scikit-learn dependencies
   - Initialize vectorizer and SVM model
   - Handle initialization errors gracefully

3. **Classification**: Process input text
   - Validate input text
   - Apply SVM-based bias detection
   - Extract bias features from text
   - Handle empty text and edge cases

4. **Result Creation**: Return standardized results
   - Map bias scores to labels
   - Convert scores to confidence values
   - Include detailed explanations in metadata
"""

import os
import re
import pickle
import importlib
from typing import Any, Dict, List, Optional, ClassVar, Union

from sifaka.classifiers.base import (
    ClassificationResult,
    ClassifierConfig,
    Classifier,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.state import ClassifierState, StateManager, create_classifier_state
from pydantic import PrivateAttr

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


class BiasDetectorImplementation:
    """
    Implementation of bias detection logic using scikit-learn.

    This implementation uses Support Vector Machines from scikit-learn to detect
    various forms of bias in text. It provides a comprehensive bias detection
    system that can identify multiple bias categories including gender, racial,
    political, age, socioeconomic, religious, cultural, educational, and
    geographical bias.

    ## Architecture

    BiasDetectorImplementation follows the composition pattern:
    1. **Core Logic**: classify_impl() implements bias detection
    2. **State Management**: Uses ClassifierState for internal state
    3. **Resource Management**: Loads and manages scikit-learn models

    ## Lifecycle

    1. **Initialization**: Set up configuration and state
       - Initialize with ClassifierConfig
       - Set up state for model and vectorizer

    2. **Warm-up**: Load scikit-learn resources
       - Load dependencies
       - Initialize vectorizer and SVM model
       - Handle initialization errors

    3. **Classification**: Process input text
       - Apply SVM-based bias detection
       - Extract bias features from text
       - Return standardized results

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    # Class constants
    DEFAULT_COST: ClassVar[float] = 2.5

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

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)

    def __init__(self, config: ClassifierConfig):
        """
        Initialize the bias detector implementation.

        Args:
            config: Configuration for the bias detector
        """
        self.config = config
        # State is managed by StateManager, no need to initialize here

    def _load_dependencies(self) -> Dict[str, Any]:
        """
        Load scikit-learn dependencies.

        Returns:
            Dictionary of loaded scikit-learn modules

        Raises:
            ImportError: If scikit-learn is not installed
            RuntimeError: If modules fail to load
        """
        try:
            # Import necessary scikit-learn modules
            sklearn_modules = {
                "feature_extraction_text": importlib.import_module(
                    "sklearn.feature_extraction.text"
                ),
                "svm": importlib.import_module("sklearn.svm"),
                "pipeline": importlib.import_module("sklearn.pipeline"),
                "calibration": importlib.import_module("sklearn.calibration"),
            }
            return sklearn_modules
        except ImportError:
            raise ImportError(
                "scikit-learn is required for BiasDetector. "
                "Install it with: pip install scikit-learn"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scikit-learn modules: {e}")

    def warm_up_impl(self) -> None:
        """
        Initialize the model if needed.

        This method loads the scikit-learn dependencies and initializes
        the vectorizer and SVM model for bias detection.
        """
        # Get state
        state = self._state_manager.get_state()

        # Check if already initialized
        if state.initialized:
            return

        # Load dependencies
        sklearn = self._load_dependencies()
        state.dependencies_loaded = True

        # Get configuration from params
        model_path = self.config.params.get("model_path")
        max_features = self.config.params.get("max_features", 3000)
        random_state = self.config.params.get("random_state", 42)

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # Create TF-IDF vectorizer with custom analyzer to catch bias keywords
            state.vectorizer = sklearn["feature_extraction_text"].TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
                ngram_range=(1, 2),
            )

            # Create SVM model with probability estimates
            svm = sklearn["svm"].LinearSVC(
                random_state=random_state,
                class_weight="balanced",
                max_iter=10000,
            )

            # Calibrate the SVM to get probability estimates
            state.model = sklearn["calibration"].CalibratedClassifierCV(
                estimator=svm,
                cv=3,
            )

            # Create pipeline
            state.pipeline = sklearn["pipeline"].Pipeline(
                [
                    ("vectorizer", state.vectorizer),
                    ("classifier", state.model),
                ]
            )

        # Mark as initialized
        state.initialized = True

    def _extract_bias_features(self, text: str) -> Dict[str, float]:
        """
        Extract bias-related features from text.

        Args:
            text: The text to analyze

        Returns:
            Dictionary of bias features and their scores
        """
        features = {}
        # Get bias keywords from params or use default empty dict
        bias_keywords = self.config.params.get("bias_keywords", {})
        for bias_type, keywords in bias_keywords.items():
            count = sum(1 for kw in keywords if re.search(rf"\b{kw}\b", text.lower()))
            features[f"bias_{bias_type}"] = count / max(len(keywords), 1)  # Avoid division by zero
        return features

    def _save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model

        Raises:
            RuntimeError: If model is not initialized
        """
        # Get state
        state = self._state_manager.get_state()

        # Check if initialized
        if not state.initialized:
            raise RuntimeError("Model not initialized")

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": state.vectorizer,
                    "model": state.model,
                    "pipeline": state.pipeline,
                    "config_params": self.config.params,
                    "feature_names": state.feature_names,
                },
                f,
            )

    def _load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        # Get state
        state = self._state_manager.get_state()

        with open(path, "rb") as f:
            data = pickle.load(f)
            state.vectorizer = data["vectorizer"]
            state.model = data["model"]
            state.pipeline = data["pipeline"]

            # Load feature names if available
            if "feature_names" in data:
                state.feature_names = data["feature_names"]

        # Mark as initialized
        state.initialized = True

    def _extract_explanations(self) -> None:
        """
        Extract feature coefficients for explanations.

        This method extracts the most important features for each bias type
        to provide explanations for classification results.
        """
        # Get state
        state = self._state_manager.get_state()

        try:
            if not hasattr(state.model, "base_estimator"):
                return

            # Get feature names
            feature_names = state.vectorizer.get_feature_names_out()

            # Store feature names in state
            state.feature_names = {
                "names": feature_names,
                "count": len(feature_names),
            }

            # Extract coefficients from the SVM
            if hasattr(state.model.base_estimator, "coef_"):
                coefficients = state.model.base_estimator.coef_

                # For each class, extract the top features
                explanations = {}
                # Use labels from config instead of bias_types
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
                state.cache["explanations"] = explanations
        except Exception as e:
            logger.warning(f"Could not extract explanations: {e}")
            state.error = f"Failed to extract explanations: {e}"

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """
        Implement bias detection logic.

        This method contains the core bias detection logic using scikit-learn.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with bias type and confidence

        Raises:
            RuntimeError: If model is not initialized
        """
        # Get state
        state = self._state_manager.get_state()

        # Check if initialized
        if not state.initialized:
            self.warm_up_impl()

        # Check if still not initialized
        if not state.initialized:
            raise RuntimeError("Model not initialized")

        # Handle empty or whitespace-only text
        if not text.strip():
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={
                    "reason": "empty_input",
                },
            )

        try:
            # Get prediction and probability
            label = state.pipeline.predict([text])[0]
            probs = state.pipeline.predict_proba([text])[0]
            confidence = max(probs)

            # Extract bias features
            bias_features = self._extract_bias_features(text)

            # Get explanations from cache
            explanations = state.cache.get("explanations", {}).get(label, {})

            # Create result
            return ClassificationResult(
                label=label,
                confidence=confidence,
                metadata={
                    "features": bias_features,
                    "explanations": explanations,
                },
            )
        except Exception as e:
            # Log the error and return a fallback result
            logger.error("Failed to classify text: %s", e)
            state.error = f"Failed to classify text: {e}"
            return ClassificationResult(
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
        Classify multiple texts in batch.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults

        Raises:
            RuntimeError: If model is not initialized
        """
        # Get state
        state = self._state_manager.get_state()

        # Check if initialized
        if not state.initialized:
            self.warm_up_impl()

        # Check if still not initialized
        if not state.pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Handle empty list
        if not texts:
            return []

        # Filter out empty texts and keep track of indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        # If all texts are empty, return unknown results
        if not valid_texts:
            return [
                ClassificationResult(
                    label="unknown",
                    confidence=0.0,
                    metadata={"reason": "empty_input"},
                )
                for _ in texts
            ]

        try:
            # Get model predictions for valid texts
            probas = state.pipeline.predict_proba(valid_texts)

            # Prepare results list with placeholders
            results = [None] * len(texts)

            # Process valid texts
            for idx, i in enumerate(valid_indices):
                # Extract bias features
                bias_features = self._extract_bias_features(texts[i])

                # Get probabilities for this text
                proba = probas[idx]
                all_probs = {self.config.labels[j]: float(prob) for j, prob in enumerate(proba)}

                # Get the most likely bias type
                label = max(all_probs.items(), key=lambda x: x[1])[0]
                confidence = all_probs[label]

                # Get min_confidence from config
                min_confidence = self.config.min_confidence

                # Get explanations from cache
                explanations = state.cache.get("explanations", {}).get(label, {})

                metadata = {
                    "probabilities": all_probs,
                    "threshold": min_confidence,
                    "is_confident": confidence >= min_confidence,
                    "bias_features": bias_features,
                    "explanations": explanations,
                }

                results[i] = ClassificationResult(
                    label=label,
                    confidence=confidence,
                    metadata=metadata,
                )

            # Fill in results for empty texts
            for i in range(len(texts)):
                if results[i] is None:
                    results[i] = ClassificationResult(
                        label="unknown",
                        confidence=0.0,
                        metadata={"reason": "empty_input"},
                    )

            return results
        except Exception as e:
            # Log the error and return fallback results
            logger.error("Failed to batch classify texts: %s", e)
            state.error = f"Failed to batch classify texts: {e}"
            return [
                ClassificationResult(
                    label="unknown",
                    confidence=0.0,
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "reason": "classification_error",
                    },
                )
                for _ in texts
            ]

    def fit_impl(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the bias detector.

        Args:
            texts: List of texts to train on
            labels: List of bias type labels

        Raises:
            ValueError: If training data is empty
        """
        if not texts or not labels:
            raise ValueError("Empty training data")

        # Get state
        state = self._state_manager.get_state()

        # Load scikit-learn dependencies
        sklearn = self._load_dependencies()
        state.dependencies_loaded = True

        # Get configuration from params
        max_features = self.config.params.get("max_features", 3000)
        random_state = self.config.params.get("random_state", 42)

        # Create vectorizer
        state.vectorizer = sklearn["feature_extraction_text"].TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
        )

        # Create and train SVM classifier
        state.model = sklearn["svm"].SVC(
            kernel="linear",
            probability=True,
            random_state=random_state,
        )

        # Create pipeline
        state.pipeline = sklearn["pipeline"].Pipeline(
            [
                ("vectorizer", state.vectorizer),
                ("classifier", state.model),
            ]
        )

        # Fit pipeline
        state.pipeline.fit(texts, labels)
        state.initialized = True

        # Extract feature explanations
        self._extract_explanations()

        # Save model if path specified
        model_path = self.config.params.get("model_path")
        if model_path:
            self._save_model(model_path)

    def get_bias_explanation(self, bias_type: str, text: str) -> Dict[str, Any]:
        """
        Get detailed explanation about why a text was classified with a particular bias.

        Args:
            bias_type: The bias type to explain
            text: The text that was classified

        Returns:
            Dictionary with explanation details

        Raises:
            ValueError: If bias type is invalid
        """
        if bias_type not in self.config.labels:
            raise ValueError(f"Invalid bias type: {bias_type}. Must be one of {self.config.labels}")

        # Get the classification result
        result = self.classify_impl(text)

        # Get state
        state = self._state_manager.get_state()

        # Extract the features that contributed to this bias type
        explanation = {}
        explanations = state.cache.get("explanations", {})
        if explanations and bias_type in explanations:
            explanation = explanations[bias_type]

        # Get bias-specific features
        bias_features = self._extract_bias_features(text)

        # Get the probability for this bias type
        bias_probability = result.metadata.get("probabilities", {}).get(bias_type, 0.0)

        # Create the explanation
        return {
            "bias_type": bias_type,
            "probability": bias_probability,
            "confidence": result.confidence,
            "is_primary_bias": bias_type == result.label,
            "contributing_features": explanation.get("positive", {}),
            "countering_features": explanation.get("negative", {}),
            "bias_specific_features": bias_features,
            "examples": self.config.params.get("bias_keywords", {}).get(bias_type, []),
        }


def create_bias_detector_with_custom_model(
    model_path: str,
    name: str = "bias_detector",
    description: str = "Detects various forms of bias in text",
    bias_types: Optional[List[str]] = None,
    bias_keywords: Optional[Dict[str, List[str]]] = None,
    min_confidence: float = 0.7,
    cache_size: int = 100,
    cost: float = BiasDetectorImplementation.DEFAULT_COST,
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Create a bias detector with a pre-trained model.

    This factory function creates a bias detector with a model loaded from disk.

    Args:
        model_path: Path to the pre-trained model
        name: Name of the classifier
        description: Description of the classifier
        bias_types: List of bias types to detect
        bias_keywords: Dictionary mapping bias types to keywords
        min_confidence: Minimum confidence threshold
        cache_size: Size of the cache for memoization
        cost: Cost of running the classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured Classifier instance with BiasDetectorImplementation

    Raises:
        FileNotFoundError: If the model file does not exist
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Prepare params
    params: Dict[str, Any] = kwargs.pop("params", {})
    params.update(
        {
            "model_path": model_path,
            "bias_types": bias_types or DEFAULT_BIAS_TYPES,
            "bias_keywords": bias_keywords or BiasDetectorImplementation.DEFAULT_BIAS_KEYWORDS,
            "min_confidence": min_confidence,
        }
    )

    # Create config
    config = ClassifierConfig(
        labels=bias_types or DEFAULT_BIAS_TYPES,
        cache_size=cache_size,
        min_confidence=min_confidence,
        cost=cost,
        params=params,
    )

    # Create implementation
    implementation = BiasDetectorImplementation(config)

    # Load the model
    implementation._load_model(model_path)

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )


def create_pretrained_bias_detector(
    texts: List[str],
    labels: List[str],
    name: str = "pretrained_bias_detector",
    description: str = "Pre-trained bias detector",
    bias_types: Optional[List[str]] = None,
    bias_keywords: Optional[Dict[str, List[str]]] = None,
    min_confidence: float = 0.7,
    max_features: int = 3000,
    random_state: int = 42,
    cache_size: int = 100,
    cost: float = BiasDetectorImplementation.DEFAULT_COST,
    model_path: Optional[str] = None,
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Create and train a bias detector in one step.

    This factory function creates and trains a bias detector with the provided
    training data.

    Args:
        texts: List of texts to train on
        labels: List of bias type labels
        name: Name of the classifier
        description: Description of the classifier
        bias_types: List of bias types to detect
        bias_keywords: Dictionary mapping bias types to keywords
        min_confidence: Minimum confidence threshold
        max_features: Maximum number of features for the vectorizer
        random_state: Random state for reproducibility
        cache_size: Size of the cache for memoization
        cost: Cost of running the classifier
        model_path: Optional path to save the trained model
        **kwargs: Additional configuration parameters

    Returns:
        Trained Classifier instance with BiasDetectorImplementation

    Raises:
        ValueError: If training data is empty
    """
    if not texts or not labels:
        raise ValueError("Empty training data")

    # Prepare params
    params: Dict[str, Any] = kwargs.pop("params", {})
    params.update(
        {
            "bias_types": bias_types or DEFAULT_BIAS_TYPES,
            "bias_keywords": bias_keywords or BiasDetectorImplementation.DEFAULT_BIAS_KEYWORDS,
            "min_confidence": min_confidence,
            "max_features": max_features,
            "random_state": random_state,
            "model_path": model_path,
        }
    )

    # Create config
    config = ClassifierConfig(
        labels=bias_types or DEFAULT_BIAS_TYPES,
        cache_size=cache_size,
        min_confidence=min_confidence,
        cost=cost,
        params=params,
    )

    # Create implementation
    implementation = BiasDetectorImplementation(config)

    # Train the implementation
    implementation.fit_impl(texts, labels)

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )


def create_bias_detector(
    name: str = "bias_detector",
    description: str = "Detects various forms of bias in text",
    bias_types: Optional[List[str]] = None,
    bias_keywords: Optional[Dict[str, List[str]]] = None,
    min_confidence: float = 0.7,
    max_features: int = 3000,
    random_state: int = 42,
    cache_size: int = 100,
    cost: float = BiasDetectorImplementation.DEFAULT_COST,
    config: Optional[Union[Dict[str, Any], ClassifierConfig]] = None,
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Create a bias detector.

    This factory function creates a bias detector with the specified
    configuration options. It follows the composition over inheritance pattern,
    creating a Classifier with a BiasDetectorImplementation.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        bias_types: List of bias types to detect
        bias_keywords: Dictionary mapping bias types to keywords
        min_confidence: Minimum confidence threshold
        max_features: Maximum number of features for the vectorizer
        random_state: Random state for reproducibility
        cache_size: Size of the cache for memoization
        cost: Cost of running the classifier
        config: Optional classifier configuration
        **kwargs: Additional configuration parameters

    Returns:
        A Classifier instance with BiasDetectorImplementation

    Examples:
        ```python
        from sifaka.classifiers.bias import create_bias_detector

        # Create a bias detector with default settings
        detector = create_bias_detector()

        # Create a bias detector with custom settings
        detector = create_bias_detector(
            name="custom_bias_detector",
            description="Custom bias detector with specific bias types",
            bias_types=["gender", "racial", "political"],
            min_confidence=0.8,
            cache_size=200
        )

        # Classify text
        result = detector.classify("Men are better at math than women.")
        print(f"Bias type: {result.label}, Confidence: {result.confidence:.2f}")
        ```
    """
    # Prepare params
    params: Dict[str, Any] = kwargs.pop("params", {})
    params.update(
        {
            "bias_types": bias_types or DEFAULT_BIAS_TYPES,
            "bias_keywords": bias_keywords or BiasDetectorImplementation.DEFAULT_BIAS_KEYWORDS,
            "min_confidence": min_confidence,
            "max_features": max_features,
            "random_state": random_state,
        }
    )

    # Create config
    if isinstance(config, ClassifierConfig):
        classifier_config = config
    elif isinstance(config, dict):
        classifier_config = ClassifierConfig(
            labels=config.get("labels", bias_types or DEFAULT_BIAS_TYPES),
            cache_size=config.get("cache_size", cache_size),
            min_confidence=config.get("min_confidence", min_confidence),
            cost=config.get("cost", cost),
            params=config.get("params", params),
        )
    else:
        classifier_config = ClassifierConfig(
            labels=bias_types or DEFAULT_BIAS_TYPES,
            cache_size=cache_size,
            min_confidence=min_confidence,
            cost=cost,
            params=params,
        )

    # Create implementation
    implementation = BiasDetectorImplementation(classifier_config)

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=classifier_config,
        implementation=implementation,
    )
