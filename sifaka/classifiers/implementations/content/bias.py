"""
Bias detection classifiers for Sifaka.

This module provides classifiers that detect various types of bias in text,
including gender bias, racial bias, and other forms of discriminatory language.
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
from sifaka.classifiers.result import ClassificationResult
from sifaka.classifiers.config import ClassifierConfig
from sifaka.utils.logging import get_logger
from sifaka.utils.config import extract_classifier_config_params

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
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """Initialize the bias detector."""
        # Create default config if not provided
        if config is None:
            params = kwargs.pop("params", {})

            # Add default bias types and keywords if not provided
            if "bias_types" not in params:
                params["bias_types"] = DEFAULT_BIAS_TYPES
            if "bias_keywords" not in params:
                params["bias_keywords"] = self.DEFAULT_BIAS_KEYWORDS

            config = ClassifierConfig(
                labels=params.get("bias_types", DEFAULT_BIAS_TYPES),
                cost=self.DEFAULT_COST,
                min_confidence=params.get("min_confidence", 0.7),
                params=params,
            )

        super().__init__(name=name, description=description, config=config)

    def _load_dependencies(self) -> Dict[str, Any]:
        """Load scikit-learn dependencies."""
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

    def warm_up(self) -> None:
        """Initialize the model if needed."""
        # Check if already initialized
        if self._state_manager.get("initialized", False):
            return

        # Load dependencies
        sklearn = self._load_dependencies()
        self._state_manager.update("dependencies_loaded", True)

        # Get configuration from params
        model_path = self.config.params.get("model_path")
        max_features = self.config.params.get("max_features", 3000)
        random_state = self.config.params.get("random_state", 42)

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
        """Extract bias-related features from text."""
        features = {}
        # Get bias keywords from params or use default empty dict
        bias_keywords = self.config.params.get("bias_keywords", {})
        for bias_type, keywords in bias_keywords.items():
            count = sum(1 for kw in keywords if re.search(rf"\b{kw}\b", text.lower()))
            features[f"bias_{bias_type}"] = count / max(len(keywords), 1)  # Avoid division by zero
        return features

    def _save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        # Check if initialized
        if not self._state_manager.get("initialized", False):
            raise RuntimeError("Model not initialized")

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self._state_manager.get("vectorizer"),
                    "model": self._state_manager.get("model"),
                    "pipeline": self._state_manager.get("pipeline"),
                    "config_params": self.config.params,
                    "feature_names": self._state_manager.get("feature_names"),
                },
                f,
            )

    def _load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._state_manager.update("vectorizer", data["vectorizer"])
            self._state_manager.update("model", data["model"])
            self._state_manager.update("pipeline", data["pipeline"])

            # Load feature names if available
            if "feature_names" in data:
                self._state_manager.update("feature_names", data["feature_names"])

            # Update config params if available in the saved model
            if "config_params" in data:
                # Create a new config with the loaded params
                self.config = ClassifierConfig(
                    labels=self.config.labels,
                    cost=self.config.cost,
                    min_confidence=self.config.min_confidence,
                    params=data["config_params"],
                )

            # Mark as initialized
            self._state_manager.update("initialized", True)

    def fit(self, texts: List[str], labels: List[str]) -> "BiasDetector":
        """Train the bias detector."""
        if not texts or not labels:
            raise ValueError("Empty training data")

        # Load scikit-learn dependencies
        sklearn = self._load_dependencies()
        self._state_manager.update("dependencies_loaded", True)

        # Get configuration from params
        max_features = self.config.params.get("max_features", 3000)
        random_state = self.config.params.get("random_state", 42)

        # Create vectorizer
        vectorizer = sklearn["feature_extraction_text"].TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
        )
        self._state_manager.update("vectorizer", vectorizer)

        # Create and train SVM classifier
        model = sklearn["svm"].SVC(
            kernel="linear",
            probability=True,
            random_state=random_state,
        )
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
        self._state_manager.update("pipeline", pipeline)
        self._state_manager.update("initialized", True)

        # Extract feature explanations
        self._extract_explanations()

        # Save model if path specified
        model_path = self.config.params.get("model_path")
        if model_path:
            self._save_model(model_path)

        return self

    def _extract_explanations(self) -> None:
        """Extract feature coefficients for explanations."""
        try:
            model = self._state_manager.get("model")
            if not hasattr(model, "base_estimator"):
                return

            # Get feature names
            vectorizer = self._state_manager.get("vectorizer")
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
        """Implement classification logic."""
        # Check if initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Check if still not initialized
        if not self._state_manager.get("initialized", False):
            raise RuntimeError("Model not initialized")

        # Get prediction and probability
        pipeline = self._state_manager.get("pipeline")
        label = pipeline.predict([text])[0]
        probs = pipeline.predict_proba([text])[0]
        confidence = max(probs)

        # Extract bias features
        bias_features = self._extract_bias_features(text)

        # Get explanations from cache
        explanations = self._state_manager.get("cache", {}).get("explanations", {}).get(label, {})

        result = ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "features": bias_features,
                "explanations": explanations,
            },
        )

        # Track statistics
        stats = self._state_manager.get("statistics", {})
        stats[label] = stats.get(label, 0) + 1
        self._state_manager.update("statistics", stats)

        return result

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts in batch.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        # Check if initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Check if still not initialized
        pipeline = self._state_manager.get("pipeline")
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
            all_probs = {self.config.labels[i]: float(prob) for i, prob in enumerate(proba)}

            # Get the most likely bias type
            label = max(all_probs.items(), key=lambda x: x[1])[0]
            confidence = all_probs[label]

            # Get min_confidence from config
            min_confidence = self.config.min_confidence

            # Get explanations from cache
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
            stats = self._state_manager.get("statistics", {})
            stats[label] = stats.get(label, 0) + 1
            self._state_manager.update("statistics", stats)

        return results

    def get_bias_explanation(self, bias_type: str, text: str) -> Dict[str, Any]:
        """
        Get detailed explanation about why a text was classified with a particular bias.

        Args:
            bias_type: The bias type to explain
            text: The text that was classified

        Returns:
            Dictionary with explanation details
        """
        if bias_type not in self.config.labels:
            raise ValueError(f"Invalid bias type: {bias_type}. Must be one of {self.config.labels}")

        # Get the classification result without explanation to avoid recursion
        result = self._classify_impl(text)

        # Extract the features that contributed to this bias type
        explanation = {}
        explanations = self._state.get("cache", {}).get("explanations", {})
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
            "classifications": self._state.get("statistics", {}),
            # Number of errors encountered
            "error_count": len(self._state.get("errors", [])),
            # Cache information
            "cache_enabled": self.config.cache_size > 0,
            "cache_size": self.config.cache_size,
            # State initialization status
            "initialized": self._state.get("initialized", False),
            # Model information
            "model_path": self.config.params.get("model_path"),
            "max_features": self.config.params.get("max_features", 3000),
            "random_state": self.config.params.get("random_state", 42),
        }

        # Add feature information if available
        feature_names = self._state.get("feature_names")
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

        # Reset statistics
        self._state.update("statistics", {})

        # Reset errors list but keep model components and initialized status
        self._state.update("errors", [])

        # Clear explanation cache but keep model, vectorizer, and pipeline
        cache = self._state.get("cache", {})
        cache.pop("explanations", None)
        self._state.update("cache", cache)

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
            params = kwargs.pop("params", {})

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
    config_dict = extract_classifier_config_params(
        labels=bias_types,
        cache_size=cache_size,
        min_confidence=min_confidence,
        cost=cost,
        provided_params=kwargs.pop("params", {}),
        default_params=default_params,
        **kwargs,
    )

    # Create config with merged parameters
    config = ClassifierConfig[str](**config_dict)

    # Create and return classifier
    return BiasDetector(
        name=name,
        description=description,
        config=config,
    )
