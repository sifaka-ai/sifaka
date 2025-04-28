"""
Bias detector using scikit-learn's SVM classifier.
"""

import importlib
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BiasConfig:
    """
    Configuration for bias detection.

    Note: This class is provided for backward compatibility.
    The preferred way to configure bias detection is to use
    ClassifierConfig with params:

    ```python
    config = ClassifierConfig(
        labels=["gender", "racial", "political", "neutral"],
        cost=2.5,
        params={
            "min_confidence": 0.7,
            "max_features": 3000,
            "random_state": 42,
            "model_path": "/path/to/model.pkl",
            "bias_types": ["gender", "racial", "political", "neutral"],
        }
    )
    ```
    """

    min_confidence: float = 0.7  # Minimum confidence threshold
    max_features: int = 3000  # Max features for vectorization
    random_state: int = 42  # For reproducibility
    model_path: Optional[str] = None  # Path to save/load the model

    # Default bias types to detect
    bias_types: List[str] = field(
        default_factory=lambda: [
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
    )

    # Additional keywords for each bias category to enhance detection
    bias_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
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
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        if not self.bias_types:
            raise ValueError("bias_types cannot be empty")
        if "neutral" not in self.bias_types:
            raise ValueError("'neutral' must be included in bias_types")


class BiasDetector(BaseClassifier):
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
    DEFAULT_COST: float = 2.5

    def __init__(
        self,
        name: str = "bias_detector",
        description: str = "Detects various forms of bias in text",
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """Initialize the bias detector."""
        super().__init__(name=name, description=description, config=config)

        # Initialize other attributes
        self._vectorizer = None
        self._model = None
        self._pipeline = None
        self._explanations = {}
        self._initialized = False

        # We'll use config.params for all configuration instead of a separate _bias_config

    def _load_dependencies(self) -> None:
        """Load scikit-learn dependencies."""
        try:
            # Import necessary scikit-learn modules
            self._sklearn_feature_extraction_text = importlib.import_module(
                "sklearn.feature_extraction.text"
            )
            self._sklearn_svm = importlib.import_module("sklearn.svm")
            self._sklearn_pipeline = importlib.import_module("sklearn.pipeline")
            self._sklearn_calibration = importlib.import_module("sklearn.calibration")
            return True
        except ImportError:
            raise ImportError(
                "scikit-learn is required for BiasDetector. "
                "Install it with: pip install scikit-learn"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scikit-learn modules: {e}")

    def warm_up(self) -> None:
        """Initialize the model if needed."""
        if not self._initialized:
            self._load_dependencies()

            # Get configuration from params
            model_path = self.config.params.get("model_path")
            max_features = self.config.params.get("max_features", 3000)
            random_state = self.config.params.get("random_state", 42)

            if model_path and os.path.exists(model_path):
                self._load_model(model_path)
            else:
                # Create TF-IDF vectorizer with custom analyzer to catch bias keywords
                self._vectorizer = self._sklearn_feature_extraction_text.TfidfVectorizer(
                    max_features=max_features,
                    stop_words="english",
                    ngram_range=(1, 2),
                )

                # Create SVM model with probability estimates
                svm = self._sklearn_svm.LinearSVC(
                    random_state=random_state,
                    class_weight="balanced",
                    max_iter=10000,
                )

                # Calibrate the SVM to get probability estimates
                self._model = self._sklearn_calibration.CalibratedClassifierCV(
                    estimator=svm,
                    cv=3,
                )

                # Create pipeline
                self._pipeline = self._sklearn_pipeline.Pipeline(
                    [
                        ("vectorizer", self._vectorizer),
                        ("classifier", self._model),
                    ]
                )

            self._initialized = True

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
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self._vectorizer,
                    "model": self._model,
                    "pipeline": self._pipeline,
                    "config_params": self.config.params,
                },
                f,
            )

    def _load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._vectorizer = data["vectorizer"]
            self._model = data["model"]
            self._pipeline = data["pipeline"]

            # Update config params if available in the saved model
            if "config_params" in data:
                # Create a new config with the loaded params
                self.config = ClassifierConfig(
                    labels=self.config.labels,
                    cost=self.config.cost,
                    min_confidence=self.config.min_confidence,
                    params=data["config_params"],
                    metadata=self.config.metadata,
                )

            self._initialized = True

    def fit(self, texts: List[str], labels: List[str]) -> "BiasDetector":
        """Train the bias detector."""
        if not texts or not labels:
            raise ValueError("Empty training data")

        # Load scikit-learn dependencies
        self._load_dependencies()

        # Get configuration from params
        max_features = self.config.params.get("max_features", 3000)
        random_state = self.config.params.get("random_state", 42)

        # Create vectorizer
        self._vectorizer = self._sklearn_feature_extraction_text.TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
        )

        # Create and train SVM classifier
        self._model = self._sklearn_svm.SVC(
            kernel="linear",
            probability=True,
            random_state=random_state,
        )

        # Create pipeline
        self._pipeline = self._sklearn_pipeline.Pipeline(
            [
                ("vectorizer", self._vectorizer),
                ("classifier", self._model),
            ]
        )

        # Fit pipeline
        self._pipeline.fit(texts, labels)
        self._initialized = True

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
            if not hasattr(self._model, "base_estimator"):
                return

            # Get feature names
            feature_names = self._vectorizer.get_feature_names_out()

            # Extract coefficients from the SVM
            if hasattr(self._model.base_estimator, "coef_"):
                coefficients = self._model.base_estimator.coef_

                # For each class, extract the top features
                self._explanations = {}
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

                    self._explanations[bias_type] = {
                        "positive": {
                            feature_names[j]: float(class_coef[j]) for j in top_pos_indices
                        },
                        "negative": {
                            feature_names[j]: float(class_coef[j]) for j in top_neg_indices
                        },
                    }
        except Exception as e:
            logger.warning(f"Could not extract explanations: {e}")

    def _classify_impl(self, text: str) -> ClassificationResult:
        """Implement classification logic."""
        if not self._initialized:
            raise RuntimeError("Model not initialized")

        # Get prediction and probability
        label = self._pipeline.predict([text])[0]
        probs = self._pipeline.predict_proba([text])[0]
        confidence = max(probs)

        # Extract bias features
        bias_features = self._extract_bias_features(text)

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "features": bias_features,
                "explanations": self._explanations.get(label, {}),
            },
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts in batch.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        if not self._pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Get model predictions for all texts
        probas = self._pipeline.predict_proba(texts)

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

            metadata = {
                "probabilities": all_probs,
                "threshold": min_confidence,
                "is_confident": confidence >= min_confidence,
                "bias_features": bias_features,
            }

            results.append(
                ClassificationResult(
                    label=label,
                    confidence=confidence,
                    metadata=metadata,
                )
            )

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
        if self._explanations and bias_type in self._explanations:
            explanation = self._explanations[bias_type]

        # Get bias-specific features
        bias_features = self._extract_bias_features(text)

        # Get the probability for this bias type
        bias_probability = result.metadata["probabilities"].get(bias_type, 0.0)

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

    @classmethod
    def create_pretrained(
        cls,
        texts: List[str],
        labels: List[str],
        name: str = "pretrained_bias_detector",
        description: str = "Pre-trained bias detector",
        bias_config: Optional[BiasConfig] = None,
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
            bias_config: Bias detection configuration (for backward compatibility)
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters

        Returns:
            Trained BiasDetector
        """
        # If bias_config is provided but config is not, create config from bias_config
        if bias_config is not None and config is None:
            # Extract params from bias_config
            params = {
                "min_confidence": bias_config.min_confidence,
                "max_features": bias_config.max_features,
                "random_state": bias_config.random_state,
                "model_path": bias_config.model_path,
                "bias_types": bias_config.bias_types,
            }

            # Create config with params
            config = ClassifierConfig(
                labels=bias_config.bias_types,
                cost=cls.DEFAULT_COST,
                min_confidence=bias_config.min_confidence,
                params=params,
            )

        # Create instance with provided configuration
        classifier = cls(name=name, description=description, config=config, **kwargs)

        # Train the classifier and return it
        return classifier.fit(texts, labels)
