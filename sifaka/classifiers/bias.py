"""
Bias detector using scikit-learn's SVM classifier.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import importlib
import logging
import pickle
import os
import re
from dataclasses import dataclass, field

from pydantic import Field
from sifaka.classifiers.base import BaseClassifier, ClassificationResult, ClassifierConfig
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BiasConfig:
    """Configuration for bias detection."""

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

    # Class constants
    DEFAULT_COST: float = 2.5

    # Additional Pydantic fields
    bias_config: BiasConfig = Field(
        default_factory=lambda: BiasConfig(),
        description="Bias detection configuration",
    )

    def __init__(
        self,
        name: str = "bias_detector",
        description: str = "Detects various forms of bias in text",
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the bias detector.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            config: Classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Extract bias-specific config from kwargs or use defaults
        bias_kwargs = {}
        for key in ["max_features", "model_path"]:
            if key in kwargs.get("additional_config", {}):
                bias_kwargs[key] = kwargs["additional_config"].pop(key)

        # Create bias config
        bias_config = BiasConfig(
            min_confidence=kwargs.get("min_confidence", 0.7),
            **bias_kwargs,
        )

        # If config is not provided, create one from bias_config values
        if config is None:
            config = ClassifierConfig(
                labels=bias_config.bias_types,
                cost=self.DEFAULT_COST,
                min_confidence=bias_config.min_confidence,
                **kwargs,
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

        # Set bias config
        self.bias_config = bias_config

        # Initialize other attributes
        self._vectorizer = None
        self._model = None
        self._pipeline = None
        self._explanations = {}
        self._initialized = False

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

            if self.bias_config.model_path and os.path.exists(self.bias_config.model_path):
                self._load_model(self.bias_config.model_path)
            else:
                # Create TF-IDF vectorizer with custom analyzer to catch bias keywords
                self._vectorizer = self._sklearn_feature_extraction_text.TfidfVectorizer(
                    max_features=self.bias_config.max_features,
                    stop_words="english",
                    ngram_range=(1, 2),
                )

                # Create SVM model with probability estimates
                svm = self._sklearn_svm.LinearSVC(
                    random_state=self.bias_config.random_state,
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
        """
        Extract bias-specific features from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of bias features and their scores
        """
        features = {}

        # Check for bias keywords in each category
        for bias_type, keywords in self.bias_config.bias_keywords.items():
            count = 0
            for keyword in keywords:
                # Case-insensitive matching
                count += len(re.findall(r"\b" + re.escape(keyword) + r"\b", text, re.IGNORECASE))

            # Normalize by text length
            features[f"bias_{bias_type}_count"] = count / (len(text.split()) + 1)

        return features

    def _save_model(self, path: str) -> None:
        """Save the model to a file."""
        try:
            model_data = {
                "pipeline": self._pipeline,
                "explanations": self._explanations,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_model(self, path: str) -> None:
        """Load the model from a file."""
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

                # Extract model data
                self._pipeline = model_data["pipeline"]
                self._explanations = model_data.get("explanations", {})

                # Extract vectorizer and model from pipeline
                self._vectorizer = self._pipeline.named_steps["vectorizer"]
                self._model = self._pipeline.named_steps["classifier"]

            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def fit(self, texts: List[str], labels: List[str]) -> "BiasDetector":
        """
        Fit the bias detector on a corpus of texts.

        Args:
            texts: List of texts to fit the model on
            labels: List of bias labels

        Returns:
            self: The fitted classifier
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

        # Validate that all labels are in the configured bias types
        invalid_labels = set(labels) - set(self.bias_config.bias_types)
        if invalid_labels:
            raise ValueError(
                f"Invalid labels found: {invalid_labels}. Must be one of {self.bias_config.bias_types}"
            )

        self.warm_up()

        # Create label mapping
        label_mapping = {label: i for i, label in enumerate(self.bias_config.bias_types)}
        numeric_labels = [label_mapping[label] for label in labels]

        # Fit the pipeline
        self._pipeline.fit(texts, numeric_labels)

        # Extract feature coefficients for explanations
        self._extract_explanations()

        # Save the model if path is provided
        if self.bias_config.model_path:
            self._save_model(self.bias_config.model_path)

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
                for i, bias_type in enumerate(self.bias_config.bias_types):
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
        """
        Implement bias detection logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with bias label and confidence
        """
        if not self._pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Extract bias features
        bias_features = self._extract_bias_features(text)

        # Get model predictions
        proba = self._pipeline.predict_proba([text])[0]
        all_probs = {self.config.labels[i]: float(prob) for i, prob in enumerate(proba)}

        # Get the most likely bias type
        label = max(all_probs.items(), key=lambda x: x[1])[0]
        confidence = all_probs[label]

        metadata = {
            "probabilities": all_probs,
            "threshold": self.bias_config.min_confidence,
            "is_confident": confidence >= self.bias_config.min_confidence,
            "bias_features": bias_features,
        }

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata=metadata,
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

            metadata = {
                "probabilities": all_probs,
                "threshold": self.bias_config.min_confidence,
                "is_confident": confidence >= self.bias_config.min_confidence,
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
            "examples": self.bias_config.bias_keywords.get(bias_type, []),
        }

    @classmethod
    def create_pretrained(
        cls,
        texts: List[str],
        labels: List[str],
        name: str = "pretrained_bias_detector",
        description: str = "Pre-trained bias detector",
        bias_config: Optional[BiasConfig] = None,
        **kwargs,
    ) -> "BiasDetector":
        """
        Create and train a bias detector in one step.

        Args:
            texts: List of texts to train on
            labels: List of bias type labels
            name: Name of the classifier
            description: Description of the classifier
            bias_config: Bias detection configuration
            **kwargs: Additional configuration parameters

        Returns:
            Trained BiasDetector
        """
        classifier = cls(name=name, description=description, bias_config=bias_config, **kwargs)
        return classifier.fit(texts, labels)
