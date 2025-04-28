"""
Spam classifier using scikit-learn's Naive Bayes.
"""

import importlib
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SpamConfig:
    """Configuration for spam classification."""

    min_confidence: float = 0.7  # Minimum confidence threshold
    max_features: int = 1000  # Max features for vectorization
    random_state: int = 42  # For reproducibility
    model_path: Optional[str] = None  # Path to save/load the model
    use_bigrams: bool = True  # Whether to use bigrams in feature extraction

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")


class SpamClassifier(BaseClassifier):
    """
    A spam classifier using Naive Bayes from scikit-learn.

    This classifier detects spam content in text using a trained Naive Bayes model.

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    def __init__(
        self,
        name: str = "spam_classifier",
        description: str = "Detects spam content in text",
        spam_config: Optional[SpamConfig] = None,
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the spam classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            spam_config: Spam classification configuration
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Store spam config
        self._spam_config = spam_config or SpamConfig()

        # Initialize other attributes
        self._vectorizer = None
        self._model = None
        self._pipeline = None
        self._initialized = False

        # Create config if not provided
        if config is None:
            # Extract params from kwargs if present
            params = kwargs.pop("params", {})

            # Add spam config to params
            params["min_confidence"] = self._spam_config.min_confidence
            params["max_features"] = self._spam_config.max_features
            params["random_state"] = self._spam_config.random_state
            params["model_path"] = self._spam_config.model_path
            params["use_bigrams"] = self._spam_config.use_bigrams

            # Create config with remaining kwargs
            config = ClassifierConfig(labels=["ham", "spam"], cost=1.5, params=params, **kwargs)

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

    def _load_dependencies(self) -> None:
        """Load scikit-learn dependencies."""
        try:
            # Import necessary scikit-learn modules
            self._sklearn_feature_extraction_text = importlib.import_module(
                "sklearn.feature_extraction.text"
            )
            self._sklearn_naive_bayes = importlib.import_module("sklearn.naive_bayes")
            self._sklearn_pipeline = importlib.import_module("sklearn.pipeline")
            return True
        except ImportError:
            raise ImportError(
                "scikit-learn is required for SpamClassifier. "
                "Install it with: pip install scikit-learn"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scikit-learn modules: {e}")

    def warm_up(self) -> None:
        """Initialize the model if needed."""
        if not self._initialized:
            self._load_dependencies()

            if self._spam_config.model_path and os.path.exists(self._spam_config.model_path):
                self._load_model(self._spam_config.model_path)
            else:
                # Create TF-IDF vectorizer
                ngram_range = (1, 2) if self._spam_config.use_bigrams else (1, 1)
                self._vectorizer = self._sklearn_feature_extraction_text.TfidfVectorizer(
                    max_features=self._spam_config.max_features,
                    stop_words="english",
                    ngram_range=ngram_range,
                )

                # Create Naive Bayes model
                self._model = self._sklearn_naive_bayes.MultinomialNB()

                # Create pipeline
                self._pipeline = self._sklearn_pipeline.Pipeline(
                    [
                        ("vectorizer", self._vectorizer),
                        ("classifier", self._model),
                    ]
                )

            self._initialized = True

    def _save_model(self, path: str) -> None:
        """Save the model to a file."""
        try:
            with open(path, "wb") as f:
                pickle.dump(self._pipeline, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_model(self, path: str) -> None:
        """Load the model from a file."""
        try:
            with open(path, "rb") as f:
                self._pipeline = pickle.load(f)
                # Extract vectorizer and model from pipeline
                self._vectorizer = self._pipeline.named_steps["vectorizer"]
                self._model = self._pipeline.named_steps["classifier"]
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def fit(self, texts: List[str], labels: List[str]) -> "SpamClassifier":
        """
        Fit the spam classifier on a corpus of texts.

        Args:
            texts: List of texts to fit the model on
            labels: List of labels ("ham" or "spam")

        Returns:
            self: The fitted classifier
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

        if not all(label in self.config.labels for label in labels):
            raise ValueError(f"Labels must be one of {self.config.labels}")

        # Convert string labels to integers for scikit-learn
        label_indices = [self.config.labels.index(label) for label in labels]

        self.warm_up()

        # Fit the pipeline
        self._pipeline.fit(texts, label_indices)

        # Save the model if path is provided
        if self._spam_config.model_path:
            self._save_model(self._spam_config.model_path)

        return self

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement spam classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with spam detection result and confidence
        """
        if not self._pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Predict probability
        proba = self._pipeline.predict_proba([text])[0]

        # Get dominant class
        pred_class = int(proba[1] >= 0.5)  # 1 if spam probability >= 0.5, else 0
        confidence = float(proba[pred_class])

        # Create metadata
        metadata = {
            "probabilities": {
                "ham": float(proba[0]),
                "spam": float(proba[1]),
            },
            "threshold": 0.5,
            "is_confident": confidence >= self._spam_config.min_confidence,
        }

        return ClassificationResult(
            label=self.config.labels[pred_class],
            confidence=confidence,
            metadata=metadata,
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts efficiently.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        self.validate_batch_input(texts)

        if not self._pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Predict probabilities for all texts
        probas = self._pipeline.predict_proba(texts)

        results = []
        for proba in probas:
            pred_class = int(proba[1] >= 0.5)
            confidence = float(proba[pred_class])

            metadata = {
                "probabilities": {
                    "ham": float(proba[0]),
                    "spam": float(proba[1]),
                },
                "threshold": 0.5,
                "is_confident": confidence >= self._spam_config.min_confidence,
            }

            results.append(
                ClassificationResult(
                    label=self.config.labels[pred_class],
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        return results

    @classmethod
    def create_pretrained(
        cls,
        texts: List[str],
        labels: List[str],
        name: str = "pretrained_spam_classifier",
        description: str = "Pre-trained spam classifier",
        spam_config: Optional[SpamConfig] = None,
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> "SpamClassifier":
        """
        Create and train a spam classifier in one step.

        Args:
            texts: List of texts to train on
            labels: List of labels ("ham" or "spam")
            name: Name of the classifier
            description: Description of the classifier
            spam_config: Spam classification configuration
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters

        Returns:
            Trained SpamClassifier
        """
        # Create instance with provided configuration
        classifier = cls(
            name=name, description=description, spam_config=spam_config, config=config, **kwargs
        )

        # Train the classifier and return it
        return classifier.fit(texts, labels)
