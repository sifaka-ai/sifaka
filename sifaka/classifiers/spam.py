"""
Spam classifier using scikit-learn's Naive Bayes.
"""

import importlib
import os
import pickle
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import PrivateAttr
from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class SpamClassifier(BaseClassifier[str, str]):
    """
    A spam classifier using Naive Bayes from scikit-learn.

    This classifier detects spam content in text using a trained Naive Bayes model.

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = ["ham", "spam"]
    DEFAULT_COST: ClassVar[int] = 1.5  # Slightly higher cost for ML-based model

    # Private attributes using PrivateAttr for state management
    _initialized: bool = PrivateAttr(default=False)
    _sklearn_feature_extraction_text: Optional[Any] = PrivateAttr(default=None)
    _sklearn_naive_bayes: Optional[Any] = PrivateAttr(default=None)
    _sklearn_pipeline: Optional[Any] = PrivateAttr(default=None)
    _vectorizer: Optional[Any] = PrivateAttr(default=None)
    _model: Optional[Any] = PrivateAttr(default=None)
    _pipeline: Optional[Any] = PrivateAttr(default=None)

    def __init__(
        self,
        name: str = "spam_classifier",
        description: str = "Detects spam content in text",
        config: Optional[ClassifierConfig[str]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the spam classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Create config if not provided
        if config is None:
            # Extract model path and features from kwargs
            model_params = {
                "model_path": kwargs.pop("model_path", None),
                "max_features": kwargs.pop("max_features", 1000),
                "use_bigrams": kwargs.pop("use_bigrams", True),
            }

            # Create config with remaining kwargs
            config = ClassifierConfig[str](
                labels=self.DEFAULT_LABELS,
                cost=self.DEFAULT_COST,
                params=model_params,
                **kwargs
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)
        self._initialized = False

        # Try to load model if path is provided
        model_path = self.config.params.get("model_path")
        if model_path and os.path.exists(model_path):
            self.warm_up()

    def _load_dependencies(self) -> bool:
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

            model_path = self.config.params.get("model_path")
            if model_path and os.path.exists(model_path):
                self._load_model(model_path)
            else:
                # Get parameters from config
                max_features = self.config.params.get("max_features", 1000)
                use_bigrams = self.config.params.get("use_bigrams", True)

                # Create TF-IDF vectorizer
                ngram_range = (1, 2) if use_bigrams else (1, 1)
                self._vectorizer = self._sklearn_feature_extraction_text.TfidfVectorizer(
                    max_features=max_features,
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
        model_path = self.config.params.get("model_path")
        if model_path:
            self._save_model(model_path)

        return self

    def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
        """
        Implement spam classification logic without caching.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with prediction details
        """
        if not self._initialized:
            self.warm_up()

        try:
            # Get prediction probabilities
            proba = self._pipeline.predict_proba([text])[0]

            # Get predicted label index and confidence
            label_idx = proba.argmax()
            confidence = float(proba[label_idx])

            # Get label from index
            label = self.config.labels[label_idx]

            return ClassificationResult[str](
                label=label,
                confidence=confidence,
                metadata={"probabilities": {l: float(p) for l, p in zip(self.config.labels, proba)}},
            )
        except Exception as e:
            logger.error(f"Failed to classify text: {e}")
            return ClassificationResult[str](
                label="unknown",
                confidence=0.0,
                metadata={"error": str(e), "reason": "classification_error"},
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult[str]]:
        """
        Classify multiple texts efficiently.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        self.validate_batch_input(texts)

        if not self._initialized:
            self.warm_up()

        try:
            # Predict probabilities for all texts
            probas = self._pipeline.predict_proba(texts)

            results = []
            for proba in probas:
                label_idx = proba.argmax()
                confidence = float(proba[label_idx])

                results.append(
                    ClassificationResult[str](
                        label=self.config.labels[label_idx],
                        confidence=confidence,
                        metadata={
                            "probabilities": {l: float(p) for l, p in zip(self.config.labels, proba)}
                        },
                    )
                )

            return results
        except Exception as e:
            logger.error(f"Failed to batch classify texts: {e}")
            return [
                ClassificationResult[str](
                    label="unknown",
                    confidence=0.0,
                    metadata={"error": str(e), "reason": "batch_classification_error"},
                )
                for _ in texts
            ]

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

        Args:
            texts: List of texts to train on
            labels: List of labels ("ham" or "spam")
            name: Name of the classifier
            description: Description of the classifier
            **kwargs: Additional configuration parameters

        Returns:
            Trained SpamClassifier
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
    model_path: Optional[str] = None,
    max_features: int = 1000,
    use_bigrams: bool = True,
    cache_size: int = 0,
    cost: int = 1.5,
    **kwargs,
) -> SpamClassifier:
    """
    Factory function to create a spam classifier.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        model_path: Path to a saved model file (optional)
        max_features: Maximum number of features to use in vectorizer
        use_bigrams: Whether to use bigrams in addition to unigrams
        cache_size: Size of the classification cache (0 to disable)
        cost: Computational cost of this classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured SpamClassifier instance
    """
    # Prepare params
    params = kwargs.pop("params", {})
    params.update({
        "model_path": model_path,
        "max_features": max_features,
        "use_bigrams": use_bigrams,
    })

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
