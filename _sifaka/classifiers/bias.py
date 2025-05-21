"""
Bias classifier for detecting politically charged language.

This module provides a classifier for detecting politically charged language
and bias in text. It uses a TF-IDF vectorizer and a logistic regression model
to classify text as biased or unbiased.
"""

import logging
import os
import pickle
from typing import Any, List, Optional

import numpy as np

# sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sifaka.classifiers import ClassificationResult, Classifier

logger = logging.getLogger(__name__)

# Sample training data for bias detection
# This would ideally be replaced with a larger, more diverse dataset
BIASED_SAMPLES = [
    "These radical leftists are destroying our country with their socialist agenda.",
    "Right-wing extremists are all racist and ignorant.",
    "Liberals are brainwashed sheep who can't think for themselves.",
    "Conservatives are all corrupt and only care about the rich.",
    "Those idiots on the other side don't understand basic economics.",
    "The other party is full of hypocrites and liars.",
    "Those fanatics are ruining everything with their irrational beliefs.",
    "The opposing party's supporters are all part of a cult.",
    "They're all stupid and don't understand how the world works.",
    "Those morons are blindly following their corrupt leaders.",
    "The other side is evil and wants to destroy our way of life.",
    "Those ignorant voters don't know what's good for them.",
    "The other party's blind followers will believe anything they're told.",
    "Those people are all sheep who can't think critically.",
    "The other side is full of idiots who don't understand basic facts.",
]

UNBIASED_SAMPLES = [
    "There are policy differences between the two parties on economic issues.",
    "Both sides have valid perspectives on healthcare reform.",
    "The debate on tax policy involves complex trade-offs.",
    "Different approaches to environmental regulation have pros and cons.",
    "There are reasonable arguments on both sides of the immigration debate.",
    "Policy experts disagree on the best approach to education reform.",
    "The discussion about government spending involves legitimate concerns.",
    "Both parties have contributed to the current situation.",
    "There are thoughtful people with different views on this issue.",
    "The debate involves complex questions without simple answers.",
    "Reasonable people can disagree on the best policy approach.",
    "There are valid concerns raised by people on both sides.",
    "The issue requires careful consideration of multiple perspectives.",
    "Different values lead to different policy preferences.",
    "The debate reflects genuine differences in priorities and values.",
]


class BiasClassifier(Classifier):
    """Classifier for detecting politically charged language and bias in text.

    This classifier uses a TF-IDF vectorizer and a logistic regression model
    to classify text as biased or unbiased.

    Attributes:
        threshold (float): The confidence threshold for classification.
        model_path (str): Path to the saved model file.
        model (Pipeline): The trained classification model.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        model_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the bias classifier.

        Args:
            threshold: The confidence threshold for classification.
            model_path: Path to the saved model file. If not provided, a new model will be trained.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._name = "bias_classifier"
        self.threshold = threshold
        self.model_path = model_path
        self.model: Any = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the classification model.

        If a model path is provided and the file exists, load the model from the file.
        Otherwise, train a new model.
        """
        try:
            if self.model_path and os.path.exists(self.model_path):
                logger.debug(f"Loading bias classifier model from {self.model_path}")
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            else:
                logger.debug("Training new bias classifier model")
                self._train_model()
        except Exception as e:
            logger.error(f"Failed to initialize bias classifier: {str(e)}")
            raise

    def _train_model(self) -> None:
        """Train a new classification model.

        This method creates a pipeline with a TF-IDF vectorizer and a logistic regression model,
        and trains it on the sample data.
        """
        try:
            # Prepare training data
            X = BIASED_SAMPLES + UNBIASED_SAMPLES
            y = [1] * len(BIASED_SAMPLES) + [0] * len(UNBIASED_SAMPLES)

            # Create and train the model
            self.model = Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            max_features=5000, ngram_range=(1, 2), stop_words="english"
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
                        ),
                    ),
                ]
            )

            self.model.fit(X, y)
            logger.debug("Successfully trained bias classifier model")

            # Save the model if a path is provided
            if self.model_path:
                logger.debug(f"Saving bias classifier model to {self.model_path}")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, "wb") as f:
                    pickle.dump(self.model, f)
        except Exception as e:
            logger.error(f"Failed to train bias classifier model: {str(e)}")
            raise

    def classify(self, text: str) -> ClassificationResult:
        """Classify text as biased or unbiased.

        Args:
            text: The text to classify.

        Returns:
            A ClassificationResult with the classification label and confidence score.
        """
        try:
            if not text.strip():
                logger.warning("Empty text provided for classification")
                return ClassificationResult(
                    label="unbiased", confidence=0.0, metadata={"reason": "empty_text"}
                )

            # Get prediction probabilities
            probabilities = self.model.predict_proba([text])[0]

            # Get the predicted class (0 = unbiased, 1 = biased)
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])

            # Map class index to label
            label = "biased" if predicted_class == 1 else "unbiased"

            logger.debug(f"Classified text as '{label}' with confidence {confidence:.2f}")

            return ClassificationResult(
                label=label,
                confidence=confidence,
                metadata={
                    "biased_probability": float(probabilities[1]),
                    "unbiased_probability": float(probabilities[0]),
                    "text_length": len(text),
                },
            )
        except Exception as e:
            logger.error(f"Failed to classify text: {str(e)}")
            return ClassificationResult(
                label="unbiased",
                confidence=0.0,
                metadata={"error": str(e), "reason": "classification_error"},
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts.

        Args:
            texts: The list of texts to classify.

        Returns:
            A list of ClassificationResults.
        """
        return [self.classify(text) for text in texts]

    @property
    def name(self) -> str:
        """Get the classifier name."""
        return "bias_classifier"

    @property
    def description(self) -> str:
        """Get the classifier description."""
        return "Detects politically charged language and bias in text"
