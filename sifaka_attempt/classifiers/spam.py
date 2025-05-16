"""
Spam classifier for categorizing text as spam or ham.

This module provides a classifier that categorizes text as spam or ham (non-spam)
using scikit-learn's Naive Bayes classifier.
"""

import os
import pickle
from typing import Dict, Any, List, Optional, Union
from ..di import inject
from . import ClassificationResult


class SpamClassifier:
    """
    Classifier that categorizes text as spam or ham.

    This classifier uses scikit-learn's Naive Bayes algorithm to detect spam content
    in text. It can be trained on custom datasets and provides detailed prediction
    probabilities.

    By default, it requires the 'scikit-learn' package to be installed.
    Install it with: pip install scikit-learn
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_features: int = 1000,
        use_bigrams: bool = True,
    ):
        """
        Initialize the spam classifier.

        Args:
            model_path: Path to a pre-trained model file
            max_features: Maximum number of features for the vectorizer
            use_bigrams: Whether to use bigrams in addition to unigrams
        """
        self.model_path = model_path
        self.max_features = max_features
        self.use_bigrams = use_bigrams
        self._vectorizer = None
        self._model = None
        self._is_initialized = False

    def _load_dependencies(self) -> bool:
        """
        Load scikit-learn dependencies.

        Returns:
            True if dependencies are successfully loaded, False otherwise
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB

            return True
        except ImportError:
            raise ImportError(
                "scikit-learn is required for SpamClassifier. "
                "Install it with: pip install scikit-learn"
            )

    def _initialize(self) -> None:
        """Initialize the model and vectorizer."""
        if self._is_initialized:
            return

        # Load dependencies
        if not self._load_dependencies():
            return

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB

        # Initialize vectorizer
        ngram_range = (1, 2) if self.use_bigrams else (1, 1)
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=ngram_range,
            strip_accents="unicode",
            min_df=2,
            stop_words="english",
        )

        # Initialize model or load pre-trained model
        if self.model_path and os.path.exists(self.model_path):
            self._load_model(self.model_path)
        else:
            self._model = MultinomialNB(alpha=1.0)

        self._is_initialized = True

    def _load_model(self, path: str) -> None:
        """
        Load a pre-trained model from disk.

        Args:
            path: Path to the model file
        """
        try:
            with open(path, "rb") as f:
                saved_data = pickle.load(f)
                self._vectorizer = saved_data.get("vectorizer")
                self._model = saved_data.get("model")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}")

    def _save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model file
        """
        if not self._is_initialized:
            raise RuntimeError("Model is not initialized")

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"vectorizer": self._vectorizer, "model": self._model}, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {path}: {str(e)}")

    def fit(self, texts: List[str], labels: List[str]) -> "SpamClassifier":
        """
        Train the classifier on labeled data.

        Args:
            texts: List of training texts
            labels: List of corresponding labels ("spam" or "ham")

        Returns:
            Self for method chaining
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

        if not texts:
            raise ValueError("Training data is empty")

        self._initialize()

        # Vectorize the texts
        X = self._vectorizer.fit_transform(texts)

        # Train the model
        self._model.fit(X, labels)

        return self

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text as spam or ham.

        Args:
            text: The text to classify

        Returns:
            A ClassificationResult with the spam/ham label and confidence score
        """
        # Initialize if needed
        if not self._is_initialized:
            try:
                self._initialize()
            except Exception as e:
                return ClassificationResult(
                    label="unknown",
                    confidence=0.0,
                    passed=False,
                    message=f"Error initializing classifier: {str(e)}",
                    metadata={"error": str(e)},
                )

        if not text.strip():
            return ClassificationResult(
                label="ham",
                confidence=0.8,
                message="Empty text is considered non-spam",
                metadata={"probabilities": {"ham": 0.8, "spam": 0.2}},
            )

        try:
            # Vectorize the text
            X = self._vectorizer.transform([text])

            # Get class probabilities
            probs = self._model.predict_proba(X)[0]

            # Get class names
            classes = self._model.classes_

            # Create probabilities dictionary
            probabilities = {cls: float(prob) for cls, prob in zip(classes, probs)}

            # Get predicted class and confidence
            predicted_idx = probs.argmax()
            label = classes[predicted_idx]
            confidence = float(probs[predicted_idx])

            # Create message
            if label == "spam":
                message = "Text appears to be spam"
            else:
                message = "Text appears to be legitimate (ham)"

            return ClassificationResult(
                label=label,
                confidence=confidence,
                message=message,
                metadata={"probabilities": probabilities},
            )
        except Exception as e:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                passed=False,
                message=f"Error classifying text: {str(e)}",
                metadata={"error": str(e)},
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts efficiently.

        Args:
            texts: The list of texts to classify

        Returns:
            A list of ClassificationResults
        """
        # Initialize if needed
        if not self._is_initialized:
            try:
                self._initialize()
            except Exception as e:
                return [
                    ClassificationResult(
                        label="unknown",
                        confidence=0.0,
                        passed=False,
                        message=f"Error initializing classifier: {str(e)}",
                        metadata={"error": str(e)},
                    )
                    for _ in texts
                ]

        results = []

        # Process empty texts
        for text in texts:
            if not text.strip():
                results.append(
                    ClassificationResult(
                        label="ham",
                        confidence=0.8,
                        message="Empty text is considered non-spam",
                        metadata={"probabilities": {"ham": 0.8, "spam": 0.2}},
                    )
                )

        # Get non-empty texts
        non_empty_texts = [text for text in texts if text.strip()]
        non_empty_indices = [i for i, text in enumerate(texts) if text.strip()]

        if non_empty_texts:
            try:
                # Vectorize the texts
                X = self._vectorizer.transform(non_empty_texts)

                # Get class probabilities
                batch_probs = self._model.predict_proba(X)

                # Get class names
                classes = self._model.classes_

                # Create results for non-empty texts
                for i, (idx, probs) in enumerate(zip(non_empty_indices, batch_probs)):
                    # Create probabilities dictionary
                    probabilities = {cls: float(prob) for cls, prob in zip(classes, probs)}

                    # Get predicted class and confidence
                    predicted_idx = probs.argmax()
                    label = classes[predicted_idx]
                    confidence = float(probs[predicted_idx])

                    # Create message
                    if label == "spam":
                        message = "Text appears to be spam"
                    else:
                        message = "Text appears to be legitimate (ham)"

                    # Insert result at the correct position
                    while len(results) <= idx:
                        results.append(None)

                    results[idx] = ClassificationResult(
                        label=label,
                        confidence=confidence,
                        message=message,
                        metadata={"probabilities": probabilities},
                    )
            except Exception as e:
                error_result = ClassificationResult(
                    label="unknown",
                    confidence=0.0,
                    passed=False,
                    message=f"Error batch classifying texts: {str(e)}",
                    metadata={"error": str(e)},
                )
                # Fill in missing results with the error
                for idx in non_empty_indices:
                    while len(results) <= idx:
                        results.append(None)
                    results[idx] = error_result

        # Fill any None values with error results
        for i in range(len(results)):
            if results[i] is None:
                results[i] = ClassificationResult(
                    label="unknown",
                    confidence=0.0,
                    passed=False,
                    message="Processing error",
                    metadata={"error": "Unknown error during batch processing"},
                )

        return results
