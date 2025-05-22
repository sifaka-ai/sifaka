"""
Spam classifier for Sifaka.

This module provides a classifier that categorizes text as spam or ham (non-spam)
using scikit-learn's Naive Bayes classifier with TF-IDF vectorization.
"""

import importlib
import os
import pickle
from typing import Any, List, Optional

from sifaka.classifiers import ClassificationResult


class SpamClassifier:
    """
    A spam classifier that categorizes text as spam or ham (non-spam).

    This classifier uses scikit-learn's Naive Bayes algorithm with TF-IDF vectorization
    to detect spam content in text. It can be trained on custom datasets and provides
    detailed prediction probabilities.

    Attributes:
        model_path: Path to a pre-trained model file.
        threshold: Confidence threshold for considering text spam.
        name: The name of the classifier.
        description: The description of the classifier.
    """

    # Type annotations for instance variables
    _name: str
    _description: str
    _model_path: Optional[str]
    _threshold: float
    _model: Any
    _vectorizer: Any
    _initialized: bool

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        name: str = "spam_classifier",
        description: str = "Classifies text as spam or ham (non-spam)",
    ):
        """
        Initialize the spam classifier.

        Args:
            model_path: Optional path to a pre-trained model file.
            threshold: Confidence threshold for considering text spam.
            name: The name of the classifier.
            description: The description of the classifier.
        """
        self._name = name
        self._description = description
        self._model_path = model_path
        self._threshold = threshold
        self._model = None
        self._vectorizer = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Get the classifier name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the classifier description."""
        return self._description

    def _load_scikit_learn(self) -> tuple[Any, Any]:
        """
        Load scikit-learn and related modules.

        Returns:
            A tuple of (sklearn, numpy) modules.

        Raises:
            ImportError: If required packages are not installed.
            RuntimeError: If initialization fails.
        """
        try:
            sklearn = importlib.import_module("sklearn")
            numpy = importlib.import_module("numpy")
            return sklearn, numpy
        except ImportError:
            raise ImportError(
                "scikit-learn and numpy packages are required for SpamClassifier. "
                "Install them with: pip install scikit-learn numpy"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scikit-learn: {e}")

    def _create_default_model(self) -> tuple[Any, Any]:
        """
        Create a default spam detection model.

        Returns:
            A tuple of (model, vectorizer).

        Raises:
            RuntimeError: If model creation fails.
        """
        try:
            _, _ = self._load_scikit_learn()

            # Import required modules
            naive_bayes = importlib.import_module("sklearn.naive_bayes")
            feature_extraction = importlib.import_module("sklearn.feature_extraction.text")

            # Create vectorizer and model
            vectorizer = feature_extraction.TfidfVectorizer(
                max_features=5000, min_df=5, max_df=0.7, stop_words="english"
            )

            model = naive_bayes.MultinomialNB(alpha=1.0)

            # Train on a small default dataset
            spam_texts = [
                "Buy now! Limited time offer! 50% off! Act fast!",
                "Congratulations! You've won a free iPhone. Click here to claim your prize!",
                "URGENT: Your account has been compromised. Verify your details now!",
                "Make money fast! Work from home and earn $5000 per week!",
                "Free Viagra! Discount medications! No prescription needed!",
                "You have been selected for a special offer. Reply now!",
                "Increase your manhood size with this amazing pill!",
                "Get rich quick! Invest now and double your money in 24 hours!",
                "Lose weight fast! No diet, no exercise, just take this pill!",
                "Your inheritance of $5,000,000 is waiting. Contact us now!",
            ]

            ham_texts = [
                "Hi, can we schedule a meeting for tomorrow at 2pm?",
                "Please find attached the report you requested.",
                "Thank you for your email. I'll get back to you soon.",
                "The project deadline has been extended to next Friday.",
                "Here are the meeting notes from yesterday's discussion.",
                "Could you please review this document and provide feedback?",
                "I'm out of office today. Will respond to emails tomorrow.",
                "The quarterly results are now available on the intranet.",
                "Please submit your expenses by the end of the month.",
                "We're hiring for a new position. Let me know if you know anyone suitable.",
            ]

            # Combine texts and create labels
            all_texts = spam_texts + ham_texts
            labels = ["spam"] * len(spam_texts) + ["ham"] * len(ham_texts)

            # Fit vectorizer and transform texts
            X = vectorizer.fit_transform(all_texts)

            # Train model
            model.fit(X, labels)

            return model, vectorizer

        except Exception as e:
            raise RuntimeError(f"Failed to create default model: {e}")

    def _load_model(self) -> tuple[Any, Any]:
        """
        Load a pre-trained model from file.

        Returns:
            A tuple of (model, vectorizer).

        Raises:
            FileNotFoundError: If model file is not found.
            RuntimeError: If model loading fails.
        """
        try:
            if not self._model_path or not os.path.exists(self._model_path):
                return self._create_default_model()

            with open(self._model_path, "rb") as f:
                model_data = pickle.load(f)

            model = model_data.get("model")
            vectorizer = model_data.get("vectorizer")

            if not model or not vectorizer:
                raise ValueError("Invalid model file: missing model or vectorizer")

            return model, vectorizer

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _initialize(self) -> None:
        """Initialize the spam classifier if needed."""
        if not self._initialized:
            self._model, self._vectorizer = self._load_model()
            self._initialized = True

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text as spam or ham (non-spam).

        Args:
            text: The text to classify.

        Returns:
            A ClassificationResult with the spam/ham label and confidence score.
        """
        # Handle empty text
        if not text or not text.strip():
            return ClassificationResult(
                label="ham",
                confidence=0.7,
                metadata={
                    "input_length": 0,
                    "reason": "empty_text",
                    "probabilities": {"spam": 0.3, "ham": 0.7},
                },
            )

        try:
            # Initialize model if needed
            self._initialize()

            # Load numpy
            _, numpy = self._load_scikit_learn()

            # After initialization, model and vectorizer should be available
            # If not, it's an error
            if self._vectorizer is None or self._model is None:
                raise RuntimeError("Failed to initialize spam classifier model")

            # Vectorize the text
            X = self._vectorizer.transform([text])

            # Get class probabilities
            probabilities = self._model.predict_proba(X)[0]

            # Get class names
            classes = self._model.classes_

            # Create probabilities dictionary
            probs_dict = {
                cls: float(prob) for cls, prob in zip(classes, probabilities, strict=False)
            }

            # Determine label and confidence
            if "spam" in probs_dict and probs_dict["spam"] > self._threshold:
                label = "spam"
                confidence = probs_dict["spam"]
            else:
                label = "ham"
                confidence = probs_dict.get("ham", 1.0 - probs_dict.get("spam", 0.0))

            # Extract spam features
            feature_names = self._vectorizer.get_feature_names_out()
            X_array = X.toarray()[0]

            # Get top features for this text
            top_features = []
            for i in numpy.argsort(X_array)[-10:]:
                if X_array[i] > 0:
                    top_features.append(
                        {
                            "word": feature_names[i],
                            "score": float(X_array[i]),
                        }
                    )

            return ClassificationResult(
                label=label,
                confidence=confidence,
                metadata={
                    "input_length": len(text),
                    "probabilities": probs_dict,
                    "top_features": top_features,
                },
            )

        except Exception as e:
            # Handle errors
            return ClassificationResult(
                label="ham",
                confidence=0.5,
                metadata={
                    "error": str(e),
                    "reason": "classification_error",
                    "input_length": len(text),
                },
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: The list of texts to classify.

        Returns:
            A list of ClassificationResults.
        """
        # Initialize model if needed
        self._initialize()

        # Handle empty list
        if not texts:
            return []

        # Handle empty texts
        results = []
        non_empty_texts = []
        non_empty_indices = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append(
                    ClassificationResult(
                        label="ham",
                        confidence=0.7,
                        metadata={
                            "input_length": 0,
                            "reason": "empty_text",
                            "probabilities": {"spam": 0.3, "ham": 0.7},
                        },
                    )
                )
            else:
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        # If there are no non-empty texts, return the results
        if not non_empty_texts:
            return results

        try:
            # Load numpy
            _, numpy = self._load_scikit_learn()

            # After initialization, model and vectorizer should be available
            # If not, it's an error
            if self._vectorizer is None or self._model is None:
                raise RuntimeError("Failed to initialize spam classifier model")

            # Vectorize the texts
            X = self._vectorizer.transform(non_empty_texts)

            # Get class probabilities
            probabilities = self._model.predict_proba(X)

            # Get class names
            classes = self._model.classes_

            # Get feature names
            feature_names = self._vectorizer.get_feature_names_out()

            # Process each result
            for i, idx in enumerate(non_empty_indices):
                text = non_empty_texts[i]
                probs = probabilities[i]

                # Create probabilities dictionary
                probs_dict = {cls: float(prob) for cls, prob in zip(classes, probs, strict=False)}

                # Determine label and confidence
                if "spam" in probs_dict and probs_dict["spam"] > self._threshold:
                    label = "spam"
                    confidence = probs_dict["spam"]
                else:
                    label = "ham"
                    confidence = probs_dict.get("ham", 1.0 - probs_dict.get("spam", 0.0))

                # Extract spam features for this text
                X_array = X[i].toarray()[0]

                # Get top features for this text
                top_features = []
                for j in numpy.argsort(X_array)[-10:]:
                    if X_array[j] > 0:
                        top_features.append(
                            {
                                "word": feature_names[j],
                                "score": float(X_array[j]),
                            }
                        )

                # Create result
                result = ClassificationResult(
                    label=label,
                    confidence=confidence,
                    metadata={
                        "input_length": len(text),
                        "probabilities": probs_dict,
                        "top_features": top_features,
                    },
                )

                # Insert result at the correct position
                results.insert(idx, result)

            return results

        except Exception:
            # Handle errors by classifying each text individually
            for text in non_empty_texts:
                results.append(self.classify(text))

            return results
