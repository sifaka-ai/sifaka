"""
Bias classifier for categorizing bias in text.

This module provides a classifier that detects various forms of bias in text,
including gender bias, racial bias, and other forms of discriminatory language.
"""

import os
import re
import pickle
from typing import Dict, Any, List, Optional, Union
from ..di import inject
from . import ClassificationResult


class BiasClassifier:
    """
    Classifier that categorizes bias in text.

    This classifier detects various forms of bias in text, including gender bias,
    racial bias, and other forms of discriminatory language using a combination
    of keyword detection and machine learning.

    By default, it requires the 'scikit-learn' package to be installed.
    Install it with: pip install scikit-learn
    """

    # Default bias types to detect
    DEFAULT_BIAS_TYPES = [
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

    # Default keywords for each bias category to enhance detection
    DEFAULT_BIAS_KEYWORDS = {
        "gender": [
            "man",
            "woman",
            "male",
            "female",
            "gender",
            "sex",
            "boy",
            "girl",
            "men",
            "women",
            "masculine",
            "feminine",
            "manly",
            "womanly",
            "gentleman",
            "lady",
            "ladies",
            "guys",
        ],
        "racial": [
            "race",
            "ethnicity",
            "black",
            "white",
            "asian",
            "hispanic",
            "african",
            "european",
            "oriental",
            "latino",
            "latina",
            "caucasian",
            "minority",
            "racist",
            "discrimination",
        ],
        "political": [
            "conservative",
            "liberal",
            "democrat",
            "republican",
            "left-wing",
            "right-wing",
            "progressive",
            "traditional",
            "socialist",
            "capitalist",
            "communist",
            "fascist",
        ],
        "age": [
            "young",
            "old",
            "elderly",
            "youth",
            "adult",
            "senior",
            "millennial",
            "boomer",
            "generation",
            "gen z",
            "gen x",
        ],
        "socioeconomic": [
            "rich",
            "poor",
            "wealthy",
            "poverty",
            "class",
            "privileged",
            "underprivileged",
            "income",
            "affluent",
            "welfare",
            "elite",
            "working class",
            "middle class",
        ],
        "religious": [
            "christian",
            "muslim",
            "jewish",
            "hindu",
            "buddhist",
            "atheist",
            "religion",
            "faith",
            "belief",
            "church",
            "mosque",
            "temple",
            "secular",
            "spiritual",
        ],
        "cultural": [
            "culture",
            "tradition",
            "heritage",
            "custom",
            "values",
            "lifestyle",
            "western",
            "eastern",
            "indigenous",
            "native",
            "foreign",
            "immigrant",
            "multicultural",
        ],
        "educational": [
            "educated",
            "uneducated",
            "academic",
            "intellectual",
            "smart",
            "dumb",
            "intelligent",
            "ignorant",
            "degree",
            "school",
            "college",
            "university",
            "professor",
            "student",
        ],
        "geographical": [
            "rural",
            "urban",
            "city",
            "country",
            "coastal",
            "inland",
            "north",
            "south",
            "east",
            "west",
            "region",
            "state",
            "nation",
            "local",
            "global",
        ],
    }

    def __init__(
        self,
        bias_types: Optional[List[str]] = None,
        bias_keywords: Optional[Dict[str, List[str]]] = None,
        threshold: float = 0.7,
        model_path: Optional[str] = None,
        max_features: int = 3000,
    ):
        """
        Initialize the bias classifier.

        Args:
            bias_types: List of bias types to detect (defaults to DEFAULT_BIAS_TYPES)
            bias_keywords: Dictionary of bias types and associated keywords
            threshold: Threshold for bias detection (0.0 to 1.0)
            model_path: Path to a pre-trained model file
            max_features: Maximum number of features for the vectorizer
        """
        self.bias_types = bias_types or self.DEFAULT_BIAS_TYPES
        self.bias_keywords = bias_keywords or self.DEFAULT_BIAS_KEYWORDS
        self.threshold = threshold
        self.model_path = model_path
        self.max_features = max_features
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
            from sklearn.svm import SVC

            return True
        except ImportError:
            raise ImportError(
                "scikit-learn is required for BiasClassifier. "
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
        from sklearn.svm import SVC

        # Initialize vectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            strip_accents="unicode",
            min_df=2,
            stop_words="english",
        )

        # Initialize model or load pre-trained model
        if self.model_path and os.path.exists(self.model_path):
            self._load_model(self.model_path)
        else:
            self._model = SVC(probability=True, kernel="linear", C=1.0)

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

    def _extract_bias_features(self, text: str) -> Dict[str, float]:
        """
        Extract bias-related features from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of bias type to score
        """
        text_lower = text.lower()
        features = {}

        # Check for keyword matches for each bias type
        for bias_type, keywords in self.bias_keywords.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            normalized_score = min(1.0, matches / max(10, len(text.split()) / 5))
            features[bias_type] = normalized_score

        return features

    def fit(self, texts: List[str], labels: List[str]) -> "BiasClassifier":
        """
        Train the classifier on labeled data.

        Args:
            texts: List of training texts
            labels: List of corresponding bias types

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
        Classify text for bias.

        Args:
            text: The text to classify

        Returns:
            A ClassificationResult with the bias type and confidence score
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
                label="neutral",
                confidence=0.9,
                message="Empty text contains no bias",
                metadata={"bias_features": {}},
            )

        try:
            # Extract bias features
            bias_features = self._extract_bias_features(text)

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

            # Boost confidence based on keyword matches
            if label in bias_features and label != "neutral":
                feature_score = bias_features.get(label, 0.0)
                # Weighted average of model confidence and feature score
                confidence = (confidence * 0.7) + (feature_score * 0.3)

            # If confidence is below threshold, default to neutral
            if confidence < self.threshold and label != "neutral":
                label = "neutral"
                confidence = 1.0 - confidence

            # Create message
            if label == "neutral":
                message = "Text appears to be neutral (no significant bias detected)"
            else:
                message = f"Text contains {label} bias"

            return ClassificationResult(
                label=label,
                confidence=confidence,
                message=message,
                metadata={"probabilities": probabilities, "bias_features": bias_features},
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
        # Process one by one for now (could be optimized in future)
        return [self.classify(text) for text in texts]
