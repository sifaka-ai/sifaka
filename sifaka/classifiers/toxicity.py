"""
Toxicity classifier for Sifaka.

This module provides a classifier that categorizes text as toxic or non-toxic
using the Detoxify library, which is based on transformer models fine-tuned
on toxic content datasets.
"""

import importlib
from typing import List, Any

from sifaka.classifiers import ClassificationResult


class ToxicityClassifier:
    """
    A toxicity classifier that categorizes text as toxic or non-toxic.

    This classifier uses the Detoxify library to detect various forms of toxic
    content in text, including general toxicity, severe toxicity, obscenity,
    threats, insults, and identity-based attacks.

    Attributes:
        threshold: Confidence threshold for considering text toxic.
        model_name: The name of the Detoxify model to use.
        name: The name of the classifier.
        description: The description of the classifier.
    """

    # Toxicity categories and their descriptions
    TOXICITY_CATEGORIES = {
        "toxic": "toxic",
        "severe_toxic": "severely toxic",
        "obscene": "obscene",
        "threat": "threatening",
        "insult": "insulting",
        "identity_attack": "identity-attacking",
    }

    # Priority order for labels (most severe first)
    LABEL_PRIORITY = ["severe_toxic", "threat", "identity_attack", "toxic", "insult", "obscene"]

    def __init__(
        self,
        threshold: float = 0.5,
        model_name: str = "original",
        name: str = "toxicity_classifier",
        description: str = "Classifies text as toxic or non-toxic",
    ):
        """
        Initialize the toxicity classifier.

        Args:
            threshold: Confidence threshold for considering text toxic.
            model_name: The name of the Detoxify model to use (original, unbiased, or multilingual).
            name: The name of the classifier.
            description: The description of the classifier.
        """
        self._name = name
        self._description = description
        self._threshold = threshold
        self._model_name = model_name
        self._model = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Get the classifier name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the classifier description."""
        return self._description

    def _load_detoxify(self) -> Any:
        """
        Load the Detoxify library and create a model.

        Returns:
            A Detoxify model instance.

        Raises:
            ImportError: If Detoxify is not installed.
            RuntimeError: If model initialization fails.
        """
        try:
            detoxify = importlib.import_module("detoxify")
            model = detoxify.Detoxify(model_type=self._model_name)
            return model
        except ImportError:
            raise ImportError(
                "detoxify package is required for ToxicityClassifier. "
                "Install it with: pip install detoxify"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Detoxify model: {e}")

    def _initialize(self) -> None:
        """Initialize the toxicity model if needed."""
        if not self._initialized:
            self._model = self._load_detoxify()
            self._initialized = True

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text as toxic or non-toxic.

        Args:
            text: The text to classify.

        Returns:
            A ClassificationResult with the toxicity label and confidence score.
        """
        # Handle empty text
        if not text or not text.strip():
            return ClassificationResult(
                label="non_toxic",
                confidence=1.0,
                metadata={"input_length": 0, "reason": "empty_text", "scores": {}},
            )

        try:
            # Initialize model if needed
            self._initialize()

            # Get toxicity scores from the model
            if self._model is None:
                self._initialize()

            # Check again after initialization
            if self._model is None:
                raise RuntimeError("Failed to initialize toxicity model")

            results = self._model.predict(text)

            # Convert results to dictionary if needed
            if hasattr(results, "items"):
                scores = dict(results)
            else:
                scores = {cat: float(score) for cat, score in results.items()}

            # Find the most severe category with a score above threshold
            selected_category = None
            max_score = 0.0

            for category in self.LABEL_PRIORITY:
                if category in scores:
                    score = scores[category]
                    if score > self._threshold and score > max_score:
                        max_score = score
                        selected_category = category

            # Determine final label and confidence
            if selected_category:
                label = selected_category
                confidence = scores[selected_category]
                message = f"Text classified as {self.TOXICITY_CATEGORIES.get(label, label)}"
            else:
                label = "non_toxic"
                # Calculate non-toxic confidence as 1 - max toxicity score
                max_toxicity = max(scores.values()) if scores else 0.0
                confidence = 1.0 - max_toxicity
                message = "Text classified as non-toxic"

            return ClassificationResult(
                label=label,
                confidence=confidence,
                metadata={
                    "input_length": len(text),
                    "scores": scores,
                    "message": message,
                },
            )

        except Exception as e:
            # Handle errors
            return ClassificationResult(
                label="non_toxic",
                confidence=0.5,
                metadata={
                    "error": str(e),
                    "reason": "classification_error",
                    "input_length": len(text),
                    "scores": {},
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
                        label="non_toxic",
                        confidence=1.0,
                        metadata={"input_length": 0, "reason": "empty_text", "scores": {}},
                    )
                )
            else:
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        # If there are no non-empty texts, return the results
        if not non_empty_texts:
            return results

        try:
            # Get toxicity scores from the model for all non-empty texts
            if self._model is None:
                self._initialize()

            # Check again after initialization
            if self._model is None:
                raise RuntimeError("Failed to initialize toxicity model")

            batch_results = self._model.predict(non_empty_texts)

            # Process each result
            for i, idx in enumerate(non_empty_indices):
                text = non_empty_texts[i]

                # Extract scores for this text
                scores = {}
                for category in self.TOXICITY_CATEGORIES:
                    if category in batch_results:
                        if isinstance(batch_results[category], list):
                            scores[category] = float(batch_results[category][i])
                        else:
                            scores[category] = float(batch_results[category])

                # Find the most severe category with a score above threshold
                selected_category = None
                max_score = 0.0

                for category in self.LABEL_PRIORITY:
                    if category in scores:
                        score = scores[category]
                        if score > self._threshold and score > max_score:
                            max_score = score
                            selected_category = category

                # Determine final label and confidence
                if selected_category:
                    label = selected_category
                    confidence = scores[selected_category]
                    message = f"Text classified as {self.TOXICITY_CATEGORIES.get(label, label)}"
                else:
                    label = "non_toxic"
                    # Calculate non-toxic confidence as 1 - max toxicity score
                    max_toxicity = max(scores.values()) if scores else 0.0
                    confidence = 1.0 - max_toxicity
                    message = "Text classified as non-toxic"

                # Create result
                result = ClassificationResult(
                    label=label,
                    confidence=confidence,
                    metadata={
                        "input_length": len(text),
                        "scores": scores,
                        "message": message,
                    },
                )

                # Insert result at the correct position
                results.insert(idx, result)

            return results

        except Exception as e:
            # Handle errors by classifying each text individually
            for i, idx in enumerate(non_empty_indices):
                text = non_empty_texts[i]
                result = ClassificationResult(
                    label="non_toxic",
                    confidence=0.5,
                    metadata={
                        "error": str(e),
                        "reason": "batch_classification_error",
                        "input_length": len(text),
                        "scores": {},
                    },
                )
                results.insert(idx, result)

            return results
