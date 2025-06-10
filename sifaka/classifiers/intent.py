"""Intent classification for detecting the purpose/intent of text.

This module provides a classifier for detecting intent in text using pretrained models
from Hugging Face transformers. Supports both direct classification models and
zero-shot classification approaches.

Detects intents like question, request, statement, command, complaint, etc.

Requires transformers library to be installed.
"""

import asyncio
import importlib
from typing import Any, Dict, List

from sifaka.classifiers.base import (
    BaseClassifier,
    CachedClassifier,
    ClassificationResult,
    TimingMixin,
)
from sifaka.utils.errors import ValidationError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Available intent classification models
INTENT_MODELS: Dict[str, Dict[str, Any]] = {
    "facebook/bart-large-mnli": {
        "description": "BART model for zero-shot intent classification using NLI",
        "method": "zero_shot",
        "candidate_labels": [
            "question",
            "statement",
            "request",
            "greeting",
            "goodbye",
            "complaint",
            "compliment",
            "command",
        ],
        "size": "large",
    },
    "cardiffnlp/tweet-topic-21-multi": {
        "description": "Multi-label topic classification model adaptable for intent",
        "method": "classification",
        "labels": {
            "LABEL_0": "statement",
            "LABEL_1": "question",
            "LABEL_2": "request",
            "LABEL_3": "greeting",
            "LABEL_4": "complaint",
        },
        "size": "base",
    },
    "microsoft/DialoGPT-medium": {
        "description": "Legacy model - not recommended for intent classification",
        "method": "classification",
        "deprecated": True,
        "warning": "DialoGPT is a generative model, not suitable for classification",
        "labels": {"LABEL_0": "statement", "LABEL_1": "question", "LABEL_2": "request"},
        "size": "medium",
    },
}


class IntentClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting intent in text using pretrained models.

    This classifier supports both direct classification models and zero-shot
    classification approaches. It can use rule-based fallbacks for simple cases
    and ML models for complex intent detection.

    Attributes:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for intent detection
        pipeline: The Hugging Face transformers pipeline
        intents: List of intents the model can detect
        method: Classification method ('zero_shot' or 'classification')
        use_rules: Whether to use rule-based fallbacks
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        threshold: float = 0.5,
        candidate_labels: List[str] = None,
        use_rules: bool = True,
        name: str = "intent_detection",
        description: str = "Detects intent in text using pretrained models and rules",
    ):
        """Initialize the intent detection classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for intent detection
            candidate_labels: List of candidate intent labels (for zero-shot)
            use_rules: Whether to use rule-based fallbacks
            name: Name of the classifier
            description: Description of the classifier

        Raises:
            ImportError: If transformers library is not installed
            ValidationError: If model loading fails
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.threshold = threshold
        self.use_rules = use_rules
        self.pipeline = None
        self.model_info = INTENT_MODELS.get(model_name, {})
        self.method = self.model_info.get("method", "classification")

        # Set up candidate labels for zero-shot or intents for classification
        if candidate_labels:
            self.intents = candidate_labels
        else:
            self.intents = self.model_info.get(
                "candidate_labels",
                self.model_info.get(
                    "intents",
                    [
                        "question",
                        "statement",
                        "request",
                        "greeting",
                        "goodbye",
                        "complaint",
                        "compliment",
                        "command",
                    ],
                ),
            )

        # Warn about deprecated models
        if self.model_info.get("deprecated"):
            logger.warning(
                "Using deprecated model for intent classification",
                extra={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "warning": self.model_info.get("warning", "Model not recommended"),
                },
            )

        self._initialize_model()

    def get_classes(self) -> list[str]:
        """Get the list of possible class labels."""
        return self.intents

    def _initialize_model(self) -> None:
        """Initialize the pretrained intent detection model."""
        # Import transformers - fail fast if not available
        transformers = importlib.import_module("transformers")

        try:
            if self.method == "zero_shot":
                # Use zero-shot classification pipeline
                self.pipeline = transformers.pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                    device=-1,  # Use CPU by default
                )
                logger.debug(
                    "Initialized zero-shot intent classifier",
                    extra={
                        "classifier": self.name,
                        "model_name": self.model_name,
                        "method": "zero_shot_classification",
                        "candidate_labels": self.intents,
                        "description": self.model_info.get("description", "Unknown model"),
                    },
                )
            else:
                # Use standard text classification pipeline
                self.pipeline = transformers.pipeline(
                    "text-classification",
                    model=self.model_name,
                    return_all_scores=True,
                    device=-1,  # Use CPU by default
                    truncation=True,
                    max_length=512,
                )
                logger.debug(
                    "Initialized intent classifier with text classification pipeline",
                    extra={
                        "classifier": self.name,
                        "model_name": self.model_name,
                        "method": "text_classification",
                        "description": self.model_info.get("description", "Unknown model"),
                        "intents": self.intents,
                    },
                )
        except Exception as e:
            logger.error(
                "Failed to initialize intent classifier model",
                extra={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "method": self.method,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise ValidationError(
                f"Failed to initialize intent classification model: {str(e)}",
                error_code="model_initialization_error",
                context={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "method": self.method,
                },
                suggestions=[
                    "Check if transformers is properly installed",
                    "Verify the model name is correct and available",
                    "Try a different model or method",
                    "Check internet connection for model download",
                ],
            ) from e

    def _classify_with_rules(self, text: str) -> ClassificationResult:
        """Apply rule-based intent classification for obvious cases.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult if a rule matches, None otherwise
        """
        text_lower = text.lower().strip()

        # Question patterns
        if text.endswith("?") or text_lower.startswith(
            (
                "what",
                "how",
                "why",
                "when",
                "where",
                "who",
                "which",
                "is ",
                "are ",
                "do ",
                "does ",
                "did ",
                "can ",
                "could ",
                "would ",
                "will ",
            )
        ):
            return self.create_classification_result(
                label="question",
                confidence=0.9,
                metadata={
                    "method": "rule_based",
                    "rule": "question_pattern",
                    "input_length": len(text),
                },
            )

        # Request patterns
        if text_lower.startswith(
            (
                "please",
                "can you",
                "could you",
                "would you",
                "help me",
                "i need",
                "i want",
                "i would like",
            )
        ):
            return self.create_classification_result(
                label="request",
                confidence=0.85,
                metadata={
                    "method": "rule_based",
                    "rule": "request_pattern",
                    "input_length": len(text),
                },
            )

        # Greeting patterns
        if text_lower in ("hello", "hi", "hey", "good morning", "good afternoon", "good evening"):
            return self.create_classification_result(
                label="greeting",
                confidence=0.95,
                metadata={
                    "method": "rule_based",
                    "rule": "greeting_pattern",
                    "input_length": len(text),
                },
            )

        # Goodbye patterns
        if text_lower in ("goodbye", "bye", "see you", "farewell", "good night"):
            return self.create_classification_result(
                label="goodbye",
                confidence=0.95,
                metadata={
                    "method": "rule_based",
                    "rule": "goodbye_pattern",
                    "input_length": len(text),
                },
            )

        # Command patterns (imperative mood)
        if text_lower.startswith(
            (
                "go ",
                "stop",
                "start",
                "run ",
                "execute",
                "do ",
                "make ",
                "create ",
                "delete ",
                "remove ",
            )
        ) and not text.endswith("?"):
            return self.create_classification_result(
                label="command",
                confidence=0.8,
                metadata={
                    "method": "rule_based",
                    "rule": "command_pattern",
                    "input_length": len(text),
                },
            )

        # No rule matched
        return None

    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text for intent asynchronously.

        Uses a hybrid approach: rule-based classification for obvious cases,
        ML models for complex cases.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with intent prediction
        """
        if not text or not text.strip():
            return self.create_empty_text_result("statement")

        with self.time_operation("intent_classification") as timer:
            try:
                # Try rule-based classification first if enabled
                if self.use_rules:
                    rule_result = self._classify_with_rules(text)
                    if rule_result and rule_result.confidence >= 0.8:
                        # High confidence rule match - use it
                        processing_time = getattr(timer, "duration_ms", 0.0)
                        rule_result.processing_time_ms = processing_time

                        logger.debug(
                            "Intent classification completed with rules",
                            extra={
                                "classifier": self.name,
                                "text_length": len(text),
                                "label": rule_result.label,
                                "confidence": rule_result.confidence,
                                "method": "rule_based",
                            },
                        )
                        return rule_result

                # Fall back to ML classification
                result = await self._classify_with_pipeline(text)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)
                result.processing_time_ms = processing_time

                logger.debug(
                    "Intent classification completed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "label": result.label,
                        "confidence": result.confidence,
                        "method": result.metadata.get("method", "unknown"),
                    },
                )

                return result

            except Exception as e:
                logger.error(
                    "Intent classification failed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise ValidationError(
                    f"Failed to classify text for intent: {str(e)}",
                    error_code="classification_error",
                    context={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error_type": type(e).__name__,
                    },
                    suggestions=[
                        "Check if transformers is properly installed",
                        "Verify input text is valid",
                        "Try with shorter text",
                        "Try disabling rule-based classification",
                    ],
                ) from e

    async def _classify_with_pipeline(self, text: str) -> ClassificationResult:
        """Classify using transformers pipeline (zero-shot or direct classification)."""

        def analyze():
            if self.method == "zero_shot":
                # Zero-shot classification with candidate labels
                return self.pipeline(text, self.intents)
            else:
                # Direct text classification
                return self.pipeline(text)

        try:
            # Use asyncio to run in thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, analyze)

            if not results:
                raise ValueError("Pipeline returned empty results")

            if self.method == "zero_shot":
                # Handle zero-shot classification results
                return self._process_zero_shot_results(results, text)
            else:
                # Handle direct classification results
                return self._process_classification_results(results, text)

        except Exception as e:
            logger.error(
                f"Pipeline intent classification failed: {e}",
                extra={"classifier": self.name, "method": self.method},
                exc_info=True,
            )
            raise

    def _process_zero_shot_results(
        self, results: Dict[str, Any], text: str
    ) -> ClassificationResult:
        """Process zero-shot classification results."""
        # Zero-shot results format: {'sequence': text, 'labels': [...], 'scores': [...]}
        if not isinstance(results, dict) or "labels" not in results or "scores" not in results:
            raise ValueError(f"Unexpected zero-shot results format: {results}")

        labels = results["labels"]
        scores = results["scores"]

        if not labels or not scores or len(labels) != len(scores):
            raise ValueError("Invalid zero-shot results: labels and scores mismatch")

        # Get the best result
        best_label = labels[0].lower()
        best_score = float(scores[0])

        # Get all results above threshold
        detected_intents = [
            {"intent": label.lower(), "confidence": float(score)}
            for label, score in zip(labels, scores)
            if float(score) >= self.threshold
        ]

        return self.create_classification_result(
            label=best_label,
            confidence=best_score,
            metadata={
                "method": "zero_shot_classification",
                "model_name": self.model_name,
                "threshold": self.threshold,
                "input_length": len(text),
                "candidate_labels": self.intents,
                "all_results": list(zip(labels, scores)),
                "detected_intents": detected_intents,
                "primary_intent": best_label,
            },
        )

    def _process_classification_results(self, results: Any, text: str) -> ClassificationResult:
        """Process direct classification results."""
        # Handle different pipeline output formats
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], dict) and "score" in results[0] and "label" in results[0]:
                # Standard format: list of dicts with score and label
                best_result = max(results, key=lambda x: x["score"])
                intent = self._map_label(best_result["label"])
                confidence = float(best_result["score"])

                # Get all intents above threshold
                detected_intents = [
                    {
                        "intent": self._map_label(result["label"]),
                        "confidence": float(result["score"]),
                    }
                    for result in results
                    if float(result["score"]) >= self.threshold
                ]
            else:
                # Alternative format - try to extract from first result
                if hasattr(results[0], "label") and hasattr(results[0], "score"):
                    intent = self._map_label(results[0].label)
                    confidence = float(results[0].score)
                    detected_intents = (
                        [{"intent": intent, "confidence": confidence}]
                        if confidence >= self.threshold
                        else []
                    )
                else:
                    raise ValueError(f"Unexpected pipeline result format: {type(results[0])}")
        else:
            raise ValueError(f"Unexpected pipeline results type: {type(results)}")

        return self.create_classification_result(
            label=intent,
            confidence=confidence,
            metadata={
                "method": "text_classification",
                "model_name": self.model_name,
                "threshold": self.threshold,
                "input_length": len(text),
                "all_results": results,
                "detected_intents": detected_intents,
                "primary_intent": intent,
            },
        )

    def _map_label(self, label: str) -> str:
        """Map model labels to standard intent labels."""
        # Handle label mapping for models that use LABEL_0, LABEL_1, etc.
        label_mapping = self.model_info.get("labels", {})
        if label in label_mapping:
            return label_mapping[label].lower()
        return label.lower()


class CachedIntentClassifier(CachedClassifier, TimingMixin):
    """Cached version of IntentClassifier with LRU caching for improved performance."""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        threshold: float = 0.5,
        candidate_labels: List[str] = None,
        use_rules: bool = True,
        cache_size: int = 128,
        name: str = "cached_intent",
        description: str = "Cached intent classifier using pretrained models and rules",
    ):
        """Initialize the cached intent classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for intent detection
            candidate_labels: List of candidate intent labels (for zero-shot)
            use_rules: Whether to use rule-based fallbacks
            cache_size: Maximum number of results to cache
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description, cache_size=cache_size)
        self.model_name = model_name
        self.threshold = threshold
        self.use_rules = use_rules
        self.pipeline = None
        self.model_info = INTENT_MODELS.get(model_name, {})
        self.method = self.model_info.get("method", "classification")

        # Set up candidate labels for zero-shot or intents for classification
        if candidate_labels:
            self.intents = candidate_labels
        else:
            self.intents = self.model_info.get(
                "candidate_labels",
                self.model_info.get(
                    "intents",
                    [
                        "question",
                        "statement",
                        "request",
                        "greeting",
                        "goodbye",
                        "complaint",
                        "compliment",
                        "command",
                    ],
                ),
            )

        # Warn about deprecated models
        if self.model_info.get("deprecated"):
            logger.warning(
                "Using deprecated model for cached intent classification",
                extra={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "warning": self.model_info.get("warning", "Model not recommended"),
                },
            )

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained intent detection model."""
        # Import transformers - fail fast if not available
        transformers = importlib.import_module("transformers")

        try:
            if self.method == "zero_shot":
                # Use zero-shot classification pipeline
                self.pipeline = transformers.pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                    device=-1,  # Use CPU by default
                )
                logger.debug(
                    "Initialized cached zero-shot intent classifier",
                    extra={
                        "classifier": self.name,
                        "model_name": self.model_name,
                        "method": "zero_shot_classification",
                        "candidate_labels": self.intents,
                        "description": self.model_info.get("description", "Unknown model"),
                    },
                )
            else:
                # Use standard text classification pipeline
                self.pipeline = transformers.pipeline(
                    "text-classification",
                    model=self.model_name,
                    return_all_scores=True,
                    device=-1,  # Use CPU by default
                    truncation=True,
                    max_length=512,
                )
                logger.debug(
                    "Initialized cached intent classifier with text classification pipeline",
                    extra={
                        "classifier": self.name,
                        "model_name": self.model_name,
                        "method": "text_classification",
                        "description": self.model_info.get("description", "Unknown model"),
                        "intents": self.intents,
                    },
                )
        except Exception as e:
            logger.error(
                "Failed to initialize cached intent classifier model",
                extra={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "method": self.method,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise ValidationError(
                f"Failed to initialize cached intent classification model: {str(e)}",
                error_code="model_initialization_error",
                context={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "method": self.method,
                },
                suggestions=[
                    "Check if transformers is properly installed",
                    "Verify the model name is correct and available",
                    "Try a different model or method",
                    "Check internet connection for model download",
                ],
            ) from e

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform intent classification without caching."""
        try:
            # Try rule-based classification first if enabled
            if self.use_rules:
                rule_result = self._classify_with_rules(text)
                if rule_result and rule_result.confidence >= 0.8:
                    # High confidence rule match - use it
                    rule_result.metadata["cached"] = True
                    return rule_result

            # Fall back to ML classification
            return self._classify_with_pipeline_sync(text)

        except Exception as e:
            logger.error(
                "Cached intent classification failed",
                extra={
                    "classifier": self.name,
                    "text_length": len(text),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise ValidationError(
                f"Failed to classify text for intent: {str(e)}",
                error_code="classification_error",
                context={
                    "classifier": self.name,
                    "text_length": len(text),
                    "error_type": type(e).__name__,
                },
                suggestions=[
                    "Check if transformers is properly installed",
                    "Verify input text is valid",
                    "Try with shorter text",
                    "Check if the model is available on Hugging Face",
                ],
            ) from e

    def _classify_with_rules(self, text: str) -> ClassificationResult:
        """Apply rule-based intent classification for obvious cases.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult if a rule matches, None otherwise
        """
        text_lower = text.lower().strip()

        # Question patterns
        if text.endswith("?") or text_lower.startswith(
            (
                "what",
                "how",
                "why",
                "when",
                "where",
                "who",
                "which",
                "is ",
                "are ",
                "do ",
                "does ",
                "did ",
                "can ",
                "could ",
                "would ",
                "will ",
            )
        ):
            return self.create_classification_result(
                label="question",
                confidence=0.9,
                metadata={
                    "method": "rule_based",
                    "rule": "question_pattern",
                    "input_length": len(text),
                },
            )

        # Request patterns
        if text_lower.startswith(
            (
                "please",
                "can you",
                "could you",
                "would you",
                "help me",
                "i need",
                "i want",
                "i would like",
            )
        ):
            return self.create_classification_result(
                label="request",
                confidence=0.85,
                metadata={
                    "method": "rule_based",
                    "rule": "request_pattern",
                    "input_length": len(text),
                },
            )

        # Greeting patterns
        if text_lower in ("hello", "hi", "hey", "good morning", "good afternoon", "good evening"):
            return self.create_classification_result(
                label="greeting",
                confidence=0.95,
                metadata={
                    "method": "rule_based",
                    "rule": "greeting_pattern",
                    "input_length": len(text),
                },
            )

        # Goodbye patterns
        if text_lower in ("goodbye", "bye", "see you", "farewell", "good night"):
            return self.create_classification_result(
                label="goodbye",
                confidence=0.95,
                metadata={
                    "method": "rule_based",
                    "rule": "goodbye_pattern",
                    "input_length": len(text),
                },
            )

        # Command patterns (imperative mood)
        if text_lower.startswith(
            (
                "go ",
                "stop",
                "start",
                "run ",
                "execute",
                "do ",
                "make ",
                "create ",
                "delete ",
                "remove ",
            )
        ) and not text.endswith("?"):
            return self.create_classification_result(
                label="command",
                confidence=0.8,
                metadata={
                    "method": "rule_based",
                    "rule": "command_pattern",
                    "input_length": len(text),
                },
            )

        # No rule matched
        return None

    def _classify_with_pipeline_sync(self, text: str) -> ClassificationResult:
        """Synchronous version of pipeline classification for cached classifier."""
        if self.method == "zero_shot":
            # Zero-shot classification with candidate labels
            results = self.pipeline(text, self.intents)
            return self._process_zero_shot_results(results, text)
        else:
            # Direct text classification
            results = self.pipeline(text)
            return self._process_classification_results(results, text)

    def _process_zero_shot_results(
        self, results: Dict[str, Any], text: str
    ) -> ClassificationResult:
        """Process zero-shot classification results."""
        # Zero-shot results format: {'sequence': text, 'labels': [...], 'scores': [...]}
        if not isinstance(results, dict) or "labels" not in results or "scores" not in results:
            raise ValueError(f"Unexpected zero-shot results format: {results}")

        labels = results["labels"]
        scores = results["scores"]

        if not labels or not scores or len(labels) != len(scores):
            raise ValueError("Invalid zero-shot results: labels and scores mismatch")

        # Get the best result
        best_label = labels[0].lower()
        best_score = float(scores[0])

        # Get all results above threshold
        detected_intents = [
            {"intent": label.lower(), "confidence": float(score)}
            for label, score in zip(labels, scores)
            if float(score) >= self.threshold
        ]

        return self.create_classification_result(
            label=best_label,
            confidence=best_score,
            metadata={
                "method": "zero_shot_classification",
                "model_name": self.model_name,
                "threshold": self.threshold,
                "input_length": len(text),
                "candidate_labels": self.intents,
                "all_results": list(zip(labels, scores)),
                "detected_intents": detected_intents,
                "primary_intent": best_label,
                "cached": True,
            },
        )

    def _process_classification_results(self, results: Any, text: str) -> ClassificationResult:
        """Process direct classification results."""
        # Handle different pipeline output formats
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], dict) and "score" in results[0] and "label" in results[0]:
                # Standard format: list of dicts with score and label
                best_result = max(results, key=lambda x: x["score"])
                intent = self._map_label(best_result["label"])
                confidence = float(best_result["score"])

                # Get all intents above threshold
                detected_intents = [
                    {
                        "intent": self._map_label(result["label"]),
                        "confidence": float(result["score"]),
                    }
                    for result in results
                    if float(result["score"]) >= self.threshold
                ]
            else:
                # Alternative format - try to extract from first result
                if hasattr(results[0], "label") and hasattr(results[0], "score"):
                    intent = self._map_label(results[0].label)
                    confidence = float(results[0].score)
                    detected_intents = (
                        [{"intent": intent, "confidence": confidence}]
                        if confidence >= self.threshold
                        else []
                    )
                else:
                    raise ValueError(f"Unexpected pipeline result format: {type(results[0])}")
        else:
            raise ValueError(f"Unexpected pipeline results type: {type(results)}")

        return self.create_classification_result(
            label=intent,
            confidence=confidence,
            metadata={
                "method": "text_classification",
                "model_name": self.model_name,
                "threshold": self.threshold,
                "input_length": len(text),
                "all_results": results,
                "detected_intents": detected_intents,
                "primary_intent": intent,
                "cached": True,
            },
        )

    def _map_label(self, label: str) -> str:
        """Map model labels to standard intent labels."""
        # Handle label mapping for models that use LABEL_0, LABEL_1, etc.
        label_mapping = self.model_info.get("labels", {})
        if label in label_mapping:
            return label_mapping[label].lower()
        return label.lower()

    def get_classes(self) -> list[str]:
        """Get the list of possible class labels."""
        return self.intents


# Factory function for easy creation
def create_intent_classifier(
    model_name: str = "facebook/bart-large-mnli",
    threshold: float = 0.5,
    candidate_labels: List[str] = None,
    use_rules: bool = True,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create an intent classifier with the specified parameters.

    Args:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for intent detection
        candidate_labels: List of candidate intent labels (for zero-shot)
        use_rules: Whether to use rule-based fallbacks
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured intent classifier
    """
    if cached:
        return CachedIntentClassifier(
            model_name=model_name,
            threshold=threshold,
            candidate_labels=candidate_labels,
            use_rules=use_rules,
            cache_size=cache_size,
        )
    else:
        return IntentClassifier(
            model_name=model_name,
            threshold=threshold,
            candidate_labels=candidate_labels,
            use_rules=use_rules,
        )
