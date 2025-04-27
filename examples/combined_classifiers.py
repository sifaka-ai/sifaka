"""
Example of using multiple classifiers together with Sifaka.

This example demonstrates how to:
1. Combine multiple classifiers
2. Use them with rules
3. Make decisions based on multiple classification results
4. Handle errors and edge cases
"""

import logging
from typing import List, Dict, Any, Protocol, TypeVar, runtime_checkable, Final
from typing_extensions import TypeGuard
from dataclasses import dataclass, field
from dotenv import load_dotenv

from sifaka import Reflector
from sifaka.models import AnthropicProvider
from sifaka.rules import ClassifierRule
from sifaka.classifiers.base import BaseClassifier, ClassificationResult, ClassifierConfig
from sifaka.critique import PromptCritique

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseClassifier")


@runtime_checkable
class TextProcessor(Protocol):
    """Protocol for text processing components."""

    def process(self, text: str) -> Dict[str, Any]: ...


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Protocol for classifier components."""

    def classify(self, text: str) -> ClassificationResult: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...


@dataclass(frozen=True)
class WordList:
    """Immutable container for word lists used in classification."""

    positive: frozenset[str]
    negative: frozenset[str]


@dataclass(frozen=True)
class ComplexityThresholds:
    """Immutable thresholds for text complexity analysis."""

    complex: float = 7.0
    moderate: float = 5.0
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.moderate >= self.complex:
            raise ValueError("moderate threshold must be less than complex threshold")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")


@runtime_checkable
class ComplexityAnalyzer(Protocol):
    """Protocol for text complexity analysis."""

    def calculate_avg_word_length(self, text: str) -> float: ...
    def calculate_sentence_length(self, text: str) -> float: ...
    def calculate_unique_words_ratio(self, text: str) -> float: ...


class DefaultComplexityAnalyzer:
    """Default implementation of ComplexityAnalyzer."""

    def calculate_avg_word_length(self, text: str) -> float:
        """Calculate average word length."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)

    def calculate_sentence_length(self, text: str) -> float:
        """Calculate average sentence length."""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)

    def calculate_unique_words_ratio(self, text: str) -> float:
        """Calculate ratio of unique words."""
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)


class ComplexityClassifier(BaseClassifier):
    """A classifier that analyzes text complexity."""

    # Class-level constants
    DEFAULT_LABELS: Final[List[str]] = ["simple", "moderate", "complex"]
    DEFAULT_COST: Final[int] = 1  # Low cost for statistical analysis

    def __init__(
        self,
        name: str = "complexity_classifier",
        description: str = "Text complexity analysis",
        thresholds: ComplexityThresholds | None = None,
        analyzer: ComplexityAnalyzer | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the complexity classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            thresholds: Complexity thresholds configuration
            analyzer: Custom complexity analyzer implementation
            **kwargs: Additional configuration parameters
        """
        config = ClassifierConfig(labels=self.DEFAULT_LABELS, cost=self.DEFAULT_COST, **kwargs)
        super().__init__(name=name, description=description, config=config)

        self._thresholds = thresholds or ComplexityThresholds()
        self._analyzer = analyzer or DefaultComplexityAnalyzer()

    def _validate_analyzer(self, analyzer: Any) -> TypeGuard[ComplexityAnalyzer]:
        """Validate that an analyzer implements the required protocol."""
        if not isinstance(analyzer, ComplexityAnalyzer):
            raise ValueError(
                f"Analyzer must implement ComplexityAnalyzer protocol, got {type(analyzer)}"
            )
        return True

    def _calculate_complexity_metrics(self, text: str) -> Dict[str, float]:
        """Calculate comprehensive complexity metrics."""
        return {
            "avg_word_length": self._analyzer.calculate_avg_word_length(text),
            "avg_sentence_length": self._analyzer.calculate_sentence_length(text),
            "unique_words_ratio": self._analyzer.calculate_unique_words_ratio(text),
        }

    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence based on metrics consistency."""
        # Higher confidence if metrics agree on complexity level
        word_complex = metrics["avg_word_length"] > self._thresholds.complex
        sent_complex = metrics["avg_sentence_length"] > 20  # Typical threshold
        unique_complex = metrics["unique_words_ratio"] > 0.8  # High vocabulary diversity

        agreement = sum([word_complex, sent_complex, unique_complex])
        base_confidence = agreement / 3.0

        return max(base_confidence, self._thresholds.min_confidence)

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement complexity classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with complexity level and confidence
        """
        try:
            # Calculate complexity metrics
            metrics = self._calculate_complexity_metrics(text)
            avg_word_length = metrics["avg_word_length"]

            # Determine complexity level
            if avg_word_length > self._thresholds.complex:
                label = "complex"
            elif avg_word_length > self._thresholds.moderate:
                label = "moderate"
            else:
                label = "simple"

            # Calculate confidence
            confidence = self._calculate_confidence(metrics)

            return ClassificationResult(
                label=label,
                confidence=confidence,
                metadata={
                    "metrics": metrics,
                    "thresholds": {
                        "complex": self._thresholds.complex,
                        "moderate": self._thresholds.moderate,
                    },
                },
            )

        except Exception as e:
            logger.error("Failed to classify text complexity: %s", e)
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={"error": str(e), "reason": "classification_error"},
            )

    @classmethod
    def create_with_custom_analyzer(
        cls,
        analyzer: ComplexityAnalyzer,
        name: str = "custom_complexity_classifier",
        description: str = "Custom complexity analyzer",
        thresholds: ComplexityThresholds | None = None,
        **kwargs,
    ) -> "ComplexityClassifier":
        """
        Factory method to create a classifier with a custom analyzer.

        Args:
            analyzer: Custom complexity analyzer implementation
            name: Name of the classifier
            description: Description of the classifier
            thresholds: Custom complexity thresholds
            **kwargs: Additional configuration parameters

        Returns:
            Configured ComplexityClassifier instance
        """
        instance = cls(
            name=name,
            description=description,
            thresholds=thresholds,
            analyzer=analyzer,
            **kwargs,
        )
        instance._validate_analyzer(analyzer)
        return instance


class SentimentClassifier(BaseClassifier, Classifier):
    """A simple sentiment classifier."""

    def __init__(
        self,
        name: str = "sentiment",
        description: str = "Sentiment analysis",
        word_list: WordList | None = None,
    ):
        super().__init__(name=name, description=description)
        default_words = WordList(
            positive=frozenset(["good", "great", "excellent", "amazing", "wonderful"]),
            negative=frozenset(["bad", "terrible", "awful", "horrible", "poor"]),
        )
        self._words = word_list or default_words

    def classify(self, text: str) -> ClassificationResult:
        """Classify text sentiment."""
        self.validate_input(text)
        text = text.lower()

        pos_count = sum(1 for word in self._words.positive if word in text)
        neg_count = sum(1 for word in self._words.negative if word in text)

        total = pos_count + neg_count
        if total == 0:
            return ClassificationResult(
                label="neutral", confidence=1.0, metadata={"reason": "no sentiment words found"}
            )

        if pos_count > neg_count:
            confidence = pos_count / total
            return ClassificationResult(
                label="positive",
                confidence=confidence,
                metadata={"positive_words": pos_count, "negative_words": neg_count},
            )
        elif neg_count > pos_count:
            confidence = neg_count / total
            return ClassificationResult(
                label="negative",
                confidence=confidence,
                metadata={"positive_words": pos_count, "negative_words": neg_count},
            )
        else:
            return ClassificationResult(
                label="neutral",
                confidence=0.5,
                metadata={"positive_words": pos_count, "negative_words": neg_count},
            )


def create_reflector(
    model: AnthropicProvider,
    sentiment_classifier: ClassifierProtocol | None = None,
    complexity_classifier: ClassifierProtocol | None = None,
    threshold: float = 0.6,
) -> Reflector:
    """Factory function to create a configured Reflector instance."""

    # Create default classifiers if not provided
    sentiment = sentiment_classifier or SentimentClassifier()
    complexity = complexity_classifier or ComplexityClassifier()

    # Create classifier rules
    sentiment_rule = ClassifierRule(
        name="sentiment_check",
        description="Checks for appropriate sentiment",
        classifier=sentiment,
        threshold=threshold,
        valid_labels=["positive", "neutral"],
    )

    complexity_rule = ClassifierRule(
        name="complexity_check",
        description="Checks text complexity",
        classifier=complexity,
        threshold=threshold,
        valid_labels=["simple", "moderate"],
    )

    # Create a critic for improving outputs that fail validation
    critic = PromptCritique(model=model)

    # Create and return a reflector with both rules
    return Reflector(
        name="content_validator",
        model=model,
        rules=[sentiment_rule, complexity_rule],
        critique=True,
        critic=critic,
    )


def main():
    """Example usage of multiple classifiers with Sifaka."""
    # Load environment variables
    load_dotenv()

    # Initialize the model provider
    model = AnthropicProvider(model_name="claude-3-haiku-20240307")

    # Create reflector with default classifiers
    reflector = create_reflector(model)

    # Example prompts
    prompts = [
        "Write a simple and positive message about learning to code.",
        "Explain quantum computing in technical terms.",
        "Write a negative review of a bad experience.",
    ]

    # Process each prompt
    for prompt in prompts:
        logger.info("\nProcessing prompt: %s", prompt)
        result = reflector.reflect(prompt)

        logger.info("Original output:")
        logger.info(result.original_output)

        if result.rule_violations:
            logger.info("\nRule violations:")
            for violation in result.rule_violations:
                logger.info("- %s: %s", violation.rule_name, violation.message)

        logger.info("\nFinal output:")
        logger.info(result.final_output)

        if result.trace:
            logger.info("\nTrace data:")
            for event in result.trace:
                logger.info("- %s: %s", event.stage, event.message)


if __name__ == "__main__":
    main()
