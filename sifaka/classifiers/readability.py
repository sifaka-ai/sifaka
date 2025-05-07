"""
Readability classifiers for Sifaka.

This module provides classifiers for assessing the readability of text content.
"""

import importlib
import statistics
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    ClassVar,
    Union,
    TypeVar,
)

from typing_extensions import TypeGuard
from pydantic import PrivateAttr

from sifaka.classifiers.base import (
    BaseClassifier,
    Classifier,
    ClassificationResult,
    ClassifierConfig,
    ClassifierImplementation,
)
from sifaka.classifiers.config import standardize_classifier_config
from sifaka.utils.logging import get_logger
from sifaka.utils.state import ClassifierState, create_classifier_state

logger = get_logger(__name__)

# Type variables for generic typing
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type


@runtime_checkable
class ReadabilityAnalyzer(Protocol):
    """Protocol for readability analysis engines."""

    @abstractmethod
    def sentence_count(self, text: str) -> int: ...
    @abstractmethod
    def syllable_count(self, text: str) -> int: ...
    @abstractmethod
    def lexicon_count(self, text: str) -> int: ...
    @abstractmethod
    def difficult_words(self, text: str) -> int: ...
    @abstractmethod
    def flesch_reading_ease(self, text: str) -> float: ...
    @abstractmethod
    def flesch_kincaid_grade(self, text: str) -> float: ...
    @abstractmethod
    def gunning_fog(self, text: str) -> float: ...
    @abstractmethod
    def smog_index(self, text: str) -> float: ...
    @abstractmethod
    def automated_readability_index(self, text: str) -> float: ...
    @abstractmethod
    def dale_chall_readability_score(self, text: str) -> float: ...


class ReadabilityMetrics:
    """Container for readability metrics."""

    def __init__(
        self,
        flesch_reading_ease: float,
        flesch_kincaid_grade: float,
        gunning_fog: float,
        smog_index: float,
        automated_readability_index: float,
        dale_chall_readability_score: float,
        lexicon_count: int,
        sentence_count: int,
        syllable_count: int,
        avg_sentence_length: float,
        sentence_length_std: float,
        avg_word_length: float,
        word_length_std: float,
        vocabulary_diversity: float,
        unique_word_count: int,
        difficult_words: int,
    ):
        self.flesch_reading_ease = flesch_reading_ease
        self.flesch_kincaid_grade = flesch_kincaid_grade
        self.gunning_fog = gunning_fog
        self.smog_index = smog_index
        self.automated_readability_index = automated_readability_index
        self.dale_chall_readability_score = dale_chall_readability_score
        self.lexicon_count = lexicon_count
        self.sentence_count = sentence_count
        self.syllable_count = syllable_count
        self.avg_sentence_length = avg_sentence_length
        self.sentence_length_std = sentence_length_std
        self.avg_word_length = avg_word_length
        self.word_length_std = word_length_std
        self.vocabulary_diversity = vocabulary_diversity
        self.unique_word_count = unique_word_count
        self.difficult_words = difficult_words


class ReadabilityClassifierImplementation:
    """
    Implementation of readability classification logic using textstat.

    This implementation uses the textstat library to analyze the readability of text.
    It provides a fast, local alternative to API-based readability analysis and
    can identify various reading levels from elementary to graduate level.

    ## Architecture

    ReadabilityClassifierImplementation follows the composition pattern:
    1. **Core Logic**: classify_impl() implements readability analysis
    2. **State Management**: Uses ClassifierState for internal state
    3. **Resource Management**: Loads and manages textstat analyzer

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Create with ClassifierConfig
       - Initialize state

    2. **Warm-up**: Prepare resources
       - Load textstat library if needed
       - Initialize analyzer

    3. **Classification**: Process inputs
       - Calculate readability metrics
       - Determine reading level
       - Calculate confidence based on metric agreement
    """

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = ["elementary", "middle", "high", "college", "graduate"]
    DEFAULT_COST: ClassVar[int] = 1  # Low cost for statistical analysis

    # Default grade level bounds
    DEFAULT_GRADE_LEVEL_BOUNDS: ClassVar[Dict[str, tuple[float, float]]] = {
        "elementary": (0.0, 6.0),
        "middle": (6.0, 9.0),
        "high": (9.0, 12.0),
        "college": (12.0, 16.0),
        "graduate": (16.0, float("inf")),
    }

    def __init__(self, config: ClassifierConfig):
        """
        Initialize the readability classifier implementation.

        Args:
            config: Configuration for the classifier
        """
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False
        self._state.cache = {}

    def _validate_analyzer(self, analyzer: Any) -> TypeGuard[ReadabilityAnalyzer]:
        """Validate that an analyzer implements the required protocol."""
        if not isinstance(analyzer, ReadabilityAnalyzer):
            raise ValueError(
                f"Analyzer must implement ReadabilityAnalyzer protocol, got {type(analyzer)}"
            )
        return True

    def _load_textstat(self) -> ReadabilityAnalyzer:
        """Load the textstat library."""
        try:
            textstat = importlib.import_module("textstat")
            self._validate_analyzer(textstat)
            return textstat
        except ImportError:
            raise ImportError(
                "textstat package is required for ReadabilityClassifier. "
                "Install it with: pip install sifaka[readability]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load textstat: {e}")

    def warm_up_impl(self) -> None:
        """Initialize the analyzer if needed."""
        # Check if already initialized
        if not self._state.initialized:
            # Get analyzer from state cache or load textstat
            analyzer = self._state.cache.get("analyzer") or self._load_textstat()

            # Store analyzer in state
            self._state.model = analyzer

            # Mark as initialized
            self._state.initialized = True

    def _get_grade_level(self, grade: float) -> str:
        """Convert grade level to readability label."""
        # Get grade level bounds from config params
        grade_level_bounds = self.config.params.get(
            "grade_level_bounds", self.DEFAULT_GRADE_LEVEL_BOUNDS
        )

        for level, (lower, upper) in grade_level_bounds.items():
            if lower <= grade < upper:
                return level
        return "graduate"  # Default to highest level if beyond all bounds

    def _get_flesch_interpretation(self, score: float) -> str:
        """Interpret Flesch reading ease score."""
        interpretations = [
            (90, "Very Easy - 5th grade"),
            (80, "Easy - 6th grade"),
            (70, "Fairly Easy - 7th grade"),
            (60, "Standard - 8th/9th grade"),
            (50, "Fairly Difficult - 10th/12th grade"),
            (30, "Difficult - College"),
            (0, "Very Difficult - College Graduate"),
        ]

        for threshold, interpretation in interpretations:
            if score >= threshold:
                return interpretation
        return interpretations[-1][1]

    def _calculate_metrics(self, text: str) -> ReadabilityMetrics:
        """Calculate comprehensive readability metrics."""
        # Get analyzer from state
        analyzer = self._state.model

        words = text.split()
        unique_words = set(word.lower() for word in words)

        # Calculate word length statistics
        word_lengths = [len(word) for word in words]
        avg_word_length = statistics.mean(word_lengths) if word_lengths else 0
        word_length_std = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0

        # Calculate sentence length statistics
        sentence_count = analyzer.sentence_count(text)
        words_per_sentence = [len(sent.split()) for sent in text.split(".") if sent.strip()]
        avg_sentence_length = statistics.mean(words_per_sentence) if words_per_sentence else 0
        sentence_length_std = (
            statistics.stdev(words_per_sentence) if len(words_per_sentence) > 1 else 0
        )

        return ReadabilityMetrics(
            flesch_reading_ease=analyzer.flesch_reading_ease(text),
            flesch_kincaid_grade=analyzer.flesch_kincaid_grade(text),
            gunning_fog=analyzer.gunning_fog(text),
            smog_index=analyzer.smog_index(text),
            automated_readability_index=analyzer.automated_readability_index(text),
            dale_chall_readability_score=analyzer.dale_chall_readability_score(text),
            lexicon_count=analyzer.lexicon_count(text),
            sentence_count=sentence_count,
            syllable_count=analyzer.syllable_count(text),
            avg_sentence_length=avg_sentence_length,
            sentence_length_std=sentence_length_std,
            avg_word_length=avg_word_length,
            word_length_std=word_length_std,
            vocabulary_diversity=len(unique_words) / len(words) if words else 0,
            unique_word_count=len(unique_words),
            difficult_words=analyzer.difficult_words(text),
        )

    def _calculate_confidence(self, metrics: ReadabilityMetrics) -> float:
        """Calculate confidence based on agreement between metrics."""
        if metrics.lexicon_count == 0:
            return 0.0

        # Get relevant grade-level metrics
        grade_metrics = [
            metrics.flesch_kincaid_grade,
            metrics.gunning_fog,
            metrics.smog_index,
            metrics.dale_chall_readability_score,
            metrics.automated_readability_index,
        ]

        # Calculate coefficient of variation
        mean = statistics.mean(grade_metrics)
        if mean == 0:
            return 0.0

        cv = statistics.stdev(grade_metrics) / mean
        confidence = max(0.0, 1.0 - cv)

        return min(1.0, confidence)

    def classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement readability classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with readability level and confidence
        """
        # Ensure initialized
        self.warm_up_impl()

        try:
            # Calculate all metrics
            metrics = self._calculate_metrics(text)

            # Calculate average grade level
            grade_metrics = [
                metrics.flesch_kincaid_grade,
                metrics.gunning_fog,
                metrics.smog_index,
                metrics.dale_chall_readability_score,
                metrics.automated_readability_index,
            ]
            avg_grade = statistics.mean(grade_metrics)

            # Get readability level and confidence
            level = self._get_grade_level(avg_grade)
            confidence = self._calculate_confidence(metrics)

            # Create detailed metadata
            metadata = {
                "grade_level": avg_grade,
                "flesch_interpretation": self._get_flesch_interpretation(
                    metrics.flesch_reading_ease
                ),
                "metrics": {
                    "flesch_reading_ease": metrics.flesch_reading_ease,
                    "flesch_kincaid_grade": metrics.flesch_kincaid_grade,
                    "gunning_fog": metrics.gunning_fog,
                    "smog_index": metrics.smog_index,
                    "automated_readability_index": metrics.automated_readability_index,
                    "dale_chall_score": metrics.dale_chall_readability_score,
                },
                "text_stats": {
                    "lexicon_count": metrics.lexicon_count,
                    "sentence_count": metrics.sentence_count,
                    "syllable_count": metrics.syllable_count,
                    "avg_sentence_length": metrics.avg_sentence_length,
                    "sentence_length_std": metrics.sentence_length_std,
                    "avg_word_length": metrics.avg_word_length,
                    "word_length_std": metrics.word_length_std,
                    "vocabulary_diversity": metrics.vocabulary_diversity,
                    "unique_word_count": metrics.unique_word_count,
                    "difficult_words": metrics.difficult_words,
                },
            }

            return ClassificationResult(label=level, confidence=confidence, metadata=metadata)

        except Exception as e:
            logger.error("Failed to classify text readability: %s", e)
            self._state.error = f"Failed to classify text readability: {e}"
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={"error": str(e), "reason": "classification_error"},
            )


def create_readability_classifier(
    name: str = "readability_classifier",
    description: str = "Analyzes text readability",
    analyzer: Optional[ReadabilityAnalyzer] = None,
    grade_level_bounds: Optional[Dict[str, tuple[float, float]]] = None,
    min_confidence: float = 0.7,
    cache_size: int = 100,
    cost: float = ReadabilityClassifierImplementation.DEFAULT_COST,
    config: Optional[Union[Dict[str, Any], ClassifierConfig]] = None,
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Create a readability classifier.

    This factory function creates a Classifier with a ReadabilityClassifierImplementation
    to analyze the readability of text. It follows the composition over inheritance pattern,
    creating a more flexible and maintainable design.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        analyzer: Custom readability analyzer implementation
        grade_level_bounds: Dictionary mapping readability levels to grade ranges
        min_confidence: Minimum confidence threshold
        cache_size: Size of the cache for memoization
        cost: Cost of running the classifier
        config: Optional classifier configuration
        **kwargs: Additional configuration parameters

    Returns:
        A Classifier instance with ReadabilityClassifierImplementation

    Examples:
        ```python
        from sifaka.classifiers.readability import create_readability_classifier

        # Create a readability classifier with default settings
        classifier = create_readability_classifier()

        # Create a readability classifier with custom settings
        classifier = create_readability_classifier(
            name="custom_readability_classifier",
            description="Custom readability analyzer with specific grade levels",
            grade_level_bounds={
                "elementary": (0.0, 5.0),
                "middle": (5.0, 8.0),
                "high": (8.0, 12.0),
                "college": (12.0, 16.0),
                "graduate": (16.0, float("inf")),
            },
            min_confidence=0.8,
            cache_size=200
        )

        # Classify text
        result = classifier.classify("This is a sample text to analyze for readability.")
        print(f"Reading level: {result.label}, Grade level: {result.metadata['grade_level']:.1f}")
        ```
    """
    # Use standardize_classifier_config to handle different config formats
    params = kwargs.pop("params", {})

    # Ensure grade_level_bounds is present
    if grade_level_bounds is not None:
        params["grade_level_bounds"] = grade_level_bounds
    elif "grade_level_bounds" not in params:
        params["grade_level_bounds"] = (
            ReadabilityClassifierImplementation.DEFAULT_GRADE_LEVEL_BOUNDS
        )

    # Create config
    classifier_config = standardize_classifier_config(
        config=config,
        labels=ReadabilityClassifierImplementation.DEFAULT_LABELS,
        min_confidence=min_confidence,
        cost=cost,
        cache_size=cache_size,
        params=params,
        **kwargs,
    )

    # Create implementation
    implementation = ReadabilityClassifierImplementation(classifier_config)

    # Store analyzer in implementation's state if provided
    if analyzer is not None:
        implementation._state.cache["analyzer"] = analyzer

    # Create and return classifier with implementation
    return Classifier(
        name=name,
        description=description,
        config=classifier_config,
        implementation=implementation,
    )
