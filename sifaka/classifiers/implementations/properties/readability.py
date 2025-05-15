"""
Readability classifiers for Sifaka.

This module provides classifiers for assessing the readability of text content.
"""

import importlib
import statistics
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Tuple, ClassVar, Type
from typing_extensions import TypeGuard
from pydantic import ConfigDict
from sifaka.classifiers.classifier import Classifier
from sifaka.core.results import ClassificationResult
from sifaka.utils.config.classifiers import ClassifierConfig, standardize_classifier_config
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_classifier_state

logger = get_logger(__name__)


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
    ) -> None:
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


class ReadabilityClassifier(Classifier):
    """
    A classifier that analyzes text readability using textstat.

    This classifier uses multiple readability metrics to determine the
    appropriate reading level of text.

    Requires the 'readability' extra to be installed:
    pip install sifaka[readability]
    """

    DEFAULT_LABELS: ClassVar[List[str]] = [
        "very_easy",
        "easy",
        "medium",
        "difficult",
        "very_difficult",
    ]
    DEFAULT_COST: ClassVar[int] = 1
    DEFAULT_GRADE_LEVEL_BOUNDS: ClassVar[Dict[str, Tuple[float, float]]] = {
        "elementary": (0.0, 6.0),
        "middle": (6.0, 9.0),
        "high": (9.0, 12.0),
        "college": (12.0, 16.0),
        "graduate": (16.0, float("inf")),
    }
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        name: str = "readability_classifier",
        description: str = "Analyzes text readability",
        analyzer: Optional[ReadabilityAnalyzer] = None,
        config: Optional[ClassifierConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the readability classifier.
        """
        if config is None:
            params = kwargs.pop("params", {})
            if "grade_level_bounds" not in params:
                params["grade_level_bounds"] = self.DEFAULT_GRADE_LEVEL_BOUNDS
            config = ClassifierConfig(params=params, **kwargs)
        # Pass self as implementation to base Classifier
        super().__init__(implementation=self, name=name, description=description, config=config)
        self._state_manager.update("initialized", False)
        cache = {}
        cache["analyzer"] = analyzer
        self._state_manager.update("cache", cache)

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
                "textstat package is required for ReadabilityClassifier. Install it with: pip install sifaka[readability]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load textstat: {e}")

    def warm_up(self) -> None:
        """Initialize the analyzer if needed."""
        if not self._state_manager.get("initialized", False):
            cache = self._state_manager.get("cache", {})
            analyzer = cache.get("analyzer") or self._load_textstat()
            self._state_manager.update("model", analyzer)
            self._state_manager.update("initialized", True)

    def _get_grade_level(self, grade: float) -> str:
        """Convert grade level to readability label."""
        grade_level_bounds = self.config.params.get(
            "grade_level_bounds", self.DEFAULT_GRADE_LEVEL_BOUNDS
        )
        for level, (lower, upper) in grade_level_bounds.items():
            if lower <= grade < upper:
                return level
        return "graduate"

    def _get_flesch_interpretation(self, score: float) -> str:  # type: ignore[no-any-return]
        """Interpret Flesch reading ease score."""
        interpretations: List[Tuple[float, str]] = [
            (90, "Very Easy - 5th grade"),
            (80, "Easy - 6th grade"),
            (70, "Fairly Easy - 7th grade"),
            (60, "Standard - 8th/9th grade"),
            (50, "Fairly Difficult - 10th/12th grade"),
            (30, "Difficult - College"),
            (0, "Very Difficult - College Graduate"),
        ]
        for threshold, interpretation in interpretations:
            interpretation: str  # type: ignore[no-redef]
            if isinstance(score, (int, float)) and score >= threshold:
                return interpretation
        return "Very Difficult - College Graduate"

    def _calculate_metrics(self, text: str) -> ReadabilityMetrics:
        """Calculate comprehensive readability metrics."""
        analyzer = self._state_manager.get("model")
        words = text.split()
        unique_words = set(word.lower() for word in words)
        word_lengths = [len(word) for word in words]
        avg_word_length = statistics.mean(word_lengths) if word_lengths else 0
        word_length_std = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0
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
        grade_metrics = [
            metrics.flesch_kincaid_grade,
            metrics.gunning_fog,
            metrics.smog_index,
            metrics.dale_chall_readability_score,
            metrics.automated_readability_index,
        ]
        mean = statistics.mean(grade_metrics)
        if mean == 0:
            return 0.0
        std = statistics.stdev(grade_metrics) if len(grade_metrics) > 1 else 0
        return max(0.0, min(1.0, 1.0 - (std / mean)))

    def _classify_impl(self, text: str) -> ClassificationResult[Any, str]:
        """Classify text based on readability metrics."""
        metrics = self._calculate_metrics(text)
        confidence = self._calculate_confidence(metrics)
        grade_level = self._get_grade_level(metrics.flesch_kincaid_grade)
        interpretation = self._get_flesch_interpretation(metrics.flesch_reading_ease)

        return ClassificationResult(
            label=grade_level,
            confidence=confidence,
            passed=True,
            message=f"Text is at {grade_level} reading level",
            metadata={
                "flesch_reading_ease": metrics.flesch_reading_ease,
                "flesch_kincaid_grade": metrics.flesch_kincaid_grade,
                "gunning_fog": metrics.gunning_fog,
                "smog_index": metrics.smog_index,
                "automated_readability_index": metrics.automated_readability_index,
                "dale_chall_readability_score": metrics.dale_chall_readability_score,
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
                "interpretation": interpretation,
            },
        )

    @classmethod
    def create_with_custom_analyzer(
        cls: Type["ReadabilityClassifier"],
        analyzer: ReadabilityAnalyzer,
        name: str = "custom_readability_classifier",
        description: str = "Custom readability analyzer",
        config: Optional[ClassifierConfig[Any]] = None,
        **kwargs: Any,
    ) -> "ReadabilityClassifier":
        """Create a readability classifier with a custom analyzer."""
        return cls(
            name=name,
            description=description,
            analyzer=analyzer,
            config=config,
            **kwargs,
        )


def create_readability_classifier(
    name: str = "readability_classifier",
    description: str = "Analyzes text readability",
    analyzer: Optional[ReadabilityAnalyzer] = None,
    grade_level_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    min_confidence: float = 0.7,
    cache_size: int = 100,
    cost: float = ReadabilityClassifier.DEFAULT_COST,
    config: Optional[ClassifierConfig[Any]] = None,
    **kwargs: Any,
) -> ReadabilityClassifier:
    """Create a readability classifier with the given configuration."""
    if config is None:
        params = kwargs.pop("params", {})
        if grade_level_bounds is not None:
            params["grade_level_bounds"] = grade_level_bounds
        config = ClassifierConfig(
            params=params,
            cache_size=cache_size,
            **kwargs,
        )
    return ReadabilityClassifier(
        name=name,
        description=description,
        analyzer=analyzer,
        config=config,
        **kwargs,
    )
