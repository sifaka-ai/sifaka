"""
Readability classifier using textstat.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING, ClassVar, Tuple
import importlib
import logging
import re
from statistics import mean, stdev

from pydantic import Field
from sifaka.classifiers.base import Classifier, ClassificationResult
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Only import type hints during type checking
if TYPE_CHECKING:
    import textstat


class ReadabilityClassifier(Classifier):
    """
    A readability classifier using textstat.

    This classifier analyzes text readability using various metrics including:
    - Flesch Reading Ease
    - Flesch-Kincaid Grade Level
    - Gunning Fog Index
    - SMOG Index
    - Dale-Chall Score
    - Automated Readability Index
    - Coleman-Liau Index
    - Linsear Write Formula
    - Rix Index
    - Spache Index

    Requires the 'readability' extra to be installed:
    pip install sifaka[readability]

    Attributes:
        min_confidence: Minimum confidence threshold
        grade_levels: Mapping of grade levels to labels
    """

    # Grade level ranges and their labels
    GRADE_LEVELS: ClassVar[Dict[str, Tuple[float, float]]] = {
        "elementary": (0, 6),  # Up to 6th grade
        "middle": (7, 9),  # 7th to 9th grade
        "high": (10, 12),  # 10th to 12th grade
        "college": (13, 16),  # College level
        "graduate": (17, float("inf")),  # Graduate level and above
    }

    # Mapping of readability scores to interpretations
    FLESCH_INTERPRETATIONS: ClassVar[Dict[Tuple[float, float], str]] = {
        (90, 100): "Very Easy - 5th grade",
        (80, 90): "Easy - 6th grade",
        (70, 80): "Fairly Easy - 7th grade",
        (60, 70): "Standard - 8th/9th grade",
        (50, 60): "Fairly Difficult - 10th/12th grade",
        (30, 50): "Difficult - College",
        (0, 30): "Very Difficult - College Graduate",
    }

    min_confidence: float = Field(default=0.5)

    def __init__(
        self,
        name: str = "readability_classifier",
        description: str = "Analyzes text readability",
        min_confidence: float = 0.5,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the readability classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            min_confidence: Minimum confidence threshold
            config: Additional configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            labels=list(self.GRADE_LEVELS.keys()),
            cost=1,  # Low cost for statistical analysis
            min_confidence=min_confidence,
            **kwargs,
        )
        self._textstat = None

    def _load_textstat(self) -> None:
        """Load the textstat module."""
        try:
            self._textstat = importlib.import_module("textstat")
        except ImportError:
            raise ImportError(
                "textstat package is required for ReadabilityClassifier. "
                "Install it with: pip install sifaka[readability]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load textstat: {e}")

    def warm_up(self) -> None:
        """Initialize textstat if needed."""
        if self._textstat is None:
            self._load_textstat()

    def _get_grade_level(self, grade: float) -> str:
        """
        Convert numerical grade level to categorical label.

        Args:
            grade: Numerical grade level

        Returns:
            Categorical grade level label
        """
        for level, (min_grade, max_grade) in self.GRADE_LEVELS.items():
            if min_grade <= grade <= max_grade:
                return level
        return "graduate"  # Default to highest level if above all ranges

    def _get_flesch_interpretation(self, score: float) -> str:
        """Get interpretation of Flesch Reading Ease score."""
        for (min_score, max_score), interpretation in self.FLESCH_INTERPRETATIONS.items():
            if min_score <= score <= max_score:
                return interpretation
        return "Very Difficult - College Graduate"

    def _calculate_rix_index(self, text: str) -> float:
        """
        Calculate RIX readability index.
        RIX = number of long words (>6 chars) / number of sentences
        """
        try:
            words = text.split()
            long_words = sum(1 for word in words if len(word) > 6)
            sentences = self._textstat.sentence_count(text)
            return long_words / sentences if sentences > 0 else 0
        except Exception as e:
            logger.warning("Failed to calculate RIX index: %s", e)
            return 0

    def _calculate_advanced_stats(self, text: str) -> Dict[str, Any]:
        """Calculate advanced text statistics."""
        try:
            words = text.split()
            sentences = text.split(".")

            # Word length statistics
            word_lengths = [len(word) for word in words]
            avg_word_length = mean(word_lengths) if word_lengths else 0
            word_length_std = stdev(word_lengths) if len(word_lengths) > 1 else 0

            # Sentence length statistics
            sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
            avg_sentence_length = mean(sentence_lengths) if sentence_lengths else 0
            sentence_length_std = stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0

            # Vocabulary diversity (unique words ratio)
            unique_words = len(set(word.lower() for word in words))
            vocabulary_diversity = unique_words / len(words) if words else 0

            return {
                "avg_word_length": avg_word_length,
                "word_length_std": word_length_std,
                "avg_sentence_length": avg_sentence_length,
                "sentence_length_std": sentence_length_std,
                "vocabulary_diversity": vocabulary_diversity,
                "unique_word_count": unique_words,
            }
        except Exception as e:
            logger.warning("Failed to calculate advanced stats: %s", e)
            return {}

    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """
        Calculate confidence based on consistency across metrics.

        Args:
            metrics: Dictionary of readability metrics

        Returns:
            Confidence score between 0 and 1
        """
        # Handle empty text
        if metrics.get("lexicon_count", 0) == 0:
            return 0.0

        # Get grade levels for each metric
        grades = [
            metrics["flesch_kincaid_grade"],
            metrics["gunning_fog"],
            metrics["smog_index"],
            metrics["coleman_liau_index"],
            metrics["automated_readability_index"],
            metrics.get("linsear_write_formula", 0),
        ]

        # Remove any None, negative, or 0 values
        grades = [g for g in grades if g is not None and g > 0]

        if not grades:
            return 0.0

        # Calculate standard deviation
        mean_grade = sum(grades) / len(grades)
        variance = sum((x - mean_grade) ** 2 for x in grades) / len(grades)
        std_dev = variance**0.5

        # Convert standard deviation to confidence score
        # Lower standard deviation means higher confidence
        # Use a more generous base confidence calculation
        base_confidence = 1 / (1 + 0.5 * std_dev)  # Reduced impact of std_dev

        # Adjust confidence based on text properties
        word_count = metrics.get("lexicon_count", 0)
        sentence_count = metrics.get("sentence_count", 0)
        avg_sentence_length = metrics.get("avg_sentence_length", 0)

        # Text length multiplier (more generous for shorter texts)
        length_multiplier = min(word_count / 10.0, 1.0)  # Scales up to 10 words
        sentence_multiplier = min(sentence_count / 2.0, 1.0)  # Scales up to 2 sentences

        # Additional boost for well-formed sentences
        if avg_sentence_length >= 3 and avg_sentence_length <= 15:
            base_confidence *= 1.2  # Boost for reasonable sentence lengths

        # Combine multipliers
        final_confidence = base_confidence * length_multiplier * sentence_multiplier

        # Cap confidence at 1.0
        return min(max(final_confidence, 0.0), 1.0)

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text readability.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with readability metrics
        """
        self.warm_up()
        try:
            # Handle empty text
            if not text.strip():
                return ClassificationResult(
                    label="college",  # Default to college level for empty text
                    confidence=0.0,
                    metadata={"error": "Empty text"},
                )

            # Calculate various readability metrics
            metrics = {
                # Standard metrics
                "flesch_reading_ease": self._textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": self._textstat.flesch_kincaid_grade(text),
                "gunning_fog": self._textstat.gunning_fog(text),
                "smog_index": self._textstat.smog_index(text),
                "coleman_liau_index": self._textstat.coleman_liau_index(text),
                "automated_readability_index": self._textstat.automated_readability_index(text),
                "dale_chall_readability_score": self._textstat.dale_chall_readability_score(text),
                # Additional metrics
                "linsear_write_formula": self._textstat.linsear_write_formula(text),
                "rix_index": self._calculate_rix_index(text),
                # Text statistics
                "text_standard": self._textstat.text_standard(text),
                "syllable_count": self._textstat.syllable_count(text),
                "lexicon_count": self._textstat.lexicon_count(text),
                "sentence_count": self._textstat.sentence_count(text),
                "avg_sentence_length": self._textstat.avg_sentence_length(text),
                "avg_syllables_per_word": self._textstat.avg_syllables_per_word(text),
                "difficult_words": self._textstat.difficult_words(text),
                "difficult_words_ratio": (
                    self._textstat.difficult_words(text) / self._textstat.lexicon_count(text)
                    if self._textstat.lexicon_count(text) > 0
                    else 0
                ),
            }

            # Add advanced statistics
            metrics.update(self._calculate_advanced_stats(text))

            # Add interpretations
            metrics["flesch_interpretation"] = self._get_flesch_interpretation(
                metrics["flesch_reading_ease"]
            )

            # Calculate average grade level using all available metrics
            grade_metrics = [
                metrics["flesch_kincaid_grade"],
                metrics["gunning_fog"],
                metrics["smog_index"],
                metrics["coleman_liau_index"],
                metrics["automated_readability_index"],
                metrics["linsear_write_formula"],
            ]

            # Filter out negative grades and calculate average
            valid_grades = [g for g in grade_metrics if g > 0]
            avg_grade = sum(valid_grades) / len(valid_grades) if valid_grades else 0

            # Get categorical grade level
            grade_level = self._get_grade_level(avg_grade)

            # Calculate confidence based on consistency across metrics
            confidence = self._calculate_confidence(metrics)

            return ClassificationResult(
                label=(
                    grade_level if confidence > 0 else "college"
                ),  # Default to college for low confidence
                confidence=confidence,
                metadata={
                    "average_grade_level": avg_grade,
                    "metrics": metrics,
                },
            )
        except Exception as e:
            logger.error("Failed to analyze readability: %s", e)
            return ClassificationResult(
                label="college",  # Default to college level
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        return [self.classify(text) for text in texts]
