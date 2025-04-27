"""
Readability classifier that analyzes text complexity.
"""

from typing import Dict, Any, List, Optional
import importlib
import logging
import statistics

from sifaka.classifiers.base import Classifier, ClassificationResult
from sifaka.utils.logging import get_logger
import textstat

logger = get_logger(__name__)


class ReadabilityClassifier(Classifier):
    """
    A classifier that analyzes text readability using textstat.

    This classifier uses multiple readability metrics to determine the
    appropriate reading level of text.

    Requires the 'readability' extra to be installed:
    pip install sifaka[readability]

    Attributes:
        min_confidence: Minimum confidence threshold
        labels: Reading level labels
    """

    def __init__(
        self,
        name: str = "readability_classifier",
        description: str = "Analyzes text readability",
        config: Dict[str, Any] = None,
        labels: List[str] = None,
        min_confidence: float = 0.5,
        **kwargs,
    ) -> None:
        """
        Initialize the readability classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            config: Additional configuration
            labels: Reading level labels
            min_confidence: Minimum confidence threshold
            **kwargs: Additional arguments
        """
        if labels is None:
            labels = ["elementary", "middle", "high", "college", "graduate"]
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            labels=labels,
            cost=1,
            min_confidence=min_confidence,
            **kwargs,
        )
        self._textstat = None

    def _load_textstat(self) -> None:
        """Load the textstat library."""
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
        """Convert grade level to readability label."""
        if grade < 6:
            return "elementary"
        elif grade < 9:
            return "middle"
        elif grade < 12:
            return "high"
        elif grade < 16:
            return "college"
        else:
            return "graduate"

    def _get_flesch_interpretation(self, score: float) -> str:
        """Interpret Flesch reading ease score."""
        if score >= 90:
            return "Very Easy - 5th grade"
        elif score >= 80:
            return "Easy - 6th grade"
        elif score >= 70:
            return "Fairly Easy - 7th grade"
        elif score >= 60:
            return "Standard - 8th/9th grade"
        elif score >= 50:
            return "Fairly Difficult - 10th/12th grade"
        elif score >= 30:
            return "Difficult - College"
        else:
            return "Very Difficult - College Graduate"

    def _calculate_rix_index(self, text: str) -> float:
        """Calculate RIX readability index."""
        words = text.split()
        if not words:
            return 0.0

        long_words = sum(1 for word in words if len(word) > 6)
        sentences = textstat.sentence_count(text)
        if sentences == 0:
            return 0.0

        return long_words / sentences

    def _calculate_advanced_stats(self, text: str) -> Dict[str, float]:
        """Calculate advanced readability statistics."""
        words = text.split()
        unique_words = set(word.lower() for word in words)

        # Calculate word length statistics
        word_lengths = [len(word) for word in words]
        avg_word_length = statistics.mean(word_lengths) if word_lengths else 0
        word_length_std = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0

        # Calculate sentence length statistics
        sentences = textstat.sentence_count(text)
        words_per_sentence = [len(sent.split()) for sent in text.split(".") if sent.strip()]
        avg_sentence_length = statistics.mean(words_per_sentence) if words_per_sentence else 0
        sentence_length_std = (
            statistics.stdev(words_per_sentence) if len(words_per_sentence) > 1 else 0
        )

        return {
            "lexicon_count": textstat.lexicon_count(text),
            "sentence_count": sentences,
            "syllable_count": textstat.syllable_count(text),
            "avg_sentence_length": avg_sentence_length,
            "sentence_length_std": sentence_length_std,
            "avg_word_length": avg_word_length,
            "word_length_std": word_length_std,
            "vocabulary_diversity": len(unique_words) / len(words) if words else 0,
            "unique_word_count": len(unique_words),
            "difficult_words": textstat.difficult_words(text),
        }

    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """
        Calculate confidence based on agreement between metrics.

        Args:
            metrics: Dictionary of readability metrics

        Returns:
            Confidence score between 0 and 1
        """
        if metrics.get("lexicon_count", 0) == 0:
            return 0.0

        # Get relevant grade-level metrics
        grade_metrics = [
            metrics.get("flesch_kincaid_grade", 0),
            metrics.get("gunning_fog", 0),
            metrics.get("smog_index", 0),
            metrics.get("dale_chall_readability_score", 0),
            metrics.get("automated_readability_index", 0),
        ]

        # Calculate coefficient of variation
        if not grade_metrics or statistics.mean(grade_metrics) == 0:
            return 0.0

        cv = statistics.stdev(grade_metrics) / statistics.mean(grade_metrics)
        confidence = max(0.0, 1.0 - cv)

        return min(1.0, confidence)

    def classify(self, text: str, **kwargs) -> ClassificationResult:
        """
        Classify text readability level.

        Args:
            text: Input text to classify
            **kwargs: Additional classification context

        Returns:
            ClassificationResult with readability level and confidence
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        if not text.strip():
            return ClassificationResult(
                label=self.labels[0],  # elementary
                confidence=0.0,
                metadata={
                    "error": "Empty input",
                    "metrics": {
                        "lexicon_count": 0,
                        "sentence_count": 0,
                        "syllable_count": 0,
                        "flesch_reading_ease": 100.0,  # Most readable
                        "flesch_kincaid_grade": 0.0,
                        "gunning_fog": 0.0,
                        "smog_index": 0.0,
                        "automated_readability_index": 0.0,
                        "coleman_liau_index": 0.0,
                    },
                },
            )

        self.warm_up()

        try:
            # Calculate basic metrics
            metrics = {
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "smog_index": textstat.smog_index(text),
                "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
                "automated_readability_index": textstat.automated_readability_index(text),
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
            }

            # Add advanced stats
            metrics.update(self._calculate_advanced_stats(text))
            metrics["rix_index"] = self._calculate_rix_index(text)

            # Calculate average grade level
            grade_level = statistics.mean(
                [
                    metrics["flesch_kincaid_grade"],
                    metrics["gunning_fog"],
                    metrics["smog_index"],
                    metrics["dale_chall_readability_score"],
                    metrics["automated_readability_index"],
                ]
            )

            # Get readability level
            label = self._get_grade_level(grade_level)

            # Calculate confidence
            confidence = self._calculate_confidence(metrics)

            return ClassificationResult(label=label, confidence=confidence, metadata=metrics)
        except Exception as e:
            return ClassificationResult(
                label=self.labels[0], confidence=0.0, metadata={"error": str(e)}  # elementary
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
