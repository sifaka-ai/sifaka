"""
Readability validator for assessing text reading difficulty.

This module provides a validator that checks the readability level of text
using various readability metrics from the textstat library.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from ..types import ValidationResult


class ReadabilityValidator:
    """
    Validator that checks text readability levels.

    This validator uses the textstat library to evaluate the readability of text
    using various established metrics like Flesch Reading Ease, Flesch-Kincaid Grade,
    and others. It assesses whether text meets desired reading level targets.

    By default, it requires the 'textstat' package to be installed.
    Install it with: pip install textstat
    """

    # Default grade level bounds for different reading levels
    DEFAULT_GRADE_LEVEL_BOUNDS = {
        "elementary": (0.0, 6.0),
        "middle": (6.0, 9.0),
        "high": (9.0, 12.0),
        "college": (12.0, 16.0),
        "graduate": (16.0, float("inf")),
    }

    # Flesch Reading Ease score interpretations
    FLESCH_INTERPRETATIONS = {
        (90, 100): "Very Easy - 5th grade",
        (80, 90): "Easy - 6th grade",
        (70, 80): "Fairly Easy - 7th grade",
        (60, 70): "Standard - 8th/9th grade",
        (50, 60): "Fairly Difficult - 10th/12th grade",
        (30, 50): "Difficult - College",
        (0, 30): "Very Difficult - College Graduate",
    }

    def __init__(
        self,
        max_grade_level: Optional[float] = None,
        min_grade_level: Optional[float] = None,
        min_flesch_reading_ease: Optional[float] = None,
        max_complexity_metrics: Optional[Dict[str, float]] = None,
        require_all_metrics: bool = False,
        target_grade_level: Optional[str] = None,
    ):
        """
        Initialize the readability validator.

        Args:
            max_grade_level: Maximum grade level (e.g., 8.0 for 8th grade)
            min_grade_level: Minimum grade level
            min_flesch_reading_ease: Minimum Flesch Reading Ease score (higher is easier)
            max_complexity_metrics: Maximum thresholds for complexity metrics like
                                   gunning_fog, smog_index, etc.
            require_all_metrics: If True, text must pass all metrics to pass validation
            target_grade_level: Target reading level ("elementary", "middle", "high",
                               "college", or "graduate")
        """
        self.max_grade_level = max_grade_level
        self.min_grade_level = min_grade_level
        self.min_flesch_reading_ease = min_flesch_reading_ease
        self.max_complexity_metrics = max_complexity_metrics or {}
        self.require_all_metrics = require_all_metrics
        self.target_grade_level = target_grade_level
        self._textstat = None  # Lazy-loaded

    @property
    def textstat(self):
        """Lazy-load the textstat library."""
        if self._textstat is None:
            try:
                import textstat

                self._textstat = textstat
            except ImportError:
                raise ImportError(
                    "textstat is required for ReadabilityValidator. "
                    "Install it with: pip install textstat"
                )
        return self._textstat

    def get_grade_level(self, grade: float) -> str:
        """Convert a numeric grade level to a named category."""
        for level, (lower, upper) in self.DEFAULT_GRADE_LEVEL_BOUNDS.items():
            if lower <= grade < upper:
                return level
        return "graduate"

    def get_flesch_interpretation(self, score: float) -> str:
        """Get a human-readable interpretation of a Flesch Reading Ease score."""
        for (lower, upper), interpretation in self.FLESCH_INTERPRETATIONS.items():
            if lower <= score < upper:
                return interpretation
        return "Unknown"

    def calculate_average_grade_level(self, metrics: Dict[str, float]) -> float:
        """Calculate the average grade level from multiple metrics."""
        grade_metrics = [
            metrics.get("flesch_kincaid_grade", 0),
            metrics.get("gunning_fog", 0),
            metrics.get("smog_index", 0),
            metrics.get("automated_readability_index", 0),
            metrics.get("dale_chall_readability_score", 0),
        ]
        return sum(grade_metrics) / len(grade_metrics)

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text for readability.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult with readability metrics
        """
        if not text.strip():
            return ValidationResult(
                passed=True,
                message="Empty text passes readability validation",
                score=1.0,
                issues=[],
                suggestions=[],
                metadata={"metrics": {}},
            )

        # Calculate metrics
        try:
            ts = self.textstat

            # Core readability metrics
            metrics = {
                "flesch_reading_ease": ts.flesch_reading_ease(text),
                "flesch_kincaid_grade": ts.flesch_kincaid_grade(text),
                "gunning_fog": ts.gunning_fog(text),
                "smog_index": ts.smog_index(text),
                "automated_readability_index": ts.automated_readability_index(text),
                "dale_chall_readability_score": ts.dale_chall_readability_score(text),
                # Text statistics
                "lexicon_count": ts.lexicon_count(text),
                "sentence_count": ts.sentence_count(text),
                "syllable_count": ts.syllable_count(text),
                "difficult_words": ts.difficult_words(text),
            }

            # Calculate average grade level for overall assessment
            avg_grade_level = self.calculate_average_grade_level(metrics)
            metrics["average_grade_level"] = avg_grade_level

            # Determine reading level category
            grade_level_category = self.get_grade_level(avg_grade_level)
            metrics["grade_level_category"] = grade_level_category

            # Get Flesch interpretation
            flesch_interpretation = self.get_flesch_interpretation(metrics["flesch_reading_ease"])
            metrics["flesch_interpretation"] = flesch_interpretation

        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Error analyzing text: {str(e)}",
                score=0.0,
                issues=[f"An error occurred during readability analysis: {str(e)}"],
                suggestions=["Try with different text or check the textstat setup"],
                metadata={"error": str(e)},
            )

        # Check validations
        issues = []
        suggestions = []

        # Check grade level bounds
        if self.max_grade_level is not None and avg_grade_level > self.max_grade_level:
            issues.append(
                f"Text grade level ({avg_grade_level:.1f}) exceeds maximum grade level ({self.max_grade_level:.1f})"
            )
            suggestions.append("Simplify text to reduce grade level")

        if self.min_grade_level is not None and avg_grade_level < self.min_grade_level:
            issues.append(
                f"Text grade level ({avg_grade_level:.1f}) is below minimum grade level ({self.min_grade_level:.1f})"
            )
            suggestions.append("Use more advanced vocabulary and sentence structures")

        # Check Flesch Reading Ease
        if (
            self.min_flesch_reading_ease is not None
            and metrics["flesch_reading_ease"] < self.min_flesch_reading_ease
        ):
            issues.append(
                f"Text reading ease ({metrics['flesch_reading_ease']:.1f}) is below minimum ({self.min_flesch_reading_ease:.1f})"
            )
            suggestions.append("Use shorter sentences and simpler words to improve readability")

        # Check complexity metrics
        for metric_name, max_value in self.max_complexity_metrics.items():
            if metric_name in metrics and metrics[metric_name] > max_value:
                issues.append(
                    f"{metric_name} ({metrics[metric_name]:.1f}) exceeds maximum ({max_value:.1f})"
                )
                suggestions.append(f"Reduce text complexity to lower {metric_name}")

        # Check target grade level
        if self.target_grade_level is not None and grade_level_category != self.target_grade_level:
            issues.append(
                f"Text reading level ({grade_level_category}) does not match target level ({self.target_grade_level})"
            )
            if (
                self.DEFAULT_GRADE_LEVEL_BOUNDS.get(self.target_grade_level, (0, 0))[0]
                > avg_grade_level
            ):
                suggestions.append("Use more advanced vocabulary and sentence structures")
            else:
                suggestions.append("Simplify text to match target reading level")

        # Determine if validation passed
        if self.require_all_metrics:
            passed = len(issues) == 0
        else:
            # Pass if at least no grade level issues (prioritize grade level checks)
            grade_level_issues = [
                i
                for i in issues
                if "grade level" in i or "reading level" in i or "reading ease" in i
            ]
            passed = len(grade_level_issues) == 0

        # Create message
        if passed:
            message = f"Text passed readability validation (Grade level: {avg_grade_level:.1f}, {grade_level_category})"
        else:
            message = f"Text failed readability validation (Grade level: {avg_grade_level:.1f}, {grade_level_category})"

        # Calculate score (1.0 = perfect match to requirements)
        score = 1.0
        if self.max_grade_level is not None and self.min_grade_level is not None:
            # Score based on how close to target range
            target_range = self.max_grade_level - self.min_grade_level
            if target_range > 0:
                if avg_grade_level < self.min_grade_level:
                    # Below range
                    distance = self.min_grade_level - avg_grade_level
                    score = max(0.0, 1.0 - (distance / 5.0))  # 5 grade levels away = 0 score
                elif avg_grade_level > self.max_grade_level:
                    # Above range
                    distance = avg_grade_level - self.max_grade_level
                    score = max(0.0, 1.0 - (distance / 5.0))  # 5 grade levels away = 0 score
        elif self.max_grade_level is not None:
            # Score based on distance below max
            if avg_grade_level > self.max_grade_level:
                distance = avg_grade_level - self.max_grade_level
                score = max(0.0, 1.0 - (distance / 5.0))
        elif self.min_grade_level is not None:
            # Score based on distance above min
            if avg_grade_level < self.min_grade_level:
                distance = self.min_grade_level - avg_grade_level
                score = max(0.0, 1.0 - (distance / 5.0))
        elif self.min_flesch_reading_ease is not None:
            # Score based on Flesch Reading Ease
            if metrics["flesch_reading_ease"] < self.min_flesch_reading_ease:
                distance = self.min_flesch_reading_ease - metrics["flesch_reading_ease"]
                score = max(0.0, 1.0 - (distance / 50.0))  # 50 points away = 0 score

        # Always return some non-zero score if passed
        if passed and score < 0.1:
            score = 0.1

        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={"metrics": metrics},
        )
