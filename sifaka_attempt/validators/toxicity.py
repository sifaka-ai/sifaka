"""
Toxicity validator for detecting toxic content in text.

This module provides a validator that checks if text contains toxic content
using the Detoxify library. It leverages the ToxicityClassifier for the core
classification functionality.
"""

from typing import Dict, Any, List, Optional, Union, cast
from ..types import ValidationResult
from ..classifiers.toxicity import ToxicityClassifier


class ToxicityValidator:
    """
    Validator that checks text for toxic content.

    This validator uses the ToxicityClassifier to detect various forms of toxic
    content in text, including general toxicity, severe toxicity, obscenity,
    threats, insults, and identity-based attacks.

    By default, it requires the 'detoxify' package to be installed.
    Install it with: pip install detoxify
    """

    def __init__(
        self,
        threshold: float = 0.5,
        severe_toxic_threshold: float = 0.7,
        threat_threshold: float = 0.7,
        check_all_categories: bool = False,
        category_thresholds: Optional[Dict[str, float]] = None,
        model_name: str = "original",
    ):
        """
        Initialize the toxicity validator.

        Args:
            threshold: General threshold for toxicity detection (0.0 to 1.0)
            severe_toxic_threshold: Threshold for severe toxicity (0.0 to 1.0)
            threat_threshold: Threshold for threats (0.0 to 1.0)
            check_all_categories: If True, text must pass all category checks
            category_thresholds: Custom thresholds for specific categories
            model_name: Detoxify model to use ('original', 'unbiased', or 'multilingual')
        """
        # Create the classifier
        self.classifier = ToxicityClassifier(
            general_threshold=threshold,
            severe_toxic_threshold=severe_toxic_threshold,
            threat_threshold=threat_threshold,
            model_name=model_name,
        )
        self.check_all_categories = check_all_categories
        self.category_thresholds = category_thresholds or {}

        # Primary categories are those that we care most about for general toxicity detection
        self.PRIMARY_CATEGORIES = ["toxicity", "severe_toxicity", "threat"]

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text for toxic content.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult with toxicity scores
        """
        if not text.strip():
            return ValidationResult(
                passed=True,
                message="Empty text passes toxicity validation",
                score=1.0,
                issues=[],
                suggestions=[],
                metadata={"scores": {}},
            )

        # Use the classifier to get toxicity information
        classification = self.classifier.classify(text)

        # Get scores from metadata
        scores = classification.metadata.get("scores", {})

        # Check each category against its threshold
        failed_categories = []
        for category, score in scores.items():
            threshold = self._get_threshold_for_category(category)
            if score > threshold:
                failed_categories.append(category)

        # Determine if validation passed based on check_all_categories
        if self.check_all_categories:
            # If we're checking all categories, any failing category means the validation fails
            passed = len(failed_categories) == 0
        else:
            # Otherwise, check only primary categories
            failed_primary = [c for c in failed_categories if c in self.PRIMARY_CATEGORIES]
            passed = len(failed_primary) == 0

        # Create issues and suggestions
        issues = []
        suggestions = []

        category_descriptions = {
            "toxicity": "general toxicity",
            "severe_toxicity": "severe toxicity",
            "obscene": "obscene content",
            "threat": "threatening content",
            "insult": "insulting content",
            "identity_attack": "identity-based attacks",
        }

        for category in failed_categories:
            category_desc = category_descriptions.get(category, category)
            threshold = self._get_threshold_for_category(category)
            score = scores.get(category, 0)
            issues.append(
                f"Text contains {category_desc} (score: {score:.2f}, threshold: {threshold:.2f})"
            )
            suggestions.append(f"Reduce {category_desc} in the text")

        # Create message
        if passed:
            message = "Text passed toxicity validation"
        else:
            message = "Text contains toxic content"

        # Calculate overall score (inverse of max toxicity)
        # Higher score = less toxic (1.0 = not toxic at all)
        toxicity_scores = [scores.get(c, 0.0) for c in category_descriptions.keys()]
        if toxicity_scores:
            # Score is the opposite of the worst toxicity score
            score = 1.0 - max(toxicity_scores)
        else:
            score = 1.0

        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={"scores": scores, "label": classification.label},
        )

    def _get_threshold_for_category(self, category: str) -> float:
        """Get the threshold for a specific category, with fallbacks."""
        # Check custom thresholds first
        if category in self.category_thresholds:
            return self.category_thresholds[category]

        # Use special thresholds for certain categories
        if category == "severe_toxicity":
            return self.classifier.severe_toxic_threshold
        elif category == "threat":
            return self.classifier.threat_threshold

        # Default to the general threshold
        return self.classifier.general_threshold
