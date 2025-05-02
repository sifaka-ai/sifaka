"""
Sentiment analysis content validation rules for Sifaka.

This module provides rules for analyzing and validating text sentiment,
including positive/negative sentiment detection and emotional content analysis.

This module is now a thin wrapper around the SentimentClassifier, which provides
more sophisticated sentiment analysis using VADER.

Usage Example:
    from sifaka.rules.content.sentiment import create_sentiment_rule

    # Create a sentiment rule using the classifier adapter
    sentiment_rule = create_sentiment_rule(threshold=0.7)

    # Validate text
    result = sentiment_rule.validate("This is a test.")
"""

from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.rules.adapters.classifier import create_classifier_rule
from sifaka.rules.base import (
    BaseValidator,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.content.base import (
    ContentAnalyzer,
    ContentValidator,
    DefaultContentAnalyzer,
)


__all__ = [
    # Factory functions
    "create_sentiment_rule",
]


def create_sentiment_rule(
    name: str = "sentiment_rule",
    description: str = "Validates text sentiment",
    threshold: float = 0.6,
    **kwargs,
) -> Rule[str, RuleResult, BaseValidator[str], RuleResultHandler[RuleResult]]:
    """
    Create a sentiment rule using the classifier adapter.

    This factory function creates a configured sentiment rule instance using the
    SentimentClassifier through the classifier adapter.

    Args:
        name: The name of the rule
        description: Description of the rule
        threshold: Threshold for sentiment detection (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured sentiment rule instance
    """
    return create_classifier_rule(
        classifier=SentimentClassifier,
        name=name,
        description=description,
        threshold=threshold,
        valid_labels=["positive", "neutral"],
        **kwargs,
    )
