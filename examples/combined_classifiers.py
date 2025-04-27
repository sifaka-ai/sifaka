"""
Example demonstrating combined use of classifiers and pattern rules.

This example shows how to:
1. Use existing classifiers (sentiment and readability)
2. Combine them with pattern rules (symmetry and repetition)
3. Process text through multiple analysis steps
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.classifiers.readability import ReadabilityClassifier
from sifaka.classifiers.base import ClassificationResult
from sifaka.rules import SymmetryRule, RepetitionRule
from sifaka.rules.base import RuleConfig, RulePriority

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze text using multiple classifiers and pattern rules.

    Args:
        text: The text to analyze

    Returns:
        Dict containing analysis results
    """
    results = {}

    # Initialize classifiers
    sentiment_classifier = SentimentClassifier(
        name="tone_analyzer", description="Analyzes the tone/sentiment of text", min_confidence=0.7
    )

    readability_classifier = ReadabilityClassifier(
        name="readability_analyzer", description="Analyzes text readability", min_confidence=0.7
    )

    # Initialize pattern rules
    symmetry_rule = SymmetryRule(
        name="symmetry_check",
        description="Checks for text symmetry patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            metadata={
                "mirror_mode": "both",
                "symmetry_threshold": 0.8,
                "preserve_whitespace": True,
                "preserve_case": True,
                "ignore_punctuation": True,
            },
        ),
    )

    repetition_rule = RepetitionRule(
        name="repetition_check",
        description="Detects repetitive patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            metadata={
                "pattern_type": "repeat",
                "pattern_length": 3,
                "case_sensitive": True,
                "allow_overlap": False,
            },
        ),
    )

    # Run sentiment analysis
    sentiment_result = sentiment_classifier.classify(text)
    results["sentiment"] = {
        "label": sentiment_result.label,
        "confidence": sentiment_result.confidence,
        "metadata": sentiment_result.metadata,
    }

    # Run readability analysis
    readability_result = readability_classifier.classify(text)
    results["readability"] = {
        "label": readability_result.label,
        "confidence": readability_result.confidence,
        "metadata": readability_result.metadata,
    }

    # Check for patterns
    symmetry_result = symmetry_rule._validate_impl(text)
    results["symmetry"] = {
        "passed": symmetry_result.passed,
        "message": symmetry_result.message,
        "metadata": symmetry_result.metadata,
    }

    repetition_result = repetition_rule._validate_impl(text)
    results["repetition"] = {
        "passed": repetition_result.passed,
        "message": repetition_result.message,
        "metadata": repetition_result.metadata,
    }

    return results


def main():
    # Example texts to analyze
    texts = [
        "The quick brown fox jumps over the lazy dog. The lazy dog lets the quick brown fox jump.",
        "A man, a plan, a canal: Panama!",
        "This is a very simple text that should be easy to read.",
        "The quantum mechanical interpretation of molecular orbital theory requires advanced understanding of mathematical principles.",
    ]

    for i, text in enumerate(texts, 1):
        logger.info("\n%s", "=" * 50)
        logger.info("Analyzing text %d: %s", i, text)

        results = analyze_text(text)

        logger.info("\nSentiment Analysis:")
        logger.info("- Label: %s", results["sentiment"]["label"])
        logger.info("- Confidence: %.2f", results["sentiment"]["confidence"])
        logger.info("- Details: %s", results["sentiment"]["metadata"])

        logger.info("\nReadability Analysis:")
        logger.info("- Level: %s", results["readability"]["label"])
        logger.info("- Confidence: %.2f", results["readability"]["confidence"])
        logger.info("- Metrics: %s", results["readability"]["metadata"])

        logger.info("\nPattern Analysis:")
        logger.info("Symmetry Check:")
        logger.info("- Passed: %s", results["symmetry"]["passed"])
        logger.info("- Message: %s", results["symmetry"]["message"])
        if "symmetry_score" in results["symmetry"]["metadata"]:
            logger.info("- Score: %.2f", results["symmetry"]["metadata"]["symmetry_score"])

        logger.info("\nRepetition Check:")
        logger.info("- Passed: %s", results["repetition"]["passed"])
        logger.info("- Message: %s", results["repetition"]["message"])
        if "patterns" in results["repetition"]["metadata"]:
            logger.info("- Patterns found: %s", results["repetition"]["metadata"]["patterns"])


if __name__ == "__main__":
    main()
