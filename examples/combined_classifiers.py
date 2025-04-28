#!/usr/bin/env python3
"""
Combined Classifiers and Rules Example for Sifaka.

This example demonstrates:
1. Using multiple classifiers together (sentiment and readability)
2. Creating classifier-based rules for validation
3. Detecting repetition patterns in text
4. Combining multiple analysis techniques

Usage:
    python combined_classifiers.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
"""

import os
import sys

# Add parent directory to system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sifaka.classifiers.readability import ReadabilityClassifier
from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.rules.pattern_rules import RepetitionRule
from sifaka.rules.adapters import ClassifierRuleAdapter
from sifaka.rules.base import RuleConfig, RulePriority
from sifaka.utils.logging import get_logger

# Initialize logger from Sifaka
logger = get_logger(__name__)


def main():
    """Run the combined classifiers example."""
    logger.info("Starting combined classifiers example...")

    # Initialize classifiers
    logger.info("Initializing classifiers and rules...")

    # Sentiment classifier
    sentiment_classifier = SentimentClassifier(
        name="tone_analyzer",
        description="Analyzes the tone/sentiment of text",
        min_confidence=0.7,
    )

    # Readability classifier
    readability_classifier = ReadabilityClassifier(
        name="readability_analyzer",
        description="Analyzes text readability",
        min_confidence=0.7,
    )

    # Create readability rule adapter
    readability_rule = ClassifierRuleAdapter(
        classifier=readability_classifier,
        rule_config=RuleConfig(
            priority=RulePriority.MEDIUM,
            cost=1.0,
            metadata={"valid_labels": ["simple", "standard"]},
        ),
    )

    # Repetition detection rule
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

    # Example texts
    texts = [
        "The quick brown fox jumps over the lazy dog. The lazy dog lets the quick brown fox jump.",
        "This is a very simple text that should be easy to read.",
        "The quantum mechanical interpretation of molecular orbital theory requires advanced understanding of mathematical principles.",
    ]

    # Process each text
    for i, text in enumerate(texts, 1):
        logger.info(f"\n===== Text {i} =====")
        logger.info(f"'{text}'")

        # Sentiment analysis
        sentiment_result = sentiment_classifier.classify(text)
        logger.info(
            f"\nSentiment: {sentiment_result.label} (confidence: {sentiment_result.confidence:.2f})"
        )

        # Readability analysis
        readability_result = readability_classifier.classify(text)
        logger.info(
            f"Readability: {readability_result.label} (confidence: {readability_result.confidence:.2f})"
        )

        # Get some readability metrics
        if readability_result.metadata:
            metrics = readability_result.metadata.get("metrics", {})
            if "flesch_kincaid_grade" in metrics:
                logger.info(f"Flesch-Kincaid Grade: {metrics['flesch_kincaid_grade']:.1f}")
            if "flesch_reading_ease" in metrics:
                logger.info(f"Flesch Reading Ease: {metrics['flesch_reading_ease']:.1f}")

        # Readability rule validation
        readability_validation = readability_rule.validate(text)
        logger.info(f"\nReadability Rule - Passed: {readability_validation.passed}")
        logger.info(f"Message: {readability_validation.message}")

        # Repetition analysis
        repetition_result = repetition_rule.validate(text)
        logger.info(f"\nRepetition Check - Passed: {repetition_result.passed}")
        logger.info(f"Message: {repetition_result.message}")

        # Show detected patterns
        if "patterns" in repetition_result.metadata and repetition_result.metadata["patterns"]:
            patterns = repetition_result.metadata["patterns"]
            logger.info("Patterns found:")
            for pattern in patterns[:3]:  # Show up to 3 patterns
                logger.info(f"- '{pattern}'")

    logger.info("\nCombined classifiers example completed.")


if __name__ == "__main__":
    main()
