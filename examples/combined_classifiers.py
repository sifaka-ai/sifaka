"""
Example demonstrating combined use of classifiers and pattern rules.

This example showcases the integration of multiple analysis techniques:
1. Text Classification
   - Sentiment analysis for tone and emotion detection
   - Readability assessment using standard metrics
   - Confidence-based classification results

2. Pattern Analysis
   - Symmetry detection with configurable thresholds
   - Repetition identification in text structures
   - Combined pattern validation approach

3. Multi-faceted Text Analysis
   - Parallel processing of multiple analysis types
   - Comprehensive result aggregation
   - Detailed metadata collection

4. Error Handling and Logging
   - Structured logging of analysis results
   - Clear validation messaging
   - Detailed pattern match reporting

Usage:
    python combined_classifiers.py

Requirements:
    - Sifaka library with classifier and pattern rule support
    - Python logging configuration
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.classifiers.readability import ReadabilityClassifier
from sifaka.classifiers.base import ClassificationResult
from sifaka.rules import SymmetryRule, RepetitionRule
from sifaka.rules.base import RuleConfig, RulePriority

import logging

# Configure logging with a consistent format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze text using multiple classifiers and pattern rules.

    This function combines multiple analysis techniques:
    1. Sentiment Analysis: Evaluates the emotional tone of the text
    2. Readability Analysis: Assesses text complexity and readability
    3. Symmetry Analysis: Detects symmetric patterns in text structure
    4. Repetition Analysis: Identifies repeated patterns and phrases

    Args:
        text (str): The text to analyze

    Returns:
        Dict[str, Any]: Analysis results containing:
            - sentiment: Sentiment classification results and confidence
            - readability: Readability level and metrics
            - symmetry: Symmetry pattern analysis results
            - repetition: Repetition pattern detection results

    Example:
        >>> results = analyze_text("The quick brown fox jumps over the lazy dog")
        >>> print(results["sentiment"]["label"])
        'neutral'
    """
    results = {}

    # Initialize sentiment classifier with confidence threshold
    sentiment_classifier = SentimentClassifier(
        name="tone_analyzer",
        description="Analyzes the tone/sentiment of text",
        min_confidence=0.7,  # High confidence threshold for reliable results
    )

    # Initialize readability classifier with confidence threshold
    readability_classifier = ReadabilityClassifier(
        name="readability_analyzer",
        description="Analyzes text readability",
        min_confidence=0.7,  # High confidence threshold for reliable results
    )

    # Configure symmetry detection with balanced thresholds
    symmetry_rule = SymmetryRule(
        name="symmetry_check",
        description="Checks for text symmetry patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            metadata={
                "mirror_mode": "both",  # Check both horizontal and vertical symmetry
                "symmetry_threshold": 0.8,  # High threshold for meaningful patterns
                "preserve_whitespace": True,  # Consider spacing in symmetry
                "preserve_case": True,  # Case-sensitive matching
                "ignore_punctuation": True,  # Ignore punctuation for better matching
            },
        ),
    )

    # Configure repetition detection with specific constraints
    repetition_rule = RepetitionRule(
        name="repetition_check",
        description="Detects repetitive patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            metadata={
                "pattern_type": "repeat",  # Look for repeated sequences
                "pattern_length": 3,  # Minimum pattern length
                "case_sensitive": True,  # Case-sensitive pattern matching
                "allow_overlap": False,  # Non-overlapping patterns only
            },
        ),
    )

    # Perform sentiment analysis
    sentiment_result = sentiment_classifier.classify(text)
    results["sentiment"] = {
        "label": sentiment_result.label,
        "confidence": sentiment_result.confidence,
        "metadata": sentiment_result.metadata,
    }

    # Perform readability analysis
    readability_result = readability_classifier.classify(text)
    results["readability"] = {
        "label": readability_result.label,
        "confidence": readability_result.confidence,
        "metadata": readability_result.metadata,
    }

    # Perform pattern analysis
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
    """
    Main function demonstrating combined classifier and pattern analysis.

    This function:
    1. Sets up example texts with varying characteristics
    2. Processes each text through multiple analysis pipelines
    3. Logs detailed results for each analysis type
    4. Demonstrates integration of different analysis techniques
    """
    # Example texts demonstrating various linguistic patterns
    texts = [
        # Text with repetitive structure and balanced tone
        "The quick brown fox jumps over the lazy dog. The lazy dog lets the quick brown fox jump.",
        # Palindrome with symmetric structure
        "A man, a plan, a canal: Panama!",
        # Simple text for readability analysis
        "This is a very simple text that should be easy to read.",
        # Complex text with technical content
        "The quantum mechanical interpretation of molecular orbital theory requires advanced understanding of mathematical principles.",
    ]

    # Process each text through the analysis pipeline
    for i, text in enumerate(texts, 1):
        logger.info("\n%s", "=" * 50)
        logger.info("Analyzing text %d: %s", i, text)

        # Perform multi-faceted analysis
        results = analyze_text(text)

        # Log sentiment analysis results
        logger.info("\nSentiment Analysis:")
        logger.info("- Label: %s", results["sentiment"]["label"])
        logger.info("- Confidence: %.2f", results["sentiment"]["confidence"])
        logger.info("- Details: %s", results["sentiment"]["metadata"])

        # Log readability analysis results
        logger.info("\nReadability Analysis:")
        logger.info("- Level: %s", results["readability"]["label"])
        logger.info("- Confidence: %.2f", results["readability"]["confidence"])
        logger.info("- Metrics: %s", results["readability"]["metadata"])

        # Log pattern analysis results
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
