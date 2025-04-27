"""
Basic usage example for Sifaka.

This example demonstrates:
1. Setting up a model provider
2. Creating rules for validation
3. Pattern detection with specialized rules (symmetry and repetition)
4. Handling validation results and pattern analysis
"""

import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from sifaka.models import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.rules import LengthRule, ProhibitedContentRule, SymmetryRule, RepetitionRule
from sifaka.rules.base import RuleConfig, RulePriority, RuleResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze text using multiple rules.

    Args:
        text: The text to analyze

    Returns:
        Dict containing analysis results
    """
    results = {}

    # Create basic validation rules
    length_rule = LengthRule(
        name="length_check",
        description="Checks if output length is within bounds",
        config={"min_length": 30, "max_length": 100, "unit": "characters"},
    )

    prohibited_terms = ProhibitedContentRule(
        name="content_filter",
        description="Checks for prohibited or inappropriate content",
        config={"prohibited_terms": ["controversial", "inappropriate"], "case_sensitive": False},
    )

    # Create pattern detection rules
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

    # Run length check
    length_result = length_rule._validate_impl(text)
    results["length"] = {
        "passed": length_result.passed,
        "message": length_result.message,
        "metadata": length_result.metadata,
    }

    # Run content check
    content_result = prohibited_terms._validate_impl(text)
    results["content"] = {
        "passed": content_result.passed,
        "message": content_result.message,
        "metadata": content_result.metadata,
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
    # Load environment variables
    load_dotenv()

    # Initialize the model provider with configuration
    config = ModelConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7,
        max_tokens=2000,
    )

    model = AnthropicProvider(
        model_name="claude-3-haiku-20240307",
        config=config,
    )

    # Example texts to analyze
    texts = [
        "The quick brown fox jumps over the lazy dog. The lazy dog lets the quick brown fox jump.",
        "A man, a plan, a canal: Panama!",
        "This is a very simple text that should be easy to read.",
        "The quantum mechanical interpretation of molecular orbital theory requires advanced understanding of mathematical principles.",
    ]

    # Process each text
    for i, text in enumerate(texts, 1):
        logger.info("\n%s", "=" * 50)
        logger.info("Analyzing text %d: %s", i, text)

        results = analyze_text(text)

        logger.info("\nLength Check:")
        logger.info("- Passed: %s", results["length"]["passed"])
        logger.info("- Message: %s", results["length"]["message"])
        logger.info("- Details: %s", results["length"]["metadata"])

        logger.info("\nContent Check:")
        logger.info("- Passed: %s", results["content"]["passed"])
        logger.info("- Message: %s", results["content"]["message"])
        logger.info("- Details: %s", results["content"]["metadata"])

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
