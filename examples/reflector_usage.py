#!/usr/bin/env python3
"""
Pattern Rules Example for Sifaka.

This example demonstrates:
1. Using SymmetryRule for detecting text symmetry
2. Using RepetitionRule for pattern detection
3. Combining pattern rules with basic validation rules

Usage:
    python reflector_usage.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
    - Anthropic API key in ANTHROPIC_API_KEY environment variable (optional)
"""

import os
import sys

# Add parent directory to system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing dotenv package. Install with: pip install python-dotenv")

from sifaka.rules import LengthRule, ProhibitedContentRule, RepetitionRule, SymmetryRule
from sifaka.rules.base import RuleConfig, RulePriority
from sifaka.utils.logging import get_logger

# Initialize logger from Sifaka
logger = get_logger(__name__)


def main():
    """Run the pattern rules example."""
    # Load environment variables (for optional model usage)
    try:
        load_dotenv()
    except:
        pass  # Not required for this example

    logger.info("Starting pattern rules example...")

    # Create pattern detection rules
    logger.info("Creating pattern detection rules...")

    # Symmetry rule
    symmetry_rule = SymmetryRule(
        name="symmetry_check",
        description="Checks for text symmetry patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            cache_size=100,
            cost=1.0,
            metadata={
                "mirror_mode": "both",
                "symmetry_threshold": 0.8,
                "preserve_whitespace": True,
                "preserve_case": True,
                "ignore_punctuation": True,
            },
        ),
    )

    # Repetition rule
    repetition_rule = RepetitionRule(
        name="repetition_check",
        description="Detects repetitive patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            cache_size=100,
            cost=1.0,
            metadata={
                "pattern_type": "repeat",
                "pattern_length": 3,
                "case_sensitive": True,
                "allow_overlap": False,
            },
        ),
    )

    # Basic validation rules
    length_rule = LengthRule(
        name="length_check",
        description="Checks if output length is within bounds",
        config={
            "min_length": 50,
            "max_length": 500,
            "unit": "characters",
            "priority": 2,
            "cost": 1.5,
            "cache_size": 100,
        },
    )

    content_rule = ProhibitedContentRule(
        name="content_filter",
        description="Checks for prohibited content",
        config={
            "prohibited_terms": ["controversial", "inappropriate"],
            "case_sensitive": False,
            "priority": 2,
            "cost": 1.5,
            "cache_size": 100,
        },
    )

    # Example texts with different pattern characteristics
    texts = [
        "A man, a plan, a canal: Panama!",  # Palindrome
        "Roses are red, red are roses, in the garden they grow, grow they in the garden.",  # Repetitive
        "Mirror mirror on the wall, who is the fairest of them all?",  # Repetitive start
        "This is just a simple text without any special patterns.",  # Control
    ]

    # Process each text
    for i, text in enumerate(texts, 1):
        logger.info(f"\n===== Text {i} =====")
        logger.info(f"'{text}'")

        # Check length
        length_result = length_rule._validate_impl(text)
        logger.info(f"\nLength Check - Passed: {length_result.passed}")
        logger.info(f"Message: {length_result.message}")

        # Check content
        content_result = content_rule._validate_impl(text)
        logger.info(f"\nContent Check - Passed: {content_result.passed}")
        logger.info(f"Message: {content_result.message}")

        # Check symmetry
        symmetry_result = symmetry_rule._validate_impl(text)
        logger.info(f"\nSymmetry Check - Passed: {symmetry_result.passed}")
        logger.info(f"Message: {symmetry_result.message}")
        if "symmetry_score" in symmetry_result.metadata:
            logger.info(f"Score: {symmetry_result.metadata['symmetry_score']:.2f}")

        # Check repetition
        repetition_result = repetition_rule._validate_impl(text)
        logger.info(f"\nRepetition Check - Passed: {repetition_result.passed}")
        logger.info(f"Message: {repetition_result.message}")
        if "patterns" in repetition_result.metadata and repetition_result.metadata["patterns"]:
            patterns = repetition_result.metadata["patterns"]
            logger.info("Patterns found:")
            for pattern in patterns[:3]:  # Show up to 3 patterns
                logger.info(f"- '{pattern}'")

    logger.info("\nPattern rules example completed.")


if __name__ == "__main__":
    main()
