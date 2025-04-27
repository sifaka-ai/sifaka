"""
Basic Usage Example for Sifaka Pattern Analysis.

This example demonstrates the fundamental usage of Sifaka's pattern analysis capabilities:
1. Setting up a model provider
   - Configuring the Anthropic model
   - Setting appropriate parameters for generation

2. Creating and applying validation rules
   - Length validation with character-based constraints
   - Content filtering for prohibited terms
   - Pattern detection using symmetry and repetition rules

3. Pattern Analysis Features
   - Text symmetry detection with configurable thresholds
   - Repetitive pattern identification with customizable parameters
   - Detailed pattern analysis reporting

4. Error Handling and Logging
   - Structured logging of validation results
   - Clear error reporting for failed validations
   - Detailed pattern match information

Usage:
    python basic_usage.py

Requirements:
    - Sifaka library with pattern rules support
    - Anthropic API key in environment variables
    - Python dotenv for environment management
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

# Configure logging with a consistent format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze text using multiple validation rules and pattern detection.

    This function applies four types of analysis:
    1. Length validation: Ensures text meets length requirements
    2. Content filtering: Checks for prohibited terms
    3. Symmetry analysis: Detects symmetric patterns in text
    4. Repetition analysis: Identifies repeated patterns

    Args:
        text (str): The text to analyze

    Returns:
        Dict[str, Any]: Analysis results containing:
            - length: Length validation results
            - content: Content filtering results
            - symmetry: Symmetry analysis results
            - repetition: Repetition analysis results

    Example:
        >>> results = analyze_text("The quick brown fox jumps over the lazy dog")
        >>> print(results["length"]["passed"])
        True
    """
    results = {}

    # Create basic validation rules with specific constraints
    length_rule = LengthRule(
        name="length_check",
        description="Checks if output length is within bounds",
        config={
            "min_length": 30,
            "max_length": 100,
            "unit": "characters",  # Using character-based length measurement
        },
    )

    prohibited_terms = ProhibitedContentRule(
        name="content_filter",
        description="Checks for prohibited or inappropriate content",
        config={
            "prohibited_terms": ["controversial", "inappropriate"],
            "case_sensitive": False,  # Case-insensitive term matching
        },
    )

    # Configure pattern detection rules with specific thresholds
    symmetry_rule = SymmetryRule(
        name="symmetry_check",
        description="Checks for text symmetry patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            metadata={
                "mirror_mode": "both",  # Check both horizontal and vertical symmetry
                "symmetry_threshold": 0.8,  # High threshold for strict symmetry
                "preserve_whitespace": True,  # Consider spacing in symmetry
                "preserve_case": True,  # Case-sensitive matching
                "ignore_punctuation": True,  # Ignore punctuation for better matching
            },
        ),
    )

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

    # Run length validation
    length_result = length_rule._validate_impl(text)
    results["length"] = {
        "passed": length_result.passed,
        "message": length_result.message,
        "metadata": length_result.metadata,
    }

    # Run content filtering
    content_result = prohibited_terms._validate_impl(text)
    results["content"] = {
        "passed": content_result.passed,
        "message": content_result.message,
        "metadata": content_result.metadata,
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
    Main function demonstrating basic usage of Sifaka's pattern analysis capabilities.

    This function:
    1. Sets up the model configuration
    2. Initializes the Anthropic provider
    3. Processes example texts through multiple analysis rules
    4. Logs detailed results of each analysis type
    """
    # Load environment variables (ANTHROPIC_API_KEY required)
    load_dotenv()

    # Initialize the Anthropic model with configuration
    config = ModelConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7,  # Balanced between creativity and consistency
        max_tokens=2000,  # Sufficient for most text generation tasks
    )

    model = AnthropicProvider(
        model_name="claude-3-haiku-20240307",
        config=config,
    )

    # Example texts demonstrating different pattern characteristics
    texts = [
        # Standard text with some repetition
        "The quick brown fox jumps over the lazy dog. The lazy dog lets the quick brown fox jump.",
        # Palindrome-like structure
        "A man, a plan, a canal: Panama!",
        # Simple text with natural repetition
        "This is a very simple text that should be easy to read and understand. It has some repeating words like read and understand.",
        # Complex text with technical terms
        "The quantum mechanical interpretation of molecular orbital theory requires advanced understanding of mathematical principles.",
        # Text with intentional repetition
        "Roses are red, red are roses, in the garden they grow, grow they in the garden.",
        # Text with mirror symmetry
        "Mirror mirror on the wall, who is the fairest of them all? All of them fairest the is who, wall the on mirror mirror?",
    ]

    # Process each text through the analysis pipeline
    for i, text in enumerate(texts, 1):
        logger.info("\n%s", "=" * 50)
        logger.info("Analyzing text %d: %s", i, text)

        # Perform multi-faceted analysis
        results = analyze_text(text)

        # Log length validation results
        logger.info("\nLength Check:")
        logger.info("- Passed: %s", results["length"]["passed"])
        logger.info("- Message: %s", results["length"]["message"])
        logger.info("- Details: %s", results["length"]["metadata"])

        # Log content filtering results
        logger.info("\nContent Check:")
        logger.info("- Passed: %s", results["content"]["passed"])
        logger.info("- Message: %s", results["content"]["message"])
        logger.info("- Details: %s", results["content"]["metadata"])

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
