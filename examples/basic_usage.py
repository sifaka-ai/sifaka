"""
Basic usage example for Sifaka.

This example demonstrates:
1. Setting up a model provider
2. Creating rules for validation
3. Using critics for content improvement
4. Pattern detection with specialized rules (symmetry and repetition)
5. Handling validation results and pattern analysis

The example shows two modes of operation:
- Validation-only mode: Uses rules to validate output without attempting improvements
- Critic mode: Uses both rules and critics to validate and improve output

Usage:
    - For strict validation without improvements, use Chain without a critic
    - For validation with automatic improvements, add a critic to the Chain
"""

import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from sifaka.models import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.rules import LengthRule, ProhibitedContentRule, SymmetryRule, RepetitionRule
from sifaka.rules.base import RuleConfig, RulePriority, RuleResult, RuleValidator
from sifaka.rules.pattern_rules import SymmetryConfig, RepetitionConfig
from sifaka.critics import PromptCritic
from sifaka.critics.prompt import PromptCriticConfig
from sifaka.chain import Chain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    # Create basic validation rules
    length_rule = LengthRule(
        name="length_check",
        description="Checks if output length is within bounds",
        config={"min_length": 100, "max_length": 500},
    )

    prohibited_terms = ProhibitedContentRule(
        name="content_filter",
        description="Checks for prohibited or inappropriate content",
        config={"prohibited_terms": ["controversial", "inappropriate"]},
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

    # Create a critic for improving outputs
    critic_config = PromptCriticConfig(
        name="content_quality_critic",
        description="Improves text quality focusing on professionalism and clarity",
        system_prompt="You are an expert editor that improves text quality, focusing on professionalism, clarity, and effectiveness.",
        temperature=0.7,
        max_tokens=1000,
    )

    critic = PromptCritic(config=critic_config, model=model)

    # Create two chains to demonstrate different modes
    validation_chain = Chain(
        model=model,
        rules=[length_rule, prohibited_terms, symmetry_rule, repetition_rule],
        max_attempts=1,  # Single attempt since no critic
    )

    critic_chain = Chain(
        model=model,
        rules=[length_rule, prohibited_terms, symmetry_rule, repetition_rule],
        critic=critic,
        max_attempts=3,  # Multiple attempts for improvement
    )

    # Example prompts to test different pattern aspects
    prompts = [
        "Write a professional email about a project update that includes a repeating call-to-action phrase",
        "Create a palindromic story that reads the same forwards and backwards",
        "Generate a poem with alternating patterns in each stanza and mirror symmetry",
        "Write a technical document with consistent section structures and repeated key points",
    ]

    # Process each prompt with both chains
    for prompt in prompts:
        logger.info("\n%s", "=" * 50)
        logger.info("Processing prompt: %s", prompt)

        # Try validation-only mode
        logger.info("\nValidation-only Mode:")
        try:
            result = validation_chain.run(prompt)
            logger.info("Generated content:")
            logger.info(result)
            logger.info("\nValidation Results:")
            for rule_result in result.rule_results:
                logger.info("\n%s:", rule_result.name)
                logger.info("- Passed: %s", rule_result.passed)
                logger.info("- Message: %s", rule_result.message)
        except ValueError as e:
            logger.info("Validation failed: %s", str(e))

        # Try critic mode
        logger.info("\nCritic Mode:")
        try:
            result = critic_chain.run(prompt)
            logger.info("\nGenerated content:")
            logger.info(result)

            # Log validation results with focus on patterns
            logger.info("\nValidation Results:")
            for rule_result in result.rule_results:
                logger.info("\n%s:", rule_result.name)
                logger.info("- Passed: %s", rule_result.passed)
                logger.info("- Message: %s", rule_result.message)

                # Detailed pattern analysis logging
                if isinstance(rule_result.metadata, dict):
                    if "symmetry_score" in rule_result.metadata:
                        logger.info(
                            "- Symmetry Score: %.2f", rule_result.metadata["symmetry_score"]
                        )
                        if "symmetric_segments" in rule_result.metadata:
                            logger.info("- Symmetric Segments:")
                            for segment in rule_result.metadata["symmetric_segments"]:
                                logger.info("  * %s", segment)

                    if "patterns" in rule_result.metadata:
                        logger.info("- Detected Patterns:")
                        for pattern in rule_result.metadata["patterns"]:
                            logger.info("  * Pattern: %s", pattern["pattern"])
                            logger.info("    Occurrences: %d", pattern["occurrences"])
                            if "locations" in pattern:
                                logger.info("    Locations: %s", pattern["locations"])

            # Log critic's feedback if available
            if hasattr(result, "critique_details"):
                logger.info("\nCritic's Analysis:")
                logger.info("- Score: %s", result.critique_details.get("score"))
                logger.info("- Feedback: %s", result.critique_details.get("feedback"))
                if result.critique_details.get("issues"):
                    logger.info("- Issues:")
                    for issue in result.critique_details["issues"]:
                        logger.info("  * %s", issue)
                if result.critique_details.get("suggestions"):
                    logger.info("- Suggestions:")
                    for suggestion in result.critique_details["suggestions"]:
                        logger.info("  * %s", suggestion)

        except ValueError as e:
            logger.error("Validation failed: %s", str(e))


if __name__ == "__main__":
    main()
