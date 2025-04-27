"""
Pattern Rules Usage Example for Sifaka.

This example demonstrates:
1. Using SymmetryRule for text symmetry validation
2. Using RepetitionRule for pattern detection
3. Combining pattern rules with other validation rules
4. Handling validation results and pattern analysis
"""

import logging
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from sifaka.models import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.rules import (
    LengthRule,
    ProhibitedContentRule,
    SymmetryRule,
    RepetitionRule,
)
from sifaka.rules.pattern_rules import (
    SymmetryConfig,
    RepetitionConfig,
)
from sifaka.critics import PromptCritic
from sifaka.critics.prompt import PromptCriticConfig
from sifaka.chain import Chain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load environment variables
    load_dotenv()

    # Initialize the model provider
    config = ModelConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7,
        max_tokens=2000,
    )

    model = AnthropicProvider(
        model_name="claude-3-haiku-20240307",
        config=config,
    )

    # Create pattern detection rules
    symmetry_rule = SymmetryRule(
        name="symmetry_check",
        description="Checks for text symmetry patterns",
        config=SymmetryConfig(
            mirror_mode="both",  # Check both horizontal and vertical symmetry
            symmetry_threshold=0.8,  # 80% symmetry required
            preserve_whitespace=True,  # Consider whitespace in symmetry
            preserve_case=True,  # Case-sensitive matching
        ),
    )

    repetition_rule = RepetitionRule(
        name="repetition_check",
        description="Detects repetitive patterns",
        config=RepetitionConfig(
            pattern_type="repeat",  # Look for repeated sequences
            pattern_length=3,  # Minimum length of pattern to detect
            max_occurrences=3,  # Maximum allowed repetitions
        ),
    )

    # Add basic validation rules
    length_rule = LengthRule(
        name="length_check",
        description="Checks if output length is within bounds",
        config={"min_length": 50, "max_length": 500},
    )

    prohibited_terms = ProhibitedContentRule(
        name="content_filter",
        description="Checks for prohibited or inappropriate content",
        config={"prohibited_terms": ["controversial", "inappropriate"]},
    )

    # Create a critic for improving outputs
    critic_config = PromptCriticConfig(
        name="pattern_critic",
        description="Improves text patterns and structure",
        system_prompt="""You are an expert at creating well-structured text with balanced patterns.
        Focus on:
        1. Creating visually appealing layouts
        2. Using repetition effectively without being excessive
        3. Maintaining symmetry where appropriate
        4. Keeping content clear and meaningful""",
        temperature=0.7,
        max_tokens=1000,
    )

    critic = PromptCritic(config=critic_config, model=model)

    # Create the chain with all components
    chain = Chain(
        model=model,
        rules=[symmetry_rule, repetition_rule, length_rule, prohibited_terms],
        critic=critic,
        max_attempts=3,
    )

    # Example prompts that test different pattern aspects
    prompts = [
        "Create a visually symmetric poem about nature",
        "Write a story with intentional repetitive elements",
        "Generate a balanced piece of text with mirrored structure",
    ]

    # Process each prompt
    for prompt in prompts:
        logger.info("\nProcessing prompt: %s", prompt)
        try:
            # Generate and validate content
            result = chain.run(prompt)
            logger.info("\nGenerated content:")
            logger.info(result)

            # Log validation results
            logger.info("\nValidation Results:")
            for rule_result in result.rule_results:
                logger.info("\n%s:", rule_result.name)
                logger.info("- Passed: %s", rule_result.passed)
                logger.info("- Message: %s", rule_result.message)
                if rule_result.metadata:
                    logger.info("- Details:")
                    for key, value in rule_result.metadata.items():
                        logger.info("  * %s: %s", key, value)

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
