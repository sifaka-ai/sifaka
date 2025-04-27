"""
Simple example demonstrating Sifaka with OpenAI in both validation-only and critic modes.

This example shows:
1. Setting up OpenAI as the provider
2. Using basic validation rules
3. Demonstrating both validation-only and critic modes
"""

import logging
import os
from dotenv import load_dotenv

from sifaka.models import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.rules import LengthRule, ProhibitedContentRule
from sifaka.critics import PromptCritic
from sifaka.critics.prompt import PromptCriticConfig
from sifaka.chain import Chain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load environment variables
    load_dotenv()

    # Initialize OpenAI provider
    config = ModelConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=100,  # Keep responses short
    )

    model = OpenAIProvider(
        model_name="gpt-4.1-mini",  # Using GPT-4.1 Mini for faster responses
        config=config,
    )

    # Create simple validation rules
    length_rule = LengthRule(
        name="length_check",
        description="Ensures the output is neither too short nor too long",
        config={"min_length": 30, "max_length": 100},  # Keep length requirements tight
    )

    prohibited_terms = ProhibitedContentRule(
        name="content_filter",
        description="Filters out inappropriate content",
        config={"prohibited_terms": ["inappropriate", "offensive", "controversial"]},
    )

    # Example prompts to test (made more specific to encourage brevity)
    prompts = [
        "Write a one-sentence product announcement for a new smartwatch",
    ]

    # Test validation-only mode
    for prompt in prompts:
        logger.info("\n%s", "=" * 50)
        logger.info("Testing prompt: %s", prompt)

        # Only use validation mode to minimize API calls
        logger.info("\nValidation-only Mode:")
        validation_chain = Chain(
            model=model,
            rules=[length_rule, prohibited_terms],
            max_attempts=1,  # Single attempt
        )

        try:
            result = validation_chain.run(prompt)
            logger.info("Success! Generated content:")
            logger.info(result.output)
            logger.info("\nValidation Results:")
            for rule_result in result.rule_results:
                logger.info("Passed: %s", rule_result.passed)
                logger.info("Message: %s", rule_result.message)
                if rule_result.metadata:
                    logger.info("Details: %s", rule_result.metadata)
        except ValueError as e:
            logger.info("Validation failed: %s", str(e))


if __name__ == "__main__":
    main()
