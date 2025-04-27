"""
Basic usage example for Sifaka.
"""

import logging
from dotenv import load_dotenv

from sifaka import Reflector
from sifaka.models import AnthropicProvider
from sifaka.rules import ToxicityRule, BiasRule, HarmfulContentRule
from sifaka.critique import PromptCritique

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load environment variables
    load_dotenv()

    # Initialize the model provider
    model = AnthropicProvider(model_name="claude-3-haiku-20240307")

    # Create rules
    toxicity_rule = ToxicityRule(
        name="toxicity_check",
        description="Checks for toxic content",
        config={"toxicity_threshold": 0.5},
    )

    bias_rule = BiasRule(
        name="bias_check", description="Checks for biased content", config={"bias_threshold": 0.3}
    )

    harmful_content_rule = HarmfulContentRule(
        name="harmful_content_check", description="Checks for harmful content"
    )

    # Create a critic for improving outputs that fail validation
    critic = PromptCritique(model=model)

    # Create a reflector with rules and critique
    reflector = Reflector(
        name="content_validator",
        model=model,
        rules=[toxicity_rule, bias_rule, harmful_content_rule],
        critique=True,
        critic=critic,
    )

    # Example prompt
    prompt = """
    Write a professional and unbiased summary of recent advancements in AI safety research.
    """

    # Run the reflector
    logger.info("Running reflector with prompt: %s", prompt)
    result = reflector.reflect(prompt)

    # Print the results
    logger.info("\nOriginal output:")
    logger.info(result.original_output)

    if result.rule_violations:
        logger.info("\nRule violations:")
        for violation in result.rule_violations:
            logger.info("- %s: %s", violation.rule_name, violation.message)

    logger.info("\nFinal output:")
    logger.info(result.final_output)

    if result.trace:
        logger.info("\nTrace data:")
        for event in result.trace:
            logger.info("- %s: %s", event.stage, event.message)


if __name__ == "__main__":
    main()
