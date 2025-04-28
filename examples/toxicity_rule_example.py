#!/usr/bin/env python3
"""
Toxicity Rule Example for Sifaka.

This example demonstrates:
1. Creating a custom ToxicityValidator
2. Creating a custom ToxicityRule that uses the validator
3. Using a critic to improve toxic content

Usage:
    python toxicity_rule_example.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
    - Sifaka toxicity extras: pip install sifaka[toxicity]
    - OpenAI API key in OPENAI_API_KEY environment variable
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
    sys.exit(1)

from sifaka.classifiers.toxicity import (
    ToxicityClassifier,
    ToxicityConfig,
    ToxicityThresholds,
)
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.base import Rule, RuleConfig, RulePriority, RuleResult, RuleValidator
from sifaka.utils.logging import get_logger

# Initialize logger from Sifaka
logger = get_logger(__name__)


# Create a validator class for toxicity detection
class ToxicityValidator(RuleValidator[str]):
    """Validator for toxicity detection."""

    def __init__(self, classifier: ToxicityClassifier):
        """Initialize the validator with a toxicity classifier."""
        self.classifier = classifier

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate text using the toxicity classifier."""
        try:
            # Classify the text
            result = self.classifier.classify(output)

            # Determine if it passes based on toxicity level
            passes = result.label == "non_toxic" or result.confidence < 0.5

            # Create an appropriate message
            if passes:
                message = f"No significant toxicity detected (Label: {result.label}, Score: {result.confidence:.2f})"
            else:
                message = f"Detected {result.label} content with confidence {result.confidence:.2f}"

            return RuleResult(
                passed=passes,
                message=message,
                metadata={"classification_result": result, "toxicity_type": result.label},
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Toxicity classification failed: {str(e)}",
                metadata={"error": str(e)},
            )

    def can_validate(self, output: str) -> bool:
        """Check if the validator can validate the output."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type:
        """Return the type of output that this validator can handle."""
        return str


# Create a custom rule that uses ToxicityClassifier
class ToxicityRule(Rule):
    """Custom rule that uses ToxicityClassifier to detect toxic content."""

    def __init__(
        self,
        name: str = "toxicity_rule",
        description: str = "Detects toxic content in text",
        config: RuleConfig = None,
        toxicity_config: ToxicityConfig = None,
    ):
        """Initialize the toxicity rule."""
        self.toxicity_config = toxicity_config or ToxicityConfig()
        self.classifier = ToxicityClassifier(
            name=f"{name}_classifier",
            description=f"Classifier for {name}",
            toxicity_config=self.toxicity_config,
        )

        # Initialize the base class with our validator
        super().__init__(name=name, description=description, config=config)

    def _create_default_validator(self) -> RuleValidator:
        """Create a default validator using our classifier."""
        return ToxicityValidator(self.classifier)

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Validate output using the toxicity validator."""
        return self._validator.validate(output, **kwargs)


def main():
    """Run the toxicity rule example."""
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    logger.info("Starting toxicity rule example...")

    # Initialize OpenAI provider
    openai_provider = OpenAIProvider(
        model_name="gpt-4-turbo-preview",
        config=ModelConfig(api_key=api_key, temperature=0.7, max_tokens=1000),
    )

    # Create a ToxicityRule with custom thresholds
    toxicity_config = ToxicityConfig(
        model_name="original",  # Options: "original", "unbiased", "multilingual"
        thresholds=ToxicityThresholds(
            severe_toxic=0.7,  # Higher threshold for severe toxicity
            threat=0.7,  # Higher threshold for threats
            general=0.5,  # General threshold for other categories
        ),
    )

    # Create our custom toxicity rule
    toxicity_rule = ToxicityRule(
        name="content_toxicity_validator",
        description="Validates text for toxic content",
        config=RuleConfig(priority=RulePriority.HIGH, cost=2.0),
        toxicity_config=toxicity_config,
    )

    # Create a prompt critic with OpenAI
    critic = PromptCritic(
        model=openai_provider,
        config=PromptCriticConfig(
            name="toxicity_improver",
            description="Improves text by removing toxic content",
            system_prompt="You are an expert editor that improves text by removing toxicity while preserving meaning.",
            temperature=0.7,
            max_tokens=1000,
        ),
    )

    # Example texts to validate (reduced set)
    example_texts = [
        "This is a normal text without any toxicity.",
        "I hate when people do stupid things like that.",
        "I'm going to destroy your reputation with this review.",
    ]

    logger.info("Testing toxicity detection and improvement...")

    for i, text in enumerate(example_texts, 1):
        logger.info(f"\nExample {i}: '{text}'")

        # Validate with toxicity rule
        result = toxicity_rule.validate(text)
        logger.info(f"Passed: {result.passed}, Message: {result.message}")

        # Get classification details
        if "classification_result" in result.metadata:
            classification = result.metadata["classification_result"]
            logger.info(f"Detected label: {classification.label}")
            logger.info(f"Confidence: {classification.confidence:.2f}")

            # Log top toxicity scores (if available)
            if "all_scores" in classification.metadata:
                scores = classification.metadata["all_scores"]
                top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                for category, score in top_scores:
                    if score > 0.1:  # Only show significant scores
                        logger.info(f"{category}: {score:.3f}")

        # If text is identified as toxic, use critic to improve it
        if not result.passed:
            logger.info("Text failed validation. Using critic to improve...")

            violations = [
                {
                    "rule": toxicity_rule.name,
                    "message": result.message,
                    "metadata": {
                        "label": classification.label,
                        "confidence": classification.confidence,
                    },
                }
            ]

            try:
                improved_text = critic.improve(text, violations)
                logger.info(f"Improved text: '{improved_text}'")

                # Validate the improved text
                improved_result = toxicity_rule.validate(improved_text)
                if "classification_result" in improved_result.metadata:
                    improved_class = improved_result.metadata["classification_result"]
                    logger.info(
                        f"New label: {improved_class.label}, Confidence: {improved_class.confidence:.2f}"
                    )
                    logger.info(f"Passed re-validation: {improved_result.passed}")
            except Exception as e:
                logger.error(f"Error improving text: {e}")
        else:
            logger.info("Text passed validation, no improvement needed.")

    logger.info("\nToxicity rule example completed.")


if __name__ == "__main__":
    main()
