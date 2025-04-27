"""
Toxicity Rule Example for Sifaka.

This example demonstrates how to:
1. Create a custom ToxicityValidator that implements the RuleValidator protocol
2. Create a custom ToxicityRule that uses the validator
3. Pass validation feedback to a language model for content improvement

Usage:
    python toxicity_rule_example.py

Requirements:
    - Sifaka library with toxicity extras: pip install sifaka[toxicity]
    - OpenAI API key in environment variables
    - Python dotenv for environment management
"""

import logging
import os
from typing import Dict, Any, List, Protocol, runtime_checkable
from dotenv import load_dotenv

from sifaka.classifiers.toxicity import ToxicityClassifier, ToxicityConfig, ToxicityThresholds
from sifaka.classifiers.base import ClassifierConfig, ClassificationResult
from sifaka.rules.base import Rule, RuleConfig, RulePriority, RuleResult, RuleValidator
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models import OpenAIProvider
from sifaka.models.base import ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

            # Return the result with metadata
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

        # Create a validator using our classifier
        validator = ToxicityValidator(self.classifier)

        # Initialize the base class with our validator
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Validate output using the toxicity validator."""
        return self._validator.validate(output, **kwargs)


def main():
    """Run the toxicity rule example."""
    # Load environment variables
    load_dotenv()

    # Initialize OpenAI provider
    openai_provider = OpenAIProvider(
        model_name="gpt-4-turbo-preview",
        config=ModelConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=1000,
        ),
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
        config=RuleConfig(
            priority=RulePriority.HIGH,
            cache_size=100,
            cost=2.0,  # Higher cost due to ML model computation
        ),
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
            min_confidence=0.6,
        ),
    )

    # Example texts to validate
    example_texts = [
        "This is a normal text without any toxicity.",
        "I hate when people do stupid things like that.",
        "This product is absolutely terrible and worthless.",
        "Please consider being more careful with your words.",
        "I'm going to destroy your reputation with this review.",
    ]

    print("\n=== Toxicity Rule Example ===\n")
    print("Testing toxicity detection and improvement with multiple examples\n")

    for i, text in enumerate(example_texts, 1):
        print(f"\n--- Example {i} ---")
        print(f'Original text: "{text}"')

        # Validate with toxicity rule
        result = toxicity_rule.validate(text)

        print(f"Toxicity validation result: {result}")

        # Check the classification details
        classification_result = result.metadata.get("classification_result")
        if classification_result:
            label = classification_result.label
            confidence = classification_result.confidence
            all_scores = classification_result.metadata.get("all_scores", {})

            print(f"Detected label: {label}")
            print(f"Confidence: {confidence:.2f}")
            print("Toxicity scores:")
            for category, score in all_scores.items():
                print(f"  - {category}: {score:.3f}")

        # If text is identified as toxic, use critic to improve it
        if not result.passed:
            violations = [
                {
                    "rule": toxicity_rule.name,
                    "message": result.message,
                    "metadata": {
                        "label": classification_result.label,
                        "confidence": classification_result.confidence,
                        "scores": classification_result.metadata.get("all_scores", {}),
                    },
                }
            ]

            print("\nImproving text with LLM critic...")
            try:
                improved_text = critic.improve(text, violations)
                print(f'Improved text: "{improved_text}"')

                # Validate the improved text
                improved_result = toxicity_rule.validate(improved_text)
                improved_classification = improved_result.metadata.get("classification_result")

                print("\nRe-validation after improvement:")
                print(f"New toxicity label: {improved_classification.label}")
                print(f"New confidence: {improved_classification.confidence:.2f}")
                print(f"Passed validation: {improved_result.passed}")

            except Exception as e:
                print(f"Error improving text: {e}")
        else:
            print("Text passed validation, no improvement needed.")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
