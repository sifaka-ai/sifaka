"""
Adapter Pattern Example

This example demonstrates how to use Sifaka's adapter pattern to integrate with external frameworks
and create custom rule adapters.

The example shows:
1. Using the ClassifierAdapter to adapt classifiers as rules
2. Creating a custom adapter for rule composition
"""

import os
from typing import List

# Import Sifaka components
from sifaka.adapters.classifier import ClassifierAdapter
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.base import Rule, RuleResult, RuleValidator, RuleProtocol, RuleConfig
from sifaka.utils.logging import get_logger

# Setup logging
logger = get_logger(__name__)

# Setup API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "demo-key")


def example_classifier_adapter():
    """Example using the ClassifierAdapter."""
    logger.info("Running ClassifierAdapter example")

    # Create classifiers using factory functions (recommended)
    from sifaka.classifiers.toxicity import create_toxicity_classifier
    from sifaka.classifiers.sentiment import create_sentiment_classifier

    toxicity_classifier = create_toxicity_classifier(
        cache_size=100,  # Enable caching for better performance
    )

    sentiment_classifier = create_sentiment_classifier(
        positive_threshold=0.1,  # More strict positive threshold
        negative_threshold=-0.1,  # More strict negative threshold
        cache_size=100,  # Enable caching for better performance
    )

    # Adapt classifiers as rules
    toxicity_rule = ClassifierAdapter(
        classifier=toxicity_classifier,
        threshold=0.5,
        valid_labels=["non_toxic"],  # Set valid labels for toxicity classifier
    )

    sentiment_rule = ClassifierAdapter(
        classifier=sentiment_classifier, threshold=0.7, valid_labels=["positive"]
    )

    # Use the adapted rules
    test_texts = [
        "I love this product, it's amazing!",
        "This is the worst thing I've ever seen.",
        "You are stupid and I hate you.",
        "The weather is nice today.",
    ]

    for text in test_texts:
        logger.info(f"\nAnalyzing: {text}")

        # Check toxicity
        toxicity_result = toxicity_rule.validate(text)
        logger.info(f"Toxicity check: {'Passed' if toxicity_result.passed else 'Failed'}")
        logger.info(f"Message: {toxicity_result.message}")

        # Check sentiment
        sentiment_result = sentiment_rule.validate(text)
        logger.info(f"Sentiment check: {'Passed' if sentiment_result.passed else 'Failed'}")
        logger.info(f"Message: {sentiment_result.message}")


class CompositeRuleAdapter(RuleProtocol):
    """
    Custom adapter that composes multiple rules with logical operations.

    This demonstrates how to create a custom adapter that works with existing rules.
    """

    def __init__(
        self,
        rules: List[Rule],
        name: str = "composite_rule",
        description: str = "Composite rule that combines multiple rules",
        operator: str = "AND",
        config: RuleConfig = None,
        **kwargs,
    ):
        """
        Initialize the composite rule adapter.

        Args:
            rules: List of rules to compose
            name: Name of the rule
            description: Description of the rule
            operator: Logical operator to use ('AND' or 'OR')
            config: Optional rule configuration
        """
        self._name = name
        self._description = description
        self._config = config or RuleConfig()
        self.rules = rules
        self.operator = operator.upper()

        if self.operator not in ["AND", "OR"]:
            raise ValueError(f"Operator must be 'AND' or 'OR', got {operator}")

    @property
    def name(self) -> str:
        """Get the rule name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the rule description."""
        return self._description

    @property
    def config(self) -> RuleConfig:
        """Get the rule configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Run all rules and combine results according to the operator."""
        results = [rule.validate(text, **kwargs) for rule in self.rules]

        if not results:
            return RuleResult(
                passed=True, message="No rules to validate with", metadata={"rules": []}
            )

        if self.operator == "AND":
            passed = all(r.passed for r in results)
        else:  # OR
            passed = any(r.passed for r in results)

        # Collect messages from failed rules
        messages = [r.message for r in results if not r.passed]
        if not messages and passed:
            messages = ["All validations passed"]

        # Combine metadata with safe rule name access
        combined_metadata = {
            "rule_results": [
                {
                    "rule_name": getattr(
                        rule, "name", type(rule).__name__
                    ),  # Safely get name or use class name
                    "passed": result.passed,
                    "message": result.message,
                }
                for rule, result in zip(self.rules, results)
            ],
            "operator": self.operator,
        }

        return RuleResult(
            passed=passed,
            message="; ".join(messages) if messages else "Validation passed",
            metadata=combined_metadata,
        )


def example_custom_adapter():
    """Example using a custom adapter."""
    logger.info("Running custom adapter example")

    # Create rules to compose
    length_rule = create_length_rule(min_chars=10, max_chars=100)

    # Create classifiers using factory functions (recommended)
    from sifaka.classifiers.toxicity import create_toxicity_classifier
    from sifaka.classifiers.sentiment import create_sentiment_classifier

    toxicity_rule = ClassifierAdapter(
        classifier=create_toxicity_classifier(cache_size=100),
        threshold=0.5,
        valid_labels=["non_toxic"],  # Set valid labels for toxicity classifier
    )

    sentiment_rule = ClassifierAdapter(
        classifier=create_sentiment_classifier(
            positive_threshold=0.1, negative_threshold=-0.1, cache_size=100
        ),
        threshold=0.6,
        valid_labels=["positive"],
    )

    # Create composite rules with different operators
    and_rule = CompositeRuleAdapter(
        rules=[length_rule, toxicity_rule],
        name="safe_content_rule",
        description="Content must be long enough and non-toxic",
        operator="AND",
    )

    or_rule = CompositeRuleAdapter(
        rules=[length_rule, sentiment_rule],
        name="engaging_content_rule",
        description="Content must be either substantive or positive",
        operator="OR",
    )

    # Test texts
    test_texts = [
        "I love this product, it's amazing and I would recommend it to everyone!",
        "This is the worst.",
        "You are stupid and I hate you for making such a terrible product.",
        "OK",
    ]

    for text in test_texts:
        logger.info(f"\nAnalyzing: {text}")

        try:
            # Test AND rule
            and_result = and_rule.validate(text)
            logger.info(f"AND rule: {'Passed' if and_result.passed else 'Failed'}")
            logger.info(f"Message: {and_result.message}")

            # Test OR rule
            or_result = or_rule.validate(text)
            logger.info(f"OR rule: {'Passed' if or_result.passed else 'Failed'}")
            logger.info(f"Message: {or_result.message}")

            # Print individual rule results from the AND composite
            for rule_result in and_result.metadata["rule_results"]:
                logger.info(
                    f"- {rule_result['rule_name']}: {'Passed' if rule_result['passed'] else 'Failed'}"
                )
        except Exception as e:
            logger.error(f"Error processing text '{text}': {str(e)}")


if __name__ == "__main__":
    logger.info("Starting adapter examples")

    try:
        # Run examples
        example_classifier_adapter()
        example_custom_adapter()
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}", exc_info=True)

    logger.info("Adapter examples completed")
