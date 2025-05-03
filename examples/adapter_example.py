"""
Adapter Pattern Example

This example demonstrates how to use Sifaka's adapter pattern to integrate with external frameworks
and create custom rule adapters.

The example shows:
1. Using the ClassifierAdapter to adapt classifiers as rules
2. Creating a custom adapter for rule composition
"""

import os
from typing import List, Dict, Any, Optional

# Import Sifaka components
from sifaka.adapters.rules import ClassifierAdapter
from sifaka.models.anthropic import AnthropicProvider
from sifaka.classifiers.toxicity import ToxicityClassifier
from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.base import Rule, RuleResult, RuleValidator
from sifaka.utils.logging import get_logger

# Setup logging
logger = get_logger(__name__)

# Setup API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "demo-key")


def example_classifier_adapter():
    """Example using the ClassifierAdapter."""
    logger.info("Running ClassifierAdapter example")

    # Create classifiers
    toxicity_classifier = ToxicityClassifier()
    sentiment_classifier = SentimentClassifier()

    # Adapt classifiers as rules
    toxicity_rule = ClassifierAdapter(
        classifier=toxicity_classifier,
        threshold=0.5,
        valid_labels=["non_toxic"]  # Set valid labels for toxicity classifier
    )

    sentiment_rule = ClassifierAdapter(
        classifier=sentiment_classifier,
        threshold=0.7,
        valid_labels=["positive"]
    )

    # Use the adapted rules
    test_texts = [
        "I love this product, it's amazing!",
        "This is the worst thing I've ever seen.",
        "You are stupid and I hate you.",
        "The weather is nice today."
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


class CompositeRuleAdapter(Rule):
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
        **kwargs
    ):
        """
        Initialize the composite rule adapter.

        Args:
            rules: List of rules to compose
            name: Name of the rule
            description: Description of the rule
            operator: Logical operator to use ('AND' or 'OR')
        """
        super().__init__(name=name, description=description, **kwargs)
        self.rules = rules
        self.operator = operator.upper()

        if self.operator not in ["AND", "OR"]:
            raise ValueError(f"Operator must be 'AND' or 'OR', got {operator}")

    def _create_default_validator(self) -> RuleValidator:
        """
        Create a default validator for this rule.

        Required implementation for the abstract Rule class.

        Returns:
            A validator that uses the validate method
        """
        class CompositeValidator(RuleValidator):
            def __init__(self, rule):
                self.rule = rule

            def validate(self, text: str, **kwargs) -> RuleResult:
                return self.rule.validate(text, **kwargs)

        return CompositeValidator(self)

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Run all rules and combine results according to the operator."""
        results = [rule.validate(text, **kwargs) for rule in self.rules]

        if not results:
            return RuleResult(
                passed=True,
                message="No rules to validate with",
                metadata={"rules": []}
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
                    "rule_name": getattr(rule, "name", type(rule).__name__),  # Safely get name or use class name
                    "passed": result.passed,
                    "message": result.message
                }
                for rule, result in zip(self.rules, results)
            ],
            "operator": self.operator
        }

        return RuleResult(
            passed=passed,
            message="; ".join(messages) if messages else "Validation passed",
            metadata=combined_metadata
        )


def example_custom_adapter():
    """Example using a custom adapter."""
    logger.info("Running custom adapter example")

    # Create rules to compose
    length_rule = create_length_rule(min_chars=10, max_chars=100)
    toxicity_rule = ClassifierAdapter(
        classifier=ToxicityClassifier(),
        threshold=0.5,
        valid_labels=["non_toxic"]  # Set valid labels for toxicity classifier
    )
    sentiment_rule = ClassifierAdapter(
        classifier=SentimentClassifier(),
        threshold=0.6,
        valid_labels=["positive"]
    )

    # Create composite rules with different operators
    and_rule = CompositeRuleAdapter(
        rules=[length_rule, toxicity_rule],
        name="safe_content_rule",
        description="Content must be long enough and non-toxic",
        operator="AND"
    )

    or_rule = CompositeRuleAdapter(
        rules=[length_rule, sentiment_rule],
        name="engaging_content_rule",
        description="Content must be either substantive or positive",
        operator="OR"
    )

    # Test texts
    test_texts = [
        "I love this product, it's amazing and I would recommend it to everyone!",
        "This is the worst.",
        "You are stupid and I hate you for making such a terrible product.",
        "OK"
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
                logger.info(f"- {rule_result['rule_name']}: {'Passed' if rule_result['passed'] else 'Failed'}")
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