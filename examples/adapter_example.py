"""
Adapter Pattern Example

This example demonstrates how to use Sifaka's adapter pattern to integrate with external frameworks
and create custom rule adapters.

The example shows:
1. Using the LangChain adapter to integrate with LangChain
2. Using the ClassifierAdapter to adapt classifiers as rules
3. Using the GuardrailsAdapter to integrate with Guardrails
4. Creating a custom adapter for rule composition
"""

import os
from typing import List, Dict, Any, Optional

# Import Sifaka components
from sifaka.adapters.langchain import LangChainAdapter
from sifaka.adapters.rules import ClassifierAdapter
from sifaka.adapters.rules.guardrails_adapter import GuardrailsAdapter
from sifaka.models.anthropic import AnthropicProvider
from sifaka.classifiers.toxicity import ToxicityClassifier
from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.base import Rule, RuleResult
from sifaka.utils.logging import get_logger

# Setup logging
logger = get_logger(__name__)

# Setup API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "demo-key")


def example_langchain_adapter():
    """Example using the LangChain adapter."""
    logger.info("Running LangChain adapter example")

    # Create a model provider
    model = AnthropicProvider(api_key=ANTHROPIC_API_KEY, model="claude-3-haiku")

    # Create a LangChain chain using the adapter
    chain = LangChainAdapter.create_chain(
        model=model,
        verbose=True,
        memory=True
    )

    # Use the chain
    result = chain.invoke({"input": "Write a short poem about AI"})
    logger.info(f"Chain result: {result}")

    # Access the underlying LangChain chain
    lc_chain = chain.get_underlying_chain()
    logger.info(f"Chain type: {type(lc_chain)}")

    return result


def example_classifier_adapter():
    """Example using the ClassifierAdapter."""
    logger.info("Running ClassifierAdapter example")

    # Create classifiers
    toxicity_classifier = ToxicityClassifier()
    sentiment_classifier = SentimentClassifier()

    # Adapt classifiers as rules
    toxicity_rule = ClassifierAdapter(
        classifier=toxicity_classifier,
        name="toxicity_rule",
        description="Checks if text contains toxic content",
        threshold=0.5
    )

    sentiment_rule = ClassifierAdapter(
        classifier=sentiment_classifier,
        name="sentiment_rule",
        description="Checks if text has positive sentiment",
        threshold=0.7,
        target_label="positive"
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


def example_guardrails_adapter():
    """Example using the GuardrailsAdapter."""
    logger.info("Running GuardrailsAdapter example")

    # Create a model provider
    model = AnthropicProvider(api_key=ANTHROPIC_API_KEY, model="claude-3-haiku")

    # Create rules to adapt for guardrails
    length_rule = create_length_rule(min_chars=20, max_chars=200)
    toxicity_rule = ClassifierAdapter(
        classifier=ToxicityClassifier(),
        name="toxicity_rule",
        threshold=0.5
    )

    # Create guardrails adapter with rules
    guardrails = GuardrailsAdapter(
        name="content_guardrails",
        rules=[length_rule, toxicity_rule]
    )

    # Test text samples
    test_inputs = [
        "Write a story about a robot",
        "Tell me how to hack a computer",
        "Hello"
    ]

    for prompt in test_inputs:
        logger.info(f"\nProcessing prompt: {prompt}")

        # Call model with guardrails
        try:
            result = guardrails.run(model=model, prompt=prompt)
            logger.info(f"Result: {result[:100]}...")
        except Exception as e:
            logger.info(f"Guardrails blocked: {str(e)}")


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

        # Combine metadata
        combined_metadata = {
            "rule_results": [
                {
                    "rule_name": rule.name,
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
        name="toxicity_rule",
        threshold=0.5
    )
    sentiment_rule = ClassifierAdapter(
        classifier=SentimentClassifier(),
        name="sentiment_rule",
        threshold=0.6,
        target_label="positive"
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


if __name__ == "__main__":
    logger.info("Starting adapter examples")

    try:
        # Run examples
        example_classifier_adapter()
        example_custom_adapter()

        # Only run these if API key is available
        if ANTHROPIC_API_KEY != "demo-key":
            example_langchain_adapter()
            example_guardrails_adapter()
        else:
            logger.info("Skipping examples that require API key")

    except Exception as e:
        logger.error(f"Error running examples: {str(e)}", exc_info=True)

    logger.info("Adapter examples completed")