"""
Example demonstrating how to prioritize rules by cost.

This example shows how to:
1. Assign different costs to rules
2. Use the prioritize_by_cost parameter in ValidationManager to execute cheaper rules first
3. Understand the impact of rule costs on execution order

This optimization can be useful when you have a mix of fast, cheap rules and slow,
expensive rules. By running the cheapest rules first, you can fail fast on simple
validations before running more complex and expensive rules.
"""

import os
import logging
from typing import List

from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.base import Rule, RuleConfig, RulePriority
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.content.prohibited import create_prohibited_content_rule
from sifaka.rules.formatting.style import create_style_rule, CapitalizationStyle
from sifaka.chain.managers.validation import ValidationManager
from sifaka.validation import ValidationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rule_cost_example")


def create_rules_with_costs() -> List[Rule]:
    """Create a set of rules with different costs."""
    rules = []

    # 1. Length rule - very fast and cheap
    length_rule = create_length_rule(
        min_words=10,
        max_words=1000,
        rule_id="length_rule",
        description="Ensures text is of appropriate length",
        config=RuleConfig(cost=1),  # Lowest cost
    )
    # Ensure the rule has the correct cost
    length_rule._config = RuleConfig(cost=1, params=length_rule.config.params)
    rules.append(length_rule)

    # 2. Style rule - medium cost
    style_rule = create_style_rule(
        capitalization=CapitalizationStyle.SENTENCE_CASE,
        rule_id="style_rule",
        description="Ensures text follows proper capitalization",
        config=RuleConfig(cost=5),  # Medium cost
    )
    # Ensure the rule has the correct cost
    style_rule._config = RuleConfig(cost=5, params=style_rule.config.params)
    rules.append(style_rule)

    # 3. Prohibited content rule - highest cost (simulating a complex rule)
    prohibited_rule = create_prohibited_content_rule(
        name="prohibited_content",
        terms=["harmful", "offensive", "illegal"],
        priority=RulePriority.HIGH,
        config=RuleConfig(cost=10),  # High cost
    )
    # Ensure the rule has the correct cost
    prohibited_rule._config = RuleConfig(cost=10, params=prohibited_rule.config.params)
    rules.append(prohibited_rule)

    return rules


def validate_with_prioritization(text: str, prioritize_by_cost: bool = False) -> None:
    """
    Validate text with an optional cost-based prioritization.

    Args:
        text: The text to validate
        prioritize_by_cost: Whether to prioritize rules by cost (cheapest first)
    """
    # Create rules
    rules = create_rules_with_costs()

    # Print the rules
    print("\nRules with costs:")
    for i, rule in enumerate(rules):
        print(f"{i+1}. {rule.name} (cost: {rule.config.cost})")

    # Create a ValidationManager with prioritize_by_cost
    validation_manager = ValidationManager[str](rules, prioritize_by_cost=prioritize_by_cost)

    # Print the execution order
    print("\nRules execution order:")
    for i, rule in enumerate(validation_manager.rules):
        print(f"{i+1}. {rule.name} (cost: {rule.config.cost})")

    # Validate the text
    print("\nRunning validation...")
    validation_result = validation_manager.validate(text)

    # Print results
    print("\nValidation results:")
    for i, rule_result in enumerate(validation_result.rule_results):
        status = "✅ Passed" if rule_result.passed else "❌ Failed"
        print(f"{i+1}. {validation_manager.rules[i].name}: {status} - {rule_result.message}")

    print(f"\nOverall validation {'passed' if validation_result.all_passed else 'failed'}")
    if not validation_result.all_passed:
        print(f"Error messages: {validation_manager.get_error_messages(validation_result)}")


def main():
    """Run the example."""
    # Example text
    text = "This is a sample text for validation. It has proper capitalization and adequate length. It contains no prohibited content."

    # Run with default rule order (no prioritization)
    print("\n=== Running validation without cost prioritization ===")
    validate_with_prioritization(text, prioritize_by_cost=False)

    # Run with cost prioritization
    print("\n=== Running validation with cost prioritization ===")
    validate_with_prioritization(text, prioritize_by_cost=True)

    # Example text that fails multiple rules
    problematic_text = "this text has improper capitalization and contains harmful content that should be detected. it violates two rules but should fail fast on the cheapest rule first."

    print("\n=== Running validation on problematic text with cost prioritization ===")
    validate_with_prioritization(problematic_text, prioritize_by_cost=True)

    print("\n=== Key insights ===")
    print("1. With cost prioritization, rules are sorted by cost (cheapest first)")
    print("2. This allows failing fast on simple validations before running more complex rules")
    print("3. Rules with lower costs (like length_rule) will run before rules with higher costs")
    print("4. This can improve performance when you have many rules or expensive validations")


if __name__ == "__main__":
    main()