"""
Example: Simple Rule Usage

This example demonstrates how to create and use rules in Sifaka for text validation.
It shows how to use built-in rules and how to combine multiple rules.

Key concepts demonstrated:
1. Creating rules using factory functions
2. Validating text with rules
3. Handling validation results
4. Combining multiple rules

Requirements:
- Sifaka 0.1.0+

To run this example:
```bash
python -m sifaka.examples.rules.simple_rules_demo
```
"""

from typing import Dict, List

# Import Sifaka components
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.formatting.style import create_style_rule, CapitalizationStyle
from sifaka.rules.content.prohibited import create_prohibited_content_rule


# SECTION 1: Creating Individual Rules
# -----------------------------------


def demonstrate_length_rule():
    """Demonstrate the usage of a length rule."""
    print("\n=== Length Rule Example ===\n")

    # Create a length rule
    length_rule = create_length_rule(min_chars=10, max_chars=50, rule_id="length_constraint")

    # Test texts
    texts = [
        "Too short.",
        "This is a good length text that meets the requirements.",
        "This text is too long because it exceeds the maximum character limit that we have set for this example to demonstrate length validation.",
    ]

    # Validate each text
    for i, text in enumerate(texts):
        result = length_rule.validate(text)
        print(f"Text {i+1} ({len(text)} chars): {text}")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        print(f"Character count: {result.metadata['char_count']}")
        print(f"Word count: {result.metadata['word_count']}")
        print()


def demonstrate_style_rule():
    """Demonstrate the usage of a style rule."""
    print("\n=== Style Rule Example ===\n")

    # Create a style rule that requires sentence case
    style_rule = create_style_rule(
        capitalization=CapitalizationStyle.SENTENCE_CASE,
        require_end_punctuation=True,
        rule_id="style_constraint",
    )

    # Test texts
    texts = [
        "This is proper sentence case with punctuation.",
        "this starts with lowercase letter.",
        "This has no end punctuation",
        "ALL CAPS TEXT IS NOT SENTENCE CASE.",
    ]

    # Validate each text
    for i, text in enumerate(texts):
        result = style_rule.validate(text)
        print(f"Text {i+1}: {text}")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        print()


def demonstrate_prohibited_content_rule():
    """Demonstrate the usage of a prohibited content rule."""
    print("\n=== Prohibited Content Rule Example ===\n")

    # Create a prohibited content rule
    prohibited_rule = create_prohibited_content_rule(
        terms=["bad", "inappropriate", "offensive"], rule_id="content_constraint"
    )

    # Test texts
    texts = [
        "This is a good text with appropriate content.",
        "This text contains a bad word that should be flagged.",
        "Content with inappropriate or offensive language should be detected.",
    ]

    # Validate each text
    for i, text in enumerate(texts):
        result = prohibited_rule.validate(text)
        print(f"Text {i+1}: {text}")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        if not result.passed:
            print(f"Prohibited terms found: {result.metadata.get('matched_terms', [])}")
        print()


# SECTION 2: Combining Multiple Rules
# ---------------------------------


def validate_with_multiple_rules(text: str, rules: List) -> Dict:
    """
    Validate text with multiple rules.

    Args:
        text: The text to validate
        rules: List of rules to apply

    Returns:
        Dictionary with validation results
    """
    results = {}
    all_passed = True

    for rule in rules:
        result = rule.validate(text)
        results[rule._name] = result
        if not result.passed:
            all_passed = False

    return {"text": text, "all_passed": all_passed, "results": results}


def demonstrate_multiple_rules():
    """Demonstrate using multiple rules together."""
    print("\n=== Multiple Rules Example ===\n")

    # Create rules
    rules = [
        create_length_rule(min_chars=10, max_chars=100, rule_id="length_rule"),
        create_style_rule(capitalization=CapitalizationStyle.SENTENCE_CASE, rule_id="style_rule"),
        create_prohibited_content_rule(terms=["bad", "inappropriate"], rule_id="content_rule"),
    ]

    # Test texts
    texts = [
        "This is a good text that passes all rules.",
        "short",
        "This contains a bad word that should be flagged.",
        "this starts with lowercase and is inappropriate.",
    ]

    # Validate each text with all rules
    for i, text in enumerate(texts):
        print(f"Text {i+1}: {text}")

        validation = validate_with_multiple_rules(text, rules)
        print(f"All rules passed: {validation['all_passed']}")

        print("Individual rule results:")
        for rule_name, result in validation["results"].items():
            print(f"  - {rule_name}: {'Passed' if result.passed else 'Failed'} - {result.message}")

        print()


# SECTION 3: Main Example
# ----------------------


def main():
    """Run the main example."""
    print("Starting Simple Rule Example...")

    # Demonstrate individual rules
    demonstrate_length_rule()
    demonstrate_style_rule()
    demonstrate_prohibited_content_rule()

    # Demonstrate multiple rules
    demonstrate_multiple_rules()

    print("Example completed successfully!")


# Run the example if this file is executed directly
if __name__ == "__main__":
    main()
