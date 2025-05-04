"""
Language Rule Demo

This example demonstrates how to use the language rule to validate
that text is in the expected language.
"""

from sifaka.rules.content.language import create_language_rule


def run_language_rule_example():
    """Run a demonstration of language rule validation."""
    print("=== Language Rule Example ===")
    print("Note: This example requires the 'langdetect' package to be installed:")
    print("pip install langdetect")

    try:
        # Create a language rule that only allows English
        english_rule = create_language_rule(
            allowed_languages=["en"],
            threshold=0.7,
            name="english_only_rule",
            description="Validates that text is in English",
        )

        # Create a language rule that allows English or French
        multilingual_rule = create_language_rule(
            allowed_languages=["en", "fr"],
            threshold=0.6,
            name="english_or_french_rule",
            description="Validates that text is in English or French",
        )

        # Test texts
        english_text = "This is a sample text in English language."
        french_text = "Ceci est un exemple de texte en français."
        spanish_text = "Este es un ejemplo de texto en español."

        # Validate with English-only rule
        print("\nEnglish-only rule:")
        result = english_rule.validate(english_text)
        print(f"English text: {'✓ Passed' if result.passed else '✗ Failed'} - {result.message}")

        result = english_rule.validate(french_text)
        print(f"French text: {'✓ Passed' if result.passed else '✗ Failed'} - {result.message}")

        result = english_rule.validate(spanish_text)
        print(f"Spanish text: {'✓ Passed' if result.passed else '✗ Failed'} - {result.message}")

        # Validate with multilingual rule
        print("\nEnglish-or-French rule:")
        result = multilingual_rule.validate(english_text)
        print(f"English text: {'✓ Passed' if result.passed else '✗ Failed'} - {result.message}")

        result = multilingual_rule.validate(french_text)
        print(f"French text: {'✓ Passed' if result.passed else '✗ Failed'} - {result.message}")

        result = multilingual_rule.validate(spanish_text)
        print(f"Spanish text: {'✓ Passed' if result.passed else '✗ Failed'} - {result.message}")

    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install the required dependencies with: pip install langdetect")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    run_language_rule_example()
