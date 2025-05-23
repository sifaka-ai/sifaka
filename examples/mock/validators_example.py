#!/usr/bin/env python3
"""Example usage of Sifaka validators.

This script demonstrates how to use the various validators available in Sifaka
to validate text content, format, and other properties.
"""

from sifaka.core.thought import Thought
from sifaka.validators import (
    ContentValidator,
    FormatValidator,
    ClassifierValidator,
    GuardrailsValidator,
    prohibited_content,
    json_format,
    markdown_format,
    custom_format,
)


def content_validation_example():
    """Demonstrate content validation."""
    print("Content Validation Example")
    print("-" * 30)

    # Create a content validator for prohibited words
    validator = prohibited_content(
        prohibited=["spam", "scam", "phishing", "malware"],
        case_sensitive=False,
        whole_word=True,
    )

    # Test texts
    texts = [
        "This is a legitimate business email.",
        "This email contains spam content.",
        "Click here for a great deal!",
        "Warning: potential phishing attempt detected.",
    ]

    for text in texts:
        thought = Thought(prompt="Validate email content", text=text)
        result = validator.validate(thought)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status}: {text[:50]}...")
        if not result.passed and result.issues:
            print(f"   Issues: {result.issues[0]}")

    print()


def format_validation_example():
    """Demonstrate format validation."""
    print("Format Validation Example")
    print("-" * 30)

    # JSON format validation
    json_validator = json_format()

    json_texts = [
        '{"name": "John", "age": 30, "city": "New York"}',
        '{"name": "John", "age": 30, "city":}',  # Invalid JSON
        "[1, 2, 3, 4, 5]",
    ]

    print("JSON Validation:")
    for text in json_texts:
        thought = Thought(prompt="Validate JSON", text=text)
        result = json_validator.validate(thought)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status}: {text[:40]}...")

    # Markdown format validation
    md_validator = markdown_format()

    md_texts = [
        "# Title\n\nThis is **bold** and *italic* text.",
        "# Title\n\n```python\nprint('hello')\n```",
        "# Title\n\n```python\nprint('hello')",  # Unmatched code block
    ]

    print("\nMarkdown Validation:")
    for text in md_texts:
        thought = Thought(prompt="Validate Markdown", text=text)
        result = md_validator.validate(thought)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status}: {text[:40].replace(chr(10), ' ')}...")

    # Custom format validation
    def is_email_format(text: str) -> bool:
        """Simple email format check."""
        return "@" in text and "." in text.split("@")[-1]

    email_validator = custom_format(is_email_format)

    email_texts = [
        "user@example.com",
        "invalid-email",
        "another.user@domain.org",
    ]

    print("\nEmail Format Validation:")
    for text in email_texts:
        thought = Thought(prompt="Validate email format", text=text)
        result = email_validator.validate(thought)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status}: {text}")

    print()


def classifier_validation_example():
    """Demonstrate classifier validation."""
    print("Classifier Validation Example")
    print("-" * 30)

    # Create a simple mock sentiment classifier
    class SimpleSentimentClassifier:
        """Mock sentiment classifier for demonstration."""

        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "love"]
        negative_words = ["bad", "terrible", "awful", "hate", "horrible", "worst"]

        def predict(self, X):
            """Predict sentiment labels."""
            predictions = []
            for text in X:
                text_lower = text.lower()
                pos_count = sum(1 for word in self.positive_words if word in text_lower)
                neg_count = sum(1 for word in self.negative_words if word in text_lower)

                if pos_count > neg_count:
                    predictions.append("positive")
                elif neg_count > pos_count:
                    predictions.append("negative")
                else:
                    predictions.append("neutral")

            return predictions

        def predict_proba(self, X):
            """Predict sentiment probabilities."""
            probabilities = []
            for text in X:
                text_lower = text.lower()
                pos_count = sum(1 for word in self.positive_words if word in text_lower)
                neg_count = sum(1 for word in self.negative_words if word in text_lower)

                total = pos_count + neg_count + 1  # +1 to avoid division by zero
                pos_prob = (pos_count + 0.1) / total
                neg_prob = (neg_count + 0.1) / total
                neu_prob = 1.0 - pos_prob - neg_prob

                probabilities.append([neg_prob, neu_prob, pos_prob])

            return probabilities

    # Create classifier validator
    classifier = SimpleSentimentClassifier()
    validator = ClassifierValidator(
        classifier=classifier,
        threshold=0.6,
        valid_labels=["positive", "neutral"],
        invalid_labels=["negative"],
    )

    # Test texts
    texts = [
        "This product is amazing and I love it!",
        "This is terrible and I hate it.",
        "The weather is okay today.",
        "Excellent service, wonderful experience!",
    ]

    for text in texts:
        thought = Thought(prompt="Analyze sentiment", text=text)
        result = validator.validate(thought)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status}: {text}")
        if result.score is not None:
            print(f"   Confidence: {result.score:.3f}")

    print()


def guardrails_validation_example():
    """Demonstrate GuardrailsAI validation."""
    print("GuardrailsAI Validation Example")
    print("-" * 30)

    try:
        # Try to create a GuardrailsAI validator
        # Note: This requires guardrails-ai to be installed and configured
        validator = GuardrailsValidator(
            validators=["GuardrailsPII"],  # Use the correct PII validator name
            validator_args={
                "GuardrailsPII": {
                    "entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]  # Required parameter
                }
            },
            api_key=None,  # Will use environment variable
        )

        # Test text with potential PII
        thought = Thought(
            prompt="Check for PII", text="My name is John Doe and my email is john.doe@example.com"
        )

        result = validator.validate(thought)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status}: PII detection test")
        print(f"   Message: {result.message}")

    except Exception as e:
        print("❌ GuardrailsAI not available or not configured")
        print(f"   Error: {str(e)}")
        print("   To use GuardrailsAI:")
        print("   1. Install: pip install guardrails-ai")
        print("   2. Set GUARDRAILS_API_KEY environment variable")
        print("   3. Configure validators as needed")

    print()


def main():
    """Run all validation examples."""
    print("Sifaka Validators Examples")
    print("=" * 50)
    print()

    content_validation_example()
    format_validation_example()
    classifier_validation_example()
    guardrails_validation_example()

    print("Examples completed!")
    print("\nFor more information, see the Sifaka documentation.")


if __name__ == "__main__":
    main()
