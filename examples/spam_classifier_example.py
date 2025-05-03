"""
Spam Classifier Example

This example demonstrates how to use the SpamClassifier to detect spam content.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from sifaka.classifiers import SpamClassifier, ClassifierConfig
from sifaka.adapters.rules.classifier import create_classifier_rule


def run_spam_example():
    """Run a simple example of spam classification."""
    print("=== Spam Classifier Example ===")

    # Training data
    ham_texts = [
        "Hey, let's meet for coffee later today.",
        "Here's the report you requested.",
        "The project deadline has been extended to next Friday.",
        "Can you please review this document when you have time?",
        "Thank you for your feedback on the proposal.",
    ]

    spam_texts = [
        "CONGRATULATIONS! You've won $1,000,000! Click here to claim now!",
        "Limited time offer: 90% off luxury watches. Buy now!",
        "Your account has been suspended. Enter your password here to restore access.",
        "Make money fast! Work from home and earn $10,000 per week!",
        "Free pills! Lose 30 pounds in just one week with this miracle drug!",
    ]

    # Labels for training data
    ham_labels = ["ham"] * len(ham_texts)
    spam_labels = ["spam"] * len(spam_texts)

    # Create and train the classifier
    print("Creating and training spam classifier...")
    classifier = SpamClassifier.create_pretrained(
        texts=ham_texts + spam_texts,
        labels=ham_labels + spam_labels,
        name="example_spam_classifier",
        description="Example spam classifier",
        config=ClassifierConfig(
            labels=["ham", "spam"],
            min_confidence=0.6,
            params={
                "max_features": 1000,
                "use_bigrams": True,
            },
        ),
    )

    # Create a rule from the classifier
    spam_rule = create_classifier_rule(
        classifier=classifier,
        name="spam_rule",
        description="Ensures text is not spam",
        threshold=0.6,
        valid_labels=["ham"],
    )

    # Test with examples
    test_texts = [
        "Let's discuss the project timeline tomorrow.",
        "FREE OFFER! Get your FREE gift card now!",
        "Here are the meeting notes from yesterday.",
        "URGENT: Your account needs verification. Click here!",
    ]

    print("\n=== Testing Classifier ===")
    for i, text in enumerate(test_texts):
        result = classifier.classify(text)
        print(f'\nText {i+1}: "{text}"')
        print(f"Classification: {result.label} (confidence: {result.confidence:.2f})")

        # Test rule validation
        rule_result = spam_rule.validate(text)
        validation = "Passed" if rule_result.passed else "Failed"
        print(f"Rule validation: {validation} - {rule_result.message}")


if __name__ == "__main__":
    run_spam_example()
