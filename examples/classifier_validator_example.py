"""
Example demonstrating the use of classifier validators in Sifaka.

This example shows how to create and use classifier validators to validate text
based on classification results.
"""

from sifaka.classifiers import SentimentClassifier
from sifaka.validators import classifier_validator
from sifaka.chain import Chain
from sifaka.factories import create_model


def main():
    """Run the classifier validator example."""
    # Create a sentiment classifier
    sentiment_classifier = SentimentClassifier()
    
    # Create a classifier validator that requires positive sentiment
    positive_validator = classifier_validator(
        classifier=sentiment_classifier,
        threshold=0.7,  # Require high confidence
        valid_labels=["positive"],  # Only accept positive sentiment
    )
    
    # Create a classifier validator that requires non-negative sentiment
    non_negative_validator = classifier_validator(
        classifier=sentiment_classifier,
        threshold=0.6,  # Moderate confidence threshold
        invalid_labels=["negative"],  # Reject negative sentiment
    )
    
    # Create a model
    model = create_model("openai:gpt-3.5-turbo")
    
    # Example 1: Generate text that should have positive sentiment
    print("Example 1: Generating text with positive sentiment")
    result1 = (
        Chain()
        .with_model(model)
        .with_prompt("Write a short paragraph about something you love.")
        .validate_with(positive_validator)
        .run()
    )
    
    print(f"Text: {result1.text}")
    print(f"Passed validation: {result1.passed}")
    if not result1.passed:
        print(f"Validation message: {result1.validation_results[0].message}")
    print()
    
    # Example 2: Generate text that should not have negative sentiment
    print("Example 2: Generating text without negative sentiment")
    result2 = (
        Chain()
        .with_model(model)
        .with_prompt("Write a neutral description of a common household object.")
        .validate_with(non_negative_validator)
        .run()
    )
    
    print(f"Text: {result2.text}")
    print(f"Passed validation: {result2.passed}")
    if not result2.passed:
        print(f"Validation message: {result2.validation_results[0].message}")
    print()
    
    # Example 3: Direct validation without a chain
    print("Example 3: Direct validation of text")
    texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is just okay. Nothing special about it.",
        "I hate this. It's terrible and doesn't work at all."
    ]
    
    for text in texts:
        result = positive_validator.validate(text)
        print(f"Text: {text}")
        print(f"Passed validation: {result.passed}")
        print(f"Message: {result.message}")
        print(f"Details: {result.details}")
        print()


if __name__ == "__main__":
    main()
