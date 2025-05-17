"""
Example demonstrating the use of classifier validators in Sifaka.

This example shows how to create and use classifier validators with different
classifiers to validate text based on classification results.

Note: Some classifiers require additional dependencies:
- LanguageClassifier: pip install langdetect
- ToxicityClassifier: pip install detoxify
- ProfanityClassifier: pip install better_profanity
- SpamClassifier: pip install scikit-learn numpy

If a dependency is missing, the classifier will raise an ImportError with
instructions on how to install the required package.
"""

from sifaka.classifiers import (
    SentimentClassifier,
    ToxicityClassifier,
    SpamClassifier,
    ProfanityClassifier,
    LanguageClassifier,
)
from sifaka.validators import classifier_validator
from sifaka.chain import Chain
from sifaka.factories import create_model


def print_validation_result(validator_name, text, result):
    """Print a validation result in a formatted way."""
    print(f"\n{validator_name} Validation:")
    print(f"Text: {text}")
    print(f"Passed: {result.passed}")
    print(f"Message: {result.message}")
    
    # Print details (limited to avoid excessive output)
    if result.details:
        print("Details:")
        for key, value in result.details.items():
            if isinstance(value, dict) and len(value) > 5:
                print(f"  {key}: {{{len(value)} items}}")
            elif isinstance(value, list) and len(value) > 5:
                print(f"  {key}: [{len(value)} items]")
            else:
                print(f"  {key}: {value}")


def main():
    """Run the classifier validators example."""
    # Create classifiers
    sentiment_classifier = SentimentClassifier()
    toxicity_classifier = ToxicityClassifier()
    spam_classifier = SpamClassifier()
    profanity_classifier = ProfanityClassifier()
    language_classifier = LanguageClassifier()
    
    # Create validators
    positive_validator = classifier_validator(
        classifier=sentiment_classifier,
        threshold=0.7,
        valid_labels=["positive"],
        name="Positive Sentiment Validator",
    )
    
    non_toxic_validator = classifier_validator(
        classifier=toxicity_classifier,
        threshold=0.5,
        invalid_labels=["toxic", "severe_toxic", "threat", "identity_attack"],
        name="Non-Toxic Content Validator",
    )
    
    non_spam_validator = classifier_validator(
        classifier=spam_classifier,
        threshold=0.6,
        invalid_labels=["spam"],
        name="Non-Spam Content Validator",
    )
    
    clean_language_validator = classifier_validator(
        classifier=profanity_classifier,
        threshold=0.5,
        valid_labels=["clean"],
        name="Clean Language Validator",
    )
    
    english_validator = classifier_validator(
        classifier=language_classifier,
        threshold=0.5,
        valid_labels=["en"],
        name="English Language Validator",
    )
    
    # Example texts
    texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is just okay. Nothing special about it.",
        "I hate this. It's terrible and doesn't work at all.",
        "Buy now! Limited time offer! 50% off! Act fast!",
        "Please review the attached document and provide feedback.",
        "Hello world! This is a test message.",
        "Bonjour le monde! C'est un message de test.",
        "Hola mundo! Este es un mensaje de prueba.",
        "This contains the word crap which might be considered mild profanity.",
    ]
    
    # Validate each text with each validator
    for text in texts:
        print("\n" + "=" * 80)
        print(f"Text: {text}")
        print("=" * 80)
        
        # Positive sentiment validation
        result = positive_validator.validate(text)
        print_validation_result("Positive Sentiment", text, result)
        
        # Non-toxic content validation
        result = non_toxic_validator.validate(text)
        print_validation_result("Non-Toxic Content", text, result)
        
        # Non-spam content validation
        result = non_spam_validator.validate(text)
        print_validation_result("Non-Spam Content", text, result)
        
        # Clean language validation
        result = clean_language_validator.validate(text)
        print_validation_result("Clean Language", text, result)
        
        # English language validation
        result = english_validator.validate(text)
        print_validation_result("English Language", text, result)
    
    # Example using validators in a chain
    print("\n\n" + "=" * 80)
    print("Using validators in a chain")
    print("=" * 80)
    
    try:
        # Create a model
        model = create_model("openai:gpt-3.5-turbo")
        
        # Create a chain with multiple validators
        chain = (
            Chain()
            .with_model(model)
            .with_prompt("Write a positive review of a product you like.")
            .validate_with(positive_validator)
            .validate_with(non_toxic_validator)
            .validate_with(non_spam_validator)
            .validate_with(clean_language_validator)
            .validate_with(english_validator)
        )
        
        # Run the chain
        result = chain.run()
        
        print(f"Chain result passed all validations: {result.passed}")
        print(f"Generated text: {result.text}")
        
        # Print validation results
        if not result.passed:
            print("\nFailed validations:")
            for validation_result in result.validation_results:
                if not validation_result.passed:
                    print(f"- {validation_result.message}")
    
    except Exception as e:
        print(f"Error running chain: {e}")
        print("Note: This example requires API keys for OpenAI to be configured.")


if __name__ == "__main__":
    main()
