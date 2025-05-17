"""
Example demonstrating the use of classifiers in Sifaka.

This example shows how to use the various classifiers provided by Sifaka
to categorize text into specific classes or labels.

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


def print_classification_result(classifier_name, text, result):
    """Print a classification result in a formatted way."""
    print(f"\n{classifier_name} Classification:")
    print(f"Text: {text}")
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence:.2f}")

    # Print metadata (limited to avoid excessive output)
    print("Metadata:")
    for key, value in result.metadata.items():
        if isinstance(value, dict) and len(value) > 5:
            print(f"  {key}: {{{len(value)} items}}")
        elif isinstance(value, list) and len(value) > 5:
            print(f"  {key}: [{len(value)} items]")
        else:
            print(f"  {key}: {value}")


def main():
    """Run the classifiers example."""
    # Create classifiers
    sentiment_classifier = SentimentClassifier()
    toxicity_classifier = ToxicityClassifier()
    spam_classifier = SpamClassifier()
    profanity_classifier = ProfanityClassifier()
    language_classifier = LanguageClassifier()

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

    # Classify each text with each classifier
    for text in texts:
        print("\n" + "=" * 80)
        print(f"Text: {text}")
        print("=" * 80)

        # Sentiment classification
        result = sentiment_classifier.classify(text)
        print_classification_result("Sentiment", text, result)

        # Toxicity classification
        result = toxicity_classifier.classify(text)
        print_classification_result("Toxicity", text, result)

        # Spam classification
        result = spam_classifier.classify(text)
        print_classification_result("Spam", text, result)

        # Profanity classification
        result = profanity_classifier.classify(text)
        print_classification_result("Profanity", text, result)

        # Language classification
        result = language_classifier.classify(text)
        print_classification_result("Language", text, result)


if __name__ == "__main__":
    main()
