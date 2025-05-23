#!/usr/bin/env python3
"""Example demonstrating the Sifaka classifiers.

This example shows how to use the various text classifiers available in Sifaka,
both standalone and integrated with validators in a chain.
"""

import sys
import os

# Add the parent directory to the path so we can import sifaka
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sifaka.classifiers import (
    BiasClassifier, 
    LanguageClassifier,
    ProfanityClassifier,
    SentimentClassifier,
    SpamClassifier,
    ToxicityClassifier,
    create_bias_validator,
    create_profanity_validator,
    create_sentiment_validator,
    create_toxicity_validator,
)
from sifaka.core.thought import Thought


def test_classifiers_standalone():
    """Test classifiers in standalone mode."""
    print("=" * 60)
    print("TESTING CLASSIFIERS STANDALONE")
    print("=" * 60)
    
    # Test texts
    test_texts = [
        "Hello, this is a friendly message!",
        "Men are naturally better at math than women.",
        "This is some damn good content, hell yeah!",
        "I love this product, it's absolutely amazing!",
        "FREE MONEY! Click here now! You won $1,000,000!",
        "You're an idiot and should kill yourself.",
        "Bonjour le monde! Comment allez-vous?",
        "This is a neutral statement about technology.",
    ]
    
    # Initialize classifiers
    classifiers = {
        "Bias": BiasClassifier(),
        "Language": LanguageClassifier(),
        "Profanity": ProfanityClassifier(),
        "Sentiment": SentimentClassifier(),
        "Spam": SpamClassifier(),
        "Toxicity": ToxicityClassifier(),
    }
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        print("-" * 50)
        
        for name, classifier in classifiers.items():
            try:
                result = classifier.classify(text)
                print(f"{name:10}: {result.label:12} (confidence: {result.confidence:.2f})")
            except Exception as e:
                print(f"{name:10}: ERROR - {e}")


def test_classifiers_with_validators():
    """Test classifiers integrated with validators."""
    print("\n" + "=" * 60)
    print("TESTING CLASSIFIERS WITH VALIDATORS")
    print("=" * 60)
    
    # Create validators
    validators = [
        create_bias_validator(threshold=0.6),
        create_profanity_validator(),
        create_sentiment_validator(forbidden_sentiments=["negative"]),
        create_toxicity_validator(threshold=0.5),
    ]
    
    # Test texts that should trigger different validators
    test_cases = [
        ("This is a perfectly fine message.", "Should pass all validators"),
        ("Women are too emotional to be good leaders.", "Should fail bias validator"),
        ("This is some damn bullshit content.", "Should fail profanity validator"),
        ("I hate this product, it's terrible and awful.", "Should fail sentiment validator"),
        ("You're an idiot and should go kill yourself.", "Should fail toxicity validator"),
    ]
    
    for text, description in test_cases:
        print(f"\nText: '{text}'")
        print(f"Expected: {description}")
        print("-" * 50)
        
        # Create a thought with the text
        thought = Thought(prompt="Test prompt", text=text)
        
        for validator in validators:
            try:
                result = validator.validate(thought)
                status = "PASS" if result.passed else "FAIL"
                print(f"{validator.name:20}: {status:4} - {result.message}")
            except Exception as e:
                print(f"{validator.name:20}: ERROR - {e}")


def test_scikit_learn_interface():
    """Test the scikit-learn compatible interface."""
    print("\n" + "=" * 60)
    print("TESTING SCIKIT-LEARN INTERFACE")
    print("=" * 60)
    
    # Test texts
    texts = [
        "This is a positive message!",
        "This is a negative message.",
        "This is neutral content.",
    ]
    
    # Test with sentiment classifier
    classifier = SentimentClassifier()
    
    print("Testing predict() method:")
    predictions = classifier.predict(texts)
    for text, prediction in zip(texts, predictions):
        print(f"'{text}' -> {prediction}")
    
    print("\nTesting predict_proba() method:")
    probabilities = classifier.predict_proba(texts)
    for text, probs in zip(texts, probabilities):
        print(f"'{text}' -> {probs}")
    
    print(f"\nAvailable classes: {classifier.get_classes()}")


def main():
    """Run all classifier examples."""
    print("Sifaka Classifiers Example")
    print("This example demonstrates text classification capabilities.")
    
    try:
        test_classifiers_standalone()
        test_classifiers_with_validators()
        test_scikit_learn_interface()
        
        print("\n" + "=" * 60)
        print("EXAMPLE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nNote: Some classifiers may show warnings about missing optional")
        print("dependencies (like scikit-learn, textblob, etc.). This is normal")
        print("and the classifiers will fall back to rule-based approaches.")
        
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
