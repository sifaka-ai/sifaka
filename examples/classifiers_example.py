"""
Classifiers Example

This example demonstrates how to use the Sifaka classifiers system with the existing implementations.
It shows how to create classifiers using the factory functions and how to use them to classify text.
"""

import sys
import os
from typing import List, Dict, Any

# Add the Sifaka package to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the classifiers factory functions
from sifaka.classifiers import (
    create_toxicity_classifier,
    create_sentiment_classifier,
    create_profanity_classifier,
)


def main():
    """Run the example."""
    # Create classifiers
    toxicity_classifier = create_toxicity_classifier(
        general_threshold=0.5,
        cache_enabled=True,
        cache_size=100
    )
    
    sentiment_classifier = create_sentiment_classifier(
        positive_threshold=0.05,
        negative_threshold=-0.05,
        cache_enabled=True,
        cache_size=100
    )
    
    profanity_classifier = create_profanity_classifier(
        threshold=0.5,
        cache_enabled=True,
        cache_size=100
    )
    
    # Sample texts
    texts = [
        "I love this product! It's amazing and works great.",
        "This is terrible. I hate it and it doesn't work at all.",
        "This is okay. It works but could be better.",
        "This product is a piece of shit and doesn't work at all.",
        "The customer service was excellent and very helpful.",
        "The customer service was awful and completely useless."
    ]
    
    # Classify texts with toxicity classifier
    print("Toxicity Classification:")
    print("-" * 50)
    for i, text in enumerate(texts):
        try:
            result = toxicity_classifier.classify(text)
            print(f"Text {i+1}: {text}")
            print(f"  Label: {result.label}")
            print(f"  Confidence: {result.confidence:.2f}")
            if result.metadata:
                print(f"  Metadata: {result.metadata}")
            print()
        except Exception as e:
            print(f"Text {i+1}: {text}")
            print(f"  Error: {str(e)}")
            print()
    
    # Classify texts with sentiment classifier
    print("\nSentiment Classification:")
    print("-" * 50)
    for i, text in enumerate(texts):
        try:
            result = sentiment_classifier.classify(text)
            print(f"Text {i+1}: {text}")
            print(f"  Label: {result.label}")
            print(f"  Confidence: {result.confidence:.2f}")
            if result.metadata:
                print(f"  Metadata: {result.metadata}")
            print()
        except Exception as e:
            print(f"Text {i+1}: {text}")
            print(f"  Error: {str(e)}")
            print()
    
    # Classify texts with profanity classifier
    print("\nProfanity Classification:")
    print("-" * 50)
    for i, text in enumerate(texts):
        try:
            result = profanity_classifier.classify(text)
            print(f"Text {i+1}: {text}")
            print(f"  Label: {result.label}")
            print(f"  Confidence: {result.confidence:.2f}")
            if result.metadata:
                print(f"  Metadata: {result.metadata}")
            print()
        except Exception as e:
            print(f"Text {i+1}: {text}")
            print(f"  Error: {str(e)}")
            print()
    
    # Batch classification
    print("\nBatch Classification:")
    print("-" * 50)
    try:
        sentiment_results = sentiment_classifier.classify_batch(texts)
        
        for i, (text, sentiment) in enumerate(zip(texts, sentiment_results)):
            print(f"Text {i+1}: {text}")
            print(f"  Sentiment: {sentiment.label} ({sentiment.confidence:.2f})")
            print()
    except Exception as e:
        print(f"Batch classification error: {str(e)}")
    
    # Print classifier statistics
    print("\nClassifier Statistics:")
    print("-" * 50)
    print("Sentiment Classifier:")
    try:
        stats = sentiment_classifier.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Error: {str(e)}")


if __name__ == "__main__":
    main()
