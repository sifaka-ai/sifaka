#!/usr/bin/env python3
"""Example demonstrating classifiers with Hugging Face transformers.

This example shows:
1. Using toxicity classifiers from Hugging Face
2. Using sentiment classifiers from Hugging Face
3. Fallback mechanisms when transformers is not available
4. Performance comparison between different models
5. Integration with the SifakaEngine

Run this example to see the classifiers in action:
    python examples/pretrained_classifiers_example.py

Note: This example requires the transformers library:
    pip install transformers torch
"""

import asyncio
from pathlib import Path

# Add the project root to the path so we can import sifaka
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sifaka.classifiers.toxicity import (
    ToxicityClassifier,
    create_toxicity_classifier,
    TOXICITY_MODELS,
)
from sifaka.classifiers.sentiment import (
    SentimentClassifier,
    create_sentiment_classifier,
    SENTIMENT_MODELS,
)
from sifaka.validators.classifier import sentiment_validator
from sifaka.core.engine import SifakaEngine
from sifaka.core.thought import SifakaThought
from sifaka.utils import (
    configure_for_development,
    get_logger,
)

# Setup logging
configure_for_development()
logger = get_logger(__name__)


async def test_toxicity():
    """Test toxicity classifiers with various models."""
    print("\n" + "=" * 60)
    print("TESTING TOXICITY CLASSIFIERS")
    print("=" * 60)

    # Sample texts with different toxicity levels
    test_texts = [
        "I love this amazing product! It's fantastic!",  # Non-toxic
        "This is a reasonable discussion about policy.",  # Non-toxic
        "You are such an idiot and should disappear",  # Toxic
        "I hate you and wish you would die",  # Very toxic
        "That's a stupid idea, but let's discuss it",  # Mildly toxic
        "",  # Empty text
    ]

    # Test different models
    models_to_test = [
        "unitary/toxic-bert-base",
        "martin-ha/toxic-comment-model",
        # "unitary/unbiased-toxic-roberta",  # Uncomment if you want to test more models
    ]

    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        print(f"Description: {TOXICITY_MODELS.get(model_name, {}).get('description', 'Unknown')}")

        try:
            classifier = ToxicityClassifier(
                model_name=model_name, threshold=0.7, name=f"toxicity_{model_name.split('/')[-1]}"
            )

            for i, text in enumerate(test_texts):
                print(f"\nTest {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")

                try:
                    result = await classifier.classify_async(text)
                    print(f"  Result: {result.label} (confidence: {result.confidence:.3f})")
                    print(f"  Method: {result.metadata.get('method', 'unknown')}")
                    print(f"  Time: {result.processing_time_ms:.1f}ms")

                    if "toxic_score" in result.metadata:
                        print(f"  Toxic score: {result.metadata['toxic_score']:.3f}")

                except Exception as e:
                    print(f"  ✗ ERROR: {type(e).__name__} - {str(e)}")

        except Exception as e:
            print(f"✗ Failed to initialize {model_name}: {type(e).__name__} - {str(e)}")


async def test_sentiment():
    """Test sentiment classifiers with various models."""
    print("\n" + "=" * 60)
    print("TESTING SENTIMENT CLASSIFIERS")
    print("=" * 60)

    # Sample texts with different sentiments
    test_texts = [
        "I absolutely love this amazing product! It's fantastic!",  # Positive
        "This is terrible and awful. I hate it completely.",  # Negative
        "The weather is okay today. Nothing special.",  # Neutral
        "I'm so excited about this new opportunity!",  # Positive
        "I'm disappointed and frustrated with the service.",  # Negative
        "The technical specifications include standard features.",  # Neutral
        "",  # Empty text
    ]

    # Test different models
    models_to_test = [
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "distilbert-base-uncased-finetuned-sst-2-english",
        # "j-hartmann/emotion-english-distilroberta-base",  # Uncomment for emotion classification
    ]

    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        print(f"Description: {SENTIMENT_MODELS.get(model_name, {}).get('description', 'Unknown')}")

        try:
            classifier = SentimentClassifier(
                model_name=model_name, name=f"sentiment_{model_name.split('/')[-1]}"
            )

            for i, text in enumerate(test_texts):
                print(f"\nTest {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")

                try:
                    result = await classifier.classify_async(text)
                    print(f"  Result: {result.label} (confidence: {result.confidence:.3f})")
                    print(f"  Method: {result.metadata.get('method', 'unknown')}")
                    print(f"  Time: {result.processing_time_ms:.1f}ms")

                    if "all_scores" in result.metadata:
                        print(f"  All scores: {result.metadata['all_scores']}")

                except Exception as e:
                    print(f"  ✗ ERROR: {type(e).__name__} - {str(e)}")

        except Exception as e:
            print(f"✗ Failed to initialize {model_name}: {type(e).__name__} - {str(e)}")


async def test_fallback_mechanisms():
    """Test fallback mechanisms when transformers is not available."""
    print("\n" + "=" * 60)
    print("TESTING FALLBACK MECHANISMS")
    print("=" * 60)

    # Test with a non-existent model to trigger fallback
    print("\n--- Testing with non-existent model ---")

    try:
        classifier = ToxicityClassifier(model_name="non-existent/model", name="fallback_test")

        test_text = "This is a test of the fallback mechanism."
        result = await classifier.classify_async(test_text)

        print(f"Text: '{test_text}'")
        print(f"Result: {result.label} (confidence: {result.confidence:.3f})")
        print(f"Method: {result.metadata.get('method', 'unknown')}")
        print("✓ Fallback mechanism working correctly")

    except Exception as e:
        print(f"✗ Fallback test failed: {type(e).__name__} - {str(e)}")


async def test_performance_comparison():
    """Compare performance between different classification methods."""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE COMPARISON")
    print("=" * 60)

    test_text = "I love this product but the customer service was disappointing."

    # Test different approaches
    classifiers = [
        ("Sentiment", SentimentClassifier()),
        ("Toxicity", ToxicityClassifier()),
    ]

    for name, classifier in classifiers:
        print(f"\n--- {name} ---")

        # Run multiple times to get average performance
        times = []
        for _ in range(3):
            try:
                result = await classifier.classify_async(test_text)
                times.append(result.processing_time_ms)
            except Exception as e:
                print(f"  ✗ ERROR: {type(e).__name__} - {str(e)}")
                break

        if times:
            avg_time = sum(times) / len(times)
            print(f"  Average time: {avg_time:.1f}ms")
            print(f"  Method: {result.metadata.get('method', 'unknown')}")
            print(f"  Result: {result.label} (confidence: {result.confidence:.3f})")


async def test_integration_with_engine():
    """Test integration with SifakaEngine using pretrained classifiers."""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION WITH SIFAKA ENGINE")
    print("=" * 60)

    try:
        # Create a custom validator using sentiment classifier
        sentiment_classifier = create_sentiment_classifier(
            model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", cached=False
        )

        # Create validator that forbids negative sentiment
        validator = sentiment_validator(
            forbidden_sentiments=["negative"],
            min_confidence=0.6,
            cached=False,
            name="pretrained_sentiment_validation",
        )

        # Test prompts
        test_prompts = [
            "Write a positive review about renewable energy",  # Should pass
            "Write about something you dislike",  # May fail sentiment validation
        ]

        for i, prompt in enumerate(test_prompts):
            print(f"\nTest {i+1}: Processing prompt: '{prompt}'")
            print("-" * 50)

            try:
                # Create a thought and test validation
                thought = SifakaThought(prompt=prompt, max_iterations=1)
                thought.current_text = "Renewable energy is amazing and will help save our planet! I love solar panels."

                # Test the validator
                validation_result = await validator.validate_async(thought)

                status = "✓ PASS" if validation_result.passed else "✗ FAIL"
                print(f"{status} Validation: {validation_result.message}")
                print(f"  Score: {validation_result.score:.3f}")

                if not validation_result.passed and validation_result.suggestions:
                    print(f"  Suggestion: {validation_result.suggestions[0]}")

            except Exception as e:
                print(f"✗ FAILED: {type(e).__name__}: {str(e)}")

    except Exception as e:
        print(f"✗ ENGINE INTEGRATION FAILED: {type(e).__name__}: {str(e)}")


async def main():
    """Run all tests."""
    logger.info("Starting pretrained classifiers example")

    print("Pretrained Classifiers Example")
    print("=" * 60)
    print("This example demonstrates pretrained classifiers using Hugging Face transformers:")
    print("- Toxicity detection with BERT/RoBERTa models")
    print("- Sentiment analysis with various pretrained models")
    print("- Fallback mechanisms for robustness")
    print("- Performance comparison and integration testing")
    print("\nNote: First run may be slow due to model downloads.")

    try:
        await test_toxicity()
        await test_sentiment()
        await test_fallback_mechanisms()
        await test_performance_comparison()
        await test_integration_with_engine()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED!")
        print("=" * 60)
        print("\nKey Benefits of Pretrained Models:")
        print("✓ Much higher accuracy than rule-based approaches")
        print("✓ No training data required")
        print("✓ State-of-the-art performance")
        print("✓ Robust fallback mechanisms")
        print("✓ Easy to swap different models")

        logger.info("Pretrained classifiers example completed successfully")

    except Exception as e:
        logger.error(
            "Example failed",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        print(f"\nExample failed: {type(e).__name__} - {str(e)}")
        raise


if __name__ == "__main__":
    print("Note: This example requires the transformers library:")
    print("  pip install transformers torch")
    print("\nFirst run will download models (~100-500MB each)")
    print("Subsequent runs will be much faster.\n")

    asyncio.run(main())
