#!/usr/bin/env python3
"""Example demonstrating the migrated validators and classifiers.

This example shows:
1. Using the new validators (length, content, format, classifier-based)
2. Using the new classifiers (sentiment analysis)
3. Integration with the SifakaEngine
4. Performance and logging features

Run this example to see the migrated components in action:
    python examples/validators_classifiers_example.py
"""

import asyncio
from pathlib import Path

# Add the project root to the path so we can import sifaka
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sifaka.core.engine import SifakaEngine
from sifaka.core.thought import SifakaThought
from sifaka.validators import (
    LengthValidator,
    ContentValidator,
    FormatValidator,
    ClassifierValidator,
    sentiment_validator,
    min_length_validator,
    max_length_validator,
    prohibited_content_validator,
    json_validator,
)
from sifaka.classifiers import (
    SentimentClassifier,
    CachedSentimentClassifier,
    create_sentiment_classifier,
)
from sifaka.utils import (
    configure_for_development,
    get_logger,
)

# Setup logging
configure_for_development()
logger = get_logger(__name__)


async def test_validators():
    """Test the various validators with sample text."""
    print("\n" + "="*60)
    print("TESTING VALIDATORS")
    print("="*60)
    
    # Sample texts for testing
    test_texts = [
        "This is a great example of positive text that should pass most validations!",
        "Bad terrible awful text",  # Negative sentiment
        "Hi",  # Too short
        "A" * 6000,  # Too long
        '{"valid": "json", "format": true}',  # Valid JSON
        '{"invalid": json}',  # Invalid JSON
        "Text with forbidden words: spam, scam, fraud",  # Prohibited content
    ]
    
    # Create validators
    validators = [
        min_length_validator(min_length=10),
        max_length_validator(max_length=5000),
        prohibited_content_validator(["spam", "scam", "fraud"]),
        json_validator(strict=False),  # Non-strict for text that might not be JSON
        sentiment_validator(forbidden_sentiments=["negative"], min_confidence=0.6),
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print("-" * 50)
        
        # Create a thought for testing
        thought = SifakaThought(prompt="Test prompt", max_iterations=1)
        thought.current_text = text
        
        for validator in validators:
            try:
                result = await validator.validate_async(thought)
                status = "✓ PASS" if result.passed else "✗ FAIL"
                print(f"  {status} {validator.name}: {result.message}")
                if not result.passed and result.suggestions:
                    print(f"    Suggestions: {result.suggestions[0]}")
            except Exception as e:
                print(f"  ✗ ERROR {validator.name}: {str(e)}")


async def test_classifiers():
    """Test the sentiment classifiers with sample text."""
    print("\n" + "="*60)
    print("TESTING CLASSIFIERS")
    print("="*60)
    
    # Sample texts with different sentiments
    test_texts = [
        "I absolutely love this amazing product! It's fantastic!",  # Positive
        "This is terrible and awful. I hate it completely.",  # Negative
        "The weather is okay today. Nothing special.",  # Neutral
        "",  # Empty text
        "The technical specifications include standard features.",  # Neutral/technical
    ]
    
    # Create classifiers
    classifiers = [
        SentimentClassifier(name="basic_sentiment"),
        CachedSentimentClassifier(name="cached_sentiment", cache_size=64),
        create_sentiment_classifier(cached=True),
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print("-" * 50)
        
        for classifier in classifiers:
            try:
                result = await classifier.classify_async(text)
                print(f"  {classifier.name}: {result.label} (confidence: {result.confidence:.3f})")
                print(f"    Method: {result.metadata.get('method', 'unknown')}")
                if hasattr(classifier, 'get_cache_info'):
                    cache_info = classifier.get_cache_info()
                    print(f"    Cache: {cache_info['hits']} hits, {cache_info['misses']} misses")
            except Exception as e:
                print(f"  ✗ ERROR {classifier.name}: {str(e)}")


async def test_integration():
    """Test integration with SifakaEngine."""
    print("\n" + "="*60)
    print("TESTING INTEGRATION WITH SIFAKA ENGINE")
    print("="*60)
    
    try:
        # Create engine with default dependencies (includes validators)
        engine = SifakaEngine()
        
        # Test prompts
        test_prompts = [
            "Write a positive review about renewable energy",  # Should pass
            "Write a negative rant about everything",  # May fail sentiment validation
            "Hi",  # May fail length validation
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest {i+1}: Processing prompt: '{prompt}'")
            print("-" * 50)
            
            try:
                with logger.performance_timer("full_thought_processing"):
                    thought = await engine.think(prompt, max_iterations=2)
                
                print(f"✓ SUCCESS: Generated {len(thought.current_text)} characters")
                print(f"  Final iteration: {thought.iteration}")
                print(f"  Validations: {len(thought.validations)}")
                print(f"  Critiques: {len(thought.critiques)}")
                
                # Show validation results
                if thought.validations:
                    print("  Validation results:")
                    for validation in thought.validations[-3:]:  # Show last 3
                        status = "✓" if validation.passed else "✗"
                        print(f"    {status} {validation.validator}: {validation.details.get('message', 'No message')}")
                
                # Show a preview of the generated text
                preview = thought.current_text[:100] + "..." if len(thought.current_text) > 100 else thought.current_text
                print(f"  Preview: {preview}")
                
            except Exception as e:
                print(f"✗ FAILED: {type(e).__name__}: {str(e)}")
                
    except Exception as e:
        print(f"✗ ENGINE CREATION FAILED: {type(e).__name__}: {str(e)}")


async def test_performance():
    """Test performance of validators and classifiers."""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE")
    print("="*60)
    
    # Create test text
    test_text = "This is a reasonably positive text that should work well for performance testing. " * 10
    
    # Create thought
    thought = SifakaThought(prompt="Test prompt", max_iterations=1)
    thought.current_text = test_text
    
    # Test validator performance
    print("\nValidator Performance:")
    print("-" * 30)
    
    validators = [
        LengthValidator(min_length=10, max_length=5000, name="length_test"),
        ContentValidator(prohibited=["spam", "scam"], name="content_test"),
        sentiment_validator(name="sentiment_test"),
    ]
    
    for validator in validators:
        times = []
        for _ in range(5):  # Run 5 times
            result = await validator.validate_async(thought)
            times.append(result.processing_time_ms)
        
        avg_time = sum(times) / len(times)
        print(f"  {validator.name}: {avg_time:.2f}ms average")
    
    # Test classifier performance
    print("\nClassifier Performance:")
    print("-" * 30)
    
    classifiers = [
        SentimentClassifier(name="basic_sentiment"),
        CachedSentimentClassifier(name="cached_sentiment"),
    ]
    
    for classifier in classifiers:
        times = []
        for _ in range(5):  # Run 5 times
            result = await classifier.classify_async(test_text)
            times.append(result.processing_time_ms)
        
        avg_time = sum(times) / len(times)
        print(f"  {classifier.name}: {avg_time:.2f}ms average")
        
        # Show cache stats for cached classifier
        if hasattr(classifier, 'get_cache_info'):
            cache_info = classifier.get_cache_info()
            print(f"    Cache hit rate: {cache_info['hit_rate']:.1%}")


async def main():
    """Run all tests."""
    logger.info("Starting validators and classifiers example")
    
    print("Sifaka Validators and Classifiers Example")
    print("=" * 60)
    print("This example demonstrates the migrated validators and classifiers:")
    print("- Length, content, format, and classifier-based validators")
    print("- Sentiment classification with caching")
    print("- Integration with SifakaEngine")
    print("- Performance testing and logging")
    
    try:
        await test_validators()
        await test_classifiers()
        await test_integration()
        await test_performance()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        logger.info("Validators and classifiers example completed successfully")
        
    except Exception as e:
        logger.error(
            "Example failed",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True
        )
        print(f"\nExample failed: {type(e).__name__} - {str(e)}")
        raise


if __name__ == "__main__":
    print("Note: This example requires API keys for the SifakaEngine integration test.")
    print("Set environment variables for your preferred providers:")
    print("- OPENAI_API_KEY")
    print("- ANTHROPIC_API_KEY") 
    print("- GOOGLE_API_KEY")
    print("- GROQ_API_KEY")
    print("\nRunning example...\n")
    
    asyncio.run(main())
