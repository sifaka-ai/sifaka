#!/usr/bin/env python3
"""
Simple test script to verify the Transformers-based toxicity classifier works.
"""

import asyncio
import sys
import os

# Add the sifaka directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sifaka'))

async def test_toxicity_classifier():
    """Test the toxicity classifier with sample texts."""
    try:
        from sifaka.classifiers.toxicity import create_toxicity_classifier
        
        print("Creating toxicity classifier...")
        classifier = create_toxicity_classifier()
        
        # Test texts
        test_texts = [
            "Hello, how are you doing today?",  # Should be non_toxic
            "You are such an idiot and I hate you!",  # Should be toxic
            "Thanks for your help with this project.",  # Should be non_toxic
            "I hope you die and suffer!",  # Should be toxic
            "The weather is nice today.",  # Should be non_toxic
        ]
        
        print("\nTesting toxicity classification:")
        print("-" * 60)
        
        for text in test_texts:
            result = await classifier.classify_async(text)
            print(f"Text: {text}")
            print(f"Label: {result.label}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Toxic score: {result.metadata.get('toxic_score', 'N/A'):.3f}")
            print(f"Non-toxic score: {result.metadata.get('non_toxic_score', 'N/A'):.3f}")
            print("-" * 60)
            
        print("✅ Toxicity classifier test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure transformers is installed: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_toxicity_classifier())
    sys.exit(0 if success else 1)
