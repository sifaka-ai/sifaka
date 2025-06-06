#!/usr/bin/env python3
"""
Simple test script to verify the Transformers-based bias classifier works.
"""

import asyncio
import sys
import os

# Add the sifaka directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sifaka'))

async def test_bias_classifier():
    """Test the bias classifier with sample texts."""
    try:
        from sifaka.classifiers.bias import create_bias_classifier
        
        print("Creating bias classifier...")
        classifier = create_bias_classifier()
        
        # Test texts
        test_texts = [
            "All people deserve equal treatment regardless of their background.",  # Should be unbiased
            "Women are naturally worse at math than men.",  # Should be biased
            "The research shows interesting results across different demographics.",  # Should be unbiased
            "People from that country are all lazy and unreliable.",  # Should be biased
            "The study included participants from diverse backgrounds.",  # Should be unbiased
        ]
        
        print("\nTesting bias classification:")
        print("-" * 70)
        
        for text in test_texts:
            result = await classifier.classify_async(text)
            print(f"Text: {text}")
            print(f"Label: {result.label}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Biased probability: {result.metadata.get('biased_probability', 'N/A'):.3f}")
            print(f"Unbiased probability: {result.metadata.get('unbiased_probability', 'N/A'):.3f}")
            print("-" * 70)
            
        print("✅ Bias classifier test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure transformers is installed: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_bias_classifier())
    sys.exit(0 if success else 1)
