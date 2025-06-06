#!/usr/bin/env python3
"""
Simple test script to verify the Transformers-based spam classifier works.
"""

import asyncio
import sys
import os

# Add the sifaka directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sifaka'))

async def test_spam_classifier():
    """Test the spam classifier with sample texts."""
    try:
        from sifaka.classifiers.spam import create_spam_classifier
        
        print("Creating spam classifier...")
        classifier = create_spam_classifier()
        
        # Test texts
        test_texts = [
            "Hi, how are you doing today?",  # Should be ham
            "URGENT! You have won $1,000,000! Click here now!",  # Should be spam
            "The meeting is scheduled for 3 PM tomorrow.",  # Should be ham
            "FREE MONEY! No strings attached! Act now!",  # Should be spam
        ]
        
        print("\nTesting spam classification:")
        print("-" * 50)
        
        for text in test_texts:
            result = await classifier.classify_async(text)
            print(f"Text: {text[:50]}...")
            print(f"Label: {result.label}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Spam probability: {result.metadata.get('spam_probability', 'N/A'):.3f}")
            print(f"Ham probability: {result.metadata.get('ham_probability', 'N/A'):.3f}")
            print("-" * 50)
            
        print("✅ Spam classifier test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure transformers is installed: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_spam_classifier())
    sys.exit(0 if success else 1)
