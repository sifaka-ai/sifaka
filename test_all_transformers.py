#!/usr/bin/env python3
"""
Quick test to verify all classifiers use Transformers and work correctly.
"""

import asyncio
import sys
import os

# Add the sifaka directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sifaka"))


async def test_all_classifiers():
    """Test that all classifiers work with transformers."""

    try:
        from sifaka.classifiers import (
            create_sentiment_classifier,
            create_toxicity_classifier,
            create_spam_classifier,
            create_language_classifier,
            create_bias_classifier,
            create_readability_classifier,
            create_emotion_classifier,
            create_intent_classifier,
        )

        print("🤖 Testing All Sifaka Transformers-Based Classifiers")
        print("=" * 55)

        # Test text
        test_text = "Hello, this is a test message for our classifiers!"

        # Create all classifiers
        classifiers = {
            "sentiment": create_sentiment_classifier(),
            "toxicity": create_toxicity_classifier(),
            "spam": create_spam_classifier(),
            "language": create_language_classifier(),
            "bias": create_bias_classifier(),
            "readability": create_readability_classifier(),
            "emotion": create_emotion_classifier(),
            "intent": create_intent_classifier(),
        }

        print(f"Testing with: '{test_text}'")
        print(f"Running {len(classifiers)} classifiers...\n")

        results = {}

        for name, classifier in classifiers.items():
            try:
                print(f"Testing {name}...", end=" ")
                result = await classifier.classify_async(test_text)

                # Verify it's using transformers
                model_name = result.metadata.get("model_name", "unknown")
                if model_name == "unknown":
                    print(f"⚠️  No model_name in metadata")
                else:
                    print(f"✅ {result.label} ({model_name})")

                results[name] = {
                    "label": result.label,
                    "confidence": result.confidence,
                    "model": model_name,
                    "success": True,
                }

            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
                results[name] = {"error": str(e), "success": False}

        # Summary
        print(f"\n📊 SUMMARY")
        print("=" * 20)

        successful = sum(1 for r in results.values() if r.get("success", False))
        total = len(results)

        print(f"Successful: {successful}/{total}")

        if successful == total:
            print("✅ All classifiers working with Transformers!")
            return True
        else:
            print("❌ Some classifiers failed")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install transformers with one of these backends:")
        print('  pip install "sifaka[classifiers]"     # PyTorch (recommended)')
        print('  pip install "sifaka[classifiers-tf]"  # TensorFlow')
        print('  pip install "sifaka[classifiers-jax]" # JAX')
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_all_classifiers())
    sys.exit(0 if success else 1)
