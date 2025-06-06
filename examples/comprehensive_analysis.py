#!/usr/bin/env python3
"""
Comprehensive Analysis Example

This example demonstrates running all available Sifaka classifiers on a static thought
to showcase the full range of text analysis capabilities. All classifiers now use
Hugging Face Transformers for state-of-the-art performance.

Requirements:
- Python 3.9+ (supports 3.9, 3.10, 3.11, 3.12)
- transformers>=4.52.0
- Backend: torch>=2.0.0 OR tensorflow>=2.15.0 OR jax>=0.4.0

Install with:
# Default PyTorch backend (recommended)
pip install "sifaka[classifiers]"

# Alternative backends
pip install "sifaka[classifiers-tf]"  # TensorFlow
pip install "sifaka[classifiers-jax]" # JAX

# Or with uv (recommended)
uv pip install "sifaka[classifiers]"
"""

import asyncio
import sys
import os
from typing import Dict, Any
import json
from datetime import datetime

# Add the sifaka directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

# Sample thoughts to analyze
SAMPLE_THOUGHTS = {
    "positive_review": """
    I absolutely love this new restaurant! The food was incredible, the service was
    outstanding, and the atmosphere was perfect for a romantic dinner. The chef clearly
    knows what they're doing - every dish was expertly prepared and beautifully presented.
    I can't wait to come back and try more items from their menu. Highly recommended!
    """,
    "technical_explanation": """
    Machine learning algorithms work by finding patterns in large datasets. Neural networks,
    inspired by the human brain, use interconnected nodes to process information. Deep learning
    uses multiple layers to extract increasingly complex features from raw data. This approach
    has revolutionized fields like computer vision, natural language processing, and speech
    recognition.
    """,
    "customer_complaint": """
    I'm extremely disappointed with my recent purchase. The product arrived damaged,
    the customer service was unhelpful, and I still haven't received a refund after
    three weeks. This is completely unacceptable and I will never shop here again.
    I'm considering filing a complaint with the Better Business Bureau.
    """,
    "multilingual_greeting": """
    Hello everyone! Bonjour tout le monde! ¡Hola a todos! Guten Tag allerseits!
    こんにちは皆さん！你好大家！مرحبا بالجميع! We're excited to welcome people
    from all around the world to our international conference.
    """,
    "simple_instruction": """
    To make a peanut butter sandwich, you need bread, peanut butter, and jelly.
    First, take two slices of bread. Then, spread peanut butter on one slice.
    Next, spread jelly on the other slice. Finally, put the slices together.
    Your sandwich is ready to eat!
    """,
}


async def analyze_text_comprehensive(text: str, text_name: str) -> Dict[str, Any]:
    """Run all available classifiers on the given text."""

    print(f"\n{'='*60}")
    print(f"ANALYZING: {text_name.upper()}")
    print(f"{'='*60}")
    print(f"Text: {text.strip()[:100]}...")
    print(f"Length: {len(text)} characters")

    results = {
        "text_name": text_name,
        "text": text.strip(),
        "text_length": len(text),
        "timestamp": datetime.now().isoformat(),
        "analyses": {},
    }

    # Create all classifiers (using cached versions for better performance)
    classifiers = {
        "sentiment": create_sentiment_classifier(cached=True),
        "toxicity": create_toxicity_classifier(cached=True),
        "spam": create_spam_classifier(cached=True),
        "language": create_language_classifier(cached=True),
        "bias": create_bias_classifier(cached=True),
        "readability": create_readability_classifier(cached=True),
        "emotion": create_emotion_classifier(cached=True),
        "intent": create_intent_classifier(cached=True),
    }

    print(f"\nRunning {len(classifiers)} classifiers...")

    # Run each classifier
    for name, classifier in classifiers.items():
        try:
            print(f"  • {name.capitalize()}...", end=" ")
            result = await classifier.classify_async(text)

            # Extract key information
            analysis = {
                "label": result.label,
                "confidence": round(result.confidence, 3),
                "processing_time_ms": round(result.processing_time_ms, 2),
                "metadata": {
                    k: v
                    for k, v in result.metadata.items()
                    if k in ["model_name", "language_name", "grade_level", "detected_language"]
                },
            }

            results["analyses"][name] = analysis
            print(f"✅ {result.label} ({result.confidence:.3f})")

        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}...")
            results["analyses"][name] = {"error": str(e), "label": "error", "confidence": 0.0}

    return results


async def main():
    """Run comprehensive analysis on all sample thoughts."""
    print("🤖 Sifaka Comprehensive Text Analysis")
    print("=====================================")
    print("All classifiers now use Hugging Face Transformers!")
    print(f"Analyzing {len(SAMPLE_THOUGHTS)} sample thoughts...")

    all_results = []

    try:
        for thought_name, thought_text in SAMPLE_THOUGHTS.items():
            result = await analyze_text_comprehensive(thought_text, thought_name)
            all_results.append(result)

            # Brief summary
            analyses = result["analyses"]
            print(f"\n📊 SUMMARY for {thought_name}:")
            print(f"   Sentiment: {analyses.get('sentiment', {}).get('label', 'N/A')}")
            print(f"   Language: {analyses.get('language', {}).get('label', 'N/A')}")
            print(f"   Emotion: {analyses.get('emotion', {}).get('label', 'N/A')}")
            print(f"   Intent: {analyses.get('intent', {}).get('label', 'N/A')}")
            print(f"   Readability: {analyses.get('readability', {}).get('label', 'N/A')}")
            print(f"   Toxicity: {analyses.get('toxicity', {}).get('label', 'N/A')}")
            print(f"   Bias: {analyses.get('bias', {}).get('label', 'N/A')}")
            print(f"   Spam: {analyses.get('spam', {}).get('label', 'N/A')}")

        # Save detailed results
        output_file = "comprehensive_analysis_results.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✅ Analysis complete! Detailed results saved to {output_file}")
        print(f"\n🎯 Key Insights:")
        print(f"   • All {len(SAMPLE_THOUGHTS)} texts analyzed successfully")
        print(f"   • {len(all_results[0]['analyses'])} different classifiers used")
        print(f"   • All classifiers use state-of-the-art Transformer models")
        print(f"   • Results include confidence scores and processing times")

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Please install required dependencies:")
        print('  pip install "sifaka[classifiers]"  # PyTorch backend')
        print('  pip install "sifaka[classifiers-tf]"  # TensorFlow backend')
        print('  pip install "sifaka[classifiers-jax]"  # JAX backend')
        print('  or: uv pip install "sifaka[classifiers]"')
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
