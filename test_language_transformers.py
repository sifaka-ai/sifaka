#!/usr/bin/env python3
"""
Simple test script to verify the Transformers-based language classifier works.
"""

import asyncio
import sys
import os

# Add the sifaka directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sifaka'))

async def test_language_classifier():
    """Test the language classifier with sample texts."""
    try:
        from sifaka.classifiers.language import create_language_classifier
        
        print("Creating language classifier...")
        classifier = create_language_classifier()
        
        # Test texts in different languages
        test_texts = [
            ("Hello, how are you doing today?", "en"),  # English
            ("Hola, ¿cómo estás hoy?", "es"),  # Spanish
            ("Bonjour, comment allez-vous aujourd'hui?", "fr"),  # French
            ("Hallo, wie geht es dir heute?", "de"),  # German
            ("Ciao, come stai oggi?", "it"),  # Italian
            ("こんにちは、今日はいかがですか？", "ja"),  # Japanese
            ("你好，你今天怎么样？", "zh"),  # Chinese
        ]
        
        print("\nTesting language classification:")
        print("-" * 80)
        
        for text, expected_lang in test_texts:
            result = await classifier.classify_async(text)
            print(f"Text: {text}")
            print(f"Expected: {expected_lang} ({classifier.get_language_name(expected_lang)})")
            print(f"Detected: {result.label} ({result.metadata.get('language_name', 'Unknown')})")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Model: {result.metadata.get('model_name', 'Unknown')}")
            
            # Show top 3 detected languages
            all_langs = result.metadata.get('all_languages', [])[:3]
            if all_langs:
                print("Top 3 languages:")
                for lang_info in all_langs:
                    print(f"  {lang_info['lang']} ({lang_info['name']}): {lang_info['score']:.3f}")
            
            print("-" * 80)
            
        print("✅ Language classifier test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure transformers is installed: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_language_classifier())
    sys.exit(0 if success else 1)
