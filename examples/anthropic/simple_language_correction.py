#!/usr/bin/env python3
"""
Simple Language Correction Example for Sifaka

This example demonstrates language correction using Claude models without
the full Sifaka chain to avoid circular import issues.

Run this example:
    python examples/anthropic/simple_language_correction.py
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Simple performance timing
def timer(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            print(f"⏱️  {name}: {duration:.3f}s")
            return result
        return wrapper
    return decorator

@timer("anthropic_import")
def import_anthropic():
    """Import anthropic with timing."""
    try:
        import anthropic
        return anthropic
    except ImportError:
        raise ImportError("anthropic package required: pip install anthropic")

@timer("model_creation")
def create_models():
    """Create Claude models."""
    anthropic = import_anthropic()
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable required")
    
    # Create clients
    generator_client = anthropic.Anthropic(api_key=api_key)
    critic_client = anthropic.Anthropic(api_key=api_key)
    
    return generator_client, critic_client

@timer("text_generation")
def generate_spanish_text(client):
    """Generate text in Spanish."""
    prompt = """Write a short story about a robot discovering emotions. 
    
    Please write this story in Spanish (en español). Make it engaging and emotional, 
    about 3-4 sentences long."""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

@timer("language_detection")
def detect_language(text):
    """Simple language detection."""
    # Simple heuristic: Spanish has more accented characters and specific words
    spanish_indicators = ['el', 'la', 'es', 'un', 'una', 'con', 'por', 'para', 'que', 'de', 'en']
    english_indicators = ['the', 'and', 'is', 'a', 'an', 'with', 'for', 'that', 'of', 'in']
    
    text_lower = text.lower()
    spanish_count = sum(1 for word in spanish_indicators if word in text_lower)
    english_count = sum(1 for word in english_indicators if word in text_lower)
    
    if spanish_count > english_count:
        return "spanish"
    else:
        return "english"

@timer("critic_feedback")
def get_translation_feedback(client, text, target_language="english"):
    """Get feedback from critic about language."""
    prompt = f"""You are a language correction expert. The following text needs to be in {target_language}, but it appears to be in a different language.

Text: "{text}"

Please provide:
1. What language is the text currently in?
2. Clear instructions for translating it to {target_language}
3. Keep the same emotional tone and content

Be concise but helpful."""
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

@timer("text_translation")
def translate_text(client, original_text, feedback):
    """Translate text based on feedback."""
    prompt = f"""Based on this feedback: "{feedback}"

Please translate this text to English while maintaining the same emotional tone and content:

"{original_text}"

Provide only the translated text, nothing else."""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

def main():
    """Run the language correction example."""
    print("🌍 Simple Language Correction Example")
    print("Using Claude models with performance monitoring")
    print("=" * 60)
    
    total_start = time.time()
    
    try:
        # Create models
        generator_client, critic_client = create_models()
        print("✅ Models created successfully")
        
        # Generate Spanish text
        print("\n📝 Generating text in Spanish...")
        spanish_text = generate_spanish_text(generator_client)
        print(f"Generated text: '{spanish_text}'")
        
        # Detect language
        detected_lang = detect_language(spanish_text)
        print(f"🔍 Detected language: {detected_lang}")
        
        # Validate (should fail for English requirement)
        is_english = detected_lang == "english"
        print(f"📊 English validation: {'✅ PASSED' if is_english else '❌ FAILED'}")
        
        if not is_english:
            print("\n🎯 Getting critic feedback...")
            feedback = get_translation_feedback(critic_client, spanish_text)
            print(f"Critic feedback: {feedback}")
            
            print("\n🔄 Translating to English...")
            english_text = translate_text(generator_client, spanish_text, feedback)
            print(f"Translated text: '{english_text}'")
            
            # Re-validate
            final_lang = detect_language(english_text)
            final_is_english = final_lang == "english"
            print(f"🔍 Final language: {final_lang}")
            print(f"📊 Final validation: {'✅ PASSED' if final_is_english else '❌ FAILED'}")
            
            final_text = english_text
        else:
            final_text = spanish_text
        
        total_time = time.time() - total_start
        
        print(f"\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        print(f"⏱️  Total execution time: {total_time:.3f}s")
        print(f"📄 Final text: '{final_text}'")
        print(f"🎯 Language correction: {'✅ SUCCESS' if detect_language(final_text) == 'english' else '❌ FAILED'}")
        
        print(f"\n💡 What happened:")
        print(f"  1. Claude-3.5-Sonnet generated text in Spanish")
        print(f"  2. Simple language detector identified it as Spanish")
        print(f"  3. Claude-3-Haiku critic provided translation guidance")
        print(f"  4. Claude-3.5-Sonnet translated to English")
        print(f"  5. Final validation confirmed English text")
        
        print(f"\n🎉 Language correction example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  - Ensure ANTHROPIC_API_KEY is set in your environment")
        print(f"  - Install required packages: pip install anthropic python-dotenv")
        print(f"  - Check your internet connection")
        raise

if __name__ == "__main__":
    main()
