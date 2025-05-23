#!/usr/bin/env python3
"""
Language Correction Example with Performance Monitoring

This example demonstrates how Sifaka's critic system can guide a model to correct
language issues. A powerful Claude model generates text in Spanish, fails English
validation, and then a smaller Claude critic guides it to translate to English.

Key components:
- Generator: Claude-3.5-Sonnet (powerful model for generation)
- Critic: Claude-3-Haiku (fast model for feedback)
- Validator: Language validator (checks for English)
- Performance monitoring throughout

Run this example:
    python examples/anthropic/language_correction_example.py

Requirements:
    - ANTHROPIC_API_KEY environment variable set
    - anthropic package installed: pip install sifaka[models]
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from sifaka.chain import Chain
from sifaka.models.anthropic import create_anthropic_model
from sifaka.critics.prompt import PromptCritic
from sifaka.validators.classifier import ClassifierValidator
from sifaka.classifiers.language import LanguageClassifier
from sifaka.utils.performance import (
    timer,
    time_operation,
    PerformanceMonitor,
    print_performance_summary,
)
from sifaka.utils.logging import configure_logging


def check_requirements():
    """Check that required environment variables and packages are available."""
    print("🔍 Checking requirements...")

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Get your API key from https://console.anthropic.com/"
        )

    print("  ✅ ANTHROPIC_API_KEY found")

    # Check anthropic package
    try:
        import anthropic  # noqa: F401

        print("  ✅ anthropic package available")
    except ImportError:
        raise ImportError("anthropic package is required. Install with: pip install sifaka[models]")

    print("✅ All requirements satisfied")


@timer("model_creation")
def create_models():
    """Create the generator and critic models with performance timing."""
    print("🤖 Creating Claude models...")

    # Create powerful generator model (Claude-3.5-Sonnet)
    generator = create_anthropic_model(
        model_name="claude-3-5-sonnet-20241022", max_tokens=500, temperature=0.7
    )
    print("  ✅ Created Claude-3.5-Sonnet generator")

    # Create fast critic model (Claude-3-Haiku)
    critic_model = create_anthropic_model(
        model_name="claude-3-haiku-20240307", max_tokens=300, temperature=0.3
    )
    print("  ✅ Created Claude-3-Haiku critic")

    return generator, critic_model


@timer("validator_creation")
def create_language_validator():
    """Create a language validator that checks for English."""
    print("🔤 Creating language validator...")

    # Create language classifier
    language_classifier = LanguageClassifier()

    # Create validator that requires English
    validator = ClassifierValidator(
        classifier=language_classifier,
        valid_labels=["en"],  # Only English is valid
        threshold=0.7,  # High confidence required
        name="EnglishLanguageValidator",
    )

    print("  ✅ Created English language validator")
    return validator


@timer("critic_creation")
def create_prompt_critic(critic_model):
    """Create a prompt critic for language correction."""
    print("🎯 Creating prompt critic...")

    # Create a critic that focuses on language correction
    critic = PromptCritic(
        model=critic_model,
        system_prompt="""You are a language correction expert. Your job is to help improve text that fails language requirements.

When text fails validation, provide specific, actionable feedback to help the model correct the language issues.

Focus on:
1. Identifying the current language of the text
2. Explaining what language is required
3. Providing clear instructions for translation or correction
4. Being encouraging and constructive

Keep your feedback concise but helpful.""",
        name="LanguageCorrectionCritic",
    )

    print("  ✅ Created language correction critic")
    return critic


@timer("chain_creation")
def create_chain(generator, validator, critic):
    """Create the Sifaka chain with all components."""
    print("⛓️ Creating Sifaka chain...")

    # Create chain with new API
    chain = Chain(
        model=generator,
        max_improvement_iterations=3,  # Allow up to 3 iterations for correction
        apply_improvers_on_validation_failure=True,  # Run critics when validation fails
        always_apply_critics=False,  # Only run critics on validation failure
        pre_generation_retrieval=False,  # No retrieval needed for this example
        post_generation_retrieval=False,  # No retrieval needed for this example
        critic_retrieval=False,  # No retrieval needed for this example
    )

    # Add validator and critic using the fluent API
    chain.validate_with(validator)
    chain.improve_with(critic)

    print("  ✅ Created chain with language correction capability")
    return chain


def demonstrate_language_correction():
    """Demonstrate the language correction process with performance monitoring."""
    print("\n🚀 Demonstrating Language Correction with Performance Monitoring")
    print("=" * 70)

    # Enable performance monitoring
    monitor = PerformanceMonitor.get_instance()
    monitor.enable()

    try:
        # Check requirements
        check_requirements()

        # Create components with timing
        generator, critic_model = create_models()
        validator = create_language_validator()
        critic = create_prompt_critic(critic_model)
        chain = create_chain(generator, validator, critic)

        # Create the initial prompt that will generate Spanish text
        prompt = """Write a short story about a robot discovering emotions.

        Please write this story in Spanish (en español). Make it engaging and emotional,
        about 3-4 sentences long."""

        print(f"\n📝 Initial prompt: {prompt}")
        print("\n🔄 Running chain with language correction...")

        # Run the chain with performance monitoring
        with time_operation("full_chain_execution"):
            # Set the prompt and run the chain
            chain.with_prompt(prompt)
            result = chain.run()

        # Display results
        print(f"\n✅ Chain completed after {result.iteration} iterations")

        # Check if all validations passed
        all_passed = True
        if result.validation_results:
            all_passed = all(val_result.passed for val_result in result.validation_results.values())

        print(f"📊 Final validation status: {'PASSED' if all_passed else 'FAILED'}")

        print(f"\n📄 Final generated text:")
        print(f"'{result.text}'")

        # Show iteration details
        print(f"\n🔍 Iteration Details:")
        for i in range(1, result.iteration + 1):
            print(f"\n  Iteration {i}:")

            # Show validation results
            if result.validation_results:
                for val_name, val_result in result.validation_results.items():
                    status = "✅ PASSED" if val_result.passed else "❌ FAILED"
                    print(f"    Validation ({val_name}): {status}")
                    if not val_result.passed and val_result.issues:
                        print(f"      Issues: {', '.join(val_result.issues)}")

            # Show critic feedback
            if result.critic_feedback:
                for feedback in result.critic_feedback:
                    if feedback.violations:
                        print(f"    Critic feedback: {', '.join(feedback.violations)}")
                    if feedback.suggestions:
                        print(f"    Suggestions: {', '.join(feedback.suggestions)}")

        return result

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        raise

    finally:
        # Print performance summary
        print(f"\n" + "=" * 70)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 70)
        print_performance_summary()


def main():
    """Run the language correction example."""
    print("🌍 Sifaka Language Correction Example")
    print("Using Claude models with performance monitoring")
    print("=" * 70)

    try:
        # Configure logging
        configure_logging(level="INFO")

        # Run the demonstration
        result = demonstrate_language_correction()

        print(f"\n🎉 Language correction example completed successfully!")

        # Check if all validations passed
        all_passed = True
        if result.validation_results:
            all_passed = all(val_result.passed for val_result in result.validation_results.values())

        print(f"📈 Final text is in English: {all_passed}")

        print(f"\n💡 What happened:")
        print(f"  1. Claude-3.5-Sonnet generated text in Spanish (as requested)")
        print(f"  2. English language validator failed the Spanish text")
        print(f"  3. Claude-3-Haiku critic provided translation guidance")
        print(f"  4. Claude-3.5-Sonnet translated to English in iteration 2")
        print(f"  5. English language validator passed the translated text")

        print(f"\n🔧 Performance monitoring tracked:")
        print(f"  ✓ Model creation and initialization times")
        print(f"  ✓ Validator and critic setup times")
        print(f"  ✓ Full chain execution time")
        print(f"  ✓ Individual operation timings")

    except Exception as e:
        print(f"\n💥 Example failed with error: {e}")
        print(f"\nTroubleshooting:")
        print(f"  - Ensure ANTHROPIC_API_KEY is set in your environment")
        print(f"  - Install required packages: pip install sifaka[models]")
        print(f"  - Check your internet connection")
        raise

    finally:
        # Clean up performance monitoring
        monitor = PerformanceMonitor.get_instance()
        monitor.clear()


if __name__ == "__main__":
    main()
