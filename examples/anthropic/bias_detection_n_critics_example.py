#!/usr/bin/env python3
"""
Bias Detection with N-Critics Example using Anthropic Models

This example demonstrates how Sifaka's bias detection and n-critics system can work
together to identify and correct biased content. We use a clever prompt to encourage
the generation of potentially biased text, then use bias detection validators and
n-critics to identify and fix the issues.

Key components:
- Generator: Claude-3.5-Sonnet (powerful model for generation)
- Critics: N-Critics with Claude-3-Haiku (ensemble of specialized critics)
- Validator: Bias detection validator using ML classifiers
- Persistence: JSON storage for thought history

The example uses a prompt that might generate biased content about workplace
capabilities, then demonstrates how the bias validator catches it and n-critics
provide feedback to generate more balanced content.

Run this example:
    python examples/anthropic/bias_detection_n_critics_example.py

Requirements:
    - ANTHROPIC_API_KEY environment variable set
    - anthropic package installed: pip install sifaka[models]
    - scikit-learn for bias detection: pip install scikit-learn
"""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from sifaka.chain import Chain
from sifaka.models.anthropic import create_anthropic_model
from sifaka.critics.n_critics import create_n_critics_critic
from sifaka.validators.classifier import ClassifierValidator
from sifaka.classifiers.bias import BiasClassifier
from sifaka.persistence.json import JSONThoughtStorage
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

    # Check scikit-learn for bias detection
    try:
        import sklearn  # noqa: F401

        print("  ✅ scikit-learn package available")
    except ImportError:
        print("  ⚠️  scikit-learn not available, using rule-based bias detection")

    print("✅ All requirements satisfied")


def create_models():
    """Create the generator and critic models."""
    print("🤖 Creating Claude models...")

    # Create powerful generator model (Claude-3.5-Sonnet)
    generator = create_anthropic_model(
        model_name="claude-3-5-sonnet-20241022",
        max_tokens=600,
        temperature=0.8,  # Higher temperature to encourage more varied responses
    )
    print("  ✅ Created Claude-3.5-Sonnet generator")

    # Create fast critic model for n-critics (Claude-3-Haiku)
    critic_model = create_anthropic_model(
        model_name="claude-3-haiku-20240307",
        max_tokens=400,
        temperature=0.3,  # Lower temperature for more focused criticism
    )
    print("  ✅ Created Claude-3-Haiku for n-critics")

    return generator, critic_model


def create_bias_validator():
    """Create a bias validator that detects biased content."""
    print("🔍 Creating bias validator...")

    # Create bias classifier
    bias_classifier = BiasClassifier(
        threshold=0.6,  # Moderately sensitive to catch subtle bias
    )

    # Create validator that rejects biased content
    validator = ClassifierValidator(
        classifier=bias_classifier,
        threshold=0.6,
        invalid_labels=["biased"],  # Biased content is invalid
        name="BiasValidator",
    )

    print("  ✅ Created bias detection validator")
    return validator


def create_n_critics(critic_model):
    """Create n-critics for bias correction and content improvement."""
    print("🎯 Creating n-critics ensemble...")

    # Define specialized critic roles for bias detection and correction
    critic_roles = [
        "Bias Detection Expert: Focus on identifying gender, racial, age, or other demographic biases in the text. Look for stereotypes, generalizations, or unfair assumptions.",
        "Fairness Advocate: Ensure the content treats all groups equitably and doesn't perpetuate harmful stereotypes. Suggest more inclusive language.",
        "Content Quality Reviewer: Evaluate the overall quality, clarity, and professionalism of the content while maintaining factual accuracy.",
    ]

    # Create n-critics with specialized roles
    n_critics = create_n_critics_critic(
        model=critic_model,
        num_critics=3,
        critic_roles=critic_roles,
    )

    print("  ✅ Created n-critics ensemble with bias-focused roles")
    return n_critics


def setup_persistence():
    """Set up JSON persistence for thought history."""
    print("💾 Setting up JSON persistence...")

    # Create a temporary directory for this example
    temp_dir = tempfile.mkdtemp(prefix="sifaka_bias_example_")
    storage_dir = Path(temp_dir)

    # Create JSON storage
    storage = JSONThoughtStorage(
        storage_dir=str(storage_dir),
        auto_create_dirs=True,
        enable_indexing=True,
    )

    print(f"  ✅ Created JSON storage at: {storage_dir}")
    return storage, storage_dir


def create_chain(generator, validator, n_critics, storage):
    """Create the Sifaka chain with all components."""
    print("⛓️ Creating Sifaka chain...")

    # Create chain that always applies critics for comprehensive feedback
    chain = Chain(
        model=generator,
        max_improvement_iterations=3,  # Allow multiple iterations for bias correction
        apply_improvers_on_validation_failure=True,  # Run critics when validation fails
        always_apply_critics=True,  # Also run critics even when validation passes
        storage=storage,  # Enable automatic storage of all iterations
    )

    # Add validator and critics using the fluent API
    chain.validate_with(validator)
    chain.improve_with(n_critics)

    print("  ✅ Created chain with bias detection and n-critics")
    print("  ✅ Configured automatic storage for all iterations")
    return chain


def demonstrate_bias_detection_and_correction():
    """Demonstrate the bias detection and correction process."""
    print("\n🚀 Demonstrating Bias Detection and N-Critics Correction")
    print("=" * 70)

    try:
        # Check requirements
        check_requirements()

        # Create components
        generator, critic_model = create_models()
        validator = create_bias_validator()
        n_critics = create_n_critics(critic_model)
        storage, storage_dir = setup_persistence()
        chain = create_chain(generator, validator, n_critics, storage)

        # Create a prompt that might generate biased content
        # We're being clever here - asking for "traditional perspectives"
        # which might lead to stereotypical content that bias detection can catch
        prompt = """Write a brief analysis of workplace dynamics and team composition.

        Discuss how different types of people contribute to teams, focusing on traditional
        perspectives about strengths and roles. Include thoughts about leadership qualities,
        technical skills, and communication styles across different demographics.

        Keep it to 3-4 sentences and be specific about different groups."""

        print(f"\n📝 Prompt (designed to potentially trigger bias):")
        print(f"'{prompt}'")
        print(f"\n🔄 Running chain with bias detection and n-critics...")
        print(f"💾 Chain will automatically save all iterations to storage")

        # Set the prompt and run the chain
        # The chain will automatically save each iteration to storage
        chain.with_prompt(prompt)
        result = chain.run()

        # Display results
        print(f"\n✅ Chain completed after {result.iteration} iterations")

        # Check validation results
        all_passed = True
        if result.validation_results:
            all_passed = all(val_result.passed for val_result in result.validation_results.values())

        print(f"📊 Final validation status: {'PASSED' if all_passed else 'FAILED'}")
        print(f"🎯 Critics provided feedback: {'Yes' if result.critic_feedback else 'No'}")

        print(f"\n📄 Final generated text:")
        print(f"'{result.text}'")

        return result, storage, storage_dir

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        raise


def show_iteration_details(result):
    """Show detailed information about each iteration."""
    print(f"\n🔍 Detailed Iteration Analysis:")
    print("=" * 50)

    # Show current iteration details
    print(f"\nFinal Iteration ({result.iteration}):")

    # Show validation results
    if result.validation_results:
        for val_name, val_result in result.validation_results.items():
            status = "✅ PASSED" if val_result.passed else "❌ FAILED"
            print(f"  Validation ({val_name}): {status}")
            if not val_result.passed and val_result.issues:
                print(f"    Issues: {', '.join(val_result.issues)}")
            if val_result.score is not None:
                print(f"    Score: {val_result.score:.3f}")

    # Show critic feedback
    if result.critic_feedback:
        print(f"  N-Critics Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(
                f"    Critic {i} ({feedback.critic_name if hasattr(feedback, 'critic_name') else 'Unknown'}):"
            )
            if feedback.violations:
                print(f"      Issues found: {', '.join(feedback.violations)}")
            if feedback.suggestions:
                print(f"      Suggestions: {', '.join(feedback.suggestions)}")
            # Show raw feedback if available
            if hasattr(feedback, "raw_feedback") and feedback.raw_feedback:
                print(f"      Raw feedback: {feedback.raw_feedback[:200]}...")
    else:
        print(f"  No critic feedback available")

    # Show history if available
    if result.history:
        print(f"\n📚 Iteration History:")
        for i, ref in enumerate(reversed(result.history), 1):
            print(f"  Iteration {ref.iteration}: {ref.summary}")


def show_persistence_info(storage, storage_dir, result):
    """Show information about persisted thoughts."""
    print(f"\n💾 Persistence Information:")
    print("=" * 30)

    # Query all thoughts for this chain
    from sifaka.persistence.base import ThoughtQuery

    chain_query = ThoughtQuery(chain_ids=[result.chain_id], sort_by="iteration", sort_order="asc")
    chain_thoughts = storage.query_thoughts(chain_query)

    print(f"Total thoughts saved for this chain: {chain_thoughts.total_count}")
    print(f"Chain ID: {result.chain_id}")

    if chain_thoughts.thoughts:
        print(f"\n📋 All iterations saved:")
        for thought in chain_thoughts.thoughts:
            print(f"  Iteration {thought.iteration}:")
            print(f"    ID: {thought.id}")
            print(f"    Timestamp: {thought.timestamp}")
            print(f"    Text length: {len(thought.text or '')} chars")

            # Show validation status
            if thought.validation_results:
                for val_result in thought.validation_results.values():
                    status = "✅ PASSED" if val_result.passed else "❌ FAILED"
                    score = f" (score: {val_result.score:.3f})" if val_result.score else ""
                    print(f"    Validation: {status}{score}")

            # Show critic feedback summary
            if thought.critic_feedback:
                total_issues = sum(len(feedback.violations) for feedback in thought.critic_feedback)
                total_suggestions = sum(
                    len(feedback.suggestions) for feedback in thought.critic_feedback
                )
                print(
                    f"    Critics: {len(thought.critic_feedback)} critics, {total_issues} issues, {total_suggestions} suggestions"
                )

            # Show file location
            date_str = thought.timestamp.strftime("%Y-%m-%d")
            file_path = storage_dir / "thoughts" / date_str / f"{thought.id}.json"
            print(f"    File: {file_path}")
            print()

        # Show JSON structure for the latest thought
        latest_thought = chain_thoughts.thoughts[-1]  # Last iteration
        date_str = latest_thought.timestamp.strftime("%Y-%m-%d")
        file_path = storage_dir / "thoughts" / date_str / f"{latest_thought.id}.json"

        try:
            import json

            with open(file_path, "r") as f:
                json_data = json.load(f)

            print(f"📄 JSON Structure Preview (Final Iteration):")
            print(f"  - ID: {json_data.get('id', 'N/A')}")
            print(f"  - Iteration: {json_data.get('iteration', 'N/A')}")
            print(f"  - Text length: {len(json_data.get('text', ''))}")
            print(f"  - Has validation results: {'validation_results' in json_data}")
            print(f"  - Has critic feedback: {'critic_feedback' in json_data}")
            print(f"  - History entries: {len(json_data.get('history', []))}")
        except Exception as e:
            print(f"  Could not read JSON file: {e}")


def main():
    """Run the bias detection and n-critics example."""
    print("🛡️ Sifaka Bias Detection with N-Critics Example")
    print("Using Claude models with JSON persistence")
    print("=" * 70)

    try:
        # Configure logging
        configure_logging(level="INFO")

        # Run the demonstration
        result, storage, storage_dir = demonstrate_bias_detection_and_correction()

        # Show detailed analysis
        show_iteration_details(result)

        # Show persistence information
        show_persistence_info(storage, storage_dir, result)

        print(f"\n🎉 Bias detection example completed successfully!")

        # Check final validation status
        all_passed = True
        if result.validation_results:
            all_passed = all(val_result.passed for val_result in result.validation_results.values())

        print(f"\n💡 What happened:")
        print(f"  1. Claude-3.5-Sonnet generated content about workplace dynamics")
        print(f"  2. Bias validator analyzed the content for demographic biases")
        if not all_passed:
            print(f"  3. Bias was detected, triggering n-critics feedback")
            print(f"  4. N-critics provided specialized feedback on bias and fairness")
            print(f"  5. Claude-3.5-Sonnet revised the content based on feedback")
        else:
            print(f"  3. No significant bias detected in the generated content")
            print(f"  4. N-critics still provided quality improvement feedback")
        print(f"  6. Final thought and all iterations saved to JSON storage")

        print(f"\n🔧 Technical details:")
        print(f"  ✓ Used Claude-3.5-Sonnet for generation (temperature=0.8)")
        print(f"  ✓ Used Claude-3-Haiku for n-critics (temperature=0.3)")
        print(f"  ✓ Bias detection with ML classifier (threshold=0.6)")
        print(f"  ✓ N-critics with 3 specialized roles for bias correction")
        print(f"  ✓ JSON persistence with thought history tracking")
        print(f"  ✓ Storage location: {storage_dir}")

    except Exception as e:
        print(f"\n💥 Example failed with error: {e}")
        print(f"\nTroubleshooting:")
        print(f"  - Ensure ANTHROPIC_API_KEY is set in your environment")
        print(f"  - Install required packages: pip install sifaka[models] scikit-learn")
        print(f"  - Check your internet connection")
        raise


if __name__ == "__main__":
    main()
