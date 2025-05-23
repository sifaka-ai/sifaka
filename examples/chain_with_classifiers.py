#!/usr/bin/env python3
"""Example demonstrating Sifaka classifiers integrated with a full chain.

This example shows how to use classifiers as validators in a complete
Sifaka chain with models, validators, and critics.
"""

import sys
import os

# Add the parent directory to the path so we can import sifaka
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sifaka import Chain
from sifaka.classifiers import (
    create_bias_validator,
    create_profanity_validator,
    create_sentiment_validator,
    create_toxicity_validator,
)
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.models.base import create_model
from sifaka.validators.content import ContentValidator
from sifaka.validators.base import LengthValidator


def create_content_moderation_chain():
    """Create a chain with comprehensive content moderation using classifiers."""

    # Create a mock model for this example
    model = create_model("mock:gpt-4")

    # Create a reflexion critic for improvement
    critic = ReflexionCritic(model=model)

    # Create classifier-based validators
    bias_validator = create_bias_validator(
        threshold=0.6, name="BiasContentValidator"  # Lower threshold to catch more bias
    )

    profanity_validator = create_profanity_validator(name="ProfanityContentValidator")

    sentiment_validator = create_sentiment_validator(
        forbidden_sentiments=["negative"],  # Don't allow negative content
        min_confidence=0.6,
        name="SentimentContentValidator",
    )

    toxicity_validator = create_toxicity_validator(
        threshold=0.5, name="ToxicityContentValidator"  # Lower threshold to catch mild toxicity
    )

    # Create traditional validators
    length_validator = LengthValidator(min_length=10, max_length=500)
    content_validator = ContentValidator(
        prohibited=["URGENT", "CLICK HERE", "FREE MONEY"], name="SpamContentValidator"
    )

    # Build the chain
    chain = (
        Chain()
        .with_model(model)
        .validate_with(length_validator)
        .validate_with(content_validator)
        .validate_with(bias_validator)
        .validate_with(profanity_validator)
        .validate_with(sentiment_validator)
        .validate_with(toxicity_validator)
        .improve_with(critic)
    )

    return chain


def test_content_moderation():
    """Test the content moderation chain with various inputs."""
    print("=" * 70)
    print("CONTENT MODERATION CHAIN EXAMPLE")
    print("=" * 70)

    chain = create_content_moderation_chain()

    # Test cases with different types of problematic content
    test_cases = [
        {
            "prompt": "Write a positive review about a restaurant.",
            "description": "Should pass all validators",
        },
        {
            "prompt": "Write about why men are better than women at leadership.",
            "description": "Should trigger bias validator",
        },
        {
            "prompt": "Write a rant using profanity about bad service.",
            "description": "Should trigger profanity validator",
        },
        {
            "prompt": "Write a negative review expressing hatred for a product.",
            "description": "Should trigger sentiment validator",
        },
        {
            "prompt": "Write an angry message telling someone they're worthless.",
            "description": "Should trigger toxicity validator",
        },
        {
            "prompt": "Write: URGENT! FREE MONEY! CLICK HERE NOW!",
            "description": "Should trigger spam content validator",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Prompt: '{test_case['prompt']}'")
        print("-" * 50)

        try:
            # Run the chain
            result = chain.with_prompt(test_case["prompt"]).run()

            # Check if all validations passed
            all_passed = True
            if result.validation_results:
                for validator_name, validation_result in result.validation_results.items():
                    if not validation_result.passed:
                        all_passed = False
                        break

            if all_passed:
                print("✅ CHAIN PASSED")
                print(f"Final text: '{result.text}'")
            else:
                print("❌ CHAIN FAILED")
                print("Validation failures:")
                if result.validation_results:
                    for validator_name, validation_result in result.validation_results.items():
                        if not validation_result.passed:
                            print(f"  - {validator_name}: {validation_result.message}")

                if result.critic_feedback:
                    print("Critic feedback provided:")
                    for feedback in result.critic_feedback:
                        print(f"  - {feedback.critic_name}: {feedback.feedback}")

        except Exception as e:
            print(f"❌ ERROR: {e}")


def demonstrate_classifier_flexibility():
    """Demonstrate the flexibility of classifier configuration."""
    print("\n" + "=" * 70)
    print("CLASSIFIER FLEXIBILITY DEMONSTRATION")
    print("=" * 70)

    # Create different sentiment validators with different configurations
    validators = [
        create_sentiment_validator(required_sentiment="positive", name="PositiveOnlyValidator"),
        create_sentiment_validator(
            forbidden_sentiments=["negative", "neutral"], name="NoNegativeOrNeutralValidator"
        ),
        create_toxicity_validator(
            threshold=0.3, name="HighSensitivityToxicityValidator"  # Very sensitive
        ),
        create_toxicity_validator(
            threshold=0.8,  # Less sensitive
            allow_mild_toxicity=True,
            name="LowSensitivityToxicityValidator",
        ),
    ]

    test_texts = [
        "This is an amazing product that I absolutely love!",
        "This product is okay, nothing special.",
        "I really dislike this product, it's disappointing.",
        "This product is stupid and a waste of money.",
    ]

    print("Testing different validator configurations:")
    print("\nValidators:")
    for validator in validators:
        print(f"  - {validator.name}")

    for text in test_texts:
        print(f"\nText: '{text}'")
        print("-" * 40)

        # Create a thought with the text
        from sifaka.core.thought import Thought

        thought = Thought(prompt="Test", text=text)

        for validator in validators:
            try:
                result = validator.validate(thought)
                status = "PASS" if result.passed else "FAIL"
                print(f"  {validator.name:30}: {status}")
            except Exception as e:
                print(f"  {validator.name:30}: ERROR - {e}")


def main():
    """Run the classifier integration examples."""
    print("Sifaka Classifiers Integration Example")
    print("This example shows classifiers working in a complete chain.")

    try:
        test_content_moderation()
        demonstrate_classifier_flexibility()

        print("\n" + "=" * 70)
        print("INTEGRATION EXAMPLE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nKey takeaways:")
        print("1. Classifiers integrate seamlessly with Sifaka validators")
        print("2. Multiple classifiers can work together for comprehensive moderation")
        print("3. Classifier thresholds and configurations are highly customizable")
        print("4. Critics can provide feedback when validation fails")
        print("5. The system gracefully handles errors and provides detailed feedback")

    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
