#!/usr/bin/env python3
"""Comprehensive Validation Demo with Mock Model.

This example demonstrates:
- Multiple validators working together
- Mock model for consistent testing
- Validation failure handling and recovery
- Comprehensive quality assurance workflow

This example shows how to use multiple validators to ensure high-quality
output that meets various criteria simultaneously.
"""

from sifaka.core.chain import Chain
from sifaka.models.base import MockModel
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.content import ContentValidator
from sifaka.validators.classifier import ClassifierValidator
from sifaka.classifiers.language import LanguageClassifier
from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.utils.logging import get_logger

# Configure logging
logger = get_logger(__name__)


def create_comprehensive_validators():
    """Create a comprehensive set of validators for demonstration."""

    validators = []

    # 1. Length Validator
    length_validator = LengthValidator(min_length=200, max_length=800)
    validators.append(length_validator)

    # 2. Content Validator - Check for prohibited content
    content_validator = ContentValidator(
        prohibited=["impossible", "never", "can't", "hate", "violence"]
    )
    validators.append(content_validator)

    # 3. Regex Validator - Ensure proper structure
    regex_validator = RegexValidator(
        required_patterns=[
            r"\b(technology|innovation|development)\b",  # Must mention key concepts
            r"\b(will|can|may|might)\b",  # Must use future/possibility language
        ],
        forbidden_patterns=[
            r"\b(never|impossible|can't)\b",  # Avoid absolute negatives
        ],
    )
    validators.append(regex_validator)

    # 4. Language Classifier Validator
    language_classifier = LanguageClassifier()
    language_validator = ClassifierValidator(
        classifier=language_classifier,
        valid_labels=["en"],
        threshold=0.9,
    )
    validators.append(language_validator)

    # 5. Sentiment Classifier Validator
    sentiment_classifier = SentimentClassifier()
    sentiment_validator = ClassifierValidator(
        classifier=sentiment_classifier,
        valid_labels=["positive", "neutral"],
        threshold=0.7,
    )
    validators.append(sentiment_validator)

    return validators


def main():
    """Run the comprehensive validation demo."""

    logger.info("Creating comprehensive validation demo")

    # Create mock model with technology-focused responses
    model = MockModel(
        model_name="Technology Education Model",
        responses=[
            "Technology is changing our world rapidly.",  # Too short, will fail length validation
            "Technology and innovation are rapidly transforming our world, creating new opportunities for development and growth. The future holds exciting possibilities as we continue to advance in various fields, enabling us to solve complex problems and improve quality of life for people everywhere.",  # Should pass most validations
            "Technology and innovation are rapidly transforming our world in unprecedented ways, creating countless new opportunities for sustainable development and exponential growth. The future holds tremendously exciting possibilities as we continue to advance in various technological fields, enabling us to solve increasingly complex global problems and dramatically improve quality of life for people everywhere through smart, efficient, and accessible solutions.",  # Should pass all validations
        ],
    )

    # Create comprehensive validators
    validators = create_comprehensive_validators()

    # Create Self-Refine critic for improvement
    critic = SelfRefineCritic(model=model)

    # Create the chain
    chain = Chain(
        model=model,
        prompt="Write about how technology and innovation will shape the future of human development and create new opportunities for progress.",
        max_improvement_iterations=3,
        apply_improvers_on_validation_failure=True,  # Apply critics when validation fails
        always_apply_critics=False,  # Only apply critics when needed
    )

    # Add all validators
    for validator in validators:
        chain.validate_with(validator)

    # Add critic
    chain.improve_with(critic)

    # Run the chain
    logger.info("Running chain with comprehensive validation...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 70)
    print("COMPREHENSIVE VALIDATION DEMO WITH MOCK MODEL")
    print("=" * 70)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nChain Execution Details:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Total Validators: {len(validators)}")

    # Show detailed validation results
    if result.validation_results:
        print(f"\nDetailed Validation Results:")
        passed_count = 0
        failed_count = 0

        for i, (validator_name, validation_result) in enumerate(
            result.validation_results.items(), 1
        ):
            status = "✓ PASSED" if validation_result.passed else "✗ FAILED"
            print(f"  {i}. {validator_name}: {status}")

            if validation_result.passed:
                passed_count += 1
                # Show success details
                if "Length" in validator_name:
                    print(f"     Text length: {len(result.text)} characters")
                elif "Content" in validator_name:
                    print(f"     Required keywords found")
                elif "Language" in validator_name:
                    print(f"     Language: English (confidence: {validation_result.score:.2f})")
                elif "Sentiment" in validator_name:
                    print(f"     Tone: Positive/Neutral")
            else:
                failed_count += 1
                print(f"     Error: {validation_result.message}")

        print(f"\nValidation Summary: {passed_count}/{len(validators)} validators passed")

        if failed_count > 0:
            print(f"  ⚠️  {failed_count} validation(s) failed - improvements were applied")
        else:
            print(f"  ✅ All validations passed!")

    # Show critic feedback if any
    if result.critic_feedback:
        print(f"\nSelf-Refine Critic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Confidence: {feedback.confidence}")
            if feedback.suggestions:
                print(f"     Improvement suggestions: {', '.join(feedback.suggestions[:3])}")
            if feedback.violations:
                print(f"     Issues: {', '.join(feedback.violations[:3])}")

    # Show validation evolution across iterations
    if result.history:
        print(f"\nValidation Evolution Across Iterations:")
        for i, historical_thought in enumerate(result.history, 1):
            print(f"  Iteration {i}: {historical_thought.summary or 'No summary'}")
    else:
        print(f"\nValidation Evolution: No history available")

    print(f"\nValidator Types Demonstrated:")
    print(f"  ✓ Length constraints (200-800 characters)")
    print(f"  ✓ Content requirements (educational keywords)")
    print(f"  ✓ Structure patterns (regex validation)")
    print(f"  ✓ Language detection (English required)")
    print(f"  ✓ Sentiment analysis (positive/neutral tone)")

    print(f"\nKey Features Demonstrated:")
    print(f"  ✓ Multiple validator coordination")
    print(f"  ✓ Validation failure handling")
    print(f"  ✓ Automatic improvement triggering")
    print(f"  ✓ Comprehensive quality assurance")
    print(f"  ✓ Iterative refinement process")

    print("\n" + "=" * 70)
    logger.info("Comprehensive validation demo completed successfully")


if __name__ == "__main__":
    main()
