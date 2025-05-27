#!/usr/bin/env python3
"""Anthropic Self-Refine with Multiple Validators Example.

This example demonstrates:
- Anthropic Claude model for text generation
- Self-Refine critic for iterative improvement
- Comprehensive set of validators for quality assurance
- Default retry behavior

The chain will generate content about software engineering best practices
and use multiple validators to ensure high-quality, comprehensive output.

Prerequisites:
- Install GuardrailsAI PII detector: guardrails hub install hub://guardrails/detect_pii
"""

import os

from dotenv import load_dotenv

from sifaka.classifiers.language import LanguageClassifier
from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.core.chain import Chain
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.models.anthropic import AnthropicModel
from sifaka.utils.logging import get_logger
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.classifier import ClassifierValidator
from sifaka.validators.content import ContentValidator
from sifaka.validators.format import FormatValidator
from sifaka.validators.guardrails import GuardrailsValidator

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def create_comprehensive_validators():
    """Create a comprehensive set of validators for quality assurance."""

    validators = []

    # 1. Length Validator - Ensure substantial content
    length_validator = LengthValidator(
        min_length=500,  # Minimum 500 characters for comprehensive coverage
        max_length=2000,  # Maximum 2000 characters to stay focused
    )
    validators.append(length_validator)

    # 2. Content Validator - Check for required topics and forbidden content
    content_validator = ContentValidator(
        required=[
            "software",
            "engineering",
            "best practices",
            "development",
            "code",
            "quality",
            "testing",
            "documentation",
        ],
        prohibited=["hack", "exploit", "vulnerability", "malicious"],
        name="Software Engineering Content Validator",
    )
    validators.append(content_validator)

    # 3. Regex Validator - Ensure proper structure and formatting
    regex_validator = RegexValidator(
        required_patterns=[
            r"\b(principle|practice|approach|method|technique)\b",  # Must mention methodologies
            r"\b(team|collaboration|communication)\b",  # Must mention teamwork aspects
            r"\b(maintain|quality|standard|review)\b",  # Must mention quality aspects
        ],
        forbidden_patterns=[
            r"\b(never|impossible|can't|won't)\b",  # Avoid absolute negative statements
            r"\b(always|must|only)\b",  # Avoid absolute positive statements
        ],
    )
    validators.append(regex_validator)

    # 4. Format Validator - Ensure readable structure
    format_validator = FormatValidator(
        format_type="custom",
        custom_validator=lambda text: len(text.split("\n\n")) >= 3,  # At least 3 paragraphs
        name="Readable Format Validator",
    )
    validators.append(format_validator)

    # 5. Language Classifier Validator - Ensure English content
    language_classifier = LanguageClassifier()
    language_validator = ClassifierValidator(
        classifier=language_classifier,
        valid_labels=["en"],  # English
        threshold=0.8,
        name="English Language Validator",
    )
    validators.append(language_validator)

    # 6. Sentiment Classifier Validator - Ensure positive/neutral tone
    sentiment_classifier = SentimentClassifier()
    sentiment_validator = ClassifierValidator(
        classifier=sentiment_classifier,
        valid_labels=["positive", "neutral"],  # Allow positive or neutral
        threshold=0.6,
        name="Professional Tone Validator",
    )
    validators.append(sentiment_validator)

    # 7. Guardrails Validator - Professional content safety
    guardrails_validator = GuardrailsValidator(
        validators=["DetectPII"],
        validator_args={"DetectPII": {"entities": ["PERSON", "EMAIL", "PHONE_NUMBER"]}},
        name="Professional Content Safety Validator",
    )
    validators.append(guardrails_validator)

    return validators


def main():
    """Run the Anthropic Self-Refine with Multiple Validators example."""

    # Ensure API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    logger.info("Creating Anthropic Self-Refine with multiple validators example")

    # Create Anthropic model
    model = AnthropicModel(model_name="claude-3-sonnet-20240229", max_tokens=1000, temperature=0.7)

    # Create Self-Refine critic for iterative improvement
    critic = SelfRefineCritic(
        model=model,
        max_refinements=2,  # Allow up to 2 refinement iterations
        name="Software Engineering Self-Refine Critic",
    )

    # Create comprehensive validators
    validators = create_comprehensive_validators()

    # Create the chain
    chain = Chain(
        model=model,
        prompt="Write a comprehensive guide on software engineering best practices that covers code quality, testing strategies, documentation standards, team collaboration, and continuous improvement processes.",
        max_improvement_iterations=3,  # Default retry behavior
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,
    )

    # Add all validators
    for validator in validators:
        chain.validate_with(validator)

    # Add critic
    chain.improve_with(critic)

    # Run the chain
    logger.info("Running chain with Self-Refine critic and comprehensive validation...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 80)
    print("ANTHROPIC SELF-REFINE WITH MULTIPLE VALIDATORS EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nIterations: {result.iteration}")
    print(f"Chain ID: {result.chain_id}")

    # Show validation results
    if result.validation_results:
        print(f"\nValidation Results ({len(validators)} validators):")
        passed_count = 0
        for i, (validator_name, validation_result) in enumerate(
            result.validation_results.items(), 1
        ):
            status = "✓ PASSED" if validation_result.passed else "✗ FAILED"
            print(f"  {i}. {validator_name}: {status}")
            if validation_result.passed:
                passed_count += 1
            else:
                print(f"     Error: {validation_result.message}")
                if validation_result.issues:
                    print(f"     Issues: {', '.join(validation_result.issues)}")
                if validation_result.suggestions:
                    print(f"     Suggestions: {', '.join(validation_result.suggestions)}")

        print(f"\nValidation Summary: {passed_count}/{len(validators)} validators passed")

    # Show Self-Refine critic feedback
    if result.critic_feedback:
        print(f"\nSelf-Refine Critic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Suggestions: {feedback.suggestions[:300]}...")
            if hasattr(feedback, "refinement_iteration"):
                print(f"     Refinement Iteration: {feedback.refinement_iteration}")

    print("\n" + "=" * 80)
    logger.info("Self-Refine with multiple validators example completed successfully")


if __name__ == "__main__":
    main()
