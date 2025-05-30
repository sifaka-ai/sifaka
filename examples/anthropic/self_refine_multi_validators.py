#!/usr/bin/env python3
"""Anthropic Self-Refine with Multiple Validators Example (PydanticAI).

This example demonstrates:
- PydanticAI Agent with Anthropic Claude Haiku model for efficient text generation
- Self-Refine critic for iterative improvement
- Comprehensive set of validators for quality assurance
- Automatic feedback summarization using T5 model
- Default retry behavior

The chain will generate content about software engineering best practices
and use multiple validators to ensure high-quality, comprehensive output.
Feedback from validators and critics is automatically summarized for more
effective improvement iterations.

Prerequisites:
- Install GuardrailsAI PII detector: guardrails hub install hub://guardrails/detect_pii
- Install transformers for feedback summarization: pip install transformers torch
"""

import os

from dotenv import load_dotenv
from pydantic_ai import Agent

from sifaka.agents import create_pydantic_chain
from sifaka.classifiers.language import LanguageClassifier
from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.models import create_model
from sifaka.storage import FileStorage
from sifaka.utils.logging import get_logger
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.classifier import ClassifierValidator
from sifaka.validators.content import ContentValidator
from sifaka.validators.format import FormatValidator

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def create_comprehensive_validators():
    """Create a comprehensive set of validators for quality assurance."""

    validators = []

    # 1. Length Validator - Ensure substantial content (CHALLENGING)
    length_validator = LengthValidator(
        min_length=1500,  # Minimum 1500 characters for comprehensive coverage
        max_length=3000,  # Maximum 3000 characters to stay focused
    )
    validators.append(length_validator)

    # 2. Content Validator - Check for required topics and forbidden content (MODERATE)
    content_validator = ContentValidator(
        required=[
            "software",
            "engineering",
            "development",
            "code",
            "quality",
            "testing",
            "documentation",
            "practices",
            "approach",
        ],
        prohibited=["hack", "exploit", "malicious"],
        name="Software Engineering Content Validator",
    )
    validators.append(content_validator)

    # 3. Regex Validator - Ensure proper structure and formatting (ACHIEVABLE)
    regex_validator = RegexValidator(
        required_patterns=[
            r"\b(practice|approach|method|technique|development|engineering)\b",  # Must mention methodologies or core terms
            r"\b(quality|standard|review|testing|documentation)\b",  # Must mention quality aspects
        ],
        forbidden_patterns=[
            r"\b(never|impossible|can't|won't)\b",  # Avoid absolute negative statements
        ],
    )
    validators.append(regex_validator)

    # 4. Format Validator - Ensure readable structure (MODERATE)
    format_validator = FormatValidator(
        format_type="custom",
        custom_validator=lambda text: (
            len(text.split("\n\n")) >= 3  # At least 3 paragraphs
            and ("# " in text or "## " in text)  # Must have some heading
        ),
        name="Readable Structure Validator",
    )
    validators.append(format_validator)

    # 5. Language Classifier Validator - Ensure English content
    language_classifier = LanguageClassifier()
    language_validator = ClassifierValidator(
        classifier=language_classifier,
        valid_labels=["en"],  # English
        threshold=0.8,
        name="LanguageValidator",  # Unique name for validation results
    )
    validators.append(language_validator)

    # 6. Sentiment Classifier Validator - Ensure positive/neutral tone (ACHIEVABLE)
    sentiment_classifier = SentimentClassifier()
    sentiment_validator = ClassifierValidator(
        classifier=sentiment_classifier,
        valid_labels=["positive", "neutral"],  # Allow positive or neutral
        threshold=0.5,  # Lower threshold for easier achievement
        name="SentimentValidator",  # Unique name for validation results
    )
    validators.append(sentiment_validator)

    # 7. Guardrails Validator - Professional content safety (REMOVED due to false PII detection)
    # The GuardrailsValidator was detecting example variable names like 'userAge' as PII
    # which caused validation errors. Removing for now.
    # guardrails_validator = GuardrailsValidator(
    #     validators=["DetectPII"],
    #     validator_args={"DetectPII": {"entities": ["PERSON", "EMAIL", "PHONE_NUMBER"]}},
    #     name="Professional Content Safety Validator",
    # )
    # validators.append(guardrails_validator)

    return validators


def main():
    """Run the Anthropic Self-Refine with Multiple Validators example using PydanticAI."""

    # Ensure API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    logger.info("Creating PydanticAI Anthropic Self-Refine with multiple validators example")

    # Create PydanticAI agent with Anthropic Claude model for main generation
    agent = Agent(
        "anthropic:claude-3-5-haiku-latest",
        system_prompt=(
            "You are a software engineering expert and technical writer. Provide comprehensive, "
            "well-structured information about software engineering best practices, development "
            "methodologies, code quality, testing approaches, and documentation standards. "
            "Use clear headings, detailed explanations, and practical examples. Maintain a "
            "positive and professional tone throughout your responses."
        ),
    )

    # Create smaller Haiku model for critic (faster and cheaper)
    critic_model = create_model("anthropic:claude-3-5-haiku-latest")

    # Create Self-Refine critic for iterative improvement using Haiku
    critic = SelfRefineCritic(
        model=critic_model,
        max_refinements=2,  # Allow up to 2 refinement iterations
        name="Software Engineering Self-Refine Critic (Haiku)",
    )

    # Create comprehensive validators
    validators = create_comprehensive_validators()

    # Debug: Print validator info
    print(f"Created {len(validators)} validators:")
    for i, validator in enumerate(validators, 1):
        validator_name = getattr(validator, "name", type(validator).__name__)
        print(f"  {i}. {validator_name}")
    print()

    # Create the PydanticAI chain with comprehensive validation
    chain = create_pydantic_chain(
        agent=agent,
        validators=validators,
        critics=[critic],
        max_improvement_iterations=3,  # Default retry behavior
        always_apply_critics=False,
        storage=FileStorage(
            "./thoughts/self_refine_multi_validators_thoughts.json",
            overwrite=True,  # Overwrite existing file instead of appending
        ),  # Save thoughts to single JSON file for debugging
    )

    # Debug: Print validator info
    print(f"Added {len(validators)} validators to PydanticAI chain:")
    for i, validator in enumerate(validators, 1):
        validator_name = getattr(validator, "name", type(validator).__name__)
        print(f"  {i}. {validator_name}")
    print()

    # Define the prompt designed to initially fail validation
    prompt = "Write about software engineering. Give me some tips."

    # Run the chain
    logger.info("Running PydanticAI chain with Self-Refine critic and comprehensive validation...")
    result = chain.run(prompt)

    # Display results
    print("\n" + "=" * 80)
    print("PYDANTIC AI ANTHROPIC SELF-REFINE WITH MULTIPLE VALIDATORS EXAMPLE")
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

        print(
            f"\nValidation Summary: {passed_count}/{len(result.validation_results)} validators passed"
        )

    # Show Self-Refine critic feedback
    if result.critic_feedback:
        print(f"\nSelf-Refine Critic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            # Show critic model if available
            if feedback.metadata and "critic_model_name" in feedback.metadata:
                print(f"     Critic Model: {feedback.metadata['critic_model_name']}")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Suggestions: {feedback.suggestions[:300]}...")
            if hasattr(feedback, "refinement_iteration"):
                print(f"     Refinement Iteration: {feedback.refinement_iteration}")

    print("\n" + "=" * 80)
    logger.info("PydanticAI Self-Refine with multiple validators example completed successfully")


if __name__ == "__main__":
    main()
