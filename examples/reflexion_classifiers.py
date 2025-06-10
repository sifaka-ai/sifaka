#!/usr/bin/env python3
"""Reflexion Critic with All Classifiers Example.

This example demonstrates:
1. Using the ReflexionCritic for self-reflection and improvement
2. All available classifiers (bias, readability, emotion, intent) as validators
3. Critics only running when validators fail (always_apply_critics=False)
4. Saving thoughts to /thoughts directory for analysis
5. Simple but comprehensive validation pipeline

The example uses a prompt that might trigger classifier failures to demonstrate
the reflexion process when validation constraints are not met.
"""

import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from sifaka import SifakaEngine
from sifaka.classifiers import (
    create_emotion_classifier,
    create_intent_classifier,
    create_readability_classifier,
)
from sifaka.graph import SifakaDependencies
from sifaka.storage.file import SifakaFilePersistence
from sifaka.utils.logging import get_logger
from sifaka.validators import ContentValidator, LengthValidator
from sifaka.validators.classifier import create_classifier_validator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def create_comprehensive_validators():
    """Create a comprehensive set of validators including all classifiers."""
    return [
        # Basic content validators
        LengthValidator(min_length=100, max_length=800, name="length_check"),
        ContentValidator(required=["technology", "future"], name="content_requirements"),
        # Classifier-based validators - using non-cached versions to avoid numpy serialization issues
        create_classifier_validator(
            classifier=create_readability_classifier(cached=False),
            threshold=0.5,
            valid_labels=["high", "college", "graduate"],  # Require high readability
            name="readability_validation",
        ),
        create_classifier_validator(
            classifier=create_emotion_classifier(cached=False),
            threshold=0.4,
            invalid_labels=["anger", "fear", "disgust"],  # Block negative emotions
            name="emotion_validation",
        ),
        create_classifier_validator(
            classifier=create_intent_classifier(cached=False),
            threshold=0.3,
            valid_labels=["informative", "educational", "explanatory"],
            name="intent_validation",
        ),
    ]


async def main():
    """Run the Reflexion critic example with all classifiers."""

    logger.info("Starting Reflexion Critic with All Classifiers Example")

    # Ensure thoughts directory exists in the root directory
    thoughts_dir = Path("../thoughts")
    thoughts_dir.mkdir(exist_ok=True)
    logger.info(f"Thoughts will be saved to: {thoughts_dir.absolute()}")

    # Create comprehensive validators with all classifiers
    validators = create_comprehensive_validators()
    logger.info(f"Created {len(validators)} validators including all classifiers")

    # Set up file-based persistence for thoughts
    persistence = SifakaFilePersistence(
        storage_dir="../thoughts", file_prefix="reflexion_classifiers_"
    )

    # Create dependencies with Reflexion critic
    # Critics only run when validators fail (always_apply_critics=False)
    dependencies = SifakaDependencies(
        generator="openai:gpt-4o-mini",  # Fast, cost-effective generator
        validators=validators,
        critics={
            "reflexion": "anthropic:claude-3-5-haiku-20241022",  # Reflexion critic
        },
        # Key configuration: critics only run when validation fails
        always_apply_critics=False,  # This is the key setting!
        validation_weight=0.7,  # Prioritize validation feedback (70%)
        critic_weight=0.3,  # Reflexion suggestions (30%)
    )

    # Create the Sifaka engine with file persistence
    engine = SifakaEngine(dependencies=dependencies, persistence=persistence)

    # Test prompt that might trigger classifier failures
    # This prompt could potentially trigger bias, emotion, or intent validation failures
    test_prompt = """
    Write a comprehensive explanation about how artificial intelligence technology
    will completely revolutionize the future of humanity. Discuss both the amazing
    benefits and the potential risks that could be catastrophic if we're not careful.
    Make sure to cover the technological aspects and future implications.
    """

    logger.info("Running generation with potential validation challenges...")
    logger.info(
        "This prompt might trigger classifier failures, causing Reflexion critic to activate"
    )

    try:
        # Run the generation process
        result = await engine.think(
            prompt=test_prompt,
            max_iterations=3,  # Allow multiple iterations for improvement
        )

        # Display results
        print("\n" + "=" * 80)
        print("REFLEXION CRITIC WITH CLASSIFIERS - RESULTS")
        print("=" * 80)

        print(f"\nValidation Passed: {result.validation_passed()}")
        print(f"Total Iterations: {result.iteration}")
        print(f"Thought ID: {result.id}")

        # Use final_text if available, otherwise current_text
        final_text = result.final_text or result.current_text or "No text generated"
        print(f"\nFinal Text ({len(final_text)} characters):")
        print("-" * 50)
        print(final_text)

        # Show validation results
        if result.validations:
            print(f"\nValidation Results:")
            print("-" * 30)
            for validation in result.validations:
                status = "✅ PASSED" if validation.passed else "❌ FAILED"
                print(f"  {validation.validator}: {status}")

        # Show critic results if any were applied
        if result.critiques:
            print(f"\nCritic Results:")
            print("-" * 30)
            for critique in result.critiques:
                print(f"  {critique.critic}: Applied (confidence: {critique.confidence})")

        # Show thought file location
        thought_file = thoughts_dir / f"reflexion_classifiers_{result.id}.json"
        print(f"\nThought Information:")
        print(f"  - Thought ID: {result.id}")
        print(f"  - Total generations: {len(result.generations)}")
        print(f"  - Total validations: {len(result.validations)}")
        print(f"  - Total critiques: {len(result.critiques)}")
        print(f"  - Techniques applied: {result.techniques_applied}")

        if thought_file.exists():
            print(f"\nThought saved to: {thought_file}")
            print("You can examine the complete thought process including:")
            print("  - All validation attempts and results")
            print("  - Reflexion critic feedback (if triggered)")
            print("  - Generation iterations and improvements")
            print("  - Classifier analysis results")
        else:
            print(f"\nNote: Thought file should be saved to {thought_file}")

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise

    logger.info("Reflexion Critic with All Classifiers Example completed")


if __name__ == "__main__":
    asyncio.run(main())
