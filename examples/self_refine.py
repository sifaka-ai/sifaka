#!/usr/bin/env python3
"""Self-Refine Example with PydanticAI-native Sifaka.

This example demonstrates:
- SifakaEngine with PydanticAI integration
- Self-Refine critic for iterative improvement
- Comprehensive set of validators for quality assurance
- Modern PydanticAI-native architecture
- Proper storage and thought inspection

The engine will generate content about software engineering best practices
and use multiple validators to ensure high-quality, comprehensive output.
The Self-Refine critic provides iterative feedback for continuous improvement.
"""

import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from sifaka import SifakaEngine
from sifaka.graph import SifakaDependencies
from sifaka.validators import LengthValidator, ContentValidator
from sifaka.storage import SifakaFilePersistence
from sifaka.utils.thought_inspector import (
    print_iteration_details,
    get_thought_overview,
)
from sifaka.utils.logging import get_logger

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def create_comprehensive_validators():
    """Create a comprehensive set of validators for quality assurance."""

    validators = []

    # 1. Length Validator - Ensure substantial content (CHALLENGING)
    length_validator = LengthValidator(
        min_length=1200,  # Minimum 1200 characters for comprehensive coverage
        max_length=3000,  # Maximum 3000 characters to stay focused
    )
    validators.append(length_validator)

    # 2. Content Validator - Check for required topics and forbidden content
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
        ],
        prohibited=["hack", "exploit", "malicious"],
        case_sensitive=False,  # Make matching case-insensitive
    )
    validators.append(content_validator)

    return validators


def show_self_refine_progression(thought):
    """Show how the text improved through Self-Refine iterations."""
    print("\n🔄 Self-Refine Progression:")

    # Show progression by iteration
    for iteration in range(thought.iteration + 1):
        # Get generations for this iteration
        iteration_generations = [g for g in thought.generations if g.iteration == iteration]
        if not iteration_generations:
            continue

        latest_gen = iteration_generations[-1]

        # Get validation status
        iteration_validations = [v for v in thought.validations if v.iteration == iteration]
        passed = sum(1 for v in iteration_validations if v.passed)
        total = len(iteration_validations)

        # Get Self-Refine feedback
        iteration_critiques = [
            c
            for c in thought.critiques
            if c.iteration == iteration and c.critic == "SelfRefineCritic"
        ]

        print(
            f"  Iteration {iteration}: {len(latest_gen.text)} chars, "
            f"validation {passed}/{total}, ",
            end="",
        )

        if iteration_critiques:
            critique = iteration_critiques[-1]
            print(f"confidence {critique.confidence:.2f}")
        else:
            print("no critique")


async def main():
    """Run the Self-Refine example using modern Sifaka architecture."""

    # Ensure API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    logger.info("Creating Self-Refine example with PydanticAI-native Sifaka")

    # Create comprehensive validators
    validators = create_comprehensive_validators()

    # Set up storage for thoughts
    storage = SifakaFilePersistence("thoughts")

    # Create dependencies with Self-Refine critic focus
    dependencies = SifakaDependencies(
        generator="anthropic:claude-3-5-haiku-20241022",
        validators=validators,
        critics={
            "self_refine": "anthropic:claude-3-5-haiku-20241022",  # Use same model for consistency
        },
        # Configuration for Self-Refine behavior
        always_apply_critics=False,  # Always apply the Self-Refine critic
        validation_weight=0.7,  # Prioritize validation feedback (70%)
        critic_weight=0.3,  # Self-Refine suggestions (30%)
    )

    # Create Sifaka engine
    engine = SifakaEngine(dependencies=dependencies, persistence=storage)

    # Define the prompt designed to initially fail validation
    prompt = "Write about software engineering. Give me some tips."

    # Run the engine with multiple iterations for Self-Refine to work
    print(f"🤖 Self-Refine Example")
    print(f"Prompt: {prompt}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    thought = await engine.think(prompt, max_iterations=3)

    # Display results using thought inspector utilities
    print("\n✅ Self-Refine Results:")

    # Show overview
    overview = get_thought_overview(thought)
    print(f"Final text: {overview['final_text_length']} characters")
    print(f"Iterations: {overview['current_iteration'] + 1}")
    print(f"Validation passed: {overview['validation_passed']}")
    print(f"Total critiques: {overview['total_critiques']}")

    # Show Self-Refine progression
    show_self_refine_progression(thought)

    # Show latest iteration details for debugging
    print("\n🔍 Latest Iteration Details:")
    print_iteration_details(thought)

    # Show storage information
    print(f"\n💾 Thought saved to: {storage.storage_dir}")
    print(f"Thought ID: {thought.id}")

    # Demonstrate thought retrieval using model_dump
    print(f"\n🔍 Thought data available via:")
    print(f"• thought.model_dump() - Full serializable dict")
    print(f"• thought.model_dump_json() - JSON string")
    print(f"• Storage backend - Automatic persistence")

    logger.info("Self-Refine example completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
