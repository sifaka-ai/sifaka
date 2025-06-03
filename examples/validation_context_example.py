#!/usr/bin/env python3
"""Validation Context Example for Sifaka.

This example demonstrates the new ValidationContext system that ensures
validation failures take priority over critic suggestions, preventing
wasted iterations and ensuring constraint compliance.

Key features demonstrated:
- Validation-aware prompting with priority hierarchy
- Automatic filtering of conflicting critic suggestions
- Critical vs regular validation failure handling
- Enhanced improvement prompts with validation context

Prerequisites:
1. Set environment variables: ANTHROPIC_API_KEY or OPENAI_API_KEY
"""

import asyncio
import os
from pathlib import Path

# Add the project root to the path so we can import sifaka
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sifaka.core.engine import SifakaEngine
from sifaka.graph.dependencies import SifakaDependencies
from sifaka.validators import LengthValidator, ContentValidator
from sifaka.critics import ReflexionCritic, ConstitutionalCritic
from sifaka.utils import (
    configure_for_development,
    get_logger,
)
from sifaka.core.thought import ValidationContext, create_validation_context

# Configure logging
configure_for_development()
logger = get_logger(__name__)


async def demonstrate_validation_context():
    """Demonstrate ValidationContext with conflicting validation and critic feedback."""
    print("\n" + "=" * 70)
    print("VALIDATION CONTEXT DEMONSTRATION")
    print("=" * 70)

    # Create validators that will likely fail
    length_validator = LengthValidator(
        min_length=50,
        max_length=200,  # Short limit to trigger length constraint
        name="strict_length",
    )

    content_validator = ContentValidator(
        required_patterns=["renewable energy", "solar"], name="content_requirements"
    )

    # Create critics that might suggest conflicting improvements
    reflexion_critic = ReflexionCritic(model_name="anthropic:claude-3-haiku-20240307")

    constitutional_critic = ConstitutionalCritic(model_name="anthropic:claude-3-haiku-20240307")

    # Create dependencies with both validators and critics
    dependencies = SifakaDependencies(
        validators=[length_validator, content_validator],
        critics={"reflexion": reflexion_critic, "constitutional": constitutional_critic},
    )

    # Create engine
    engine = SifakaEngine(dependencies=dependencies)

    print("✅ Created SifakaEngine with validation-aware critics")
    print(f"   - Validators: {len(dependencies.validators)}")
    print(f"   - Critics: {len(dependencies.critics)}")
    print(f"   - Length limit: {length_validator.max_length} characters")

    # Test with a prompt that will likely generate long text
    prompt = """Write a comprehensive guide about renewable energy sources including
    detailed explanations, examples, case studies, and implementation strategies."""

    print(f"\n📝 Processing prompt (likely to exceed {length_validator.max_length} char limit):")
    print(f"   '{prompt[:60]}...'")

    try:
        # Process the thought
        thought = await engine.think(prompt, max_iterations=3)

        print(f"\n✅ Thought processing completed!")
        print(f"   Final iteration: {thought.iteration}")
        print(f"   Final text length: {len(thought.final_text) if thought.final_text else 0}")

        # Analyze the validation context behavior
        await analyze_validation_context_behavior(thought)

        return thought

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        logger.error("Validation context demonstration failed", exc_info=True)
        return None


async def analyze_validation_context_behavior(thought):
    """Analyze how ValidationContext influenced the thought processing."""
    print(f"\n" + "=" * 50)
    print("VALIDATION CONTEXT ANALYSIS")
    print("=" * 50)

    # Check each iteration for validation context behavior
    for iteration in range(thought.iteration + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Get validations for this iteration
        iteration_validations = [v for v in thought.validations if v.iteration == iteration]
        iteration_critiques = [c for c in thought.critiques if c.iteration == iteration]

        if iteration_validations:
            print(f"Validations ({len(iteration_validations)}):")
            for validation in iteration_validations:
                status = "✅ PASSED" if validation.passed else "❌ FAILED"
                print(f"  - {validation.validator}: {status}")
                if not validation.passed:
                    print(f"    Details: {validation.details}")

        if iteration_critiques:
            print(f"Critiques ({len(iteration_critiques)}):")
            for critique in iteration_critiques:
                print(f"  - {critique.critic}: {len(critique.suggestions)} suggestions")
                if "filtered" in critique.feedback.lower():
                    print(f"    🔍 Suggestions were filtered due to validation constraints")

                # Show first few suggestions
                for i, suggestion in enumerate(critique.suggestions[:2]):
                    print(f"    {i+1}. {suggestion[:60]}...")

        # Demonstrate ValidationContext extraction
        if iteration < thought.iteration:  # Don't analyze final iteration
            print(f"\n🔍 ValidationContext Analysis:")

            # Create a temporary thought state for this iteration
            temp_thought = thought
            temp_thought.iteration = iteration

            validation_context = ValidationContext.extract_constraints(temp_thought)
            if validation_context:
                has_critical = ValidationContext.has_critical_constraints(validation_context)
                print(f"  - Has validation constraints: Yes")
                print(f"  - Has critical constraints: {'Yes' if has_critical else 'No'}")
                print(f"  - Failed validations: {validation_context.get('total_failures', 0)}")

                if has_critical:
                    print(f"  - 🚨 Critical validation failures detected!")
                    print(f"  - Critic suggestions would be filtered/deprioritized")
            else:
                print(f"  - Has validation constraints: No")


async def demonstrate_suggestion_filtering():
    """Demonstrate how ValidationContext filters conflicting suggestions."""
    print(f"\n" + "=" * 50)
    print("SUGGESTION FILTERING DEMONSTRATION")
    print("=" * 50)

    # Create a mock validation context with length constraints
    mock_constraints = {
        "type": "validation_failures",
        "failed_validations": [
            {
                "validator_name": "LengthValidator",
                "message": "Text too long: 350 characters (maximum: 200)",
                "suggestions": ["Reduce content length", "Remove unnecessary details"],
                "issues": ["exceeds_length_limit"],
                "score": 0.0,
            }
        ],
        "total_failures": 1,
    }

    # Mock critic suggestions that would conflict with length constraints
    conflicting_suggestions = [
        "Add more detailed examples and case studies",
        "Provide comprehensive explanations for each point",
        "Include additional background information",
        "Expand on the technical details",
        "Simplify the language for better clarity",  # This one should pass
        "Focus on the main points",  # This one should pass
        "Remove redundant information",  # This one should pass
    ]

    print("Original critic suggestions:")
    for i, suggestion in enumerate(conflicting_suggestions, 1):
        print(f"  {i}. {suggestion}")

    # Filter suggestions
    filtered_suggestions = ValidationContext.filter_conflicting_suggestions(
        conflicting_suggestions, mock_constraints
    )

    print(
        f"\nFiltered suggestions (removed {len(conflicting_suggestions) - len(filtered_suggestions)} conflicting):"
    )
    for i, suggestion in enumerate(filtered_suggestions, 1):
        print(f"  {i}. {suggestion}")

    # Show what was filtered out
    removed_suggestions = [s for s in conflicting_suggestions if s not in filtered_suggestions]
    if removed_suggestions:
        print(f"\nRemoved suggestions (conflict with length constraints):")
        for i, suggestion in enumerate(removed_suggestions, 1):
            print(f"  {i}. {suggestion}")


async def main():
    """Run the ValidationContext demonstration."""
    print("Sifaka ValidationContext Example")
    print("=" * 70)
    print("This example demonstrates validation-aware prompting that ensures")
    print("validation constraints take priority over critic suggestions.")
    print("=" * 70)

    # Check for API keys
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("⚠️  Warning: No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        print("   The example will still demonstrate suggestion filtering.")

        # Just run the filtering demonstration
        await demonstrate_suggestion_filtering()
        return

    try:
        # Run the full demonstration
        thought = await demonstrate_validation_context()

        # Run the filtering demonstration
        await demonstrate_suggestion_filtering()

        if thought:
            print(f"\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"✅ ValidationContext successfully demonstrated!")
            print(f"   - Processed {thought.iteration + 1} iterations")
            print(f"   - Applied {len(thought.validations)} validations")
            print(f"   - Applied {len(thought.critiques)} critiques")
            print(
                f"   - Final text: {len(thought.final_text) if thought.final_text else 0} characters"
            )

            if thought.final_text and len(thought.final_text) <= 200:
                print(f"   - ✅ Length constraint satisfied!")
            elif thought.final_text:
                print(f"   - ⚠️  Length constraint not fully satisfied")

    except Exception as e:
        logger.error("ValidationContext example failed", exc_info=True)
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
