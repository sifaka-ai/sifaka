#!/usr/bin/env python3
"""Simple example demonstrating the always_apply_critics feature.

This example shows how to use the new always_apply_critics option to run
critics even when validation passes, eliminating the need for artificial
validation failures just to trigger critic feedback.
"""

import os
from sifaka.chain import Chain
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.models.base import create_model
from sifaka.utils.logging import configure_logging
from sifaka.validators.base import LengthValidator


def main():
    """Demonstrate always_apply_critics functionality."""
    print("🎯 Always Apply Critics Example")
    print("=" * 50)

    # Configure logging
    configure_logging(level="INFO")

    # Set up a simple model (mock if no API key)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ Using OpenAI GPT-3.5-Turbo")
        model = create_model("openai:gpt-3.5-turbo", api_key=openai_key)
    else:
        print("⚠️  Using mock model (no OPENAI_API_KEY found)")
        model = create_model("mock:gpt-3.5-turbo")

    # Set up a simple validator that will pass
    length_validator = LengthValidator(min_length=50, max_length=1000)

    # Set up a constitutional critic
    constitutional_principles = [
        "The content should be clear and well-structured.",
        "The content should be engaging and informative.",
        "The content should use specific examples where appropriate.",
    ]

    critic = ConstitutionalCritic(
        model=model,
        principles=constitutional_principles,
        strict_mode=False,
    )

    # Simple prompt that should pass validation
    prompt = "Write a brief explanation of what machine learning is."

    print(f"\n📝 Prompt: {prompt}")
    print(f"✅ Validator: Length between 50-1000 characters (should pass)")
    print(f"⚖️  Critic: Constitutional critic with {len(constitutional_principles)} principles")

    # Test 1: Traditional behavior (critics only run on validation failure)
    print(f"\n🧪 Test 1: Traditional behavior (apply_improvers_on_validation_failure=True)")

    chain1 = Chain(
        model=model,
        prompt=prompt,
        max_improvement_iterations=2,
        apply_improvers_on_validation_failure=True,
        always_apply_critics=False,  # Traditional behavior
    )
    chain1.validate_with(length_validator)
    chain1.improve_with(critic)

    result1 = chain1.run()

    print(f"   Result: {len(result1.text)} characters")
    print(f"   Validation passed: {all(v.passed for v in result1.validation_results.values())}")
    print(f"   Critics ran: {'Yes' if result1.critic_feedback else 'No'}")
    print(f"   Iterations: {result1.iteration + 1}")

    # Test 2: New behavior (critics always run)
    print(f"\n🧪 Test 2: New behavior (always_apply_critics=True)")

    chain2 = Chain(
        model=model,
        prompt=prompt,
        max_improvement_iterations=2,
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,  # NEW: Always run critics!
    )
    chain2.validate_with(length_validator)
    chain2.improve_with(critic)

    result2 = chain2.run()

    print(f"   Result: {len(result2.text)} characters")
    print(f"   Validation passed: {all(v.passed for v in result2.validation_results.values())}")
    print(f"   Critics ran: {'Yes' if result2.critic_feedback else 'No'}")
    print(f"   Iterations: {result2.iteration + 1}")

    # Test 3: Critics without any validators
    print(f"\n🧪 Test 3: Critics without validators (always_apply_critics=True)")

    chain3 = Chain(
        model=model,
        prompt=prompt,
        max_improvement_iterations=2,
        always_apply_critics=True,  # Run critics even with no validators
    )
    # No validators added!
    chain3.improve_with(critic)

    result3 = chain3.run()

    print(f"   Result: {len(result3.text)} characters")
    print(
        f"   Validators: {len(result3.validation_results) if result3.validation_results else 0} (none added)"
    )
    print(f"   Critics ran: {'Yes' if result3.critic_feedback else 'No'}")
    print(f"   Iterations: {result3.iteration + 1}")

    # Summary
    print(f"\n📊 Summary:")
    print(f"   Test 1 (traditional): Critics ran = {'Yes' if result1.critic_feedback else 'No'}")
    print(f"   Test 2 (always_apply): Critics ran = {'Yes' if result2.critic_feedback else 'No'}")
    print(f"   Test 3 (no validators): Critics ran = {'Yes' if result3.critic_feedback else 'No'}")

    print(f"\n🎉 Example completed!")
    print(f"💡 Key insight: always_apply_critics=True allows critics to improve text")
    print(f"   even when validation passes or when no validators are present!")


if __name__ == "__main__":
    main()
