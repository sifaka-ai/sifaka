#!/usr/bin/env python3
"""Test script for validation constraint prioritization across all critics.

This script tests that all enhanced critics (SelfRefineCritic, ConstitutionalCritic,
PromptCritic, SelfConsistencyCritic) properly prioritize validation constraints.
"""

import os

from sifaka.core.chain import Chain
from sifaka.models.gemini import GeminiModel
from sifaka.validators import LengthValidator


def test_critic_validation_prioritization(critic_name, critic, model):
    """Test a specific critic's validation constraint prioritization."""

    print(f"\n🧪 Testing {critic_name}")
    print("=" * 50)

    # Create validator with tight constraint
    validator = LengthValidator(min_length=50, max_length=200)
    print("✓ Created length validator (50-200 chars)")

    # Test prompt that will generate long content
    prompt = (
        "Write a detailed explanation of quantum computing, including its principles, "
        "applications, advantages over classical computing, and future prospects."
    )

    # Create chain
    chain = (
        Chain(model=model)
        .with_prompt(prompt)
        .validate_with(validator)
        .improve_with(critic)
        .with_options(
            max_iterations=2, always_apply_critics=False, apply_improvers_on_validation_failure=True
        )
    )
    print(f"✓ Created chain with {critic_name}")

    print(f"\n📝 Test prompt: {prompt[:60]}...")
    print(f"🎯 Target: 50-200 characters")

    # Run the chain
    print(f"\n🚀 Running chain with {critic_name}...")
    result = chain.run()

    # Analyze results
    length = len(result.text)
    within_range = 50 <= length <= 200

    print(f"\n📊 Results:")
    print(f"  Final length: {length} characters")
    print(f"  Target range: 50-200 characters")
    print(f"  Within range: {'✅ YES' if within_range else '❌ NO'}")
    print(f"  Iterations: {result.iteration}")

    # Show validation results
    if hasattr(result, "validation_results") and result.validation_results:
        for name, validation_result in result.validation_results.items():
            status = "✅ PASSED" if validation_result.passed else "❌ FAILED"
            print(f"  Validation: {status}")

    print(f"\n📄 Final Text ({length} chars):")
    print("-" * 30)
    print(result.text)
    print("-" * 30)

    return within_range, length, result.iteration


def main():
    """Test all critics for validation constraint prioritization."""

    print("🧪 Testing All Critics - Validation Constraint Prioritization")
    print("=" * 70)

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        return

    # Create model - using fast Gemini Flash model
    model = GeminiModel(
        model_name="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )
    print("✓ Created Gemini model (gemini-1.5-flash)")

    # Import all enhanced critics
    from sifaka.critics.meta_rewarding import MetaRewardingCritic
    from sifaka.critics.n_critics import NCriticsCritic
    from sifaka.critics.reflexion import ReflexionCritic
    from sifaka.critics.self_rag import SelfRAGCritic

    # Test only the critics we weren't able to test due to network timeout
    print("🎯 Testing only the critics that failed due to network timeout...")
    critics_to_test = [
        ("ReflexionCritic", ReflexionCritic(model=model)),
        ("MetaRewardingCritic", MetaRewardingCritic(model=model)),
        (
            "NCriticsCritic",
            NCriticsCritic(
                model=model, critic_roles=["clarity", "accuracy"]  # Reduced for faster testing
            ),
        ),
        ("SelfRAGCritic", SelfRAGCritic(model=model)),
    ]

    results = {}

    try:
        for critic_name, critic in critics_to_test:
            within_range, length, iterations = test_critic_validation_prioritization(
                critic_name, critic, model
            )
            results[critic_name] = {
                "within_range": within_range,
                "length": length,
                "iterations": iterations,
            }

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return

    # Summary
    print(f"\n🎉 Test Summary")
    print("=" * 50)

    successful_critics = []
    failed_critics = []

    for critic_name, result in results.items():
        status = "✅ SUCCESS" if result["within_range"] else "❌ FAILED"
        print(
            f"{critic_name:20} | {status} | {result['length']:3d} chars | {result['iterations']} iterations"
        )

        if result["within_range"]:
            successful_critics.append(critic_name)
        else:
            failed_critics.append(critic_name)

    print(f"\n📊 Overall Results:")
    print(f"✅ Successful: {len(successful_critics)}/{len(results)} critics")
    print(f"❌ Failed: {len(failed_critics)}/{len(results)} critics")

    if successful_critics:
        print(f"✅ Working critics: {', '.join(successful_critics)}")

    if failed_critics:
        print(f"❌ Need adjustment: {', '.join(failed_critics)}")

    # Final assessment
    success_rate = len(successful_critics) / len(results)
    if success_rate >= 0.75:
        print(
            f"\n🎉 EXCELLENT: {success_rate:.0%} of critics successfully prioritize validation constraints!"
        )
    elif success_rate >= 0.5:
        print(f"\n👍 GOOD: {success_rate:.0%} of critics working, some need fine-tuning")
    else:
        print(f"\n⚠️  NEEDS WORK: Only {success_rate:.0%} of critics working properly")


if __name__ == "__main__":
    main()
