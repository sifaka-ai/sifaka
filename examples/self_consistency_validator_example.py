#!/usr/bin/env python3
"""Self-Consistency Critic with Validator Integration Example.

This example demonstrates:
- Self-consistency critic with multiple independent critique attempts
- Validator integration for validation-aware consistency analysis
- Consensus building across different evaluation perspectives
- Cross-validation consistency checking for improved reliability

The self-consistency critic performs multiple independent assessments and builds
consensus while analyzing validation consistency across attempts.
"""

import asyncio
import os
from datetime import datetime

# Simple imports - no complex dependencies needed
import sifaka
from sifaka.fluent import Sifaka


def create_self_consistency_validator_sifaka(prompt):
    """Create self-consistency critic Sifaka instance with validator integration."""

    # Use the fluent API to create a Sifaka instance with self-consistency critic
    sifaka_instance = (
        Sifaka(prompt)
        .generator("openai:gpt-4o-mini")  # Fast, reliable model
        .with_self_consistency("openai:gpt-4o-mini")  # Use self-consistency critic
        .min_length(50)  # Minimum content length for validation
        .max_length(300)  # Maximum content length for validation
        .max_iterations(2)  # Allow iterations for consistency improvement
    )

    return sifaka_instance


async def main():
    """Run the Self-Consistency Critic with validator integration example."""

    # Ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    print("🔄 Self-Consistency Critic with Validator Integration")
    print("=" * 60)

    print("✅ Using Self-Consistency Critic with built-in validation")
    print(f"   Model: openai:gpt-4o-mini")
    print(f"   Critics: self_consistency with fluent API")
    print(f"   Validators: length validation (min: 50, max: 300)")
    print(f"   Consistency attempts: 3 independent assessments (default)")
    print(f"   Max iterations: 2 for consistency improvement")
    print(f"   Note: Self-consistency critic builds consensus across multiple attempts")

    # Test cases for self-consistency evaluation
    test_cases = [
        {
            "name": "Renewable Energy Overview",
            "prompt": "Write a brief overview of renewable energy sources and their benefits",
            "expected": "Multiple consistent assessments with validation-aware consensus",
        },
        {
            "name": "Short Energy Text",
            "prompt": "Write about renewable energy in exactly 25 words",
            "expected": "Consistency analysis with length validation failures",
        },
        {
            "name": "Incomplete Energy Draft",
            "prompt": "Write about renewable energy but include TODO placeholders for later completion",
            "expected": "Consistency analysis with content validation failures",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*60}")

        print(f"\n📝 Prompt: {test_case['prompt']}")
        print(f"💡 Expected: {test_case['expected']}")

        # Generate and analyze with self-consistency critic
        print(f"\n🔄 Running self-consistency critic with validator integration...")
        try:
            # Create Sifaka instance for this specific prompt
            sifaka_instance = create_self_consistency_validator_sifaka(test_case["prompt"])
            thought = await sifaka_instance.improve()

            # Display results using simple built-in information
            print(f"\n✅ Self-Consistency Critic Results:")
            print(f"Final text: {len(thought.final_text or thought.current_text)} characters")
            print(f"Iterations: {thought.iteration}")
            print(f"Validation passed: {thought.validation_passed()}")
            print(f"Total critiques: {len(thought.critiques)}")

            # Show validation results
            if thought.validations:
                print(f"\n🔍 Validation Results:")
                passed_validations = sum(1 for v in thought.validations if v.passed)
                total_validations = len(thought.validations)
                print(f"   Passed: {passed_validations}/{total_validations} validations")

                # Show failed validations
                failed_validations = [v for v in thought.validations if not v.passed]
                if failed_validations:
                    print(f"   Failed validations:")
                    for validation in failed_validations[-3:]:  # Show last 3 failures
                        print(f"     - {validation.validator}: {validation.feedback}")

            # Show self-consistency critic feedback
            if thought.critiques:
                consistency_critiques = [
                    c for c in thought.critiques if "SelfConsistency" in c.critic
                ]
                if consistency_critiques:
                    print(f"\n🔄 Self-Consistency Critic Feedback:")
                    latest_critique = consistency_critiques[-1]
                    print(f"   Needs improvement: {latest_critique.needs_improvement}")
                    print(f"   Confidence: {latest_critique.confidence:.2f}")
                    if latest_critique.feedback:
                        print(f"   Consensus feedback: {latest_critique.feedback[:200]}...")

                    # Show consistency metadata if available
                    if (
                        hasattr(latest_critique, "critic_metadata")
                        and latest_critique.critic_metadata
                    ):
                        metadata = latest_critique.critic_metadata
                        if "critique_attempts" in metadata:
                            print(f"   Critique attempts: {metadata['critique_attempts']}")
                            print(f"   Successful attempts: {metadata['successful_attempts']}")
                            print(f"   Improvement agreement: {metadata['improvement_agreement']}")

                        if "validation_consistency_enabled" in metadata:
                            print(
                                f"   Validation consistency: {metadata['validation_consistency_enabled']}"
                            )
                            if metadata.get("validation_consensus_score"):
                                print(
                                    f"   Validation consensus: {metadata['validation_consensus_score']:.2f}"
                                )

            # Show final text sample
            final_text = thought.final_text or thought.current_text
            if final_text:
                print(f"\n📄 Final Text Sample:")
                print(f"   {final_text[:150]}{'...' if len(final_text) > 150 else ''}")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print(f"   Type: {type(e).__name__}")

        print(f"\n⏱️ Completed at: {datetime.now().strftime('%H:%M:%S')}")

    # Summary
    print(f"\n{'='*70}")
    print("📋 Self-Consistency with Validator Integration Summary")
    print(f"{'='*70}")
    print(f"✅ Self-consistency critic with validator integration demonstrated")
    print(f"🔄 Multiple independent assessments with consensus building")
    print(f"🛡️ Validation-aware consistency analysis across attempts")
    print(f"📊 Cross-validation consistency checking for improved reliability")

    print("\n✅ Self-Consistency with Validator Integration completed!")
    print("Key Benefits:")
    print("• Multiple independent critique attempts for improved reliability")
    print("• Consensus building across different evaluation perspectives")
    print("• Validation-aware consistency analysis and confidence scoring")
    print("• Cross-validation consistency checking for enhanced accuracy")
    print("• Comprehensive metadata for observability and debugging")


if __name__ == "__main__":
    asyncio.run(main())
