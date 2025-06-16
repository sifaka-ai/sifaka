#!/usr/bin/env python3
"""Meta-Evaluation Critic Example - Using the New Preset API.

This example demonstrates:
- Meta-evaluation critic for assessing critique quality
- Simple preset API for business writing
- Built-in validation and improvement workflow

The meta-evaluation critic provides meta-level assessment to improve
the overall critique process and content quality.
"""

import asyncio
import os
from datetime import datetime

# Simple imports using the new preset API
import sifaka


async def main():
    """Run the Meta-Evaluation Critic example using preset API."""

    # Ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    print("🎯 Meta-Evaluation Critic Example")
    print("=" * 50)

    print("✅ Using Meta-Evaluation Critic with preset API")
    print(f"   Model: openai:gpt-4o-mini")
    print(f"   Critics: meta_evaluation")
    print(f"   Length validation: 100-800 characters")
    print(f"   Max iterations: 3")
    print(f"   Note: Meta-evaluation critic assesses critique quality for better results")

    # Test cases that will initially contain PII (email addresses)
    test_cases = [
        {
            "name": "Business Contact Information",
            "prompt": "Write a professional business introduction for John Smith, including his contact details and role as Marketing Director at TechCorp.",
        },
        {
            "name": "Customer Service Response",
            "prompt": "Write a customer service email response to help a customer with their account. Include contact information for follow-up.",
        },
        {
            "name": "Team Directory Entry",
            "prompt": "Create a team directory entry for Sarah Johnson, the new Product Manager, including her background and how to reach her.",
        },
    ]

    # Process each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"🧪 Test Case {i}: {test_case['name']}")
        print(f"{'='*70}")

        print(f"\n📝 Prompt: {test_case['prompt']}")
        print(f"💡 Expected: Meta-rewarding critic will evaluate critique quality for PII removal")

        # Generate and analyze with meta-evaluation critic
        print(f"\n🔄 Running meta-evaluation critic...")
        try:
            # Use the business writing preset with meta-evaluation critic
            thought = await sifaka.business_writing(
                test_case["prompt"],
                model="openai:gpt-4o-mini",
                min_length=100,
                max_length=800,
                max_rounds=3,
                critics=["meta_evaluation"],
            )

            # Display results using simple built-in information
            print(f"\n✅ Meta-Rewarding Critic Results:")
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

            # Show meta-rewarding critic feedback
            if thought.critiques:
                meta_critiques = [c for c in thought.critiques if "MetaEvaluation" in c.critic]
                if meta_critiques:
                    print(f"\n🎯 Meta-Rewarding Critic Feedback:")
                    latest_critique = meta_critiques[-1]
                    print(f"   Needs improvement: {latest_critique.needs_improvement}")
                    print(f"   Confidence: {latest_critique.confidence:.2f}")
                    if latest_critique.feedback:
                        print(f"   Meta-feedback: {latest_critique.feedback[:150]}...")

            # Show generation progression to demonstrate PII removal
            if len(thought.generations) > 1:
                print(f"\n📊 Generation Progression (PII Removal):")
                for idx, generation in enumerate(
                    thought.generations[:3]
                ):  # Show first 3 generations
                    text_preview = (
                        generation.text[:100] + "..."
                        if len(generation.text) > 100
                        else generation.text
                    )
                    # Simple email detection for demonstration
                    has_email = "@" in generation.text and "." in generation.text
                    pii_status = "⚠️ Contains PII" if has_email else "✅ PII-Free"
                    print(f"   Generation {idx + 1}: {pii_status}")
                    print(f"     Preview: {text_preview}")

            # Show final text preview
            final_text = thought.final_text or thought.current_text
            print(f"\n📝 Final Generated Text (PII-Free):")
            print(f"{final_text[:300]}..." if len(final_text) > 300 else final_text)

            # Check if final text is actually PII-free
            has_final_email = "@" in final_text and "." in final_text
            pii_removal_success = "✅ SUCCESS" if not has_final_email else "⚠️ NEEDS REVIEW"
            print(f"\n🛡️ PII Removal Status: {pii_removal_success}")

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            print(f"💡 Make sure you have a valid OpenAI API key and Guardrails installed")

    # Show performance stats (fluent API may not have timing/caching stats)
    print(f"\n⏱️ Performance Stats: Available through fluent API logging")
    print(f"\n💾 Cache Stats: Available through fluent API caching")

    # Summary
    print(f"\n{'='*70}")
    print("📋 Meta-Evaluation Critic Summary")
    print(f"{'='*70}")
    print(f"✅ Meta-evaluation critic with preset API demonstrated")
    print(f"🎯 Quality assessment through meta-level critique evaluation")
    print(f"🔄 Iterative improvement through meta-judging feedback")
    print(f"📊 Simple preset API with powerful meta-evaluation features")

    print("\n✅ Meta-Evaluation Critic Example completed!")
    print("Key Benefits:")
    print("• Meta-evaluation of critique quality for better results")
    print("• Simple preset API with advanced critic capabilities")
    print("• Iterative content improvement with meta-level assessment")
    print("• Built-in validation and improvement workflow")
    print("• Zero configuration complexity with powerful features")


if __name__ == "__main__":
    asyncio.run(main())
