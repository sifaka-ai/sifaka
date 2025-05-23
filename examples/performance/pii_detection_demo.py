#!/usr/bin/env python3
"""
PII Detection Performance Demo with OpenAI and Guardrails

This example demonstrates:
1. Using OpenAI to generate content that likely contains PII (phone numbers, emails)
2. Using Guardrails PII validator to detect the PII
3. Using ReflexionCritic to help remove PII from the content
4. Performance monitoring throughout the entire process

This showcases real-world usage where AI-generated content needs to be sanitized
for privacy compliance.
"""

import json
import os
from dotenv import load_dotenv

from sifaka.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators.guardrails import GuardrailsValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.retrievers.base import MockRetriever

# Load environment variables
load_dotenv()


def main():
    print("🔒 PII Detection Performance Demo - Sifaka Framework")
    print("=" * 65)

    # Check for required API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    guardrails_key = os.getenv("GUARDRAILS_API_KEY")

    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Please set your OpenAI API key to run this demo")
        return

    print("✅ OpenAI API key found")

    if not guardrails_key:
        print("⚠️  GUARDRAILS_API_KEY not found - will try without it")
    else:
        print("✅ Guardrails API key found")

    print("\n📦 Setting up components...")

    # Create OpenAI model for generation - MUST use OpenAI
    model = create_model("openai:gpt-4")
    print("✅ OpenAI GPT-4 model created")

    # Create Guardrails PII validator
    try:
        pii_validator = GuardrailsValidator(
            validators=["GuardrailsPII"],
            validator_args={
                "GuardrailsPII": {"entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "SSN"]}
            },
            api_key=guardrails_key,
            name="PII_Detector",
        )
        print("✅ Guardrails PII validator created")
    except Exception as e:
        print(f"❌ Failed to create Guardrails validator: {e}")
        print("   This demo requires guardrails-ai to be installed")
        print("   Install with: pip install guardrails-ai")
        return

    # Create ReflexionCritic using OpenAI for PII removal
    critic = ReflexionCritic(model_name="openai:gpt-4")
    print("✅ ReflexionCritic with OpenAI GPT-4 created")

    # Create retriever for context
    retriever = MockRetriever()

    print("✅ All components created successfully!")

    # Create a prompt that will definitely generate PII
    prompt = """Write a realistic customer service email response that includes:

1. A customer's full name (make one up)
2. A direct phone number for support
3. An email address for follow-up
4. A customer account number or ID

Make it sound like a real customer service response with actual contact details."""

    print(f"\n🔗 Creating chain with PII-generating prompt...")

    # Create and configure chain
    chain = Chain(model=model, prompt=prompt, retriever=retriever)

    # Add PII validator
    chain.validate_with(pii_validator)

    # Add ReflexionCritic for PII removal
    chain.improve_with(critic)

    # Configure chain to apply critics when validation fails
    chain.with_options(
        always_apply_critics=False,
        apply_improvers_on_validation_failure=True,
        max_improvement_iterations=3,
    )

    print("✅ Chain configured with OpenAI + PII detection + ReflexionCritic!")

    # Clear performance data
    chain.clear_performance_data()

    print("\n🏃 Running chain with performance monitoring...")
    print("   Expected flow:")
    print("   1. OpenAI generates content with PII")
    print("   2. Guardrails detects PII → validation fails")
    print("   3. ReflexionCritic removes PII")
    print("   4. Re-validate until clean")

    # Run the chain
    thought = chain.run()
    print("✅ Chain execution completed!")

    # Display results
    print(f"\n📊 Results Summary:")
    print("=" * 50)
    print(f"Final Iteration: {thought.iteration + 1}")
    print(f"Text Length: {len(thought.text)} characters")
    print(f"Validation Results: {len(thought.validation_results)}")
    print(f"Critic Feedback: {len(thought.critic_feedback) if thought.critic_feedback else 0}")

    # Show PII validation results
    print(f"\n🔍 PII Validation Results:")
    for validator_name, result in thought.validation_results.items():
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"  • {validator_name}: {status}")
        if not result.passed and result.issues:
            for issue in result.issues[:3]:
                print(f"    - {issue}")

    # Show final text (first 400 chars)
    print(f"\n📝 Final Generated Text (OpenAI):")
    print("-" * 50)
    final_text = thought.text[:400] + "..." if len(thought.text) > 400 else thought.text
    print(final_text)
    print("-" * 50)

    # Performance analysis
    print(f"\n📊 Performance Analysis:")
    print("=" * 40)

    performance_summary = chain.get_performance_summary()

    print(f"Total Operations: {len(performance_summary['operations'])}")
    print(f"Total Execution Time: {performance_summary.get('total_time', 0):.3f}s")

    # Show OpenAI-specific timings
    operations = performance_summary["operations"]
    openai_operations = ["text_generation", "improvement_generation"]

    print(f"\n🤖 OpenAI Operation Timings:")
    for op_name in openai_operations:
        if op_name in operations:
            metrics = operations[op_name]
            avg_time = metrics.get("avg_time", 0)
            call_count = metrics.get("call_count", 0)
            print(f"  • {op_name}: {avg_time:.3f}s avg ({call_count} calls)")

    # Show PII detection timings
    pii_operations = [op for op in operations.keys() if "PII" in op or "validation" in op]
    if pii_operations:
        print(f"\n🔒 PII Detection Timings:")
        for op_name in pii_operations:
            metrics = operations[op_name]
            avg_time = metrics.get("avg_time", 0)
            call_count = metrics.get("call_count", 0)
            print(f"  • {op_name}: {avg_time:.3f}s avg ({call_count} calls)")

    # Show critic timings
    critic_operations = [op for op in operations.keys() if "critic" in op]
    if critic_operations:
        print(f"\n🧠 ReflexionCritic Timings:")
        for op_name in critic_operations:
            metrics = operations[op_name]
            avg_time = metrics.get("avg_time", 0)
            call_count = metrics.get("call_count", 0)
            print(f"  • {op_name}: {avg_time:.3f}s avg ({call_count} calls)")

    # Bottleneck analysis
    print(f"\n🐌 Performance Bottlenecks:")
    bottlenecks = chain.get_performance_bottlenecks()
    if bottlenecks:
        for bottleneck in bottlenecks:
            print(f"  • {bottleneck}")
    else:
        print("  • No significant bottlenecks detected")

    # Privacy compliance check
    print(f"\n🔒 Privacy Compliance Check:")
    print("=" * 40)

    final_validation = thought.validation_results.get("PII_Detector")
    if final_validation and final_validation.passed:
        print("✅ SUCCESS: PII successfully removed by ReflexionCritic")
        print("   Content is now privacy-compliant")
    elif final_validation and not final_validation.passed:
        print("❌ WARNING: PII still detected in final output")
        print("   May need additional improvement iterations")
    else:
        print("⚠️  No PII validation data available")

    # Save performance report
    print(f"\n💾 Saving performance report...")
    with open("pii_detection_performance_report.json", "w") as f:
        json.dump(performance_summary, f, indent=2, default=str)
    print("✅ Report saved to pii_detection_performance_report.json")

    print(f"\n🎉 OpenAI + PII Detection Demo completed!")
    print(f"   Demonstrated: OpenAI generation → PII detection → ReflexionCritic removal")


if __name__ == "__main__":
    main()
