#!/usr/bin/env python3
"""
Performance Monitoring Demo

This example demonstrates the new performance monitoring capabilities
added to the Sifaka Chain class.
"""

import json
from sifaka.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.retrievers.base import MockRetriever


def main():
    print("🚀 Performance Monitoring Demo - Sifaka Framework")
    print("=" * 60)

    # Create components
    print("\n📦 Setting up components...")
    model = create_model("mock:performance-test")
    retriever = MockRetriever()

    # Create validators
    length_validator = LengthValidator(min_length=50, max_length=500)
    regex_validator = RegexValidator(required_patterns=[r"performance|monitoring|metrics"])

    # Create critic
    critic = ReflexionCritic(model_name="mock:critic")

    print("✅ Components created successfully!")

    # Create and configure chain
    print("\n🔗 Creating and configuring chain...")
    chain = Chain(
        model=model,
        prompt="Write a paragraph about performance monitoring in AI systems.",
        retriever=retriever,
    )

    # Configure chain
    chain.validate_with(length_validator)
    chain.validate_with(regex_validator)
    chain.improve_with(critic)

    print("✅ Chain configured with performance monitoring!")

    # Clear any previous performance data
    chain.clear_performance_data()

    # Run the chain
    print("\n🏃 Running the chain with performance monitoring...")
    thought = chain.run()

    print("✅ Chain execution completed!")

    # Get performance summary
    print("\n📊 Performance Summary:")
    print("=" * 40)

    performance_summary = chain.get_performance_summary()

    if performance_summary.get("operations"):
        print(f"Total Operations Tracked: {len(performance_summary['operations'])}")
        print(f"Total Execution Time: {performance_summary.get('total_time', 0):.3f}s")
        print()

        # Show detailed operation timings
        print("🔍 Operation Timings:")
        operations = performance_summary["operations"]

        for op_name, metrics in sorted(
            operations.items(), key=lambda x: x[1].get("avg_time", 0), reverse=True
        ):
            avg_time = metrics.get("avg_time", 0)
            total_time = metrics.get("total_time", 0)
            call_count = metrics.get("call_count", 0)

            print(f"  • {op_name}:")
            print(f"    - Average Time: {avg_time:.3f}s")
            print(f"    - Total Time: {total_time:.3f}s")
            print(f"    - Call Count: {call_count}")
            print()
    else:
        print("No performance data available.")

    # Identify bottlenecks
    print("🐌 Performance Bottlenecks:")
    bottlenecks = chain.get_performance_bottlenecks()
    if bottlenecks:
        for bottleneck in bottlenecks:
            print(f"  • {bottleneck}")
    else:
        print("  • No significant bottlenecks detected (all operations < 100ms)")

    # Show final result
    print(f"\n📝 Final Result:")
    print(f"  • Text Length: {len(thought.text)} characters")
    print(f"  • Iterations: {thought.iteration + 1}")
    print(f"  • Validation Results: {len(thought.validation_results)}")
    print(f"  • Critic Feedback: {len(thought.critic_feedback) if thought.critic_feedback else 0}")

    # Save performance data to file
    print(f"\n💾 Saving performance data...")
    with open("performance_report.json", "w") as f:
        json.dump(performance_summary, f, indent=2, default=str)
    print("✅ Performance report saved to performance_report.json")

    print("\n🎉 Performance monitoring demo completed successfully!")


if __name__ == "__main__":
    main()
