"""Basic chain example for Sifaka.

This example demonstrates how to use the Sifaka framework to create a simple chain
that generates text, validates it, and improves it using critics.

The example uses mock components for simplicity, but the same approach can be used
with real models, validators, and critics.
"""

import logging

from sifaka.chain import Chain
from sifaka.core.thought import Thought
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.models.base import create_model
from sifaka.retrievers.base import MockRetriever
from sifaka.utils.logging import configure_logging
from sifaka.validators.base import LengthValidator, RegexValidator


def print_thought_details(thought: Thought) -> None:
    """Print details about a thought.

    Args:
        thought: The thought to print details for.
    """
    print("\n" + "=" * 80)
    print(f"Thought (Iteration {thought.iteration})")
    print("=" * 80)

    print(f"Prompt: {thought.prompt}")

    if thought.pre_generation_context:
        print("\nPre-Generation Context:")
        for i, doc in enumerate(thought.pre_generation_context):
            print(f"  Document {i+1}: {doc.text[:100]}...")

    if thought.text:
        print(f"\nGenerated Text: {thought.text[:200]}...")

    if thought.validation_results:
        print("\nValidation Results:")
        for name, result in thought.validation_results.items():
            status = "PASSED" if result.passed else "FAILED"
            print(f"  {name}: {status} - {result.message}")
            if result.issues:
                print("    Issues:")
                for issue in result.issues:
                    print(f"      - {issue}")
            if result.suggestions:
                print("    Suggestions:")
                for suggestion in result.suggestions:
                    print(f"      - {suggestion}")

    if thought.critique:
        print("\nCritique:")
        if "issues" in thought.critique:
            print("  Issues:")
            for issue in thought.critique["issues"]:
                print(f"    - {issue}")
        if "suggestions" in thought.critique:
            print("  Suggestions:")
            for suggestion in thought.critique["suggestions"]:
                print(f"    - {suggestion}")

    if thought.post_generation_context:
        print("\nPost-Generation Context:")
        for i, doc in enumerate(thought.post_generation_context):
            print(f"  Document {i+1}: {doc.text[:100]}...")

    print("\n" + "-" * 80)


def main() -> None:
    """Run the example."""
    print("🚀 Basic Chain Example - Sifaka Framework")
    print("=" * 60)

    # Configure logging
    configure_logging(level=logging.INFO)

    # Create components
    print("\n📦 Setting up components...")

    # Create a mock model
    model = create_model("mock:example-model")

    # Create validators
    length_validator = LengthValidator(min_length=50, max_length=1000)
    regex_validator = RegexValidator(
        required_patterns=[r"artificial intelligence|AI", r"machine learning|ML"],
        forbidden_patterns=[r"harmful", r"dangerous"],
    )

    # Create a critic
    critic = ReflexionCritic(model_name="mock:critic-model")

    # Create a retriever with relevant documents
    retriever = MockRetriever(
        documents=[
            "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines.",
            "Machine Learning is a subset of AI that enables systems to learn and improve from experience.",
            "Natural Language Processing (NLP) is a field of AI that focuses on interactions between computers and human language.",
            "Deep Learning is a subset of machine learning that uses neural networks with many layers.",
            "Computer Vision is a field of AI that enables computers to derive meaningful information from digital images and videos.",
            "The future of AI includes advances in autonomous systems, personalized medicine, and smart cities.",
            "Ethical AI development focuses on fairness, transparency, and accountability in machine learning systems.",
            "AI applications are expanding into healthcare, finance, education, and environmental sustainability.",
        ]
    )

    # Create a prompt
    prompt = "Write a comprehensive paragraph about the future of artificial intelligence and machine learning technologies."

    print(f"✅ Components created successfully!")
    print(f"📝 Prompt: {prompt}")

    # Create and configure the chain using the new API
    print("\n🔗 Creating and configuring chain...")

    chain = Chain(
        model=model,
        prompt=prompt,
        retriever=retriever,  # Use the same retriever for all stages
        max_improvement_iterations=2,
        apply_improvers_on_validation_failure=True,
    )

    # Add validators and critics using the fluent API
    chain.validate_with(length_validator).validate_with(regex_validator).improve_with(critic)

    print("✅ Chain configured with validators and critics!")

    # Run the chain
    print("\n🏃 Running the chain...")
    result = chain.run()

    print("✅ Chain execution completed!")

    # Print the result
    print_thought_details(result)

    # Print history if available
    if result.history:
        print("\n📚 Thought History:")
        print("-" * 40)
        for i, historical_thought in enumerate(result.history):
            print(f"\n📖 Historical Thought {i+1} (Iteration {historical_thought.iteration}):")
            print(f"   Prompt: {historical_thought.prompt}")
            if historical_thought.text:
                print(f"   Text: {historical_thought.text[:100]}...")
            if historical_thought.validation_results:
                passed_count = sum(
                    1 for r in historical_thought.validation_results.values() if r.passed
                )
                total_count = len(historical_thought.validation_results)
                print(f"   Validations: {passed_count}/{total_count} passed")

    print(f"\n🎉 Example completed successfully!")


if __name__ == "__main__":
    main()
