#!/usr/bin/env python3
"""Test script to verify if README examples actually work."""

import sys

# Import path is handled by proper package installation


def test_basic_readme_example():
    """Test the basic README example with Chain-orchestrated retrieval."""
    print("Testing basic README example...")

    try:
        from sifaka.core.chain import Chain
        from sifaka.validators.base import LengthValidator, RegexValidator
        from sifaka.critics.reflexion import ReflexionCritic
        from sifaka.models.base import create_model
        from sifaka.retrievers import MockRetriever

        print("✅ All imports successful")

        # Create a model
        model = create_model("mock:default")
        print("✅ Model creation successful")

        # Create validators and critics
        length_validator = LengthValidator(min_length=50, max_length=1000)
        content_validator = RegexValidator(forbidden_patterns=["violent", "harmful"])
        critic = ReflexionCritic(model=model)
        print("✅ Validators and critics creation successful")

        # Create a retriever
        retriever = MockRetriever()
        print("✅ Retriever creation successful")

        # Create a chain with model, prompt, retriever, validators and critics
        # The Chain orchestrates ALL retrieval automatically
        prompt = "Write a short story about a robot."
        chain = Chain(
            model=model,
            prompt=prompt,
            model_retrievers=[retriever],  # Chain handles model retrieval
            max_improvement_iterations=3,
            apply_improvers_on_validation_failure=True,
        )

        chain.validate_with(length_validator)
        chain.validate_with(content_validator)
        chain.improve_with(critic)
        print("✅ Chain creation and configuration successful")

        # Run the chain - it handles all retrieval automatically
        result = chain.run()
        print("✅ Chain execution successful")

        # Check the result
        print(f"Generated text: {result.text}")

        # Access validation results
        for name, validation_result in result.validation_results.items():
            print(f"{name}: {'Passed' if validation_result.passed else 'Failed'}")
            if validation_result.issues:
                print(f"Issues: {validation_result.issues}")

        print("✅ Basic README example works!")
        return True

    except Exception as e:
        print(f"❌ Basic README example failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_chain_orchestrated_retrieval():
    """Test the new Chain-orchestrated retrieval approach."""
    print("\nTesting Chain-orchestrated retrieval...")

    try:
        from sifaka.retrievers import InMemoryRetriever
        from sifaka.models.base import create_model
        from sifaka.critics.reflexion import ReflexionCritic
        from sifaka.core.chain import Chain

        # Create a retriever
        retriever = InMemoryRetriever()

        # Add documents to the retriever
        retriever.add_document(
            "doc1", "Robots are machines that can be programmed to perform tasks."
        )
        retriever.add_document(
            "doc2", "Asimov's Three Laws of Robotics are rules for robots in his science fiction."
        )
        retriever.add_document(
            "doc3", "Machine learning allows robots to learn from data and improve over time."
        )
        print("✅ Retriever setup successful")

        # Create a model and critic (no retriever needed - Chain handles it)
        model = create_model("mock:default")
        critic = ReflexionCritic(model=model)
        print("✅ Model and critic creation successful")

        # Create a Chain with the retriever - it orchestrates ALL retrieval
        chain = Chain(
            model=model,
            prompt="Write a short story about a robot that learns.",
            model_retrievers=[retriever],  # Chain handles model retrieval automatically
        )

        chain.improve_with(critic)
        print("✅ Chain setup successful")

        # Run the chain - it handles all retrieval automatically
        result = chain.run()
        print("✅ Chain execution successful")

        # Print the retrieved context
        print("Pre-generation context:")
        for doc in result.pre_generation_context:
            print(f"- {doc.text} (score: {doc.score})")

        print("\nPost-generation context:")
        for doc in result.post_generation_context:
            print(f"- {doc.text} (score: {doc.score})")

        print("✅ Chain-orchestrated retrieval works!")
        return True

    except Exception as e:
        print(f"❌ Chain-orchestrated retrieval failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_basic_readme_example()
    success2 = test_chain_orchestrated_retrieval()

    if success1 and success2:
        print("\n🎉 All README examples work as documented!")
        sys.exit(0)
    else:
        print("\n💥 Some README examples don't work")
        sys.exit(1)
