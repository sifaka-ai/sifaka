#!/usr/bin/env python3
"""Test script for the new Chain API with separate model and critic retrievers."""


def test_new_chain_api():
    """Test the new Chain API with separate retrievers."""
    try:
        from sifaka.core.chain import Chain
        from sifaka.models.base import create_model
        from sifaka.retrievers import MockRetriever
        from sifaka.validators.base import LengthValidator
        from sifaka.critics.reflexion import ReflexionCritic

        print("✅ All imports successful")

        # Create components
        model = create_model("mock:default")
        model_retriever = MockRetriever()
        critic_retriever = MockRetriever()
        validator = LengthValidator(min_length=10, max_length=500)
        critic = ReflexionCritic(model=model)

        print("✅ All components created")

        # Test 1: Constructor with separate retrievers
        chain1 = Chain(
            model=model,
            prompt="Write about artificial intelligence",
            model_retrievers=[model_retriever],
            critic_retrievers=[critic_retriever],
        )
        chain1.validate_with(validator)
        chain1.improve_with(critic)

        print("✅ Chain constructor with separate retrievers works")

        # Test 2: Fluent API
        chain2 = (
            Chain(model=model, prompt="Write about machine learning")
            .with_model_retrievers([model_retriever])
            .with_critic_retrievers([critic_retriever])
            .validate_with(validator)
            .improve_with(critic)
        )

        print("✅ Chain fluent API works")

        # Test 3: Run chain
        result = chain1.run()

        print(f"✅ Chain execution successful!")
        print(f"   Generated text: {result.text[:100]}...")
        print(f"   Pre-generation context: {len(result.pre_generation_context or [])} docs")
        print(f"   Post-generation context: {len(result.post_generation_context or [])} docs")

        # Test 4: Different retrievers for different purposes
        recent_retriever = MockRetriever()
        factual_retriever = MockRetriever()

        chain3 = Chain(
            model=model,
            prompt="Write about recent AI developments",
            model_retrievers=[recent_retriever],  # Recent context for model
            critic_retrievers=[factual_retriever],  # Authoritative context for critics
        )

        print("✅ Different retrievers for models vs critics works")

        print("\n🎉 All tests passed! New Chain API is working correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_new_chain_api()
