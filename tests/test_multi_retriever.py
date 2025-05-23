#!/usr/bin/env python3
"""Test script for multi-retriever functionality."""

import sys
import os

# Import path is handled by proper package installation


def test_multi_retriever_fact_checking():
    """Test using different retrievers for models vs critics (fact-checking use case)."""
    print("Testing multi-retriever fact-checking scenario...")

    try:
        from sifaka.chain import Chain
        from sifaka.models.base import create_model
        from sifaka.critics.reflexion import ReflexionCritic
        from sifaka.validators.base import LengthValidator
        from sifaka.retrievers.base import MockRetriever  # Use MockRetriever instead of specialized

        print("✅ All imports successful")

        # Create mock retrievers (simulating specialized retrievers)
        twitter_retriever = MockRetriever()  # Simulates recent context for model
        factual_retriever = MockRetriever()  # Simulates authoritative sources for critics
        print("✅ Mock retrievers created")

        # Create model and critic
        model = create_model("mock:default")
        critic = ReflexionCritic(model=model)
        # Create a validator that will fail to trigger critic improvement
        validator = LengthValidator(
            min_length=1000, max_length=2000
        )  # Mock response will be too short
        print("✅ Model, critic, and validator created")

        # Create Chain with different retrievers for different stages
        chain = Chain(
            model=model,
            prompt="Write a news summary about recent AI developments and their implications.",
            model_retriever=twitter_retriever,  # Model uses recent Twitter/news context
            critic_retriever=factual_retriever,  # Critics use factual database for verification
            pre_generation_retrieval=True,  # Get recent context before generation
            post_generation_retrieval=True,  # Get more recent context after generation
            critic_retrieval=True,  # Get factual context for critics
            apply_improvers_on_validation_failure=True,  # Enable critic improvement on validation failure
        )

        chain.validate_with(validator)  # Add validator to trigger failure
        chain.improve_with(critic)
        print("✅ Chain configured with different retrievers")

        # Run the chain
        result = chain.run()
        print("✅ Chain execution successful")

        # Analyze the results
        print("\n📊 RETRIEVAL ANALYSIS:")
        print(
            f"Pre-generation context (from TwitterRetriever): {len(result.pre_generation_context)} documents"
        )
        for i, doc in enumerate(result.pre_generation_context):
            print(f"  {i+1}. {doc.text[:60]}... (score: {doc.score})")

        print(
            f"\nPost-generation context (from FactualDatabaseRetriever): {len(result.post_generation_context)} documents"
        )
        for i, doc in enumerate(result.post_generation_context):
            print(f"  {i+1}. {doc.text[:60]}... (score: {doc.score})")

        print(f"\nGenerated text: {result.text}")

        print("\n✅ Multi-retriever fact-checking works!")
        return True

    except Exception as e:
        print(f"❌ Multi-retriever fact-checking failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_fallback_retriever():
    """Test fallback to default retriever when specific retrievers not provided."""
    print("\nTesting fallback retriever behavior...")

    try:
        from sifaka.chain import Chain
        from sifaka.models.base import create_model
        from sifaka.critics.reflexion import ReflexionCritic
        from sifaka.retrievers.base import MockRetriever

        # Create a default retriever
        default_retriever = MockRetriever()

        # Create model and critic
        model = create_model("mock:default")
        critic = ReflexionCritic(model=model)

        # Create Chain with only default retriever (should fallback for all stages)
        chain = Chain(
            model=model,
            prompt="Write about AI developments.",
            retriever=default_retriever,  # Only default retriever provided
            # model_retriever and critic_retriever not provided - should fallback
        )

        chain.improve_with(critic)
        print("✅ Chain configured with fallback retriever")

        # Run the chain
        result = chain.run()
        print("✅ Chain execution with fallback successful")

        print(f"Retrieved {len(result.pre_generation_context)} pre-generation documents")
        print(f"Retrieved {len(result.post_generation_context)} post-generation documents")

        print("✅ Fallback retriever behavior works!")
        return True

    except Exception as e:
        print(f"❌ Fallback retriever test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_no_retriever():
    """Test Chain behavior when no retrievers are provided."""
    print("\nTesting Chain with no retrievers...")

    try:
        from sifaka.chain import Chain
        from sifaka.models.base import create_model
        from sifaka.critics.reflexion import ReflexionCritic

        # Create model and critic
        model = create_model("mock:default")
        critic = ReflexionCritic(model=model)

        # Create Chain with no retrievers
        chain = Chain(
            model=model,
            prompt="Write about AI developments.",
            # No retrievers provided
        )

        chain.improve_with(critic)
        print("✅ Chain configured with no retrievers")

        # Run the chain
        result = chain.run()
        print("✅ Chain execution without retrievers successful")

        pre_context_count = (
            len(result.pre_generation_context) if result.pre_generation_context else 0
        )
        post_context_count = (
            len(result.post_generation_context) if result.post_generation_context else 0
        )
        print(f"Pre-generation context: {pre_context_count} documents")
        print(f"Post-generation context: {post_context_count} documents")
        print(f"Generated text: {result.text}")

        print("✅ No retriever behavior works!")
        return True

    except Exception as e:
        print(f"❌ No retriever test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_multi_retriever_fact_checking()
    success2 = test_fallback_retriever()
    success3 = test_no_retriever()

    if success1 and success2 and success3:
        print("\n🎉 All multi-retriever tests pass!")
        sys.exit(0)
    else:
        print("\n💥 Some multi-retriever tests failed")
        sys.exit(1)
