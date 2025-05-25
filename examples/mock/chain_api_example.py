#!/usr/bin/env python3
"""
Example demonstrating the new Chain API with separate model and critic retrievers.

This example shows how to use different retrievers for models vs critics,
enabling powerful use cases like fact-checking where the model gets recent
context while critics get authoritative sources.
"""

from sifaka.core.chain import Chain
from sifaka.models.base import create_model
from sifaka.retrievers import MockRetriever
from sifaka.validators.base import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic


def main():
    """Demonstrate the new Chain API with separate retrievers."""
    print("🔗 New Chain API Example: Separate Model and Critic Retrievers")
    print("=" * 70)

    # Create model and components
    model = create_model("mock:default")
    validator = LengthValidator(min_length=50, max_length=500)
    critic = ReflexionCritic(model=model)

    # Create specialized retrievers
    recent_retriever = MockRetriever(
        documents=[
            "Latest AI news: GPT-4 shows impressive reasoning capabilities",
            "Recent breakthrough: New transformer architecture improves efficiency",
            "Twitter buzz: AI researchers excited about multimodal models",
        ]
    )

    factual_retriever = MockRetriever(
        documents=[
            "Scientific paper: Attention mechanisms in neural networks (Vaswani et al., 2017)",
            "Authoritative source: Deep learning fundamentals from MIT textbook",
            "Peer-reviewed research: Transformer models achieve state-of-the-art results",
        ]
    )

    print("✅ Created specialized retrievers:")
    print("   📰 Recent retriever: Latest news and social media")
    print("   📚 Factual retriever: Scientific papers and authoritative sources")

    # Example 1: Constructor API
    print("\n🔧 Example 1: Constructor API")
    chain1 = Chain(
        model=model,
        prompt="Write about recent developments in AI transformers",
        model_retrievers=[recent_retriever],  # Model gets recent context
        critic_retrievers=[factual_retriever],  # Critics get authoritative context
        max_improvement_iterations=2,
        apply_improvers_on_validation_failure=True,
    )
    chain1.validate_with(validator)
    chain1.improve_with(critic)

    print("✅ Chain created with constructor API")

    # Example 2: Fluent API
    print("\n🔧 Example 2: Fluent API")
    chain2 = (
        Chain(model=model, prompt="Explain transformer attention mechanisms")
        .with_model_retrievers([recent_retriever])
        .with_critic_retrievers([factual_retriever])
        .validate_with(validator)
        .improve_with(critic)
    )

    print("✅ Chain created with fluent API")

    # Example 3: Same retriever for both (traditional approach)
    print("\n🔧 Example 3: Same retriever for both model and critics")
    general_retriever = MockRetriever(
        documents=[
            "General AI knowledge: Machine learning is a subset of AI",
            "Common fact: Neural networks are inspired by biological neurons",
        ]
    )

    chain3 = Chain(
        model=model,
        prompt="Write a general introduction to AI",
        model_retrievers=[general_retriever],
        critic_retrievers=[general_retriever],  # Same retriever for both
    )

    print("✅ Chain created with same retriever for both purposes")

    # Example 4: Model-only retrieval (no critic retrieval)
    print("\n🔧 Example 4: Model-only retrieval")
    chain4 = Chain(
        model=model,
        prompt="Write about AI without fact-checking",
        model_retrievers=[recent_retriever],  # Only model gets context
        # No critic_retrievers - critics work without additional context
    )

    print("✅ Chain created with model-only retrieval")

    print("\n🎉 All examples demonstrate the flexibility of the new Chain API!")
    print("\nKey Benefits:")
    print("  🎯 Targeted context: Models get recent info, critics get authoritative sources")
    print("  🔄 Flexible configuration: Use constructor or fluent API")
    print("  ⚡ Performance: Only retrieve what each component needs")
    print("  🛡️ Quality: Critics can fact-check against authoritative sources")


if __name__ == "__main__":
    main()
