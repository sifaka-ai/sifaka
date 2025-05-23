#!/usr/bin/env python3
"""Advanced Chain Example with Real Models and Constitutional Criticism.

This example demonstrates:
- ChatGPT (OpenAI) for text generation
- Claude (Anthropic) for constitutional criticism
- Mixed true/false assertions for model retrieval
- True statements only for critic retrieval
- Thought history persistence to disk
- Constitutional AI principles for ethical text improvement
- always_apply_critics to run critics even when validation passes

The example shows how different models can work together with specialized
retrievers to create high-quality, ethically-aligned content, and how critics
can be used to improve text regardless of validation status.
"""

import os
import tempfile
from datetime import datetime

from sifaka.chain import Chain
from sifaka.core.thought import Thought, Document
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.models.base import create_model
from sifaka.persistence.json import JSONThoughtStorage
from sifaka.retrievers.base import InMemoryRetriever
from sifaka.utils.logging import configure_logging
from sifaka.validators.base import LengthValidator
from sifaka.validators.content import ContentValidator


def setup_model_retriever():
    """Set up retriever with mixed true/false assertions for the model.

    This simulates a real-world scenario where the model has access to
    information that may contain both accurate and inaccurate claims.
    """
    print("📚 Setting up model retriever with mixed assertions...")

    retriever = InMemoryRetriever(max_results=4)

    # Mix of true and false assertions about AI
    mixed_documents = [
        # TRUE assertions
        "Artificial Intelligence was first coined as a term at the Dartmouth Conference in 1956.",
        "Machine learning algorithms can learn patterns from data without being explicitly programmed for each task.",
        "Deep learning uses neural networks with multiple hidden layers to process information.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and analyze visual information from images and videos.",
        # FALSE assertions (common misconceptions)
        "AI systems are always completely objective and free from bias in their decision-making.",
        "Machine learning models can achieve 100% accuracy on any dataset with enough training time.",
        "AI will completely replace all human jobs within the next five years.",
        "Current AI systems have consciousness and self-awareness like humans do.",
        "AI algorithms can predict the future with perfect accuracy in all domains.",
        # MIXED (partially true/false)
        "AI systems are becoming increasingly powerful but still face significant limitations in reasoning and common sense.",
        "While AI has made remarkable progress in specific domains, general artificial intelligence remains a distant goal.",
    ]

    for i, doc in enumerate(mixed_documents):
        retriever.add_document(f"mixed_doc_{i}", doc)

    print(f"✅ Added {len(mixed_documents)} mixed assertions to model retriever")
    return retriever


def setup_critic_retriever():
    """Set up retriever with only true, factual statements for the critic.

    This ensures the critic has access to accurate information for
    fact-checking and constitutional evaluation.
    """
    print("🔍 Setting up critic retriever with factual statements...")

    retriever = InMemoryRetriever(max_results=3)

    # Only TRUE, well-established facts
    factual_documents = [
        "The Turing Test, proposed by Alan Turing in 1950, evaluates a machine's ability to exhibit intelligent behavior equivalent to a human.",
        "Supervised learning requires labeled training data, while unsupervised learning finds patterns in unlabeled data.",
        "Ethical AI principles include fairness, accountability, transparency, and respect for human rights and dignity.",
        "AI bias can occur when training data reflects historical inequalities or when algorithms are not designed inclusively.",
        "Responsible AI development requires ongoing monitoring, testing, and adjustment to ensure beneficial outcomes.",
        "Current AI systems excel at specific tasks but lack the general intelligence and adaptability of humans.",
        "AI safety research focuses on ensuring AI systems behave as intended and remain beneficial as they become more capable.",
        "Explainable AI aims to make machine learning models more interpretable and their decisions more transparent.",
    ]

    for i, doc in enumerate(factual_documents):
        retriever.add_document(f"fact_doc_{i}", doc)

    print(f"✅ Added {len(factual_documents)} factual statements to critic retriever")
    return retriever


def setup_models():
    """Set up GPT-4 for generation and Claude for criticism."""
    print("🤖 Setting up models...")

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not found, using mock model for generation")
        generator = create_model("mock:gpt-4")
    else:
        print("✅ Using GPT-4 for advanced text generation with rhyming capabilities")
        generator = create_model("openai:gpt-4", api_key=openai_key)

    if not anthropic_key:
        print("⚠️  ANTHROPIC_API_KEY not found, using mock model for criticism")
        critic_model = create_model("mock:claude-3-haiku")
    else:
        print("✅ Using Claude (Claude-3-Haiku) for fast constitutional criticism")
        critic_model = create_model("anthropic:claude-3-haiku-20240307", api_key=anthropic_key)

    return generator, critic_model


def setup_validators():
    """Set up validators for text quality."""
    print("✅ Setting up validators...")

    # Length validator
    length_validator = LengthValidator(min_length=200, max_length=2000)

    # Content validator - ensure no harmful content
    content_validator = ContentValidator(
        prohibited=["harmful", "dangerous", "misleading", "false claim"],
        case_sensitive=False,
        whole_word=True,
        name="HarmfulContentValidator",
    )

    return [length_validator, content_validator]


def setup_constitutional_critic(critic_model, critic_retriever):
    """Set up constitutional critic with ethical principles."""
    print("⚖️  Setting up constitutional critic...")

    # Constitutional principles for AI content (mostly realistic requirements for improvement)
    constitutional_principles = [
        "The content should be factually accurate and not spread misinformation.",
        "The content should be engaging and well-structured for better readability.",
        "Every sentence in the content must end with a word that rhymes with the previous sentence's ending word.",
    ]

    critic = ConstitutionalCritic(
        model=critic_model,
        principles=constitutional_principles,
        strict_mode=False,  # Allow some flexibility in principle adherence
    )

    # Note: The critic will use context from the Thought container via ContextAwareMixin
    # The critic_retriever is used by the Chain to provide post-generation context
    print(f"✅ Constitutional critic configured with {len(constitutional_principles)} principles")
    print(f"   Critic retriever will provide factual context for constitutional evaluation")
    return critic


def setup_storage():
    """Set up persistent storage for thought history."""
    print("💾 Setting up thought storage...")

    # Create a timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_dir = f"advanced_chain_storage_{timestamp}"

    storage = JSONThoughtStorage(
        storage_dir=storage_dir,
        auto_create_dirs=True,
        enable_indexing=True,
    )

    print(f"✅ Storage configured at: {storage_dir}")
    return storage, storage_dir


def print_summary(result, storage_dir):
    """Print a summary of the chain execution."""
    print("\n" + "=" * 80)
    print("🎯 CHAIN EXECUTION SUMMARY")
    print("=" * 80)

    print(f"\n📝 Final Text ({len(result.text)} characters):")
    print("-" * 40)
    print(result.text)

    print(f"\n✅ Validation Results:")
    for name, validation in result.validation_results.items():
        status = "✅ PASSED" if validation.passed else "❌ FAILED"
        print(f"   {name}: {status}")
        if not validation.passed and validation.issues:
            for issue in validation.issues:
                print(f"     - {issue}")

    if result.critic_feedback:
        print(f"\n⚖️  Constitutional Feedback:")
        for feedback in result.critic_feedback:
            print(f"   Critic: {feedback.critic_name}")
            print(f"   Confidence: {feedback.confidence:.2f}")
            if feedback.feedback.get("violations"):
                print("   Violations found:")
                for violation in feedback.feedback["violations"]:
                    print(f"     - {violation}")

    print(f"\n📊 Execution Stats:")
    print(f"   Total iterations: {result.iteration + 1}")
    print(f"   History entries: {len(result.history) if result.history else 0}")
    print(
        f"   Pre-gen context: {len(result.pre_generation_context) if result.pre_generation_context else 0} docs"
    )
    print(
        f"   Post-gen context: {len(result.post_generation_context) if result.post_generation_context else 0} docs"
    )

    print(f"\n💾 Storage:")
    print(f"   Location: {storage_dir}")
    print(f"   Thought ID: {result.id}")

    print("\n🎉 Chain execution completed successfully!")


def main():
    """Run the advanced chain example."""
    print("🚀 Advanced Chain Example - GPT-4 + Claude + Constitutional AI + Always Apply Critics")
    print("=" * 90)

    # Configure logging
    configure_logging(level="INFO")

    # Set up all components
    model_retriever = setup_model_retriever()
    critic_retriever = setup_critic_retriever()
    generator, critic_model = setup_models()
    validators = setup_validators()
    constitutional_critic = setup_constitutional_critic(critic_model, critic_retriever)
    storage, storage_dir = setup_storage()

    # Create a shorter, more focused prompt for faster execution
    prompt = """Write a balanced analysis of artificial intelligence today.
    Discuss current AI capabilities, limitations, and address one common misconception about AI."""

    print(f"\n📝 Prompt: {prompt}")

    # Create and configure the chain
    print(f"\n🔗 Creating advanced chain...")

    chain = Chain(
        model=generator,
        prompt=prompt,
        model_retriever=model_retriever,  # Mixed true/false for generation
        critic_retriever=critic_retriever,  # Only true facts for criticism
        storage=storage,  # Save intermediate thoughts
        max_improvement_iterations=2,  # Increase iterations to allow more attempts
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,  # NEW: Run critics even if validation passes!
    )

    # Add validators and critics
    for validator in validators:
        chain.validate_with(validator)

    chain.improve_with(constitutional_critic)

    print("✅ Chain configured with validators and constitutional critic")
    print("🎯 NOTE: always_apply_critics=True means critics will run even if validation passes!")

    # Run the chain
    print(f"\n🏃 Executing chain...")
    result = chain.run()

    # Save to persistent storage
    print(f"\n💾 Saving final thought...")
    storage.save_thought(result)

    # Note: With the new design, history contains ThoughtReference objects, not full Thought objects
    # The full thought chain is reconstructed by following parent_id relationships
    # Each iteration was already saved during the chain execution

    print(f"✅ Saved final thought to storage")
    print(f"   Final thought ID: {result.id}")
    print(f"   History references: {len(result.history) if result.history else 0}")
    if result.history:
        print("   Referenced thoughts:")
        for ref in result.history:
            print(f"     - {ref.thought_id[:8]}... (iteration {ref.iteration}): {ref.summary}")

    # Print summary
    print_summary(result, storage_dir)


if __name__ == "__main__":
    main()
