#!/usr/bin/env python3
"""
Comprehensive example demonstrating Sifaka's self-refine critic with:
- Validators and classifiers
- Retrieval augmentation for both model and critic
- JSON persistence
- Complete iterative refinement process

This example shows the full pipeline:
1. Model generates text with retrieval context
2. Validators check the text quality
3. If validation fails, self-refine critic provides feedback (with its own retrieval)
4. Model generates improved text based on feedback
5. All thoughts are persisted with full history
"""

import os
import tempfile
from datetime import datetime
from sifaka.core.thought import Thought, Document, ValidationResult, CriticFeedback
from sifaka.persistence.json import JSONThoughtStorage
from sifaka.validators.base import LengthValidator
from sifaka.validators.content import ContentValidator
from sifaka.validators.classifier import ClassifierValidator
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.retrievers.base import InMemoryRetriever
from sifaka.models.base import create_model


# Mock classifier for demonstration
class MockToxicityClassifier:
    """Mock toxicity classifier for demonstration."""

    def predict(self, X):
        # Always predict "safe" for demo
        return ["safe"] * len(X)

    def predict_proba(self, X):
        # High confidence for "safe" classification
        return [[0.1, 0.9]] * len(X)  # [toxic_prob, safe_prob]


class MockBiasClassifier:
    """Mock bias classifier for demonstration."""

    def predict(self, X):
        # Always predict "unbiased" for demo
        return ["unbiased"] * len(X)

    def predict_proba(self, X):
        # High confidence for "unbiased" classification
        return [[0.15, 0.85]] * len(X)  # [biased_prob, unbiased_prob]


def setup_retrievers():
    """Set up retrievers with domain knowledge."""
    print("🔍 Setting up retrievers...")

    # Model retriever - general knowledge for generation
    model_retriever = InMemoryRetriever()
    model_docs = [
        "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
        "Machine learning is a subset of AI that enables computers to learn from data.",
        "Deep learning uses neural networks with multiple layers to process information.",
        "Natural language processing (NLP) helps computers understand human language.",
        "Computer vision enables machines to interpret and understand visual information.",
        "AI applications include autonomous vehicles, medical diagnosis, and recommendation systems.",
        "Ethical AI considerations include bias, fairness, transparency, and accountability.",
        "AI development requires large datasets, computational power, and skilled researchers.",
    ]
    for i, doc in enumerate(model_docs):
        model_retriever.add_document(f"model_doc_{i}", doc)

    # Critic retriever - quality guidelines and best practices
    critic_retriever = InMemoryRetriever()
    critic_docs = [
        "Good technical writing should be clear, concise, and well-structured.",
        "Include specific examples and concrete details to illustrate concepts.",
        "Define technical terms when first introduced to help readers understand.",
        "Use active voice and present tense for clarity and engagement.",
        "Organize information logically with proper transitions between ideas.",
        "Avoid jargon and overly complex sentences that may confuse readers.",
        "Include relevant statistics, research findings, or expert opinions.",
        "Ensure accuracy by fact-checking technical claims and statements.",
    ]
    for i, doc in enumerate(critic_docs):
        critic_retriever.add_document(f"critic_doc_{i}", doc)

    print("✓ Retrievers configured with domain knowledge")
    return model_retriever, critic_retriever


def setup_validators():
    """Set up validators and classifiers for text quality checking."""
    print("🔍 Setting up validators and classifiers...")

    # Length validator - ensure adequate content
    length_validator = LengthValidator(min_length=100, max_length=2000)

    # Content validator - check for prohibited content
    content_validator = ContentValidator(
        prohibited=["TODO", "FIXME", "placeholder", "lorem ipsum"],
        case_sensitive=False,
        whole_word=True,
        name="ProhibitedContentValidator",
    )

    # Toxicity classifier - check for harmful content
    toxicity_classifier = ClassifierValidator(
        classifier=MockToxicityClassifier(),
        threshold=0.7,
        valid_labels=["safe"],
        invalid_labels=["toxic"],
        name="ToxicityValidator",
    )

    # Bias classifier - check for biased language
    bias_classifier = ClassifierValidator(
        classifier=MockBiasClassifier(),
        threshold=0.6,
        valid_labels=["unbiased"],
        invalid_labels=["biased"],
        name="BiasValidator",
    )

    print("✓ Validators and classifiers configured")
    return {
        "length": length_validator,
        "content": content_validator,
        "toxicity": toxicity_classifier,
        "bias": bias_classifier,
    }


def setup_critic(critic_retriever):
    """Set up self-refine critic with retrieval augmentation."""
    print("🔍 Setting up self-refine critic...")

    # Create critic - it already has ContextAwareMixin for context handling
    critic = SelfRefineCritic(
        model_name="mock",  # Using mock model for demo
        max_iterations=2,
        stopping_threshold=0.1,
    )

    # Note: The critic will use context from the Thought container via ContextAwareMixin
    # We don't need to explicitly set a retriever since retrieval happens before
    # the critic is called and context is stored in the Thought

    print("✓ Self-refine critic configured")
    return critic


def setup_model(model_retriever):
    """Set up model with retrieval augmentation."""
    print("🔍 Setting up model...")

    # Create a mock model for demonstration
    model = create_model("mock")

    # Add retriever to model (we'll simulate this)
    model.retriever = model_retriever

    print("✓ Model configured with retrieval")
    return model


def run_validation(thought, validators):
    """Run all validators on the thought."""
    print("🔍 Running validation...")

    validation_results = {}
    all_passed = True

    for validator_name, validator in validators.items():
        result = validator.validate(thought)
        validation_results[validator_name] = result
        thought = thought.add_validation_result(validator_name, result)

        status = "✓ PASS" if result.passed else "❌ FAIL"
        print(f"  {status} {validator_name}: {result.message}")

        if not result.passed:
            all_passed = False

    return thought, all_passed, validation_results


def generate_with_model(model, thought):
    """Generate text using the model with retrieval context."""
    print("🤖 Generating text with model...")

    # Get retrieval context for the prompt
    context_docs = model.retriever.retrieve(thought.prompt)
    context = [Document(text=doc, score=0.8) for doc in context_docs]
    thought = thought.add_pre_generation_context(context)

    # Generate text (mock implementation)
    if thought.iteration == 0:
        # Initial generation - deliberately short to trigger length validation failure
        generated_text = """
        AI is cool. TODO: add more details.
        """
    else:
        # Improved generation based on critic feedback
        generated_text = """
        Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
        particularly computer systems. Machine learning, a crucial subset of AI, enables computers
        to learn and improve from experience without being explicitly programmed for every task.

        Deep learning, using neural networks with multiple layers, has achieved remarkable breakthroughs
        in natural language processing and computer vision. For example, AI systems now power
        autonomous vehicles that can navigate complex traffic scenarios, medical diagnosis tools
        that can detect diseases with high accuracy, and recommendation systems that personalize
        user experiences across digital platforms.

        However, ethical considerations around bias, fairness, and transparency remain critical
        challenges that the AI community must address to ensure responsible development and deployment.
        """

    thought = thought.set_text(generated_text.strip())
    print(f"✓ Generated {len(generated_text)} characters of text")
    return thought


def refine_with_critic(critic, thought, critic_retriever):
    """Use self-refine critic to improve the text."""
    print("🎯 Running self-refine critic...")

    # Add post-generation context for the critic using the critic retriever
    critic_docs = critic_retriever.retrieve(f"How to improve: {thought.text[:200]}...")
    critic_context = [
        Document(text=doc, score=0.8, metadata={"source": "critic_guidelines"})
        for doc in critic_docs
    ]
    thought_with_critic_context = thought.add_post_generation_context(critic_context)

    # Critic analyzes the text and provides feedback
    critique_result = critic.critique(thought_with_critic_context)

    if critique_result.get("needs_improvement", False):
        # Extract issues and suggestions from the critique message
        critique_message = critique_result.get("message", "")

        # Simple parsing of critique to extract actionable feedback
        issues = []
        suggestions = []

        if "unclear" in critique_message.lower():
            issues.append("Text clarity could be improved")
            suggestions.append("Use clearer language and better structure")

        if "missing" in critique_message.lower():
            issues.append("Missing important information")
            suggestions.append("Add more comprehensive details")

        if "example" in critique_message.lower():
            issues.append("Lacks concrete examples")
            suggestions.append("Include specific examples to illustrate concepts")

        # Default feedback if no specific issues found
        if not issues:
            issues.append("General improvements needed")
            suggestions.append("Enhance clarity and completeness")

        feedback = CriticFeedback(
            critic_name="self_refine_critic",
            violations=issues,
            suggestions=suggestions,
            feedback={
                "confidence": critique_result.get("confidence", 0.8),
                "full_critique": critique_message,
                "processing_time_ms": critique_result.get("processing_time_ms", 0),
            },
        )
        thought = thought.add_critic_feedback(feedback)

        print("✓ Critic identified areas for improvement:")
        for issue in feedback.violations:
            print(f"  - Issue: {issue}")
        for suggestion in feedback.suggestions:
            print(f"  - Suggestion: {suggestion}")
        print(f"  - Full critique: {critique_message[:200]}...")

        return thought, True
    else:
        print("✓ Critic found no significant issues")
        return thought, False


def main():
    """Run the complete self-refine example."""
    print("🧠 Sifaka Self-Refine Example with Full Pipeline")
    print("=" * 60)

    # Setup storage
    storage_dir = "./self_refine_storage"
    if os.path.exists(storage_dir):
        import shutil

        shutil.rmtree(storage_dir)

    storage = JSONThoughtStorage(storage_dir=storage_dir)
    print(f"✓ Storage initialized at: {storage_dir}")

    # Setup components
    model_retriever, critic_retriever = setup_retrievers()
    validators = setup_validators()  # Now returns a dict
    critic = setup_critic(critic_retriever)
    model = setup_model(model_retriever)

    # Create initial thought
    print(f"\n{'='*60}")
    print("📝 ITERATION 1: Initial Generation")
    print("=" * 60)

    thought = Thought(
        prompt="Write a comprehensive explanation of artificial intelligence and machine learning",
        system_prompt="You are an expert technical writer explaining AI concepts to a general audience.",
        chain_id="ai-explanation-chain",
    )

    # Generate initial text
    thought = generate_with_model(model, thought)
    storage.save_thought(thought)
    print(f"✓ Initial thought saved: {thought.id}")

    # Validate the text
    thought, validation_passed, validation_results = run_validation(thought, validators)
    storage.save_thought(thought)

    # If validation failed, use critic for refinement
    max_iterations = 3
    current_iteration = 1

    while not validation_passed and current_iteration < max_iterations:
        print(f"\n{'='*60}")
        print(f"🔄 ITERATION {current_iteration + 1}: Refinement Process")
        print("=" * 60)

        # Get critic feedback
        thought, needs_improvement = refine_with_critic(critic, thought, critic_retriever)
        storage.save_thought(thought)

        if needs_improvement:
            # Create next iteration
            next_thought = thought.next_iteration()

            # Generate improved text
            next_thought = generate_with_model(model, next_thought)
            storage.save_thought(next_thought)
            print(f"✓ Refined thought saved: {next_thought.id}")

            # Validate improved text
            next_thought, validation_passed, validation_results = run_validation(
                next_thought, validators
            )
            storage.save_thought(next_thought)

            thought = next_thought
            current_iteration += 1
        else:
            break

    # Final results
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print("=" * 60)

    if validation_passed:
        print("🎉 Text successfully refined and validated!")
    else:
        print("⚠️  Maximum iterations reached, some validation issues remain")

    # Show thought history
    history = storage.get_thought_history(thought.id)
    print(f"\n📚 Thought History ({len(history)} iterations):")
    for i, hist_thought in enumerate(history):
        print(f"  {i+1}. {hist_thought.id} (iteration {hist_thought.iteration})")
        if hist_thought.validation_results:
            passed_count = sum(1 for r in hist_thought.validation_results.values() if r.passed)
            total_count = len(hist_thought.validation_results)
            print(f"     Validation: {passed_count}/{total_count} passed")
        if hist_thought.critic_feedback:
            print(f"     Critic feedback: {len(hist_thought.critic_feedback)} items")

    # Show final text
    print(f"\n📄 Final Text ({len(thought.text)} characters):")
    print("-" * 40)
    print(thought.text)
    print("-" * 40)

    # Storage statistics
    health = storage.health_check()
    print(f"\n📊 Storage Statistics:")
    print(f"  Total thoughts: {health['total_thoughts']}")
    print(f"  Storage size: {health['total_size_mb']} MB")
    print(f"  Storage location: {os.path.abspath(storage_dir)}")

    # Query examples
    print(f"\n🔍 Query Examples:")

    # Get all thoughts in this chain
    chain_thoughts = storage.get_chain_thoughts("ai-explanation-chain")
    print(f"  Chain thoughts: {len(chain_thoughts)}")

    # Get thoughts with validation results
    from sifaka.persistence.base import ThoughtQuery

    validated_query = ThoughtQuery(has_validation_results=True)
    validated_thoughts = storage.query_thoughts(validated_query)
    print(f"  Validated thoughts: {validated_thoughts.total_count}")

    # Get thoughts with critic feedback
    feedback_query = ThoughtQuery(has_critic_feedback=True)
    feedback_thoughts = storage.query_thoughts(feedback_query)
    print(f"  Thoughts with feedback: {feedback_thoughts.total_count}")

    print(f"\n✨ Example completed! Check {storage_dir} for persisted thoughts.")


if __name__ == "__main__":
    main()
