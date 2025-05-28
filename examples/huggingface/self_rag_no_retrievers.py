#!/usr/bin/env python3
"""HuggingFace Self-RAG without Additional Retrievers Example.

This example demonstrates:
- HuggingFace model for text generation (using DeepSeek as preferred)
- Self-RAG critic without additional retrievers (internal knowledge only)
- Default retry behavior
- Remote HuggingFace Inference API usage

The chain will generate content about quantum computing using only the model's
internal knowledge and Self-RAG's built-in retrieval capabilities.
"""

import os

from dotenv import load_dotenv

from sifaka.core.chain import Chain
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.models.huggingface import HuggingFaceModel
from sifaka.utils.logging import get_logger
import logging
from sifaka.storage import FileStorage

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)

# Enable debug logging to see what's happening with the critic
logging.getLogger("sifaka.critics.self_rag").setLevel(logging.DEBUG)
logging.getLogger("sifaka.core.chain").setLevel(logging.DEBUG)
logging.getLogger("sifaka.core.chain.executor").setLevel(logging.DEBUG)


def main():
    """Run the HuggingFace Self-RAG without additional retrievers example."""

    # Ensure API key is available
    if not os.getenv("HUGGINGFACE_API_KEY"):
        raise ValueError("HUGGINGFACE_API_KEY environment variable is required")

    logger.info("Creating HuggingFace Self-RAG without additional retrievers example")

    # Create HuggingFace model using Meta Llama-3.3-70B-Instruct
    model = HuggingFaceModel(
        model_name="meta-llama/Llama-3.3-70B-Instruct",  # Meta Llama-3.3-70B-Instruct model
        api_token=os.getenv("HUGGINGFACE_API_KEY"),
        temperature=0.7,
        max_tokens=800,
        use_inference_api=True,  # Use HuggingFace Inference Providers API
    )

    # Test if HuggingFace API is available
    try:
        # Simple health check by attempting a small generation
        test_response = model.generate("Test", max_tokens=5)
        logger.info("HuggingFace API is available")
    except Exception as e:
        logger.error(f"HuggingFace API not available: {e}")
        print("Error: HuggingFace API is not accessible. Please check your API key and try again.")
        return

    # Create Self-RAG critic without additional retrievers
    # Self-RAG will use its internal retrieval mechanisms only
    critic = SelfRAGCritic(
        model=model,
        retriever=None,  # No additional retrievers as specified
        use_internal_knowledge=True,  # Rely on model's internal knowledge
        name="Internal Knowledge Self-RAG Critic",
        use_reflection_tokens=True,  # Enable reflection tokens for better feedback
    )

    # Create the chain without any retrievers
    chain = Chain(
        model=model,
        prompt="Explain the fundamental principles of quantum computing, including quantum bits (qubits), superposition, entanglement, and quantum gates. Discuss how quantum computers differ from classical computers and what potential applications they might have in the future.",
        model_retrievers=None,  # No additional retrievers for model
        critic_retrievers=None,  # No additional retrievers for critic
        max_improvement_iterations=1,  # Retry once
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,  # Always apply the critic
        storage=FileStorage(
            "./thoughts/self_rag_no_retrievers_thoughts.json",
            overwrite=True,  # Overwrite existing file instead of appending
        ),  # Save thoughts to single JSON file for debugging
    )

    # Add Self-RAG critic (no validators specified)
    print(f"DEBUG: Adding Self-RAG critic to chain...")
    chain = chain.improve_with(critic)
    print(f"DEBUG: Chain now has {len(chain._config.critics)} critics")

    # Run the chain
    logger.info("Running chain with Self-RAG critic using internal knowledge only...")
    print("DEBUG: About to run chain with Self-RAG critic...")
    result = chain.run()
    print("DEBUG: Chain completed, checking for critic feedback...")

    # Display results
    print("\n" + "=" * 80)
    print("HUGGINGFACE SELF-RAG WITHOUT ADDITIONAL RETRIEVERS EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nProcessing Details:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Model: HuggingFace Llama-3.3-70B-Instruct (remote)")
    print(f"  Retrievers: None (internal knowledge only)")

    # Show Self-RAG critic feedback
    if result.critic_feedback:
        print(f"\nSelf-RAG Critic Feedback (Internal Knowledge):")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Internal Knowledge Assessment: {feedback.suggestions[:350]}...")

            # Show Self-RAG specific metrics if available
            if hasattr(feedback, "relevance_score"):
                print(f"     Relevance Score: {feedback.relevance_score}")
            if hasattr(feedback, "factuality_score"):
                print(f"     Factuality Score: {feedback.factuality_score}")

    # Verify no external retrieval was used
    print(f"\nRetrieval Verification:")
    if hasattr(result, "pre_generation_context"):
        if result.pre_generation_context:
            print(f"  WARNING: Unexpected pre-generation context found")
        else:
            print(f"  ✓ No pre-generation context (as expected)")
    else:
        print(f"  ✓ No pre-generation context (as expected)")

    if hasattr(result, "post_generation_context"):
        if result.post_generation_context:
            print(f"  WARNING: Unexpected post-generation context found")
        else:
            print(f"  ✓ No post-generation context (as expected)")
    else:
        print(f"  ✓ No post-generation context (as expected)")

    print(f"\nSystem Features:")
    print(f"  - HuggingFace Inference API")
    print(f"  - Meta Llama-3.3-70B-Instruct model")
    print(f"  - Self-RAG internal knowledge only")
    print(f"  - No external retrievers")
    print(f"  - Pure model-based generation and critique")

    print(f"\nQuantum Computing Coverage Analysis:")
    quantum_terms = ["qubit", "superposition", "entanglement", "quantum gate", "quantum computer"]
    covered_terms = [term for term in quantum_terms if term.lower() in result.text.lower()]
    print(f"  Key terms covered: {len(covered_terms)}/{len(quantum_terms)}")
    for term in covered_terms:
        print(f"    ✓ {term}")

    print("\n" + "=" * 80)
    logger.info("Self-RAG without additional retrievers example completed successfully")


if __name__ == "__main__":
    main()
