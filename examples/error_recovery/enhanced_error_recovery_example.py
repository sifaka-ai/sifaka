#!/usr/bin/env python3
"""Enhanced Error Recovery Example for Sifaka.

This example demonstrates the enhanced error recovery features including:
1. Circuit breaker pattern for external services
2. Retry mechanisms with exponential backoff
3. Fallback chains for graceful degradation
4. Resilient model and retriever wrappers
5. Health monitoring and recovery

The example shows how these features work together to provide robust
error handling and automatic recovery in production environments.

Requirements:
- Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
- Optional: Redis running on localhost:6379 for retriever examples

Usage:
    python examples/error_recovery/enhanced_error_recovery_example.py
"""

import os
import time
from typing import List

from sifaka.core.chain import Chain
from sifaka.models.base import create_model
from sifaka.models.resilient import ResilientModel
from sifaka.retrievers.simple import InMemoryRetriever, MockRetriever
from sifaka.retrievers.resilient import ResilientRetriever
from sifaka.validators.base import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.utils import (
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryConfig,
    FallbackConfig,
    BackoffStrategy,
    get_logger,
    retry_with_backoff,
    DEFAULT_RETRY_CONFIG,
)

# Configure logger
logger = get_logger(__name__)


def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker pattern."""
    print("\n" + "=" * 60)
    print("CIRCUIT BREAKER DEMONSTRATION")
    print("=" * 60)

    # Configure circuit breaker
    config = CircuitBreakerConfig(
        failure_threshold=3, recovery_timeout=5.0, expected_exception=Exception
    )

    breaker = CircuitBreaker("demo-service", config)

    # Simulate a failing service
    call_count = 0

    def unreliable_service():
        nonlocal call_count
        call_count += 1
        if call_count <= 5:  # First 5 calls fail
            raise ConnectionError(f"Service unavailable (call {call_count})")
        return f"Success on call {call_count}"

    # Test circuit breaker behavior
    for i in range(10):
        try:
            with breaker.protect_call():
                result = unreliable_service()
                print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")

        # Show circuit breaker stats
        stats = breaker.get_stats()
        print(
            f"  State: {breaker.state.value}, Failures: {stats.failed_requests}, "
            f"Successes: {stats.successful_requests}"
        )

        time.sleep(1)  # Brief pause between calls


def demonstrate_retry_mechanism():
    """Demonstrate retry mechanisms with different strategies."""
    print("\n" + "=" * 60)
    print("RETRY MECHANISM DEMONSTRATION")
    print("=" * 60)

    # Test different retry configurations
    configs = [
        ("Default", DEFAULT_RETRY_CONFIG),
        (
            "Exponential",
            RetryConfig(
                max_attempts=4,
                base_delay=0.5,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                backoff_factor=2.0,
            ),
        ),
        (
            "Linear",
            RetryConfig(max_attempts=3, base_delay=1.0, backoff_strategy=BackoffStrategy.LINEAR),
        ),
        (
            "Fixed",
            RetryConfig(max_attempts=3, base_delay=1.0, backoff_strategy=BackoffStrategy.FIXED),
        ),
    ]

    for name, config in configs:
        print(f"\nTesting {name} retry strategy:")

        attempt_count = 0

        @retry_with_backoff(config)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:  # Fail first 2 attempts
                raise ConnectionError(f"Temporary failure (attempt {attempt_count})")
            return f"Success on attempt {attempt_count}"

        try:
            result = flaky_function()
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Failed: {e}")

        attempt_count = 0  # Reset for next test


def demonstrate_fallback_chain():
    """Demonstrate fallback chains for graceful degradation."""
    print("\n" + "=" * 60)
    print("FALLBACK CHAIN DEMONSTRATION")
    print("=" * 60)

    # Create mock services with different reliability
    def primary_service(query: str) -> List[str]:
        if "fail" in query.lower():
            raise ConnectionError("Primary service is down")
        return [f"Primary result for: {query}"]

    def fallback_service_1(query: str) -> List[str]:
        if "total_fail" in query.lower():
            raise ConnectionError("Fallback 1 is also down")
        return [f"Fallback 1 result for: {query}"]

    def fallback_service_2(query: str) -> List[str]:
        return [f"Fallback 2 (always works) result for: {query}"]

    # Create fallback chain
    from sifaka.utils.fallback import FallbackChain

    config = FallbackConfig(max_fallbacks=3, track_performance=True)
    chain = FallbackChain("demo-service", config)

    chain.add_primary(primary_service, "primary")
    chain.add_fallback(fallback_service_1, priority=1, name="fallback-1")
    chain.add_fallback(fallback_service_2, priority=2, name="fallback-2")

    # Test different scenarios
    test_queries = [
        "normal query",
        "fail primary",
        "total_fail everything",
    ]

    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        try:
            result = chain.execute(query)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Failed: {e}")

        # Show stats
        stats = chain.get_stats()
        print(f"  Usage: {stats.fallback_usage}")


def demonstrate_resilient_model():
    """Demonstrate resilient model wrapper."""
    print("\n" + "=" * 60)
    print("RESILIENT MODEL DEMONSTRATION")
    print("=" * 60)

    # Check for API keys
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("Skipping resilient model demo - no API keys found")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this demo")
        return

    try:
        # Create models
        if os.getenv("OPENAI_API_KEY"):
            primary_model = create_model("openai:gpt-3.5-turbo")
        else:
            primary_model = create_model("anthropic:claude-3-haiku")

        # Create resilient model directly
        if os.getenv("ANTHROPIC_API_KEY"):
            fallback_model = create_model("anthropic:claude-3-haiku")
        else:
            fallback_model = create_model("openai:gpt-3.5-turbo")

        resilient_model = ResilientModel(
            primary_model=primary_model,
            fallback_models=[fallback_model],
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2),
            retry_config=RetryConfig(max_attempts=2),
            fallback_config=FallbackConfig(max_fallbacks=1),
        )

        # Test generation
        prompt = "Write a brief explanation of error recovery in software systems."

        print("Testing resilient model generation...")
        try:
            response = resilient_model.generate(prompt, max_tokens=100)
            print(f"Generated response: {response[:200]}...")
        except Exception as e:
            print(f"Generation failed: {e}")

        # Show health status
        health = resilient_model.get_health_status()
        print(f"Model health: {health['is_healthy']}")

    except Exception as e:
        print(f"Failed to create resilient model: {e}")


def demonstrate_resilient_retriever():
    """Demonstrate resilient retriever wrapper."""
    print("\n" + "=" * 60)
    print("RESILIENT RETRIEVER DEMONSTRATION")
    print("=" * 60)

    # Create retrievers
    primary_retriever = InMemoryRetriever()
    fallback_retriever = MockRetriever(max_results=3)

    # Add some documents to primary retriever
    docs = {
        "doc1": "Error recovery is essential for robust software systems.",
        "doc2": "Circuit breakers prevent cascading failures in distributed systems.",
        "doc3": "Retry mechanisms help handle transient failures gracefully.",
    }

    for doc_id, text in docs.items():
        primary_retriever.add_document(doc_id, text)

    # Create resilient retriever
    resilient_retriever = ResilientRetriever(
        primary_retriever=primary_retriever,
        fallback_retrievers=[fallback_retriever],
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2),
        retry_config=RetryConfig(max_attempts=2),
        fallback_config=FallbackConfig(max_fallbacks=1),
    )

    # Test retrieval
    queries = ["error recovery", "circuit breaker", "retry mechanisms"]

    for query in queries:
        print(f"\nTesting query: '{query}'")
        try:
            results = resilient_retriever.retrieve(query)
            print(f"  Results: {len(results)} documents found")
            for i, result in enumerate(results[:1]):  # Show first result
                print(f"    {i+1}: {result[:100]}...")
        except Exception as e:
            print(f"  Retrieval failed: {e}")

    # Show health status
    health = resilient_retriever.get_health_status()
    print(f"\nRetriever health: {health['is_healthy']}")


def demonstrate_integrated_chain():
    """Demonstrate integrated chain with error recovery."""
    print("\n" + "=" * 60)
    print("INTEGRATED CHAIN WITH ERROR RECOVERY")
    print("=" * 60)

    # Check for API keys
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("Skipping integrated chain demo - no API keys found")
        return

    try:
        # Create resilient components
        if os.getenv("OPENAI_API_KEY"):
            model = create_model("openai:gpt-3.5-turbo")
        else:
            model = create_model("anthropic:claude-3-haiku")

        # Create resilient retriever
        primary_retriever = InMemoryRetriever()
        fallback_retriever = MockRetriever(max_results=3)

        # Add documents
        docs = {
            "doc1": "Error recovery patterns include circuit breakers, retries, and fallbacks.",
            "doc2": "Resilient systems gracefully handle failures and recover automatically.",
            "doc3": "Monitoring and health checks are essential for error recovery.",
        }

        for doc_id, text in docs.items():
            primary_retriever.add_document(doc_id, text)

        resilient_retriever = ResilientRetriever(
            primary_retriever=primary_retriever, fallback_retrievers=[fallback_retriever]
        )

        # Create chain with resilient components
        chain = Chain(
            model=model,
            prompt="Explain the importance of error recovery in software systems",
            retrievers=[resilient_retriever],
            max_improvement_iterations=2,
            apply_improvers_on_validation_failure=True,
        )
        chain.validate_with(LengthValidator(min_length=50, max_length=500))

        # Add critic if we have a model
        try:
            critic = ReflexionCritic(model=model, name="ErrorRecoveryCritic")
            chain.improve_with(critic)
        except Exception as e:
            print(f"Skipping critic due to: {e}")

        # Run chain
        print("Running chain with error recovery...")
        try:
            result = chain.run()
            print(f"Chain completed successfully!")
            print(f"Final response: {result.text[:200]}...")
            print(f"Iterations: {result.iteration}")
        except Exception as e:
            print(f"Chain failed: {e}")

    except Exception as e:
        print(f"Failed to create integrated chain: {e}")


def main():
    """Run all error recovery demonstrations."""
    print("Enhanced Error Recovery Demonstration for Sifaka")
    print("This example shows various error recovery patterns and mechanisms.")

    try:
        demonstrate_circuit_breaker()
        demonstrate_retry_mechanism()
        demonstrate_fallback_chain()
        demonstrate_resilient_model()
        demonstrate_resilient_retriever()
        demonstrate_integrated_chain()

        print("\n" + "=" * 60)
        print("ERROR RECOVERY DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("All error recovery features demonstrated successfully!")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
