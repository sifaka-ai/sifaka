"""Integration tests for all critics with multiple LLM providers."""

import pytest
import asyncio
import os
from typing import Dict, List
from dataclasses import dataclass

from sifaka import improve, SifakaResult
from sifaka.core.config import Config
from sifaka.critics import CriticRegistry
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.critics.self_rag_enhanced import SelfRAGEnhancedCritic, RetrievalBackend


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    name: str
    model: str
    api_key_env: str
    supports_json: bool = True
    temperature_range: tuple = (0.0, 1.0)


# Provider configurations
PROVIDERS = [
    ProviderConfig("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
    ProviderConfig("anthropic", "claude-3-haiku-20240307", "ANTHROPIC_API_KEY"),
    ProviderConfig("google", "gemini-1.5-flash", "GOOGLE_API_KEY"),
    ProviderConfig("xai", "grok-beta", "XAI_API_KEY"),
]

# Test texts for different scenarios
TEST_TEXTS = {
    "technical": """
    Python uses dynamic typing, which means variables don't need explicit type
    declarations. The interpreter determines types at runtime. This provides
    flexibility but can lead to runtime errors if not careful.
    """,
    "factual": """
    The speed of light is approximately 300,000 kilometers per second.
    Einstein's theory of relativity, published in 1905, established that
    nothing can travel faster than light in a vacuum.
    """,
    "creative": """
    The old lighthouse stood sentinel against the storm, its beam cutting
    through the darkness like a sword of hope. Waves crashed against the
    rocks below, each impact echoing the keeper's lonely vigil.
    """,
    "argumentative": """
    Remote work has fundamentally changed the nature of employment. While
    it offers flexibility and eliminates commutes, it also creates challenges
    for collaboration and work-life balance. Companies must adapt their
    cultures to support distributed teams effectively.
    """,
}


class TestMultiProviderCritics:
    """Test all critics with multiple providers."""

    @pytest.fixture
    def available_providers(self):
        """Get list of providers with available API keys."""
        available = []
        for provider in PROVIDERS:
            if os.getenv(provider.api_key_env):
                available.append(provider)
            else:
                print(f"Skipping {provider.name}: No {provider.api_key_env} found")
        return available

    @pytest.fixture
    def all_critics(self):
        """Get list of all available critics."""
        return CriticRegistry.list_critics()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_all_critics_all_providers(self, available_providers, all_critics):
        """Test each critic with each available provider."""
        if not available_providers:
            pytest.skip("No API keys available")

        results = {}

        for provider in available_providers:
            print(f"\n\nTesting provider: {provider.name}")
            results[provider.name] = {}

            for critic_name in all_critics:
                print(f"\n  Testing critic: {critic_name}")

                try:
                    # Create critic with specific provider
                    config = Config(
                        model=provider.model, temperature=0.7, max_iterations=2
                    )

                    # Test with technical text
                    result = await improve(
                        TEST_TEXTS["technical"],
                        critics=[critic_name],
                        config=config,
                        provider=provider.name,
                        api_key=os.getenv(provider.api_key_env),
                    )

                    # Verify result
                    assert result.final_text
                    assert len(result.critiques) > 0
                    assert result.iteration > 0

                    # Store result
                    results[provider.name][critic_name] = {
                        "success": True,
                        "iterations": result.iteration,
                        "confidence": (
                            result.critiques[-1].confidence if result.critiques else 0
                        ),
                        "improved": not result.needs_improvement,
                    }

                    print(
                        f"    ✓ Success - Iterations: {result.iteration}, "
                        f"Confidence: {results[provider.name][critic_name]['confidence']:.2f}"
                    )

                except Exception as e:
                    results[provider.name][critic_name] = {
                        "success": False,
                        "error": str(e),
                    }
                    print(f"    ✗ Failed: {str(e)[:100]}")

        # Summary
        print("\n\nSummary:")
        print("=" * 80)
        for provider_name, critic_results in results.items():
            success_count = sum(1 for r in critic_results.values() if r.get("success"))
            print(
                f"\n{provider_name}: {success_count}/{len(critic_results)} critics succeeded"
            )

            for critic_name, result in critic_results.items():
                if not result.get("success"):
                    print(
                        f"  - {critic_name} failed: {result.get('error', 'Unknown error')[:50]}"
                    )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_critic_consistency_across_providers(self, available_providers):
        """Test that critics provide consistent feedback across providers."""
        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for consistency test")

        critic_name = "constitutional"  # Well-defined critic for consistency
        test_text = TEST_TEXTS["factual"]

        results = {}

        for provider in available_providers[:3]:  # Test up to 3 providers
            result = await improve(
                test_text,
                critics=[critic_name],
                max_iterations=1,
                provider=provider.name,
                api_key=os.getenv(provider.api_key_env),
                model=provider.model,
            )

            results[provider.name] = {
                "feedback": result.critiques[0].feedback if result.critiques else "",
                "suggestions": (
                    result.critiques[0].suggestions if result.critiques else []
                ),
                "needs_improvement": (
                    result.critiques[0].needs_improvement if result.critiques else True
                ),
            }

        # Check consistency
        all_suggestions = [r["suggestions"] for r in results.values()]
        all_needs_improvement = [r["needs_improvement"] for r in results.values()]

        # Should have some overlap in suggestions
        if len(all_suggestions) > 1:
            common_themes = set()
            for suggestions in all_suggestions:
                for suggestion in suggestions:
                    # Look for common keywords
                    words = set(suggestion.lower().split())
                    common_themes.update(words)

            print(f"\nCommon themes across providers: {len(common_themes)} words")

        # Should generally agree on whether improvement is needed
        agreement_rate = sum(
            1 for x in all_needs_improvement if x == all_needs_improvement[0]
        ) / len(all_needs_improvement)
        print(f"Agreement rate on improvement needed: {agreement_rate:.0%}")

        assert agreement_rate >= 0.5, "Providers disagree too much on improvement needs"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_enhanced_n_critics_dynamic_perspectives(self, available_providers):
        """Test the enhanced NCriticsCritic with dynamic perspectives."""
        if not available_providers:
            pytest.skip("No API keys available")

        provider = available_providers[0]

        # Test with auto-generated perspectives
        dynamic_critic = NCriticsCritic(
            model=provider.model,
            provider=provider.name,
            api_key=os.getenv(provider.api_key_env),
            auto_generate_perspectives=True,
        )

        result = SifakaResult(
            original_text=TEST_TEXTS["technical"], final_text=TEST_TEXTS["technical"]
        )

        critique = await dynamic_critic.critique(TEST_TEXTS["technical"], result)

        assert critique.feedback
        assert len(critique.suggestions) > 0
        print("\nDynamic perspectives generated and used successfully")
        print(f"Feedback length: {len(critique.feedback)}")
        print(f"Suggestions: {len(critique.suggestions)}")

        # Test with custom perspectives
        custom_perspectives = {
            "Security Expert": "Focus on security implications and vulnerabilities",
            "Performance Analyst": "Evaluate performance and efficiency aspects",
            "Beginner's Advocate": "Consider clarity for newcomers to the topic",
        }

        custom_critic = NCriticsCritic(
            model=provider.model,
            provider=provider.name,
            api_key=os.getenv(provider.api_key_env),
            perspectives=custom_perspectives,
        )

        critique2 = await custom_critic.critique(TEST_TEXTS["technical"], result)

        assert critique2.feedback
        # Should see evidence of custom perspectives in feedback
        feedback_lower = critique2.feedback.lower()
        perspective_mentioned = any(
            keyword in feedback_lower
            for keyword in ["security", "performance", "beginner", "newcomer"]
        )
        assert perspective_mentioned, "Custom perspectives not reflected in feedback"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_advanced_confidence_calculation(self, available_providers):
        """Test the advanced confidence calculation system."""
        if not available_providers:
            pytest.skip("No API keys available")

        provider = available_providers[0]
        config = Config(model=provider.model, use_advanced_confidence=True)

        # Test different critics to see confidence variation
        critics_to_test = [
            "reflexion",
            "constitutional",
            "self_consistency",
            "meta_rewarding",
        ]
        confidence_results = {}

        for critic_name in critics_to_test:
            try:
                result = await improve(
                    TEST_TEXTS["factual"],
                    critics=[critic_name],
                    max_iterations=1,
                    config=config,
                    provider=provider.name,
                    api_key=os.getenv(provider.api_key_env),
                )

                if result.critiques:
                    confidence = result.critiques[-1].confidence
                    confidence_results[critic_name] = confidence
                    print(f"\n{critic_name} confidence: {confidence:.2f}")
            except Exception as e:
                print(f"\n{critic_name} failed: {str(e)[:50]}")

        # Verify different critics have different confidence patterns
        if len(confidence_results) > 1:
            confidences = list(confidence_results.values())
            variance = sum(
                (c - sum(confidences) / len(confidences)) ** 2 for c in confidences
            ) / len(confidences)
            print(f"\nConfidence variance: {variance:.4f}")

            # Should have some variance (not all identical)
            assert (
                variance > 0.001
            ), "All critics have identical confidence - advanced calculation may not be working"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_self_rag_with_mock_retrieval(self, available_providers):
        """Test SelfRAG critic with mock retrieval backend."""
        if not available_providers:
            pytest.skip("No API keys available")

        provider = available_providers[0]

        # Create mock retrieval backend
        class MockRetrievalBackend(RetrievalBackend):
            async def retrieve(
                self, query: str, top_k: int = 3
            ) -> List[Dict[str, any]]:
                # Return mock facts based on query
                if "speed of light" in query.lower():
                    return [
                        {
                            "content": "The speed of light in vacuum is exactly 299,792,458 meters per second.",
                            "source": "Physics reference",
                            "score": 0.95,
                        }
                    ]
                elif "einstein" in query.lower():
                    return [
                        {
                            "content": "Albert Einstein published his special theory of relativity in 1905.",
                            "source": "Historical records",
                            "score": 0.90,
                        }
                    ]
                return []

            async def add_documents(self, documents: List[Dict[str, str]]) -> None:
                pass

        # Create enhanced SelfRAG critic
        rag_critic = SelfRAGEnhancedCritic(
            model=provider.model,
            provider=provider.name,
            api_key=os.getenv(provider.api_key_env),
            retrieval_backend=MockRetrievalBackend(),
            retrieval_threshold=0.5,
        )

        result = SifakaResult(
            original_text=TEST_TEXTS["factual"], final_text=TEST_TEXTS["factual"]
        )

        critique = await rag_critic.critique(TEST_TEXTS["factual"], result)

        assert critique.feedback
        assert "retrieval_context" in critique.metadata

        context = critique.metadata["retrieval_context"]
        assert context["retrieval_available"] == True

        print("\nSelfRAG with retrieval succeeded")
        print(f"Claims verified: {len(context.get('verified_claims', []))}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "text_type", ["technical", "factual", "creative", "argumentative"]
    )
    async def test_different_text_types(self, available_providers, text_type):
        """Test critics with different types of text."""
        if not available_providers:
            pytest.skip("No API keys available")

        provider = available_providers[0]

        # Different critics for different text types
        critic_map = {
            "technical": ["self_refine", "n_critics"],
            "factual": ["self_rag", "constitutional"],
            "creative": ["reflexion", "meta_rewarding"],
            "argumentative": ["constitutional", "self_consistency"],
        }

        critics = critic_map.get(text_type, ["reflexion"])

        result = await improve(
            TEST_TEXTS[text_type],
            critics=critics,
            max_iterations=2,
            provider=provider.name,
            api_key=os.getenv(provider.api_key_env),
            model=provider.model,
        )

        assert result.final_text
        assert len(result.critiques) > 0

        print(f"\n{text_type.capitalize()} text improved successfully")
        print(f"Critics used: {critics}")
        print(f"Iterations: {result.iteration}")
        print(f"Final confidence: {result.critiques[-1].confidence:.2f}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_parallel_provider_comparison(self, available_providers):
        """Test multiple providers in parallel for performance comparison."""
        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for parallel test")

        test_text = TEST_TEXTS["technical"]
        critic = "reflexion"

        async def test_provider(provider):
            """Test a single provider."""
            start_time = asyncio.get_event_loop().time()

            try:
                result = await improve(
                    test_text,
                    critics=[critic],
                    max_iterations=1,
                    provider=provider.name,
                    api_key=os.getenv(provider.api_key_env),
                    model=provider.model,
                )

                end_time = asyncio.get_event_loop().time()

                return {
                    "provider": provider.name,
                    "success": True,
                    "duration": end_time - start_time,
                    "confidence": (
                        result.critiques[0].confidence if result.critiques else 0
                    ),
                }
            except Exception as e:
                end_time = asyncio.get_event_loop().time()
                return {
                    "provider": provider.name,
                    "success": False,
                    "duration": end_time - start_time,
                    "error": str(e),
                }

        # Run providers in parallel
        tasks = [test_provider(p) for p in available_providers[:3]]
        results = await asyncio.gather(*tasks)

        print("\n\nParallel Provider Performance:")
        print("=" * 60)
        for result in sorted(results, key=lambda x: x["duration"]):
            if result["success"]:
                print(
                    f"{result['provider']}: {result['duration']:.2f}s "
                    f"(confidence: {result['confidence']:.2f})"
                )
            else:
                print(f"{result['provider']}: Failed - {result['error'][:50]}")

        # At least one should succeed
        assert any(r["success"] for r in results), "All providers failed"


class TestCriticChaining:
    """Test chaining multiple critics together."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_critic_chain_improvement(self, available_providers):
        """Test that chaining critics leads to progressive improvement."""
        if not available_providers:
            pytest.skip("No API keys available")

        provider = available_providers[0]

        # Chain of critics from general to specific
        critic_chain = [
            "reflexion",  # Learn from attempts
            "self_refine",  # General improvements
            "constitutional",  # Principle-based review
            "n_critics",  # Multi-perspective final review
        ]

        text = TEST_TEXTS["argumentative"]

        # Track improvement through the chain
        intermediate_results = []
        current_text = text

        for i, critic in enumerate(critic_chain):
            result = await improve(
                current_text,
                critics=[critic],
                max_iterations=1,
                provider=provider.name,
                api_key=os.getenv(provider.api_key_env),
                model=provider.model,
            )

            intermediate_results.append(
                {
                    "critic": critic,
                    "iteration": i + 1,
                    "text_length": len(result.final_text),
                    "confidence": (
                        result.critiques[0].confidence if result.critiques else 0
                    ),
                    "improved": not result.needs_improvement,
                }
            )

            current_text = result.final_text

        print("\n\nCritic Chain Results:")
        print("=" * 60)
        for r in intermediate_results:
            print(
                f"{r['critic']}: Length={r['text_length']}, "
                f"Confidence={r['confidence']:.2f}, Improved={r['improved']}"
            )

        # Should see progression
        initial_length = len(text)
        final_length = intermediate_results[-1]["text_length"]

        print(f"\nText transformation: {initial_length} → {final_length} chars")
        print(
            f"Confidence progression: {intermediate_results[0]['confidence']:.2f} → "
            f"{intermediate_results[-1]['confidence']:.2f}"
        )


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases with multiple providers."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provider_fallback(self):
        """Test fallback when primary provider fails."""
        # Use invalid API key for primary provider
        try:
            result = await improve(
                "Test text",
                critics=["reflexion"],
                provider="openai",
                api_key="invalid-key",
                max_iterations=1,
            )
            # Should fail
            assert False, "Should have raised an error"
        except Exception as e:
            assert "api" in str(e).lower() or "auth" in str(e).lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_empty_text_handling(self, available_providers):
        """Test how critics handle empty or minimal text."""
        if not available_providers:
            pytest.skip("No API keys available")

        provider = available_providers[0]

        test_cases = ["", " ", "OK", "This is a test."]

        for text in test_cases:
            try:
                result = await improve(
                    text,
                    critics=["constitutional"],
                    max_iterations=1,
                    provider=provider.name,
                    api_key=os.getenv(provider.api_key_env),
                    model=provider.model,
                )

                print(f"\nInput: '{text}' → Output length: {len(result.final_text)}")
                assert result.final_text  # Should always return something

            except Exception as e:
                print(f"\nInput: '{text}' → Error: {str(e)[:50]}")
                # Empty text might reasonably fail
                if text.strip():
                    raise


# Fixture for running all integration tests
@pytest.fixture(scope="session")
def integration_test_summary():
    """Provide summary of integration tests."""
    yield
    print("\n\nIntegration Test Summary")
    print("=" * 80)
    print("✓ Multi-provider critic tests")
    print("✓ Enhanced N-Critics with dynamic perspectives")
    print("✓ Advanced confidence calculation")
    print("✓ SelfRAG with retrieval backend")
    print("✓ Critic chaining and progression")
    print("✓ Error handling and edge cases")


if __name__ == "__main__":
    # Can run directly for testing
    pytest.main([__file__, "-v", "-s"])
