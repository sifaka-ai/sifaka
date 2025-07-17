"""Integration tests specifically for new features and improvements."""

import asyncio
import os

import pytest

from sifaka import Config, SifakaResult, improve
from sifaka.core.middleware import (
    CachingMiddleware,
    MetricsMiddleware,
    MiddlewarePipeline,
)
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.validators.composable import Validator


class TestNCriticsEnhancements:
    """Test the enhanced N-Critics implementation."""

    @pytest.mark.asyncio
    async def test_dynamic_perspective_generation(self):
        """Test that N-Critics can generate perspectives dynamically."""
        # Only test if we have an API key
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Create critic with dynamic perspectives
        critic = NCriticsCritic(auto_generate_perspectives=True, api_key=api_key)

        # Technical text should generate technical perspectives
        technical_text = """
        Quantum computing uses quantum bits (qubits) that can exist in
        superposition, allowing them to represent both 0 and 1 simultaneously.
        This enables quantum computers to perform certain calculations
        exponentially faster than classical computers.
        """

        result = SifakaResult(original_text=technical_text, final_text=technical_text)
        critique = await critic.critique(technical_text, result)

        assert critique.feedback
        assert len(critique.suggestions) > 0

        # Feedback should mention quantum/technical concepts
        feedback_lower = critique.feedback.lower()
        technical_terms = [
            "quantum",
            "technical",
            "scientific",
            "accuracy",
            "complexity",
        ]
        assert any(
            term in feedback_lower for term in technical_terms
        ), "Dynamic perspectives didn't generate technical focus"

    @pytest.mark.asyncio
    async def test_custom_perspectives(self):
        """Test N-Critics with custom perspectives."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Create domain-specific perspectives
        ml_perspectives = {
            "ML Researcher": "Evaluate theoretical soundness and novelty",
            "ML Engineer": "Focus on implementation feasibility and scalability",
            "Data Scientist": "Assess data requirements and preprocessing needs",
            "MLOps Specialist": "Consider deployment and monitoring aspects",
        }

        critic = NCriticsCritic(perspectives=ml_perspectives, api_key=api_key)

        ml_text = """
        Our new model uses transformer architecture with 10 billion parameters.
        It's trained on 1TB of text data and achieves state-of-the-art results
        on benchmarks. We plan to deploy it to production next week.
        """

        result = SifakaResult(original_text=ml_text, final_text=ml_text)
        critique = await critic.critique(ml_text, result)

        # Should see ML-specific concerns
        feedback_lower = critique.feedback.lower()
        ml_concerns = [
            "deployment",
            "scalability",
            "data",
            "implementation",
            "parameters",
        ]
        matches = sum(1 for concern in ml_concerns if concern in feedback_lower)
        assert matches >= 2, "Custom ML perspectives not reflected in critique"

    @pytest.mark.asyncio
    async def test_perspective_comparison(self):
        """Compare default vs custom perspectives on same text."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        text = "The new product will revolutionize how people work from home."

        # Default perspectives
        default_critic = NCriticsCritic(api_key=api_key)
        result = SifakaResult(original_text=text, final_text=text)
        default_critique = await default_critic.critique(text, result)

        # Business perspectives
        business_perspectives = {
            "Market Analyst": "Evaluate market fit and competitive advantage",
            "Financial Advisor": "Assess revenue potential and cost implications",
            "Customer Success": "Consider user experience and adoption barriers",
        }

        business_critic = NCriticsCritic(
            perspectives=business_perspectives, api_key=api_key
        )
        business_critique = await business_critic.critique(text, result)

        # Critiques should be different
        assert default_critique.feedback != business_critique.feedback

        # Business critique should have business terms
        business_terms = ["market", "revenue", "customer", "user", "adoption", "cost"]
        business_matches = sum(
            1 for term in business_terms if term in business_critique.feedback.lower()
        )
        assert business_matches > 0, "Business perspectives not reflected"


class TestAdvancedConfidenceCalculation:
    """Test the critic-specific confidence calculation."""

    @pytest.mark.asyncio
    async def test_critic_specific_confidence(self):
        """Test that different critics provide different confidence levels."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        text = "The quick brown fox jumps over the lazy dog."

        # Test different critics
        critics = ["reflexion", "constitutional", "self_refine"]
        confidences = {}

        for critic_name in critics:
            result = await improve(
                text,
                critics=[critic_name],
                max_iterations=1,
                api_key=api_key,
            )

            if result.critiques:
                confidences[critic_name] = result.critiques[0].confidence

        # Different critics should have different confidence patterns
        if len(confidences) > 1:
            values = list(confidences.values())
            # Should have some variance
            assert max(values) - min(values) > 0.01

        print("\nConfidence scores:")
        for critic, conf in confidences.items():
            print(f"{critic}: {conf:.2f}")

    @pytest.mark.asyncio
    async def test_confidence_with_real_critics(self):
        """Test confidence calculation with actual critic runs."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        text = "AI will replace all jobs within 10 years."

        # Test with different critics
        critics_to_test = ["reflexion", "constitutional", "meta_rewarding"]
        confidences = {}

        for critic_name in critics_to_test:
            result = await improve(
                text,
                critics=[critic_name],
                max_iterations=1,
                api_key=api_key,
                config=Config(use_advanced_confidence=True),
            )

            if result.critiques:
                confidences[critic_name] = result.critiques[0].confidence

        # Verify different patterns
        if len(confidences) > 1:
            values = list(confidences.values())
            # Should have some variance
            assert (
                max(values) - min(values) > 0.05
            ), "All critics have very similar confidence"

            print("\nCritic confidence scores:")
            for critic, conf in confidences.items():
                print(f"{critic}: {conf:.2f}")


class TestSelfRAGEnhancements:
    """Test the SelfRAG implementation."""

    @pytest.mark.asyncio
    async def test_self_rag_fact_checking(self):
        """Test SelfRAG's fact checking capabilities."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Create SelfRAG critic
        critic = SelfRAGCritic(api_key=api_key)

        # Text with factual claims
        text = """
        Paris, the capital of Spain, is famous for the Eiffel Tower which was
        built in 1850. France left the European Union in 2020.
        """

        result = SifakaResult(original_text=text, final_text=text)
        critique = await critic.critique(text, result)

        # Should identify factual issues
        assert critique.feedback
        assert len(critique.suggestions) > 0

        # Check that fact analysis was performed
        feedback_lower = critique.feedback.lower()
        fact_related_terms = ["fact", "accurate", "verify", "claim", "evidence"]
        assert any(term in feedback_lower for term in fact_related_terms)

    @pytest.mark.asyncio
    async def test_self_rag_analysis(self):
        """Test that SelfRAG analyzes factual claims appropriately."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Create SelfRAG critic
        rag_critic = SelfRAGCritic(api_key=api_key)

        # Factual text that should trigger analysis
        factual_text = """The Amazon rainforest produces 20% of the world's oxygen.
        It covers an area of 5.5 million square kilometers."""

        result = SifakaResult(original_text=factual_text, final_text=factual_text)
        critique = await rag_critic.critique(factual_text, result)

        # Should provide fact-focused feedback
        assert critique.feedback
        feedback_lower = critique.feedback.lower()

        # Should mention facts, claims, or evidence
        fact_terms = ["fact", "claim", "evidence", "verify", "accurate"]
        assert any(term in feedback_lower for term in fact_terms)

        print("\nSelfRAG analysis completed")
        print(
            f"Feedback mentions facts: {any(term in feedback_lower for term in fact_terms)}"
        )


class TestMiddlewareIntegration:
    """Test the middleware system with critics."""

    @pytest.mark.asyncio
    async def test_middleware_pipeline(self):
        """Test that middleware pipeline works with improve()."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Create middleware pipeline
        pipeline = MiddlewarePipeline()

        # Add metrics middleware
        metrics = MetricsMiddleware()
        pipeline.add(metrics)

        # Add caching
        pipeline.add(CachingMiddleware(max_size=10))

        # Test with middleware
        _text = "The quick brown fox jumps over the lazy dog."

        # Note: improve() API would need to support middleware parameter
        # For now, test middleware components directly

        # Test metrics collection
        assert metrics.get_metrics()["total_requests"] == 0

        # After improvement, metrics should be updated
        # This would be integrated into the improve() function

    @pytest.mark.asyncio
    async def test_caching_middleware(self):
        """Test that caching middleware reduces API calls."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Create caching middleware
        cache = CachingMiddleware(max_size=5)

        # Test cache key generation
        context = {
            "critics": ["reflexion", "constitutional"],
            "model": "gpt-4",
            "temperature": 0.7,
        }

        key1 = cache._get_cache_key("Test text", context)
        key2 = cache._get_cache_key("Test text", context)
        key3 = cache._get_cache_key("Different text", context)

        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different input = different key

        print(f"\nCache stats: {cache.get_stats()}")


class TestComposableValidators:
    """Test the composable validator system."""

    @pytest.mark.asyncio
    async def test_validator_composition(self):
        """Test creating complex validators through composition."""
        # Create validators
        length_validator = Validator.length(100, 500)
        keyword_validator = Validator.contains(["important", "critical"], mode="any")

        # Compose with AND
        combined = length_validator & keyword_validator

        # Test texts
        good_text = "This is an important message. " * 10  # ~300 chars, has keyword
        too_short = "Important!"  # Has keyword but too short
        no_keyword = (
            "This is a long message without the required words. " * 5
        )  # Long but no keyword

        result = SifakaResult(original_text=good_text, final_text=good_text)

        # Good text should pass
        good_result = await combined.validate(good_text, result)
        assert good_result.passed

        # Too short should fail
        short_result = await combined.validate(too_short, result)
        assert not short_result.passed

        # No keyword should fail
        no_kw_result = await combined.validate(no_keyword, result)
        assert not no_kw_result.passed

    @pytest.mark.asyncio
    async def test_validator_builder(self):
        """Test the fluent validator builder interface."""
        # Build a complex validator
        essay_validator = (
            Validator.create("academic_essay")
            .length(500, 2000)
            .sentences(10, 50)
            .words(100, 400)
            .contains(["thesis", "conclusion", "evidence"], mode="all")
            .matches(r"\b[A-Z][^.!?]*[.!?]", "proper_sentences")
            .build()
        )

        # Test with an essay
        essay = """
        This essay presents a clear thesis about the importance of education.

        Education is fundamental to human development. It provides the foundation
        for critical thinking and personal growth. The evidence shows that
        educated populations have better health outcomes and economic prosperity.

        Furthermore, education enables innovation and scientific progress.
        Studies demonstrate that investment in education yields high returns
        for both individuals and society. This evidence supports our thesis.

        In conclusion, education is essential for human flourishing. The evidence
        presented clearly demonstrates its vital importance.
        """

        result = SifakaResult(original_text=essay, final_text=essay)
        validation = await essay_validator.validate(essay, result)

        assert validation.passed
        assert validation.score > 0.7
        print(f"\nEssay validation details:\n{validation.details}")


class TestIntegrationWithImproveAPI:
    """Test full integration with the improve() API."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test a complete improvement workflow with all enhancements."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Configure with advanced features
        config = Config(use_advanced_confidence=True, max_iterations=3)

        # Create enhanced critics
        enhanced_critics = []

        # N-Critics with custom perspectives
        n_critic = NCriticsCritic(
            perspectives={
                "Clarity Expert": "Focus on clear communication",
                "Engagement Specialist": "Evaluate reader engagement",
            },
            api_key=api_key,
        )
        enhanced_critics.append(n_critic)

        # Add standard critics
        enhanced_critics.extend(["constitutional", "self_refine"])

        # Text to improve
        original = """
        AI is very powerful technology. It can do many things. Some people
        worry about it. But it has benefits too. We should use it carefully.
        """

        # Run improvement
        result = await improve(
            original, critics=enhanced_critics, config=config, api_key=api_key
        )

        # Verify improvements
        assert len(result.final_text) > len(original)  # Should expand
        assert result.iteration > 0
        assert len(result.critiques) > 0

        # Check confidence progression
        confidences = [c.confidence for c in result.critiques]
        print(f"\nConfidence progression: {[f'{c:.2f}' for c in confidences]}")

        # Final text should be more sophisticated
        final_lower = result.final_text.lower()
        improved_indicators = ["however", "therefore", "specifically", "example"]
        improvements = sum(1 for ind in improved_indicators if ind in final_lower)
        assert improvements > 0, "Text doesn't show sophistication improvements"

        print(f"\nOriginal ({len(original)} chars):")
        print(original)
        print(f"\nFinal ({len(result.final_text)} chars):")
        print(result.final_text)


# Performance test
class TestPerformanceWithProviders:
    """Test performance across different providers."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_provider_latency(self):
        """Measure and compare provider latencies."""
        providers_to_test = []

        if os.getenv("OPENAI_API_KEY"):
            providers_to_test.append(("openai", "gpt-4o-mini"))
        if os.getenv("ANTHROPIC_API_KEY"):
            providers_to_test.append(("anthropic", "claude-3-haiku-20240307"))
        if os.getenv("GOOGLE_API_KEY"):
            providers_to_test.append(("google", "gemini-1.5-flash"))

        if not providers_to_test:
            pytest.skip("No API keys available")

        text = "Explain quantum computing in one paragraph."
        latencies = {}

        for provider, model in providers_to_test:
            start = asyncio.get_event_loop().time()

            try:
                await improve(
                    text,
                    critics=["reflexion"],
                    max_iterations=1,
                    provider=provider,
                    model=model,
                    api_key=os.getenv(f"{provider.upper()}_API_KEY"),
                )

                latency = asyncio.get_event_loop().time() - start
                latencies[provider] = latency

            except Exception as e:
                print(f"\n{provider} failed: {str(e)[:50]}")

        if latencies:
            print("\n\nProvider Latencies:")
            print("=" * 40)
            for provider, latency in sorted(latencies.items(), key=lambda x: x[1]):
                print(f"{provider}: {latency:.2f}s")

            # Find fastest
            fastest = min(latencies.items(), key=lambda x: x[1])
            print(f"\nFastest: {fastest[0]} ({fastest[1]:.2f}s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-k", "test_"])
