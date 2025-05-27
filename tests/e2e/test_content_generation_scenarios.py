#!/usr/bin/env python3
"""End-to-end tests for content generation scenarios.

This module contains comprehensive end-to-end tests that simulate real-world
content generation workflows using the complete Sifaka pipeline.
"""

from unittest.mock import Mock

import pytest

from sifaka.core.chain import Chain
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.critics.prompt import PromptCritic
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.models.base import MockModel
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.content import ContentValidator
from tests.utils.assertions import assert_thought_valid, assert_validation_results


@pytest.mark.e2e
class TestContentGenerationScenarios:
    """Test realistic content generation scenarios."""

    def test_blog_post_generation(self):
        """Test complete blog post generation workflow."""
        # Set up model with realistic blog post content
        model = MockModel(
            model_name="blog-writer",
            response_text="Artificial Intelligence is revolutionizing how we work and live. "
            "From machine learning algorithms that power recommendation systems "
            "to natural language processing that enables chatbots, AI technologies "
            "are becoming increasingly sophisticated and accessible. This transformation "
            "brings both exciting opportunities and important challenges that we must "
            "carefully consider as we move forward into an AI-powered future.",
        )

        # Create chain with comprehensive validation and improvement
        chain = Chain(model=model, prompt="Write a blog post about the impact of AI on society.")

        # Add content validators
        chain = chain.validate_with(LengthValidator(min_length=100, max_length=2000))
        chain = chain.validate_with(
            RegexValidator(required_patterns=[r"[Aa]rtificial [Ii]ntelligence"])
        )
        chain = chain.validate_with(
            ContentValidator(prohibited=["hate", "violence"], name="Safety Filter")
        )

        # Add quality critics
        critic_model = MockModel(
            model_name="critic", response_text="Well-structured and informative content."
        )
        chain = chain.improve_with(ReflexionCritic(model=critic_model))
        chain = chain.improve_with(SelfRefineCritic(model=critic_model))

        # Execute the complete workflow
        result = chain.run()

        # Validate results
        assert_thought_valid(result, min_length=100)
        assert_validation_results(result, expected_count=3, expected_passed=True)

    def test_technical_documentation_generation(self):
        """Test technical documentation generation workflow."""
        model = MockModel(
            model_name="tech-writer",
            response_text="API Documentation: The GET /users endpoint retrieves user information. "
            "Request parameters include 'id' (required) and 'include_profile' (optional). "
            "Response format is JSON with user data including name, email, and profile. "
            "Authentication required via Bearer token in Authorization header. "
            "Rate limiting applies: 100 requests per minute per API key.",
        )

        chain = Chain(model=model, prompt="Write API documentation for a user management endpoint.")

        # Technical documentation requirements
        chain = chain.validate_with(LengthValidator(min_length=50, max_length=1000))
        chain = chain.validate_with(
            RegexValidator(required_patterns=[r"API", r"GET|POST|PUT|DELETE", r"Response"])
        )
        chain = chain.validate_with(
            ContentValidator(prohibited=["TODO", "FIXME"], name="Completeness Check")
        )

        # Technical review
        critic_model = MockModel(
            model_name="tech-critic", response_text="Clear and comprehensive documentation."
        )
        chain = chain.improve_with(ReflexionCritic(model=critic_model))

        result = chain.run()

        assert_thought_valid(result, min_length=50)
        assert_validation_results(result, expected_count=3, expected_passed=True)

    def test_creative_writing_scenario(self):
        """Test creative writing with style and content validation."""
        model = MockModel(
            model_name="creative-writer",
            response_text="In a world where technology and magic coexist, young Maya discovered "
            "she could communicate with AI spirits that lived within the quantum realm. "
            "These digital entities, born from the intersection of advanced tech and "
            "ancient mystical forces, held the key to solving the great energy crisis "
            "that threatened both the physical and virtual worlds.",
        )

        chain = Chain(
            model=model,
            prompt="Write a short fantasy story that incorporates modern technology themes.",
        )

        # Creative writing validation
        chain = chain.validate_with(LengthValidator(min_length=80, max_length=500))
        chain = chain.validate_with(
            RegexValidator(required_patterns=[r"technology|tech", r"magic|fantasy"])
        )

        # Creative enhancement
        critic_model = MockModel(
            model_name="creative-critic", response_text="Engaging narrative with good pacing."
        )
        chain = chain.improve_with(SelfRefineCritic(model=critic_model))
        chain = chain.improve_with(PromptCritic(model=critic_model))

        result = chain.run()

        assert_thought_valid(result, min_length=80)
        assert_validation_results(result, expected_count=2, expected_passed=True)


@pytest.mark.e2e
class TestComplexWorkflowScenarios:
    """Test complex multi-step workflow scenarios."""

    def test_research_and_summary_workflow(self):
        """Test a workflow that involves research and summarization."""
        # Research phase
        research_model = MockModel(
            model_name="researcher",
            response_text="Climate change research shows significant impacts on global weather patterns. "
            "Rising temperatures lead to more frequent extreme weather events. "
            "Ocean acidification affects marine ecosystems. Renewable energy adoption "
            "is accelerating but needs faster implementation to meet climate goals.",
        )

        research_chain = Chain(
            model=research_model,
            prompt="Research the current state of climate change and its impacts.",
        )
        research_chain = research_chain.validate_with(
            LengthValidator(min_length=100, max_length=800)
        )

        research_result = research_chain.run()

        # Summary phase using research result
        summary_model = MockModel(
            model_name="summarizer",
            response_text="Climate change causes extreme weather and ocean acidification. "
            "Renewable energy adoption is growing but needs acceleration.",
        )

        summary_chain = Chain(
            model=summary_model, prompt=f"Summarize this research: {research_result.text}"
        )
        summary_chain = summary_chain.validate_with(LengthValidator(min_length=50, max_length=200))
        summary_chain = summary_chain.improve_with(
            SelfRefineCritic(
                model=MockModel(
                    model_name="summary-critic", response_text="Clear and concise summary."
                )
            )
        )

        summary_result = summary_chain.run()

        # Validate both phases
        assert_thought_valid(research_result, min_length=100)
        assert_thought_valid(summary_result, min_length=50)
        assert len(summary_result.text) < len(research_result.text)  # Summary should be shorter

    def test_iterative_improvement_workflow(self):
        """Test iterative content improvement workflow."""
        model = MockModel(
            model_name="product-writer",
            response_text="Our new smartphone features advanced camera technology, long battery life, "
            "and premium design. The device offers excellent performance for daily tasks "
            "and professional photography. Available in multiple colors with competitive pricing.",
        )

        chain = Chain(model=model, prompt="Write a product description for a new smartphone.")

        # Progressive validation requirements
        chain = chain.validate_with(LengthValidator(min_length=50, max_length=300))
        chain = chain.validate_with(
            ContentValidator(prohibited=["cheap", "basic"], name="Quality Filter")
        )

        # Multiple critics for iterative improvement
        critic_model = MockModel(
            model_name="product-critic",
            response_text="Good features but needs more specific details.",
        )
        chain = chain.improve_with(ReflexionCritic(model=critic_model))
        chain = chain.improve_with(
            ConstitutionalCritic(
                model=critic_model, principles=["Be accurate", "Be persuasive", "Be clear"]
            )
        )

        result = chain.run()

        assert_thought_valid(result, min_length=50)
        assert_validation_results(result, expected_count=2, expected_passed=True)


@pytest.mark.e2e
class TestErrorRecoveryScenarios:
    """Test error recovery and resilience scenarios."""

    def test_validation_failure_recovery(self):
        """Test recovery from validation failures."""
        # Model that initially produces content that fails validation
        model = MockModel(
            model_name="recovery-test", response_text="Short"  # Will fail length validation
        )

        chain = Chain(
            model=model, prompt="Write a detailed explanation of machine learning.", max_retries=3
        )
        chain = chain.validate_with(LengthValidator(min_length=50, max_length=500))

        # Add a critic so there's something to apply during retries
        critic_model = MockModel(
            model_name="recovery-critic", response_text="Needs more detail and examples."
        )
        chain = chain.improve_with(ReflexionCritic(model=critic_model))

        result = chain.run()

        # Should complete even if validation fails
        assert result is not None
        assert result.validation_results is not None

    def test_critic_failure_handling(self):
        """Test handling of critic failures."""
        model = MockModel(
            model_name="base-model", response_text="Base content for testing critic failures."
        )

        # Create a failing critic
        failing_critic_model = Mock()
        failing_critic_model.generate.side_effect = Exception("Critic failure")
        failing_critic = ReflexionCritic(model=failing_critic_model)

        # Create a working critic
        working_critic_model = MockModel(
            model_name="working-critic", response_text="Good content structure."
        )
        working_critic = SelfRefineCritic(model=working_critic_model)

        chain = Chain(model=model, prompt="Test critic error handling.", always_apply_critics=True)

        # Add both failing and working critics
        chain = chain.improve_with(failing_critic)
        chain = chain.improve_with(working_critic)

        result = chain.run()

        # Should complete despite critic failure
        assert result is not None
        # Working critic should still provide feedback
        assert result.critic_feedback is not None


@pytest.mark.e2e
@pytest.mark.slow
class TestPerformanceScenarios:
    """Test performance-related scenarios."""

    def test_high_throughput_scenario(self):
        """Test high-throughput content generation."""
        model = MockModel(
            model_name="throughput-test",
            response_text="Brief summary content for throughput testing.",
        )

        results = []
        for i in range(5):
            chain = Chain(model=model, prompt=f"Write a brief summary about topic {i}.")
            chain = chain.validate_with(LengthValidator(min_length=20, max_length=200))

            result = chain.run()
            results.append(result)

        # All results should be valid
        assert len(results) == 5
        for result in results:
            assert_thought_valid(result, min_length=20)

    def test_complex_validation_performance(self):
        """Test performance with complex validation scenarios."""
        model = MockModel(
            model_name="complex-test",
            response_text="This comprehensive response covers artificial intelligence, machine learning, "
            "neural networks, and deep learning technologies in detail. The content provides "
            "accurate information about modern AI systems and their applications in various "
            "industries including healthcare, finance, and autonomous vehicles.",
        )

        chain = Chain(model=model, prompt="Write a comprehensive overview of AI technologies.")

        # Multiple validators
        chain = chain.validate_with(LengthValidator(min_length=100, max_length=1000))
        chain = chain.validate_with(
            RegexValidator(required_patterns=[r"artificial", r"learning", r"neural"])
        )
        chain = chain.validate_with(
            ContentValidator(prohibited=["error", "fail"], name="Error Filter")
        )

        # Multiple critics
        critic_model = MockModel(
            model_name="complex-critic", response_text="Comprehensive and well-structured content."
        )
        chain = chain.improve_with(ReflexionCritic(model=critic_model))
        chain = chain.improve_with(NCriticsCritic(model=critic_model, num_critics=3))

        result = chain.run()

        assert_thought_valid(result, min_length=100)
        assert_validation_results(result, expected_count=3, expected_passed=True)


@pytest.mark.e2e
class TestSpecializedContentScenarios:
    """Test specialized content generation scenarios."""

    def test_customer_support_response(self):
        """Test customer support response generation."""
        model = MockModel(
            model_name="support-agent",
            response_text="Thank you for contacting our support team. I understand your concern about "
            "the billing issue. I'm here to help resolve this matter quickly. Please provide "
            "your account number so I can review your billing details and find the best solution.",
        )

        chain = Chain(
            model=model,
            prompt="Generate a helpful customer support response for a billing inquiry.",
        )

        # Support response requirements
        chain = chain.validate_with(LengthValidator(min_length=50, max_length=400))
        chain = chain.validate_with(RegexValidator(required_patterns=[r"[Tt]hank", r"help"]))
        chain = chain.validate_with(
            ContentValidator(prohibited=["sorry", "unfortunately"], name="Positive Tone")
        )

        # Ensure helpful and professional tone
        critic_model = MockModel(
            model_name="support-critic", response_text="Professional and solution-focused response."
        )
        chain = chain.improve_with(
            ConstitutionalCritic(
                model=critic_model,
                principles=["Be helpful", "Be professional", "Be solution-focused"],
            )
        )

        result = chain.run()

        assert_thought_valid(result, min_length=50)
        assert_validation_results(result, expected_count=3, expected_passed=True)

    def test_educational_content_creation(self):
        """Test educational content creation workflow."""
        model = MockModel(
            model_name="educator",
            response_text="Python programming is a versatile language perfect for beginners. "
            "Start with basic syntax: variables store data, functions organize code, "
            "and loops repeat actions. For example: 'for i in range(5): print(i)' "
            "prints numbers 0 through 4. Practice with simple projects like calculators "
            "or text games to build confidence and understanding.",
        )

        chain = Chain(
            model=model,
            prompt="Create educational content explaining Python programming basics to beginners.",
        )

        # Educational content requirements
        chain = chain.validate_with(LengthValidator(min_length=100, max_length=800))
        chain = chain.validate_with(
            RegexValidator(required_patterns=[r"Python", r"programming", r"example"])
        )

        # Educational quality review
        critic_model = MockModel(
            model_name="education-critic", response_text="Clear explanations with good examples."
        )
        chain = chain.improve_with(ReflexionCritic(model=critic_model))
        chain = chain.improve_with(
            ConstitutionalCritic(
                model=critic_model, principles=["Be clear", "Be educational", "Include examples"]
            )
        )

        result = chain.run()

        assert_thought_valid(result, min_length=100)
        assert_validation_results(result, expected_count=2, expected_passed=True)
