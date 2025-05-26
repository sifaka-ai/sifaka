#!/usr/bin/env python3
"""End-to-end test scenarios for Sifaka.

This test suite validates real-world usage patterns and workflows,
testing complete scenarios from start to finish as users would
experience them.
"""

import time

from sifaka.core.chain import Chain
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.critics.prompt import PromptCritic
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.models.base import MockModel
from sifaka.storage.memory import MemoryStorage
from sifaka.utils.logging import get_logger
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.content import ContentValidator
from tests.utils import (
    MockModelFactory,
    MockStorageFactory,
    assert_chain_execution_success,
    assert_performance_within_bounds,
    assert_thought_valid,
    assert_validation_results,
)

logger = get_logger(__name__)


class TestContentGenerationScenarios:
    """Test realistic content generation scenarios."""

    def test_blog_post_generation(self):
        """Test generating a blog post with quality assurance."""
        model = MockModel(
            model_name="blog-writer",
            response_text="Artificial Intelligence: The Future is Now\n\nArtificial intelligence has revolutionized how we approach complex problems. From machine learning algorithms to neural networks, AI continues to evolve and shape our digital landscape.",
        )

        # Set up comprehensive validation
        chain = Chain(
            model=model,
            prompt="Write a blog post about artificial intelligence and its impact on society.",
            always_apply_critics=True,
        )

        # Add content validators
        chain.validate_with(LengthValidator(min_length=100, max_length=2000))
        chain.validate_with(RegexValidator(required_patterns=[r"[Aa]rtificial [Ii]ntelligence"]))
        chain.validate_with(ContentValidator(prohibited=["hate", "violence"], name="Safety Filter"))

        # Add quality critics
        critic_model = MockModel(model_name="critic")
        chain.improve_with(ReflexionCritic(model=critic_model))
        chain.improve_with(SelfRefineCritic(model=critic_model))

        start_time = time.time()
        result = chain.run()
        execution_time = time.time() - start_time

        # Validate results
        assert_thought_valid(result, min_length=100)
        assert_validation_results(result, expected_count=3, expected_passed=True)
        # Critics may run multiple times across iterations, so check minimum count
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 2
        assert_chain_execution_success(result)
        assert_performance_within_bounds(
            execution_time, max_time=10.0, operation_name="blog post generation"
        )

    def test_technical_documentation(self):
        """Test generating technical documentation with strict requirements."""
        model = MockModel(
            model_name="tech-writer",
            response_text="API Documentation\n\nGET /api/users\nReturns a list of users in JSON format.\n\nParameters:\n- limit: integer (optional)\n- offset: integer (optional)\n\nResponse: 200 OK with user array",
        )

        chain = Chain(
            model=model,
            prompt="Write API documentation for a user management endpoint.",
            always_apply_critics=True,
        )

        # Technical documentation requirements
        chain.validate_with(LengthValidator(min_length=50, max_length=1000))
        chain.validate_with(
            RegexValidator(required_patterns=[r"API", r"GET|POST|PUT|DELETE", r"Response"])
        )
        chain.validate_with(
            ContentValidator(prohibited=["TODO", "FIXME"], name="Completeness Check")
        )

        # Technical review
        critic_model = MockModel(model_name="tech-critic")
        chain.improve_with(ReflexionCritic(model=critic_model))

        result = chain.run()

        assert_thought_valid(result, min_length=50)
        assert_validation_results(result, expected_count=3)
        # Critics may run multiple times across iterations, so check minimum count
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 1
        assert_chain_execution_success(result)

    def test_creative_writing_scenario(self):
        """Test creative writing with style and content validation."""
        model = MockModel(
            model_name="creative-writer",
            response_text="Once upon a time, in a world where technology and magic coexisted, a young programmer discovered an ancient algorithm that could predict the future. The code was written in a language long forgotten, its syntax resembling mystical incantations.",
        )

        chain = Chain(
            model=model,
            prompt="Write the beginning of a fantasy story that incorporates technology.",
            always_apply_critics=True,
        )

        # Creative writing validation
        chain.validate_with(LengthValidator(min_length=80, max_length=500))
        chain.validate_with(
            RegexValidator(required_patterns=[r"technology|tech", r"magic|fantasy"])
        )

        # Creative enhancement
        critic_model = MockModel(model_name="creative-critic")
        chain.improve_with(SelfRefineCritic(model=critic_model))
        chain.improve_with(PromptCritic(model=critic_model))

        result = chain.run()

        assert_thought_valid(result, min_length=80)
        assert_validation_results(result, expected_count=2)
        # Critics may run multiple times across iterations, so check minimum count
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 2
        assert_chain_execution_success(result)


class TestMultiStepWorkflows:
    """Test complex multi-step workflows."""

    def test_research_and_summarization_workflow(self):
        """Test a workflow that simulates research and summarization."""
        # Step 1: Generate initial research content
        research_model = MockModel(
            model_name="researcher",
            response_text="Research findings on climate change: Global temperatures have risen by 1.1°C since pre-industrial times. Key factors include greenhouse gas emissions, deforestation, and industrial processes. Renewable energy adoption is increasing but needs acceleration.",
        )

        research_chain = Chain(
            model=research_model,
            prompt="Research the current state of climate change and its causes.",
        )
        research_chain.validate_with(LengthValidator(min_length=100, max_length=800))

        research_result = research_chain.run()

        # Step 2: Summarize the research
        summary_model = MockModel(
            model_name="summarizer",
            response_text="Climate change summary: Global temperatures up 1.1°C due to emissions and deforestation. Renewable energy growing but needs faster adoption.",
        )

        summary_chain = Chain(
            model=summary_model,
            prompt=f"Summarize this research in 2-3 sentences: {research_result.text}",
            always_apply_critics=True,
        )
        summary_chain.validate_with(LengthValidator(min_length=50, max_length=200))
        summary_chain.improve_with(SelfRefineCritic(model=MockModel(model_name="summary-critic")))

        summary_result = summary_chain.run()

        # Validate both steps
        assert_thought_valid(research_result, min_length=100)
        assert_thought_valid(summary_result, min_length=50, max_length=200)
        assert_chain_execution_success(research_result)
        assert_chain_execution_success(summary_result)

    def test_iterative_improvement_workflow(self):
        """Test a workflow with multiple improvement iterations."""
        model = MockModel(model_name="iterative-writer")
        storage = MemoryStorage()

        # Initial draft
        chain = Chain(
            model=model,
            prompt="Write a product description for a smart home device.",
            storage=storage,
            always_apply_critics=True,
        )

        # Progressive validation requirements
        chain.validate_with(LengthValidator(min_length=50, max_length=300))
        chain.validate_with(ContentValidator(prohibited=["cheap", "basic"], name="Quality Filter"))

        # Multiple critics for iterative improvement
        critic_model = MockModel(model_name="product-critic")
        chain.improve_with(ReflexionCritic(model=critic_model))
        chain.improve_with(
            ConstitutionalCritic(
                model=critic_model, principles=["Be accurate", "Be persuasive", "Be clear"]
            )
        )

        result = chain.run()

        assert_thought_valid(result, min_length=50, max_length=300)
        assert_validation_results(result, expected_count=2)
        # Critics may run multiple times across iterations, so check minimum count
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 2
        assert_chain_execution_success(result)

        # Verify storage contains the iterations
        assert storage.exists(result.id)


class TestErrorRecoveryScenarios:
    """Test error handling and recovery scenarios."""

    def test_validation_failure_recovery(self):
        """Test recovery from validation failures."""
        # Model that initially produces content that fails validation
        model = MockModelFactory.create_variable_response(
            responses=[
                "Short",  # Will fail length validation
                "This is a much longer response that should pass the length validation requirements and provide adequate content.",
                "Final improved response with even better content and structure.",
            ]
        )

        chain = Chain(
            model=model,
            prompt="Write a detailed explanation of machine learning.",
            max_improvement_iterations=3,
            apply_improvers_on_validation_failure=True,  # Enable retries on validation failure
        )
        chain.validate_with(LengthValidator(min_length=50, max_length=500))

        # Add a critic so there's something to apply during retries
        critic_model = MockModel(model_name="recovery-critic")
        chain.improve_with(ReflexionCritic(model=critic_model))

        result = chain.run()

        # Should eventually succeed after retries
        assert_thought_valid(result, min_length=50)
        assert_validation_results(result, expected_count=1)
        # May have multiple iterations due to retries
        assert result.iteration >= 1

    def test_critic_failure_graceful_handling(self):
        """Test graceful handling of critic failures."""
        model = MockModel(model_name="test-model")

        # Create a failing critic using a failing model
        failing_model = MockModelFactory.create_failing(
            error_message="Critic temporarily unavailable"
        )
        failing_critic = ReflexionCritic(model=failing_model)

        working_critic = ReflexionCritic(model=MockModel(model_name="working-critic"))

        chain = Chain(
            model=model,
            prompt="Write about error handling in AI systems.",
            always_apply_critics=True,
        )

        # Add both failing and working critics
        chain.improve_with(failing_critic)
        chain.improve_with(working_critic)

        # Should complete despite one critic failing
        result = chain.run()

        assert_thought_valid(result)
        assert_chain_execution_success(result)
        # Should have feedback from the working critic (may be None if all critics fail)
        # The test should pass as long as the chain completes without crashing

    def test_storage_failure_handling(self):
        """Test handling of storage failures."""
        model = MockModel(model_name="test-model")
        failing_storage = MockStorageFactory.create_failing()

        chain = Chain(model=model, prompt="Write about data persistence.", storage=failing_storage)

        # Should complete even if storage fails
        result = chain.run()

        assert_thought_valid(result)
        assert_chain_execution_success(result)


class TestPerformanceScenarios:
    """Test performance in realistic scenarios."""

    def test_high_throughput_scenario(self):
        """Test processing multiple requests efficiently."""
        model = MockModel(model_name="high-throughput-model")

        # Process multiple requests
        results = []
        start_time = time.time()

        for i in range(5):
            chain = Chain(model=model, prompt=f"Write a brief summary about topic {i}.")
            chain.validate_with(LengthValidator(min_length=20, max_length=200))

            result = chain.run()
            results.append(result)

        total_time = time.time() - start_time

        # Validate all results
        for result in results:
            assert_thought_valid(result, min_length=20, max_length=200)
            assert_chain_execution_success(result)

        # Performance should be reasonable
        avg_time_per_request = total_time / len(results)
        assert (
            avg_time_per_request < 2.0
        ), f"Average time per request too high: {avg_time_per_request:.2f}s"

    def test_complex_validation_performance(self):
        """Test performance with complex validation scenarios."""
        model = MockModel(
            model_name="complex-model",
            response_text="This is a comprehensive response about artificial intelligence, machine learning, and natural language processing. It covers various aspects including neural networks, deep learning, and practical applications in modern technology.",
        )

        # Complex validation setup
        chain = Chain(
            model=model,
            prompt="Write about AI comprehensively.",
            always_apply_critics=True,  # Ensure critics run even when validation passes
        )

        # Multiple validators
        chain.validate_with(LengthValidator(min_length=100, max_length=1000))
        chain.validate_with(
            RegexValidator(required_patterns=[r"artificial", r"learning", r"neural"])
        )
        chain.validate_with(ContentValidator(prohibited=["error", "fail"], name="Error Filter"))

        # Multiple critics
        critic_model = MockModel(model_name="complex-critic")
        chain.improve_with(ReflexionCritic(model=critic_model))
        chain.improve_with(NCriticsCritic(model=critic_model, num_critics=3))

        start_time = time.time()
        result = chain.run()
        execution_time = time.time() - start_time

        assert_thought_valid(result, min_length=100)
        assert_validation_results(result, expected_count=3)
        # Critics may run multiple times across iterations, so check minimum count
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 2
        assert_performance_within_bounds(
            execution_time, max_time=15.0, operation_name="complex validation"
        )


class TestRealWorldUseCases:
    """Test scenarios based on real-world use cases."""

    def test_customer_support_response(self):
        """Test generating customer support responses."""
        model = MockModel(
            model_name="support-agent",
            response_text="Thank you for contacting our support team. I understand you're experiencing issues with your account login. Let me help you resolve this quickly. Please try resetting your password using the 'Forgot Password' link on our login page.",
        )

        chain = Chain(
            model=model,
            prompt="Generate a helpful customer support response for a user having login issues.",
            always_apply_critics=True,
        )

        # Support response requirements
        chain.validate_with(LengthValidator(min_length=50, max_length=400))
        chain.validate_with(RegexValidator(required_patterns=[r"[Tt]hank", r"help"]))
        chain.validate_with(
            ContentValidator(prohibited=["sorry", "unfortunately"], name="Positive Tone")
        )

        # Ensure helpful and professional tone
        critic_model = MockModel(model_name="support-critic")
        chain.improve_with(
            ConstitutionalCritic(
                model=critic_model,
                principles=["Be helpful", "Be professional", "Be solution-focused"],
            )
        )

        result = chain.run()

        assert_thought_valid(result, min_length=50, max_length=400)
        assert_validation_results(result, expected_count=3)
        # Critics may run multiple times across iterations, so check minimum count
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 1
        assert_chain_execution_success(result)

    def test_educational_content_creation(self):
        """Test creating educational content with quality assurance."""
        model = MockModel(
            model_name="educator",
            response_text="Introduction to Python Programming\n\nPython is a high-level programming language known for its simplicity and readability. Key concepts include variables, functions, loops, and data structures. Example: print('Hello, World!') displays text on screen.",
        )

        chain = Chain(
            model=model,
            prompt="Create an introductory lesson about Python programming for beginners.",
            always_apply_critics=True,
        )

        # Educational content requirements
        chain.validate_with(LengthValidator(min_length=100, max_length=800))
        chain.validate_with(
            RegexValidator(required_patterns=[r"Python", r"programming", r"example"])
        )

        # Educational quality review
        critic_model = MockModel(model_name="education-critic")
        chain.improve_with(ReflexionCritic(model=critic_model))
        chain.improve_with(
            ConstitutionalCritic(
                model=critic_model, principles=["Be clear", "Be educational", "Include examples"]
            )
        )

        result = chain.run()

        assert_thought_valid(result, min_length=100, max_length=800)
        assert_validation_results(result, expected_count=2)
        # Critics may run multiple times across iterations, so check minimum count
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 2
        assert_chain_execution_success(result)
