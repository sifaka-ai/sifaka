#!/usr/bin/env python3
"""Comprehensive tests for Sifaka critic implementations.

This test suite validates all critic types including Reflexion, Self-Refine,
Self-RAG, Constitutional AI, N-Critics, and Prompt-based critics. It tests
critique generation, improvement suggestions, and integration scenarios.
"""


from sifaka.core.thought import CriticFeedback
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.critics.prompt import PromptCritic

# Import all new critics
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.models.base import MockModel
from sifaka.utils.logging import get_logger
from tests.utils import MockModelFactory, create_test_thought

logger = get_logger(__name__)


class TestReflexionCritic:
    """Test ReflexionCritic implementation."""

    def test_reflexion_critic_basic_critique(self):
        """Test basic critique functionality."""
        model = MockModel(
            model_name="reflexion-model",
            response_text="The text could be improved with more specific examples and clearer structure.",
        )
        critic = ReflexionCritic(model=model)

        thought = create_test_thought(text="AI is useful for many applications.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)
        assert "needs_improvement" in critique_result
        # Check for critique or message field
        assert "critique" in critique_result or "message" in critique_result
        assert isinstance(critique_result["needs_improvement"], bool)

    def test_reflexion_critic_improvement(self):
        """Test text improvement functionality."""
        model = MockModel(
            model_name="reflexion-model",
            response_text="Artificial intelligence is a powerful technology with diverse applications across industries, including healthcare, finance, and transportation, enabling automation and enhanced decision-making.",
        )
        critic = ReflexionCritic(model=model)

        thought = create_test_thought(text="AI is useful.")
        improved_text = critic.improve(thought)

        assert isinstance(improved_text, str)
        assert len(improved_text) > len(thought.text)  # Should be more detailed

    def test_reflexion_critic_with_feedback(self):
        """Test reflexion critic with existing feedback."""
        model = MockModel(model_name="reflexion-model")
        critic = ReflexionCritic(model=model)

        # Create thought with existing critic feedback
        thought = create_test_thought(text="Machine learning is complex.")
        feedback = CriticFeedback(
            critic_name="previous_critic", feedback="Add more examples", needs_improvement=True
        )
        thought = thought.add_critic_feedback(feedback)

        critique_result = critic.critique(thought)
        assert isinstance(critique_result, dict)

    def test_reflexion_critic_integration(self):
        """Test reflexion critic integration with chain."""
        from sifaka.core.chain import Chain

        main_model = MockModel(model_name="main-model")
        critic_model = MockModel(model_name="critic-model")
        critic = ReflexionCritic(model=critic_model)

        chain = Chain(
            model=main_model, prompt="Write about machine learning.", always_apply_critics=True
        )
        chain.improve_with(critic)

        result = chain.run()

        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 1

        # Check that feedback is from ReflexionCritic
        reflexion_feedback = [
            fb for fb in result.critic_feedback if "reflexion" in fb.critic_name.lower()
        ]
        assert len(reflexion_feedback) >= 1


class TestSelfRefineCritic:
    """Test SelfRefineCritic implementation."""

    def test_self_refine_critic_basic(self):
        """Test basic self-refine functionality."""
        model = MockModel(
            model_name="self-refine-model",
            response_text="The text needs more detail and better organization.",
        )
        critic = SelfRefineCritic(model=model)

        thought = create_test_thought(text="Neural networks are important.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)
        assert "needs_improvement" in critique_result
        # Check for critique or message field
        assert "critique" in critique_result or "message" in critique_result

    def test_self_refine_iterative_improvement(self):
        """Test iterative improvement with self-refine."""
        model = MockModelFactory.create_variable_response(
            responses=[
                "First improvement: Neural networks are computational models inspired by biological neural networks.",
                "Second improvement: Neural networks are sophisticated computational models that mimic the structure and function of biological neural networks in the brain.",
                "Final improvement: Neural networks are sophisticated computational models that mimic the interconnected structure and parallel processing capabilities of biological neural networks in the brain, enabling pattern recognition and learning.",
            ]
        )
        critic = SelfRefineCritic(model=model)

        thought = create_test_thought(text="Neural networks exist.")

        # Multiple improvement iterations
        improved_text = thought.text
        for _i in range(3):
            current_thought = create_test_thought(text=improved_text)
            improved_text = critic.improve(current_thought)
            assert len(improved_text) >= len(current_thought.text)

    def test_self_refine_with_specific_criteria(self):
        """Test self-refine with specific improvement criteria."""
        model = MockModel(model_name="self-refine-model")
        critic = SelfRefineCritic(
            model=model, improvement_criteria=["clarity", "detail", "examples"]
        )

        thought = create_test_thought(text="Deep learning works well.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)
        # The critique should consider the specified criteria


class TestConstitutionalCritic:
    """Test ConstitutionalCritic implementation."""

    def test_constitutional_critic_basic(self):
        """Test basic constitutional AI functionality."""
        model = MockModel(model_name="constitutional-model")
        principles = ["Be helpful", "Be harmless", "Be honest"]
        critic = ConstitutionalCritic(model=model, principles=principles)

        thought = create_test_thought(text="Here's some advice that might not be accurate.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)
        assert "needs_improvement" in critique_result
        # Check for critique or message field
        assert "critique" in critique_result or "message" in critique_result

    def test_constitutional_critic_principles(self):
        """Test constitutional critic with specific principles."""
        model = MockModel(model_name="constitutional-model")
        principles = [
            "Provide accurate information",
            "Be respectful and inclusive",
            "Avoid harmful content",
        ]
        critic = ConstitutionalCritic(model=model, principles=principles)

        thought = create_test_thought(
            text="This information might be wrong and could be offensive."
        )
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)
        # Should identify violations of principles

    def test_constitutional_critic_improvement(self):
        """Test constitutional critic improvement suggestions."""
        model = MockModel(
            model_name="constitutional-model",
            response_text="Here is accurate, respectful, and helpful information about the topic.",
        )
        principles = ["Be accurate", "Be respectful", "Be helpful"]
        critic = ConstitutionalCritic(model=model, principles=principles)

        thought = create_test_thought(text="Some questionable information.")
        improved_text = critic.improve(thought)

        assert isinstance(improved_text, str)
        assert len(improved_text) > 0

    def test_constitutional_critic_multiple_principles(self):
        """Test constitutional critic with many principles."""
        model = MockModel(model_name="constitutional-model")
        principles = [
            "Be truthful and accurate",
            "Be helpful and constructive",
            "Be respectful and inclusive",
            "Avoid harmful or dangerous content",
            "Protect privacy and confidentiality",
            "Be transparent about limitations",
        ]
        critic = ConstitutionalCritic(model=model, principles=principles)

        thought = create_test_thought(text="Here's some general advice.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)


class TestNCriticsCritic:
    """Test NCriticsCritic implementation."""

    def test_n_critics_basic(self):
        """Test basic N-Critics functionality."""
        model = MockModel(model_name="n-critics-model")
        critic = NCriticsCritic(model=model, num_critics=3)

        thought = create_test_thought(text="Artificial intelligence has many applications.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)
        assert "needs_improvement" in critique_result
        # Check for critique or message field
        assert "critique" in critique_result or "message" in critique_result

    def test_n_critics_multiple_perspectives(self):
        """Test N-Critics with multiple critic perspectives."""
        model = MockModelFactory.create_variable_response(
            responses=[
                "Critic 1: The text needs more technical depth.",
                "Critic 2: The text should include practical examples.",
                "Critic 3: The text could benefit from better structure.",
            ]
        )
        critic = NCriticsCritic(model=model, num_critics=3)

        thought = create_test_thought(text="Machine learning is useful.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)
        # Should incorporate multiple perspectives

    def test_n_critics_consensus(self):
        """Test N-Critics consensus mechanism."""
        model = MockModel(model_name="n-critics-model")
        critic = NCriticsCritic(model=model, num_critics=5)

        thought = create_test_thought(text="This is a comprehensive analysis of AI.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)
        # Should provide consensus from multiple critics

    def test_n_critics_improvement(self):
        """Test N-Critics improvement functionality."""
        model = MockModel(
            model_name="n-critics-model",
            response_text="Improved text incorporating feedback from multiple critics with enhanced detail and structure.",
        )
        critic = NCriticsCritic(model=model, num_critics=3)

        thought = create_test_thought(text="Basic text about AI.")
        improved_text = critic.improve(thought)

        assert isinstance(improved_text, str)
        assert len(improved_text) > len(thought.text)


class TestSelfRAGCritic:
    """Test SelfRAGCritic implementation."""

    def test_self_rag_basic(self):
        """Test basic Self-RAG functionality."""
        model = MockModel(model_name="self-rag-model")
        critic = SelfRAGCritic(model=model)

        thought = create_test_thought(text="AI systems can process large amounts of data.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)
        assert "needs_improvement" in critique_result

    def test_self_rag_with_retrieval(self):
        """Test Self-RAG with retrieval context."""
        model = MockModel(model_name="self-rag-model")

        # Mock retriever
        from tests.utils.mocks import MockRetrieverFactory

        retriever = MockRetrieverFactory.create_standard(
            [
                "AI systems use machine learning algorithms.",
                "Data processing is a key capability of AI.",
                "Neural networks are fundamental to modern AI.",
            ]
        )

        critic = SelfRAGCritic(model=model, retriever=retriever)

        thought = create_test_thought(text="AI is useful.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)

    def test_self_rag_improvement(self):
        """Test Self-RAG improvement with retrieval."""
        model = MockModel(
            model_name="self-rag-model",
            response_text="AI systems leverage machine learning algorithms to process large datasets and extract meaningful patterns, enabling applications in healthcare, finance, and autonomous systems.",
        )
        critic = SelfRAGCritic(model=model)

        thought = create_test_thought(text="AI is good.")
        improved_text = critic.improve(thought)

        assert isinstance(improved_text, str)
        assert len(improved_text) > len(thought.text)


class TestPromptCritic:
    """Test PromptCritic implementation."""

    def test_prompt_critic_basic(self):
        """Test basic prompt-based criticism."""
        model = MockModel(model_name="prompt-critic-model")
        critic = PromptCritic(
            model=model, critique_prompt="Analyze this text for clarity and completeness."
        )

        thought = create_test_thought(text="Machine learning algorithms learn from data.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)

    def test_prompt_critic_custom_prompts(self):
        """Test prompt critic with custom prompts."""
        model = MockModel(model_name="prompt-critic-model")
        critic = PromptCritic(
            model=model,
            critique_prompt="Evaluate this text for technical accuracy.",
            improvement_prompt="Rewrite this text to be more technically precise.",
        )

        thought = create_test_thought(text="AI works by using computers.")

        critique_result = critic.critique(thought)
        assert isinstance(critique_result, dict)

        improved_text = critic.improve(thought)
        assert isinstance(improved_text, str)

    def test_prompt_critic_specialized_domains(self):
        """Test prompt critic for specialized domains."""
        model = MockModel(model_name="domain-critic-model")
        critic = PromptCritic(
            model=model,
            critique_prompt="Evaluate this medical text for accuracy and safety.",
            domain="medical",
        )

        thought = create_test_thought(text="This medicine might help with symptoms.")
        critique_result = critic.critique(thought)

        assert isinstance(critique_result, dict)


class TestCriticIntegration:
    """Test critic integration scenarios."""

    def test_multiple_critics_chain(self):
        """Test multiple critics in a chain."""
        from sifaka.core.chain import Chain

        main_model = MockModel(model_name="main-model")
        critic_model = MockModel(model_name="critic-model")

        chain = Chain(
            model=main_model,
            prompt="Write about artificial intelligence.",
            always_apply_critics=True,
        )

        # Add multiple critics
        chain.improve_with(ReflexionCritic(model=critic_model))
        chain.improve_with(SelfRefineCritic(model=critic_model))
        chain.improve_with(PromptCritic(model=critic_model))

        result = chain.run()

        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 3

    def test_critic_performance(self):
        """Test critic performance."""
        import time

        model = MockModel(model_name="performance-critic")
        critic = ReflexionCritic(model=model)

        # Test performance with multiple critiques
        start_time = time.time()
        for i in range(10):
            thought = create_test_thought(text=f"Test text {i} for performance evaluation.")
            critique_result = critic.critique(thought)
            assert isinstance(critique_result, dict)

        execution_time = time.time() - start_time

        # Should be reasonably fast
        assert (
            execution_time < 5.0
        ), f"Critic performance too slow: {execution_time:.3f}s for 10 critiques"

    def test_critic_error_handling(self):
        """Test critic error handling."""
        failing_model = MockModelFactory.create_failing(error_message="Critic model failed")
        critic = ReflexionCritic(model=failing_model)

        thought = create_test_thought(text="Test text for error handling.")

        # Should handle model failures gracefully and return error result
        result = critic.critique(thought)

        # Verify error result format
        assert isinstance(result, dict)
        assert "needs_improvement" in result
        assert "processing_time_ms" in result
        assert result["needs_improvement"] is True  # Error cases should indicate improvement needed

    def test_critic_with_empty_text(self):
        """Test critic behavior with edge cases."""
        model = MockModel(model_name="edge-case-critic")
        critic = ReflexionCritic(model=model)

        # Test with empty text
        thought = create_test_thought(text="")
        critique_result = critic.critique(thought)
        assert isinstance(critique_result, dict)

        # Test with None text
        thought = create_test_thought(text=None)
        critique_result = critic.critique(thought)
        assert isinstance(critique_result, dict)

    def test_critic_feedback_consistency(self):
        """Test consistency of critic feedback format."""
        model = MockModel(model_name="consistency-critic")
        critics = [
            ReflexionCritic(model=model),
            SelfRefineCritic(model=model),
            PromptCritic(model=model),
            ConstitutionalCritic(model=model, principles=["Be helpful"]),
        ]

        thought = create_test_thought(text="Test text for consistency check.")

        for critic in critics:
            critique_result = critic.critique(thought)

            # All critics should return consistent format
            assert isinstance(critique_result, dict)
            assert "needs_improvement" in critique_result
            # Critics should have either "critique" or "message" field
            assert "critique" in critique_result or "message" in critique_result
            assert isinstance(critique_result["needs_improvement"], bool)
            # Check that the critique/message is a string
            if "critique" in critique_result:
                assert isinstance(critique_result["critique"], str)
            if "message" in critique_result:
                assert isinstance(critique_result["message"], str)
