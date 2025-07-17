"""Tests for Reflexion critic."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.config import Config
from sifaka.core.models import CritiqueResult, Generation, SifakaResult
from sifaka.critics.reflexion import (
    ReflexionCritic,
    ReflexionResponse,
)


class TestReflexionResponse:
    """Test the ReflexionResponse model."""

    def test_creation_minimal(self):
        """Test creating response with minimal fields."""
        response = ReflexionResponse(
            feedback="Text needs improvement in clarity",
            suggestions=["Simplify complex sentences"],
            needs_improvement=True,
        )
        assert response.feedback == "Text needs improvement in clarity"
        assert len(response.suggestions) == 1
        assert response.confidence == 0.7  # default
        assert response.evolution_summary == ""  # default

    def test_creation_full(self):
        """Test creating response with all fields."""
        response = ReflexionResponse(
            feedback="Significant improvement from previous iteration",
            suggestions=["Add concrete examples", "Strengthen conclusion"],
            needs_improvement=True,
            confidence=0.85,
            evolution_summary="Text has become clearer and more structured",
            key_learnings=["Opening is now stronger", "Flow improved"],
            metadata={"iterations_analyzed": 3},
        )

        assert response.confidence == 0.85
        assert "clearer and more structured" in response.evolution_summary
        assert len(response.key_learnings) == 2
        assert response.metadata["iterations_analyzed"] == 3

    def test_confidence_bounds(self):
        """Test confidence bounds validation."""
        response = ReflexionResponse(
            feedback="Test", suggestions=[], needs_improvement=False, confidence=1.0
        )
        assert response.confidence == 1.0

        response2 = ReflexionResponse(
            feedback="Test", suggestions=[], needs_improvement=True, confidence=0.0
        )
        assert response2.confidence == 0.0


class TestReflexionCritic:
    """Test the ReflexionCritic class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Final text")

    @pytest.fixture
    def result_with_history(self):
        """Create a SifakaResult with critique history."""
        result = SifakaResult(
            original_text="Original text", final_text="Current text version"
        )

        # Add some critiques
        result.critiques.append(
            CritiqueResult(
                critic="reflexion",
                feedback="Initial feedback: needs more clarity",
                suggestions=["Simplify opening"],
                needs_improvement=True,
                confidence=0.6,
            )
        )
        result.critiques.append(
            CritiqueResult(
                critic="reflexion",
                feedback="Better, but still needs work on structure",
                suggestions=["Reorganize middle section"],
                needs_improvement=True,
                confidence=0.7,
            )
        )

        # Add some generations
        result.generations.append(
            Generation(text="First version", model="gpt-4o-mini", iteration=1)
        )
        result.generations.append(
            Generation(text="Second version", model="gpt-4o-mini", iteration=2)
        )

        return result

    def test_initialization_default(self):
        """Test default initialization."""
        critic = ReflexionCritic()
        assert critic.name == "reflexion"
        assert critic.model == "gpt-4o-mini"
        assert critic.temperature == 0.7

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Config()
        config.temperature = 0.5
        critic = ReflexionCritic(config=config)
        assert critic.config == config

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        critic = ReflexionCritic(model="gpt-4", temperature=0.8, api_key="test-key")
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.8

    def test_get_response_type(self):
        """Test that critic uses ReflexionResponse."""
        critic = ReflexionCritic()
        assert critic._get_response_type() == ReflexionResponse

    def test_get_system_prompt(self):
        """Test system prompt generation."""
        critic = ReflexionCritic()
        prompt = critic._get_system_prompt()
        assert "Reflexion technique" in prompt
        assert "self-improvement" in prompt
        assert "iterative reflection" in prompt

    def test_build_context_no_history(self, sample_result):
        """Test context building with no history."""
        critic = ReflexionCritic()
        context = critic._build_context(sample_result)
        assert "first iteration" in context
        assert "No previous feedback" in context

    def test_build_context_with_history(self, result_with_history):
        """Test context building with critique history."""
        critic = ReflexionCritic()
        context = critic._build_context(result_with_history)

        assert "Previous reflections" in context
        assert "Iteration 1" in context
        assert "needs more clarity" in context
        assert "Iteration 2" in context
        assert "structure" in context
        assert "2 iterations" in context

    def test_build_context_truncates_old_critiques(self):
        """Test that context only includes last 3 critiques."""
        result = SifakaResult(original_text="Original", final_text="Final")

        # Add 5 critiques
        for i in range(5):
            result.critiques.append(
                CritiqueResult(
                    critic="reflexion",
                    feedback=f"Feedback {i+1}",
                    suggestions=[f"Suggestion {i+1}"],
                    needs_improvement=True,
                )
            )

        critic = ReflexionCritic()
        context = critic._build_context(result)

        # Should only see last 3
        assert "Feedback 3" in context
        assert "Feedback 4" in context
        assert "Feedback 5" in context
        assert "Feedback 1" not in context
        assert "Feedback 2" not in context

    @pytest.mark.asyncio
    async def test_create_messages_first_iteration(self, sample_result):
        """Test message creation for first iteration."""
        critic = ReflexionCritic()
        messages = await critic._create_messages("Test text", sample_result)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "expert text critic" in messages[0]["content"]

        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert "Test text" in user_content
        assert "first iteration" in user_content
        assert "Reflect on this text" in user_content

    @pytest.mark.asyncio
    async def test_create_messages_with_history(self, result_with_history):
        """Test message creation with history."""
        critic = ReflexionCritic()
        messages = await critic._create_messages("Current text", result_with_history)

        user_content = messages[1]["content"]
        assert "Current text" in user_content
        assert "Previous reflections" in user_content
        assert "needs more clarity" in user_content
        assert "structure" in user_content

    @pytest.mark.asyncio
    async def test_critique_success_first_iteration(self, sample_result):
        """Test successful critique on first iteration."""
        critic = ReflexionCritic()

        # Mock the LLM response
        mock_response = ReflexionResponse(
            feedback="The text lacks clarity and structure",
            suggestions=["Add topic sentences", "Improve transitions"],
            needs_improvement=True,
            confidence=0.6,
            evolution_summary="First iteration - establishing baseline",
            key_learnings=["Needs fundamental restructuring"],
            metadata={"analysis_type": "initial"},
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Test text", sample_result)

            assert result.critic == "reflexion"
            assert result.feedback == mock_response.feedback
            assert len(result.suggestions) == 2
            assert result.needs_improvement is True
            assert result.confidence == 0.6
            assert "evolution_summary" in result.metadata
            assert "key_learnings" in result.metadata

    @pytest.mark.asyncio
    async def test_critique_with_improvement_trajectory(self, result_with_history):
        """Test critique showing improvement over iterations."""
        critic = ReflexionCritic()

        mock_response = ReflexionResponse(
            feedback="Significant progress - structure much improved",
            suggestions=["Minor polish on conclusion"],
            needs_improvement=True,
            confidence=0.9,
            evolution_summary="Text has evolved from unclear to well-structured",
            key_learnings=[
                "Opening simplification was effective",
                "Middle reorganization improved flow",
                "Conclusion still needs minor work",
            ],
            metadata={"improvement_rate": "high"},
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Improved text", result_with_history)

            assert result.confidence == 0.9  # Higher confidence
            assert "Significant progress" in result.feedback
            assert len(result.metadata["key_learnings"]) == 3

    @pytest.mark.asyncio
    async def test_critique_no_improvement_needed(self, result_with_history):
        """Test critique when text no longer needs improvement."""
        critic = ReflexionCritic()

        mock_response = ReflexionResponse(
            feedback="Text has reached high quality through iterations",
            suggestions=[],
            needs_improvement=False,
            confidence=0.95,
            evolution_summary="Successfully refined through 3 iterations",
            key_learnings=[
                "Iterative refinement was effective",
                "All major issues addressed",
            ],
            metadata={"final_iteration": True},
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Final text", result_with_history)

            assert result.needs_improvement is False
            assert len(result.suggestions) == 0
            assert result.confidence == 0.95
            assert result.metadata["final_iteration"] is True

    def test_provider_configuration(self):
        """Test provider configuration."""
        from sifaka.core.llm_client import Provider

        critic = ReflexionCritic(provider=Provider.ANTHROPIC)
        assert critic.provider == Provider.ANTHROPIC

        critic = ReflexionCritic(provider="openai")
        assert critic.provider == Provider.OPENAI

    @pytest.mark.asyncio
    async def test_reflexion_learning_pattern(self):
        """Test that reflexion shows learning across iterations."""
        critic = ReflexionCritic()
        result = SifakaResult(original_text="Original", final_text="Current")

        # Simulate 3 iterations with improving confidence
        confidences = [0.5, 0.7, 0.9]
        for i, conf in enumerate(confidences):
            mock_response = ReflexionResponse(
                feedback=f"Iteration {i+1} feedback",
                suggestions=[f"Suggestion {i+1}"] if i < 2 else [],
                needs_improvement=i < 2,
                confidence=conf,
                evolution_summary=f"Progress level {i+1}",
                key_learnings=[f"Learning from iteration {i+1}"],
            )

            mock_agent_result = Mock()
            mock_agent_result.output = mock_response

            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_agent_result)

            with patch.object(critic.client, "create_agent", return_value=mock_agent):
                critique = await critic.critique(f"Text v{i+1}", result)
                result.critiques.append(critique)

                # Verify increasing confidence
                assert critique.confidence == conf

        # Final iteration should not need improvement
        assert not result.critiques[-1].needs_improvement
        assert len(result.critiques[-1].suggestions) == 0
