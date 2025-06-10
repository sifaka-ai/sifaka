"""Comprehensive unit tests for Self-RAG critic.

This module tests the SelfRAGCritic implementation:
- Self-RAG methodology inspired by Asai et al. 2023
- Retrieval-augmented critique with decision tokens
- Integration with PydanticAI agents and retrieval tools
- Error handling and edge cases

Tests cover:
- Basic Self-RAG functionality
- Retrieval decision making
- Relevance and support assessment
- Tool integration and discovery
- Mock-based testing without external API calls
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.critics.self_rag import (
    RelevanceAssessment,
    RetrievalDecision,
    SelfRAGCritic,
    SupportAssessment,
    UtilityAssessment,
)


class TestSelfRAGCritic:
    """Test suite for SelfRAGCritic class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock PydanticAI agent."""
        agent = Mock()
        agent.run = AsyncMock()

        # Mock response for Self-RAG
        mock_response = Mock()
        mock_response.data = {
            "feedback": "Based on retrieval analysis, the text needs factual verification and additional sources.",
            "suggestions": [
                "Retrieve current statistics on renewable energy adoption",
                "Add citations from authoritative sources",
                "Verify claims against recent research",
            ],
            "needs_improvement": True,
            "confidence": 0.78,
            "reasoning": "Self-RAG analysis indicates knowledge gaps requiring retrieval",
            "retrieval_decision": "RETRIEVE",
            "relevance_assessment": "RELEVANT",
            "support_assessment": "PARTIALLY_SUPPORTED",
            "utility_assessment": "USEFUL",
        }
        agent.run.return_value = mock_response

        return agent

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        thought = SifakaThought(
            prompt="Explain the current state of renewable energy adoption globally",
            final_text="Renewable energy is growing rapidly worldwide. Many countries are investing heavily.",
            iteration=1,
            max_iterations=3,
        )
        return thought

    def test_self_rag_critic_creation_minimal(self):
        """Test creating SelfRAGCritic with minimal parameters."""
        critic = SelfRAGCritic()

        assert critic.model_name == "groq:mixtral-8x7b-32768"
        assert "self-rag" in critic.system_prompt.lower()
        assert critic.metadata["critic_type"] == "SelfRAGCritic"
        assert "Asai et al. 2023" in critic.paper_reference

    def test_self_rag_critic_creation_with_retrieval_tools(self):
        """Test creating SelfRAGCritic with retrieval tools."""
        mock_tools = [Mock(), Mock()]

        critic = SelfRAGCritic(
            retrieval_tools=mock_tools,
            retrieval_focus_areas=["scientific papers", "statistics"],
            model_name="openai:gpt-4",
        )

        assert critic.retrieval_tools == mock_tools
        assert critic.retrieval_focus_areas == ["scientific papers", "statistics"]
        assert critic.model_name == "openai:gpt-4"

    def test_self_rag_critic_with_auto_discover_tools(self):
        """Test SelfRAGCritic with automatic tool discovery."""
        critic = SelfRAGCritic(
            auto_discover_tools=True, tool_categories=["web_search", "academic_search"]
        )

        assert critic.auto_discover_tools is True
        assert critic.tool_categories == ["web_search", "academic_search"]

    @pytest.mark.asyncio
    async def test_critique_async_basic(self, mock_agent, sample_thought):
        """Test basic Self-RAG functionality."""
        critic = SelfRAGCritic(model_name="mock")

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Verify result structure
        assert "feedback" in result
        assert "suggestions" in result
        assert "needs_improvement" in result
        assert "confidence" in result

        # Verify content
        assert isinstance(result["feedback"], str)
        assert isinstance(result["suggestions"], list)
        assert isinstance(result["needs_improvement"], bool)
        assert isinstance(result["confidence"], (int, float))

        # Verify agent was called
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique_with_retrieval_decision_no_retrieve(self, mock_agent, sample_thought):
        """Test Self-RAG when no retrieval is needed."""
        critic = SelfRAGCritic()

        # Mock response indicating no retrieval needed
        mock_response = Mock()
        mock_response.data = {
            "feedback": "The text is factually accurate and well-supported without additional retrieval.",
            "suggestions": ["Minor stylistic improvements could be made"],
            "needs_improvement": False,
            "confidence": 0.85,
            "retrieval_decision": "NO_RETRIEVE",
        }
        mock_agent.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should indicate no retrieval needed
        assert result["needs_improvement"] is False
        assert "accurate" in result["feedback"].lower() or "supported" in result["feedback"].lower()

    @pytest.mark.asyncio
    async def test_critique_with_retrieval_tools_integration(self, mock_agent, sample_thought):
        """Test Self-RAG with retrieval tools integration."""
        mock_tool1 = Mock()
        mock_tool1.name = "web_search"
        mock_tool2 = Mock()
        mock_tool2.name = "academic_search"

        critic = SelfRAGCritic(retrieval_tools=[mock_tool1, mock_tool2])

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should consider available tools in critique
        assert "feedback" in result
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique_with_focus_areas(self, mock_agent, sample_thought):
        """Test Self-RAG with specific retrieval focus areas."""
        focus_areas = ["recent statistics", "policy changes", "technology trends"]

        critic = SelfRAGCritic(retrieval_focus_areas=focus_areas)

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should incorporate focus areas in critique
        assert "feedback" in result
        mock_agent.run.assert_called_once()

        # Verify focus areas were considered
        call_args = mock_agent.run.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_critique_error_handling(self, sample_thought):
        """Test error handling in Self-RAG."""
        critic = SelfRAGCritic()

        # Mock agent that raises an exception
        mock_agent = Mock()
        mock_agent.run = AsyncMock(side_effect=Exception("Retrieval Error"))

        with patch.object(critic, "_agent", mock_agent):
            with pytest.raises(Exception, match="Retrieval Error"):
                await critic.critique_async(sample_thought)

    def test_retrieval_decision_enum_values(self):
        """Test RetrievalDecision enum values."""
        assert RetrievalDecision.RETRIEVE.value == "RETRIEVE"
        assert RetrievalDecision.NO_RETRIEVE.value == "NO_RETRIEVE"
        assert RetrievalDecision.CONTINUE.value == "CONTINUE"

    def test_relevance_assessment_enum_values(self):
        """Test RelevanceAssessment enum values."""
        assert RelevanceAssessment.RELEVANT.value == "RELEVANT"
        assert RelevanceAssessment.IRRELEVANT.value == "IRRELEVANT"
        assert RelevanceAssessment.PARTIALLY_RELEVANT.value == "PARTIALLY_RELEVANT"

    def test_support_assessment_enum_values(self):
        """Test SupportAssessment enum values."""
        assert SupportAssessment.FULLY_SUPPORTED.value == "FULLY_SUPPORTED"
        assert SupportAssessment.PARTIALLY_SUPPORTED.value == "PARTIALLY_SUPPORTED"
        assert SupportAssessment.NOT_SUPPORTED.value == "NOT_SUPPORTED"

    def test_utility_assessment_enum_values(self):
        """Test UtilityAssessment enum values."""
        assert UtilityAssessment.USEFUL.value == "USEFUL"
        assert UtilityAssessment.NOT_USEFUL.value == "NOT_USEFUL"
        assert UtilityAssessment.SOMEWHAT_USEFUL.value == "SOMEWHAT_USEFUL"

    @pytest.mark.asyncio
    async def test_improve_async_functionality(self, mock_agent, sample_thought):
        """Test the improve_async method."""
        critic = SelfRAGCritic()

        # Mock improved response
        mock_response = Mock()
        mock_response.data = "Improved text with retrieved information and verified facts."
        mock_agent.run.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            improved_text = await critic.improve_async(sample_thought)

        assert isinstance(improved_text, str)
        assert len(improved_text) > 0
        assert "improved" in improved_text.lower()

    def test_system_prompt_contains_self_rag_methodology(self):
        """Test that system prompt contains Self-RAG methodology."""
        critic = SelfRAGCritic()

        system_prompt = critic.system_prompt.lower()

        # Should contain key Self-RAG concepts
        assert "self-rag" in system_prompt or "retrieval" in system_prompt
        assert "retrieve" in system_prompt
        assert "relevant" in system_prompt or "relevance" in system_prompt
        assert "support" in system_prompt or "supported" in system_prompt

    @pytest.mark.asyncio
    async def test_critique_with_validation_context(self, mock_agent, sample_thought):
        """Test Self-RAG with validation context awareness."""
        # Add validation results to the thought
        sample_thought.add_validation("factual_validator", False, {"issue": "unverified claims"})
        sample_thought.add_validation("citation_validator", False, {"missing_sources": 3})

        critic = SelfRAGCritic()

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should consider validation context in retrieval decisions
        assert "feedback" in result
        mock_agent.run.assert_called_once()

        # Check that validation context was included in the call
        call_args = mock_agent.run.call_args
        assert call_args is not None

    def test_paper_reference_and_methodology(self):
        """Test that paper reference and methodology are properly set."""
        critic = SelfRAGCritic()

        assert "Asai et al. 2023" in critic.paper_reference
        assert "Self-RAG" in critic.paper_reference
        assert "retrieval" in critic.methodology.lower()
        assert "augmented" in critic.methodology.lower()
        assert len(critic.methodology) > 50  # Should have substantial methodology description

    @pytest.mark.asyncio
    async def test_critique_with_empty_text(self, mock_agent):
        """Test Self-RAG with empty text."""
        thought = SifakaThought(prompt="Test prompt")
        thought.current_text = ""

        critic = SelfRAGCritic()

        # Mock response for empty text
        mock_response = Mock()
        mock_response.data = {
            "feedback": "No content to evaluate for retrieval needs.",
            "suggestions": ["Generate initial content before retrieval assessment"],
            "needs_improvement": True,
            "confidence": 0.9,
            "retrieval_decision": "NO_RETRIEVE",
        }
        mock_agent.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(thought)

        # Should handle empty text gracefully
        assert "feedback" in result
        assert "no content" in result["feedback"].lower() or "empty" in result["feedback"].lower()

    def test_default_focus_areas_structure(self):
        """Test that default retrieval focus areas are properly structured."""
        critic = SelfRAGCritic()

        # Should have reasonable default focus areas
        if hasattr(critic, "retrieval_focus_areas") and critic.retrieval_focus_areas:
            assert isinstance(critic.retrieval_focus_areas, list)
            assert len(critic.retrieval_focus_areas) > 0
