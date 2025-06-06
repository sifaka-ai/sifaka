"""Comprehensive unit tests for Constitutional critic.

This module tests the ConstitutionalCritic implementation:
- Constitutional AI critique methodology
- Principle-based feedback generation
- Integration with PydanticAI agents
- Error handling and edge cases

Tests cover:
- Basic critique functionality
- Constitutional principle application
- Feedback quality and structure
- Performance characteristics
- Mock-based testing without external API calls
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.core.thought import SifakaThought


class TestConstitutionalCritic:
    """Test the ConstitutionalCritic implementation."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock PydanticAI agent for testing."""
        agent = AsyncMock()
        
        # Mock successful response
        mock_result = Mock()
        mock_result.data = {
            "feedback": "The text lacks balanced perspective and could benefit from constitutional principles",
            "suggestions": [
                "Include multiple viewpoints on the topic",
                "Ensure respect for individual rights",
                "Add evidence-based reasoning"
            ],
            "needs_improvement": True,
            "confidence": 0.85,
            "reasoning": "The text presents only one side of the argument without considering alternative perspectives"
        }
        agent.run.return_value = mock_result
        
        return agent

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        thought = SifakaThought(prompt="Discuss the benefits of renewable energy")
        thought.current_text = "Renewable energy is the only solution to climate change. Everyone should switch immediately."
        thought.add_generation(thought.current_text, "gpt-4", None)
        return thought

    def test_critic_initialization_default(self):
        """Test critic initialization with default parameters."""
        critic = ConstitutionalCritic()
        
        assert critic.name == "constitutional"
        assert critic.model_name is not None
        assert hasattr(critic, 'principles')
        assert len(critic.principles) > 0

    def test_critic_initialization_custom(self):
        """Test critic initialization with custom parameters."""
        custom_principles = [
            "Respect individual autonomy",
            "Promote fairness and equality",
            "Ensure transparency"
        ]
        
        critic = ConstitutionalCritic(
            model_name="gpt-3.5-turbo",
            principles=custom_principles,
            name="custom-constitutional"
        )
        
        assert critic.name == "custom-constitutional"
        assert critic.model_name == "gpt-3.5-turbo"
        assert critic.principles == custom_principles

    def test_critic_initialization_invalid_params(self):
        """Test critic initialization with invalid parameters."""
        # Empty principles
        with pytest.raises(ValueError):
            ConstitutionalCritic(principles=[])
        
        # Invalid model name
        with pytest.raises(ValueError):
            ConstitutionalCritic(model_name="")

    @pytest.mark.asyncio
    async def test_critique_basic_functionality(self, mock_agent, sample_thought):
        """Test basic critique functionality."""
        critic = ConstitutionalCritic(model_name="test-model")
        
        # Mock the agent
        with patch.object(critic, '_agent', mock_agent):
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
    async def test_critique_with_no_text(self, mock_agent):
        """Test critique when thought has no current text."""
        thought = SifakaThought(prompt="Test prompt")
        # No current_text set
        
        critic = ConstitutionalCritic()
        
        with patch.object(critic, '_agent', mock_agent):
            result = await critic.critique_async(thought)
        
        # Should handle gracefully
        assert "feedback" in result
        # Might indicate no text to critique
        assert "no text" in result["feedback"].lower() or result["needs_improvement"] is False

    @pytest.mark.asyncio
    async def test_critique_with_empty_text(self, mock_agent):
        """Test critique with empty text."""
        thought = SifakaThought(prompt="Test prompt")
        thought.current_text = ""
        
        critic = ConstitutionalCritic()
        
        with patch.object(critic, '_agent', mock_agent):
            result = await critic.critique_async(thought)
        
        # Should handle empty text gracefully
        assert "feedback" in result
        assert result["needs_improvement"] is False or "empty" in result["feedback"].lower()

    @pytest.mark.asyncio
    async def test_critique_with_good_text(self, mock_agent, sample_thought):
        """Test critique with text that follows constitutional principles."""
        # Modify the thought to have balanced text
        sample_thought.current_text = """
        Renewable energy offers significant benefits for addressing climate change, including reduced carbon emissions 
        and energy independence. However, the transition involves challenges such as infrastructure costs and 
        intermittency issues. A balanced approach considering both environmental benefits and economic factors 
        would be most effective for sustainable energy policy.
        """
        
        # Mock response for good text
        mock_result = Mock()
        mock_result.data = {
            "feedback": "The text presents a balanced perspective with multiple viewpoints",
            "suggestions": [],
            "needs_improvement": False,
            "confidence": 0.9,
            "reasoning": "The text considers both benefits and challenges"
        }
        mock_agent.run.return_value = mock_result
        
        critic = ConstitutionalCritic()
        
        with patch.object(critic, '_agent', mock_agent):
            result = await critic.critique_async(sample_thought)
        
        assert result["needs_improvement"] is False
        assert len(result["suggestions"]) == 0
        assert result["confidence"] > 0.8

    @pytest.mark.asyncio
    async def test_critique_error_handling(self, sample_thought):
        """Test critique error handling when agent fails."""
        critic = ConstitutionalCritic()
        
        # Mock agent that raises an exception
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("API Error")
        
        with patch.object(critic, '_agent', mock_agent):
            result = await critic.critique_async(sample_thought)
        
        # Should handle error gracefully
        assert "feedback" in result
        assert "error" in result["feedback"].lower()
        assert result["needs_improvement"] is False  # Conservative default
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_critique_with_conversation_history(self, mock_agent):
        """Test critique considering conversation history."""
        thought = SifakaThought(prompt="Continue the discussion about renewable energy")
        thought.current_text = "As I mentioned before, solar is the best option."
        
        # Add previous generation to simulate conversation
        thought.add_generation("Solar energy has many advantages.", "gpt-4", None)
        thought.iteration = 1
        thought.add_generation(thought.current_text, "gpt-4", None)
        
        critic = ConstitutionalCritic()
        
        with patch.object(critic, '_agent', mock_agent):
            result = await critic.critique_async(thought)
        
        # Should consider conversation context
        assert "feedback" in result
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_improve_functionality(self, mock_agent, sample_thought):
        """Test the improve_async method."""
        # Mock improved text response
        mock_result = Mock()
        mock_result.data = """
        Renewable energy offers significant benefits for addressing climate change, including reduced greenhouse gas 
        emissions and increased energy security. However, the transition to renewable energy also presents challenges 
        such as initial infrastructure costs, grid stability concerns, and the need for energy storage solutions. 
        A comprehensive approach that weighs both the environmental benefits and practical considerations would be 
        most effective for developing sustainable energy policies.
        """
        mock_agent.run.return_value = mock_result
        
        critic = ConstitutionalCritic()
        
        with patch.object(critic, '_agent', mock_agent):
            improved_text = await critic.improve_async(sample_thought)
        
        assert isinstance(improved_text, str)
        assert len(improved_text) > len(sample_thought.current_text)
        assert "however" in improved_text.lower() or "although" in improved_text.lower()  # Indicates balance

    @pytest.mark.asyncio
    async def test_improve_error_handling(self, sample_thought):
        """Test improve method error handling."""
        critic = ConstitutionalCritic()
        
        # Mock agent that raises an exception
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("API Error")
        
        with patch.object(critic, '_agent', mock_agent):
            improved_text = await critic.improve_async(sample_thought)
        
        # Should return original text or error message
        assert isinstance(improved_text, str)
        assert len(improved_text) > 0

    def test_principle_application(self):
        """Test that constitutional principles are properly applied."""
        critic = ConstitutionalCritic()
        
        # Verify default principles are reasonable
        assert len(critic.principles) > 0
        
        # Check that principles cover key constitutional concepts
        principles_text = " ".join(critic.principles).lower()
        constitutional_concepts = ["fairness", "rights", "equality", "freedom", "justice", "respect"]
        
        # At least some constitutional concepts should be present
        found_concepts = [concept for concept in constitutional_concepts if concept in principles_text]
        assert len(found_concepts) > 0

    @pytest.mark.asyncio
    async def test_critique_consistency(self, mock_agent, sample_thought):
        """Test that critique results are consistent for the same input."""
        critic = ConstitutionalCritic()
        
        with patch.object(critic, '_agent', mock_agent):
            result1 = await critic.critique_async(sample_thought)
            result2 = await critic.critique_async(sample_thought)
        
        # Results should be consistent (same mock response)
        assert result1["feedback"] == result2["feedback"]
        assert result1["needs_improvement"] == result2["needs_improvement"]
        assert result1["confidence"] == result2["confidence"]

    @pytest.mark.asyncio
    async def test_critique_with_metadata(self, mock_agent, sample_thought):
        """Test that critique includes proper metadata."""
        critic = ConstitutionalCritic(model_name="gpt-4")
        
        with patch.object(critic, '_agent', mock_agent):
            result = await critic.critique_async(sample_thought)
        
        # Should include metadata about the critique process
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        
        # May include additional metadata
        if "reasoning" in result:
            assert isinstance(result["reasoning"], str)
            assert len(result["reasoning"]) > 0

    def test_critic_string_representation(self):
        """Test string representation of critic."""
        critic = ConstitutionalCritic(model_name="gpt-4")
        
        str_repr = str(critic)
        assert "ConstitutionalCritic" in str_repr
        assert "gpt-4" in str_repr

    def test_critic_equality(self):
        """Test critic equality comparison."""
        critic1 = ConstitutionalCritic(model_name="gpt-4")
        critic2 = ConstitutionalCritic(model_name="gpt-4")
        critic3 = ConstitutionalCritic(model_name="gpt-3.5-turbo")
        
        # Same configuration should be equal
        assert critic1.name == critic2.name
        assert critic1.model_name == critic2.model_name
        
        # Different configuration should not be equal
        assert critic1.model_name != critic3.model_name

    @pytest.mark.asyncio
    async def test_critique_performance(self, mock_agent, sample_thought):
        """Test critique performance characteristics."""
        critic = ConstitutionalCritic()
        
        import time
        start_time = time.time()
        
        with patch.object(critic, '_agent', mock_agent):
            result = await critic.critique_async(sample_thought)
        
        end_time = time.time()
        
        # Should complete quickly (mock should be fast)
        assert (end_time - start_time) < 1.0
        assert "feedback" in result

    @pytest.mark.asyncio
    async def test_critique_with_complex_thought(self, mock_agent):
        """Test critique with a complex thought containing multiple elements."""
        thought = SifakaThought(prompt="Complex analysis prompt")
        thought.current_text = "Complex text with multiple aspects to analyze."
        
        # Add validation results
        thought.add_validation("length-validator", True, {"word_count": 50})
        thought.add_validation("content-validator", False, {"issue": "needs examples"})
        
        # Add previous critiques
        thought.add_critique(
            "reflexion-critic",
            "Previous feedback",
            ["Previous suggestion"],
            needs_improvement=True
        )
        
        critic = ConstitutionalCritic()
        
        with patch.object(critic, '_agent', mock_agent):
            result = await critic.critique_async(thought)
        
        # Should handle complex thought structure
        assert "feedback" in result
        assert isinstance(result["suggestions"], list)
        
        # Agent should have been called with appropriate context
        mock_agent.run.assert_called_once()
