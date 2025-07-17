"""Tests for Prompt critic."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.config import Config
from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.critics.prompt import (
    PromptCritic,
    PromptResponse,
    create_academic_critic,
    create_business_critic,
    create_creative_critic,
)


class TestPromptResponse:
    """Test the PromptResponse model."""

    def test_creation_minimal(self):
        """Test creating response with minimal fields."""
        response = PromptResponse(
            feedback="Text needs improvement in clarity",
            suggestions=["Simplify language"],
            needs_improvement=True,
        )
        assert response.feedback == "Text needs improvement in clarity"
        assert len(response.suggestions) == 1
        assert response.confidence == 0.7  # default

    def test_creation_full(self):
        """Test creating response with all fields."""
        criteria1 = {
            "criterion": "Clarity",
            "assessment": "Good clarity overall",
            "score": 0.8,
            "improvements": ["Minor simplifications"],
        }
        criteria2 = {
            "criterion": "Structure",
            "assessment": "Well-organized",
            "score": 0.9,
            "improvements": [],
        }

        response = PromptResponse(
            feedback="Strong text with minor improvements needed",
            suggestions=["Simplify introduction", "Add conclusion"],
            needs_improvement=True,
            confidence=0.85,
            metadata={
                "custom_criteria_results": [criteria1, criteria2],
                "overall_score": 0.85,
                "key_findings": ["Good structure", "Clear arguments"],
                "evaluation_type": "academic",
            },
        )

        assert len(response.metadata["custom_criteria_results"]) == 2
        assert response.metadata["overall_score"] == 0.85
        assert len(response.metadata["key_findings"]) == 2
        assert response.metadata["evaluation_type"] == "academic"


class TestPromptCritic:
    """Test the PromptCritic class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Final text")

    def test_initialization_default(self):
        """Test default initialization."""
        critic = PromptCritic()
        assert critic.name == "prompt"
        assert critic.model == "gpt-3.5-turbo"  # Default from Config
        assert critic.temperature == 0.7
        assert "Evaluate this text" in critic.custom_prompt

    def test_initialization_with_custom_prompt(self):
        """Test initialization with custom prompt."""
        custom_prompt = "Check for technical accuracy and completeness"
        critic = PromptCritic(custom_prompt=custom_prompt)
        assert critic.custom_prompt == custom_prompt

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Config()
        config.temperature = 0.5
        critic = PromptCritic(config=config)
        assert critic.config == config

    def test_initialization_with_params(self):
        """Test initialization with all parameters."""
        critic = PromptCritic(
            custom_prompt="Custom evaluation prompt",
            model="gpt-4",
            temperature=0.8,
            api_key="test-key",
        )
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.8
        assert critic.custom_prompt == "Custom evaluation prompt"

    def test_get_response_type(self):
        """Test that critic uses PromptResponse."""
        critic = PromptCritic()
        assert critic._get_response_type() == PromptResponse

    @pytest.mark.asyncio
    async def test_create_messages(self, sample_result):
        """Test message creation with custom prompt."""
        custom_prompt = "Evaluate for business impact"
        critic = PromptCritic(custom_prompt=custom_prompt)
        messages = await critic._create_messages("Test business text", sample_result)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "customizable text critic" in messages[0]["content"]

        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert custom_prompt in user_content
        assert "Test business text" in user_content
        assert "specific, actionable feedback" in user_content

    @pytest.mark.asyncio
    async def test_create_messages_with_context(self, sample_result):
        """Test message creation with previous context."""
        # Add a previous critique
        sample_result.critiques.append(
            CritiqueResult(
                critic="prompt",
                feedback="Previous custom feedback",
                suggestions=["Previous suggestion"],
            )
        )

        critic = PromptCritic()
        messages = await critic._create_messages("Test text", sample_result)

        user_content = messages[1]["content"]
        assert "Previous feedback:" in user_content
        assert "Previous custom feedback" in user_content

    @pytest.mark.asyncio
    async def test_critique_success(self, sample_result):
        """Test successful critique with custom criteria."""
        critic = PromptCritic(custom_prompt="Evaluate for clarity and impact")

        # Mock the LLM response
        mock_response = PromptResponse(
            feedback="Text has good clarity but lacks emotional impact",
            suggestions=["Add compelling examples", "Use stronger verbs"],
            needs_improvement=True,
            confidence=0.8,
            metadata={
                "custom_criteria_results": [
                    {
                        "criterion": "Clarity",
                        "assessment": "Clear and well-structured",
                        "score": 0.85,
                        "improvements": [],
                    },
                    {
                        "criterion": "Impact",
                        "assessment": "Lacks emotional resonance",
                        "score": 0.6,
                        "improvements": ["Add personal stories", "Use vivid language"],
                    },
                ],
                "overall_score": 0.725,
                "key_findings": ["Strong clarity", "Weak emotional impact"],
                "prompt_type": "custom",
            },
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response  # For backward compatibility
        mock_agent_result.data = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Test text", sample_result)

            assert result.critic == "prompt"
            assert result.feedback == mock_response.feedback
            assert len(result.suggestions) == 2
            assert result.needs_improvement is True
            assert result.confidence == 0.8
            assert "custom_criteria_results" in result.metadata
            assert len(result.metadata["custom_criteria_results"]) == 2
            assert result.metadata["overall_score"] == 0.725

    @pytest.mark.asyncio
    async def test_critique_no_improvement(self, sample_result):
        """Test critique when no improvement needed."""
        critic = PromptCritic(custom_prompt="Check professional tone")

        mock_response = PromptResponse(
            feedback="Excellent professional tone throughout",
            suggestions=[],
            needs_improvement=False,
            confidence=0.95,
            metadata={
                "custom_criteria_results": [
                    {
                        "criterion": "Professional Tone",
                        "assessment": "Consistently professional",
                        "score": 0.95,
                        "improvements": [],
                    }
                ],
                "overall_score": 0.95,
                "key_findings": ["Perfect professional tone"],
                "tone_analysis": "formal",
            },
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response  # For backward compatibility
        mock_agent_result.data = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Professional text", sample_result)

            assert result.needs_improvement is False
            assert len(result.suggestions) == 0
            assert result.confidence == 0.95
            assert result.metadata["overall_score"] == 0.95

    def test_create_academic_critic(self):
        """Test creating an academic critic."""
        critic = create_academic_critic()
        assert isinstance(critic, PromptCritic)
        assert "academic paper" in critic.custom_prompt.lower()
        assert "citations" in critic.custom_prompt
        assert "argumentation" in critic.custom_prompt

    def test_create_academic_critic_with_params(self):
        """Test creating an academic critic with custom params."""
        critic = create_academic_critic(model="gpt-4", temperature=0.5)
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.5

    def test_create_business_critic(self):
        """Test creating a business critic."""
        critic = create_business_critic()
        assert isinstance(critic, PromptCritic)
        assert "business document" in critic.custom_prompt.lower()
        assert "Professional tone" in critic.custom_prompt
        assert "Value proposition" in critic.custom_prompt

    def test_create_creative_critic(self):
        """Test creating a creative writing critic."""
        critic = create_creative_critic()
        assert isinstance(critic, PromptCritic)
        assert "creative writing" in critic.custom_prompt.lower()
        assert "Narrative flow" in critic.custom_prompt
        assert "Character development" in critic.custom_prompt

    def test_provider_configuration(self):
        """Test provider configuration."""
        from sifaka.core.llm_client import Provider

        critic = PromptCritic(provider=Provider.ANTHROPIC)
        assert critic.provider == Provider.ANTHROPIC

        critic = PromptCritic(provider="openai")
        assert critic.provider == Provider.OPENAI

    @pytest.mark.asyncio
    async def test_multiple_criteria_evaluation(self, sample_result):
        """Test evaluation with multiple custom criteria."""
        prompt = """Evaluate this text on:
        1. Technical accuracy
        2. Readability
        3. Completeness
        4. Practical value"""

        critic = PromptCritic(custom_prompt=prompt)

        mock_response = PromptResponse(
            feedback="Mixed results across criteria",
            suggestions=["Improve technical details", "Simplify language"],
            needs_improvement=True,
            confidence=0.75,
            metadata={
                "custom_criteria_results": [
                    {
                        "criterion": "Technical accuracy",
                        "assessment": "Some inaccuracies found",
                        "score": 0.6,
                        "improvements": ["Verify facts", "Update statistics"],
                    },
                    {
                        "criterion": "Readability",
                        "assessment": "Too complex for target audience",
                        "score": 0.5,
                        "improvements": ["Simplify jargon", "Shorter sentences"],
                    },
                    {
                        "criterion": "Completeness",
                        "assessment": "All topics covered",
                        "score": 0.9,
                        "improvements": [],
                    },
                    {
                        "criterion": "Practical value",
                        "assessment": "Highly practical",
                        "score": 0.85,
                        "improvements": ["Add more examples"],
                    },
                ],
                "overall_score": 0.7125,  # Average of scores
                "key_findings": [
                    "Technical inaccuracies",
                    "Poor readability",
                    "Complete coverage",
                    "Good practical value",
                ],
            },
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response  # For backward compatibility
        mock_agent_result.data = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Technical document", sample_result)

            assert len(result.metadata["custom_criteria_results"]) == 4
            assert len(result.metadata["key_findings"]) == 4
            # Verify the worst-scoring criterion
            criteria_scores = [
                c["score"] for c in result.metadata["custom_criteria_results"]
            ]
            assert min(criteria_scores) == 0.5  # Readability was worst
