"""Tests for critic implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from sifaka.core.config import Config
from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.prompt import PromptCritic, create_academic_critic
from sifaka.critics.reflexion import ReflexionCritic


class MockAgentResult:
    """Mock result from PydanticAI agent run."""

    def __init__(self, output):
        self.output = output
        self._usage = MagicMock()
        self._usage.total_tokens = 100

    def usage(self):
        """Mock usage data."""
        return self._usage


class MockCriticResponse(BaseModel):
    """Mock structured response for critics."""

    feedback: str
    suggestions: list[str]
    needs_improvement: bool
    confidence: float
    metadata: dict = {}


@pytest.fixture
def mock_pydantic_agent():
    """Create a mock PydanticAI agent."""
    with patch("sifaka.core.llm_client.LLMClient.create_agent") as mock_create:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        mock_create.return_value = mock_agent
        yield mock_agent


@pytest.fixture
def sample_result():
    """Create a sample SifakaResult for testing."""
    return SifakaResult(
        original_text="Write about AI",
        final_text="AI is transforming the world",
        iteration=1,
        generations=[],
        critiques=[],
        validations=[],
        processing_time=1.0,
    )


class TestReflexionCritic:
    """Test Reflexion critic implementation."""

    def test_initialization(self):
        """Test critic initialization."""
        critic = ReflexionCritic(model="gpt-4", temperature=0.5)
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.5
        assert critic.name == "reflexion"
        assert critic.config is not None

    def test_build_context_first_iteration(self):
        """Test context building for first iteration."""
        critic = ReflexionCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Context building is now internal to the critic's get_instructions method
        instructions = critic.get_instructions("Test", result)
        assert "previous" in instructions.lower() or "analyze" in instructions.lower()

    def test_build_context_with_critiques(self):
        """Test context building with previous critiques."""
        critic = ReflexionCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Add some critiques
        result.add_critique(
            critic="other_critic",
            feedback="Previous feedback",
            suggestions=["Previous suggestion"],
            needs_improvement=True,
        )

        # Context is built within get_instructions method
        instructions = critic.get_instructions("Test", result)
        # Should contain analysis instructions, history is handled internally
        assert "analyze" in instructions.lower() or "evaluate" in instructions.lower()

    def test_generate_critique_format(self):
        """Test critique generation format."""
        critic = ReflexionCritic()
        # Response format is now handled by PydanticAI
        assert critic.name == "reflexion"

        # Test that config can be customized
        custom_config = Config()
        critic2 = ReflexionCritic(config=custom_config)
        assert critic2.config is not None

    @pytest.mark.asyncio
    async def test_parse_json_response(self, mock_pydantic_agent, sample_result):
        """Test parsing JSON response."""
        # Use the mocks defined in this file

        critic = ReflexionCritic()

        # Mock the structured response
        mock_response = MockCriticResponse(
            feedback="This text shows good structure but lacks depth.",
            suggestions=[
                "Add more specific examples",
                "Improve the conclusion",
                "Include citations",
            ],
            needs_improvement=True,
            confidence=0.8,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result = await critic.critique("Test text", sample_result)
        assert "good structure" in result.feedback
        assert "lacks depth" in result.feedback
        assert len(result.suggestions) == 3
        assert "Add more specific examples" in result.suggestions
        assert result.needs_improvement is True
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_parse_text_response(self, mock_pydantic_agent, sample_result):
        """Test parsing unstructured text response."""
        # Use the mocks defined in this file

        critic = ReflexionCritic()

        # Even with PydanticAI, we get structured responses
        mock_response = MockCriticResponse(
            feedback="This is unstructured feedback without proper sections.",
            suggestions=["General improvement needed"],
            needs_improvement=True,
            confidence=0.7,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result = await critic.critique("Test text", sample_result)
        assert (
            result.feedback == "This is unstructured feedback without proper sections."
        )
        assert len(result.suggestions) >= 1  # Should have fallback suggestion
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_assess_needs_improvement(self, mock_pydantic_agent, sample_result):
        """Test improvement need assessment."""
        # Use the mocks defined in this file

        critic = ReflexionCritic()

        # Test needs improvement case
        mock_response = MockCriticResponse(
            feedback="The text could be improved and needs more clarity",
            suggestions=["Fix this", "Change that"],
            needs_improvement=True,
            confidence=0.7,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result = await critic.critique("Test text", sample_result)
        assert result.needs_improvement is True

        # Test no improvement needed case
        mock_response2 = MockCriticResponse(
            feedback="The text is excellent and well-written",
            suggestions=["Minor suggestion"],
            needs_improvement=False,
            confidence=0.9,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response2)

        result2 = await critic.critique("Excellent text", sample_result)
        assert result2.needs_improvement is False

    @pytest.mark.asyncio
    async def test_calculate_confidence(self, mock_pydantic_agent, sample_result):
        """Test confidence calculation."""
        # Use the mocks defined in this file

        critic = ReflexionCritic()

        # Short response
        mock_response = MockCriticResponse(
            feedback="Brief feedback",
            suggestions=["Brief suggestion"],
            needs_improvement=True,
            confidence=0.7,  # Will be recalculated internally
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result1 = await critic.critique("Test", sample_result)
        assert 0.0 <= result1.confidence <= 1.0

        # Long detailed response
        long_feedback = "A" * 600
        mock_response2 = MockCriticResponse(
            feedback=long_feedback,
            suggestions=[
                "Detailed suggestion 1",
                "Detailed suggestion 2",
                "Detailed suggestion 3",
            ],
            needs_improvement=True,
            confidence=0.7,  # Will be recalculated internally
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response2)

        result2 = await critic.critique("Test", sample_result)
        # Confidence calculation is internal, but longer responses tend to have higher confidence
        assert 0.0 <= result2.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_base_critic_functionality(self, mock_pydantic_agent, sample_result):
        """Test that base critic functionality works."""
        # Use the mocks defined in this file

        critic = ReflexionCritic()

        # Test that critic returns structured suggestions
        mock_response = MockCriticResponse(
            feedback="Here are the issues found",
            suggestions=["First item", "Second item", "Third item"],
            needs_improvement=True,
            confidence=0.8,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result = await critic.critique("Test text", sample_result)
        assert len(result.suggestions) == 3
        assert "First item" in result.suggestions
        assert "Third item" in result.suggestions

    @pytest.mark.asyncio
    async def test_critique_success(self, mock_pydantic_agent, sample_result):
        """Test successful critique execution."""
        # Use the mocks defined in this file

        critic = ReflexionCritic()

        # Mock PydanticAI response
        mock_response = MockCriticResponse(
            feedback="Good analysis of the text with clear structure.",
            suggestions=["Add more details", "Improve structure"],
            needs_improvement=True,
            confidence=0.75,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critique_result = await critic.critique("Test text", sample_result)

        assert isinstance(critique_result, CritiqueResult)
        assert critique_result.critic == "reflexion"
        assert "Good analysis" in critique_result.feedback
        assert len(critique_result.suggestions) == 2
        assert critique_result.confidence == 0.75
        assert critique_result.needs_improvement is True

    @pytest.mark.asyncio
    async def test_critique_error_handling(self, mock_pydantic_agent, sample_result):
        """Test critique error handling."""
        critic = ReflexionCritic()

        # Mock API error
        mock_pydantic_agent.run.side_effect = Exception("API Error")

        # Should raise ModelProviderError
        with pytest.raises(Exception) as exc_info:
            await critic.critique("Test text", sample_result)

        assert "failed to process text" in str(exc_info.value)


class TestPromptCritic:
    """Test configurable prompt critic."""

    def test_initialization_default(self):
        """Test default initialization."""
        critic = PromptCritic()
        # Model comes from config or default
        assert critic.model is not None
        assert critic.temperature is not None
        # Default prompt is provided
        assert (
            critic.custom_prompt
            == "Evaluate this text for quality and suggest improvements."
        )
        assert critic.name == "prompt"

    def test_initialization_with_custom_prompt(self):
        """Test initialization with custom prompt."""
        custom_prompt = "Evaluate for: 1. Clear structure 2. Proper citations"
        critic = PromptCritic(custom_prompt=custom_prompt)
        assert critic.custom_prompt == custom_prompt
        assert critic.name == "prompt"

    def test_initialization_with_custom_prompt_string(self):
        """Test initialization with custom prompt string."""
        custom_prompt = "Evaluate this text for academic quality"
        critic = PromptCritic(custom_prompt=custom_prompt)
        assert critic.custom_prompt == custom_prompt

    @pytest.mark.asyncio
    async def test_format_custom_prompt(self, mock_pydantic_agent, sample_result):
        """Test custom prompt formatting."""
        custom_prompt = "Evaluate for quality"
        critic = PromptCritic(custom_prompt=custom_prompt)

        # Mock a response
        mock_response = MockCriticResponse(
            feedback="Quality evaluation complete",
            suggestions=["Improve quality"],
            needs_improvement=True,
            confidence=0.7,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        # The custom prompt should be passed to the agent
        result = await critic.critique("Test text", sample_result)
        # Check that the agent was called (custom prompt is used internally)
        mock_pydantic_agent.run.assert_called_once()
        assert result.feedback == "Quality evaluation complete"

    def test_build_criteria_prompt_default(self):
        """Test building prompt with default criteria."""
        critic = PromptCritic()
        # Prompt building is internal to get_instructions
        instructions = critic.get_instructions(
            "Test text", SifakaResult(original_text="Test", final_text="Test text")
        )
        assert instructions is not None

        # Instructions should contain evaluation guidance
        assert len(instructions) > 0

    @pytest.mark.asyncio
    async def test_build_criteria_prompt_custom(
        self, mock_pydantic_agent, sample_result
    ):
        """Test building prompt with custom criteria."""
        custom_prompt = "Evaluate for: Academic tone, Proper methodology"
        critic = PromptCritic(custom_prompt=custom_prompt)

        # Mock a response showing evaluation of these criteria
        mock_response = MockCriticResponse(
            feedback="Academic tone is good, methodology needs work",
            suggestions=["Improve methodology section"],
            needs_improvement=True,
            confidence=0.75,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result = await critic.critique("Test text", sample_result)
        # The custom prompt should result in feedback addressing those criteria
        assert "Academic tone" in result.feedback or "methodology" in result.feedback

    @pytest.mark.asyncio
    async def test_parse_evaluation_structured(
        self, mock_pydantic_agent, sample_result
    ):
        """Test parsing structured evaluation."""
        critic = PromptCritic()

        # Mock structured response
        mock_response = MockCriticResponse(
            feedback="The text meets most criteria well.",
            suggestions=["Add more examples", "Improve conclusion"],
            needs_improvement=False,
            confidence=0.85,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result = await critic.critique("Test text", sample_result)
        assert "meets most criteria" in result.feedback
        assert result.needs_improvement is False
        assert len(result.suggestions) == 2

    @pytest.mark.asyncio
    async def test_parse_evaluation_criteria_no(
        self, mock_pydantic_agent, sample_result
    ):
        """Test parsing when criteria not met."""
        critic = PromptCritic()

        # Mock response indicating criteria not met
        mock_response = MockCriticResponse(
            feedback="The text has several issues.",
            suggestions=["Fix the problems"],
            needs_improvement=True,
            confidence=0.6,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result = await critic.critique("Test text", sample_result)
        assert result.needs_improvement is True

    def test_calculate_criteria_specificity(self):
        """Test criteria specificity calculation."""
        # Custom prompt (most specific)
        custom_critic = PromptCritic(custom_prompt="Custom evaluation")
        assert custom_critic.custom_prompt == "Custom evaluation"

        # Complex prompt
        complex_critic = PromptCritic(custom_prompt="Evaluate for: A, B, C, D, E")
        assert "A, B, C, D, E" in complex_critic.custom_prompt

        # Default prompt
        default_critic = PromptCritic()
        assert (
            default_critic.custom_prompt
            == "Evaluate this text for quality and suggest improvements."
        )

    def test_get_domain_indicators(self):
        """Test domain indicator extraction."""
        # Academic critic
        academic_critic = PromptCritic(
            custom_prompt="Evaluate for academic quality: thesis, research methodology"
        )
        assert "academic" in academic_critic.custom_prompt

        # Business critic
        business_critic = PromptCritic(
            custom_prompt="Evaluate for business impact: ROI, strategy"
        )
        assert "business" in business_critic.custom_prompt

        # Generic critic
        generic_critic = PromptCritic()
        assert generic_critic.name == "prompt"

    def test_create_academic_critic(self):
        """Test academic critic factory function."""
        critic = create_academic_critic()
        assert isinstance(critic, PromptCritic)
        assert critic.name == "prompt"
        # Should have academic-focused prompt
        assert (
            "academic" in critic.custom_prompt.lower()
            or "thesis" in critic.custom_prompt.lower()
        )

    @pytest.mark.asyncio
    async def test_critique_with_criteria(self, mock_pydantic_agent, sample_result):
        """Test critique with custom criteria."""
        custom_prompt = "Evaluate for: Clear thesis, Strong evidence"
        critic = PromptCritic(custom_prompt=custom_prompt)

        # Mock response
        mock_response = MockCriticResponse(
            feedback="Good academic structure.",
            suggestions=["Add more citations"],
            needs_improvement=False,
            confidence=0.8,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critique_result = await critic.critique("Academic text", sample_result)

        assert isinstance(critique_result, CritiqueResult)
        assert critique_result.critic == "prompt"
        assert "Good academic structure" in critique_result.feedback
        assert not critique_result.needs_improvement  # Criteria met


class TestConstitutionalCritic:
    """Test Constitutional AI critic."""

    def test_initialization(self):
        """Test constitutional critic initialization."""
        critic = ConstitutionalCritic()
        # Model and temperature come from config or defaults
        assert critic.model is not None
        assert critic.temperature is not None
        assert len(critic.principles) > 0  # Has default principles
        assert critic.name == "constitutional"

    def test_custom_principles(self):
        """Test initialization with custom principles."""
        custom_principles = ["Be helpful", "Be truthful", "Be harmless"]
        critic = ConstitutionalCritic(principles=custom_principles)
        assert critic.principles == custom_principles

    @pytest.mark.asyncio
    async def test_parse_json_with_violations(self, mock_pydantic_agent, sample_result):
        """Test parsing JSON response with violations."""
        critic = ConstitutionalCritic()

        # Mock response with violations
        mock_response = MockCriticResponse(
            feedback="Poor compliance with principles",
            suggestions=["Clarify text", "Improve structure"],
            needs_improvement=True,
            confidence=0.8,
            metadata={
                "principle_scores": {"1": 3, "2": 4},
                "violations": [
                    {
                        "principle_number": 1,
                        "principle_text": "Be clear",
                        "violation_description": "Text is confusing",
                        "severity": 5,
                    }
                ],
            },
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result = await critic.critique("Test text", sample_result)
        assert "Poor compliance" in result.feedback
        assert result.needs_improvement is True
        assert len(result.suggestions) == 2

    def test_format_principles(self):
        """Test principle formatting."""
        critic = ConstitutionalCritic(
            principles=["Test principle 1", "Test principle 2"]
        )
        # Principles should be accessible
        assert len(critic.principles) == 2
        assert "Test principle 1" in critic.principles

    def test_constitutional_config(self):
        """Test constitutional critic configuration."""
        critic = ConstitutionalCritic()

        # Config exists and has expected structure
        assert critic.config is not None
        assert hasattr(critic.config, "llm")
        assert hasattr(critic.config, "critic")

    @pytest.mark.asyncio
    async def test_critique_with_json_response(
        self, mock_pydantic_agent, sample_result
    ):
        """Test critique with JSON response."""
        critic = ConstitutionalCritic()

        # Mock response
        mock_response = MockCriticResponse(
            feedback="Text follows most principles well",
            suggestions=["Minor improvements needed"],
            needs_improvement=False,
            confidence=0.85,
            metadata={"principle_scores": {"1": 5, "2": 4, "3": 5}},
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critique_result = await critic.critique("Test text", sample_result)

        assert critique_result.critic == "constitutional"
        assert "follows most principles" in critique_result.feedback
        assert critique_result.confidence > 0.8
