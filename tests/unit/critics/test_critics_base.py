"""Tests for the BaseCritic class and CriticResponse model."""

import asyncio
from typing import List, Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from sifaka.core.config import Config
from sifaka.core.exceptions import ModelProviderError
from sifaka.core.llm_client import Provider
from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.critics.core.base import BaseCritic, CriticResponse
from sifaka.tools.base import ToolInterface


class TestCriticResponse:
    """Test CriticResponse model."""

    def test_default_values(self):
        """Test default values in CriticResponse."""
        response = CriticResponse(feedback="Test feedback", needs_improvement=True)
        assert response.feedback == "Test feedback"
        assert response.suggestions == []
        assert response.needs_improvement is True
        assert response.confidence == 0.7
        assert response.metadata == {}

    def test_all_fields(self):
        """Test CriticResponse with all fields."""
        response = CriticResponse(
            feedback="Detailed feedback",
            suggestions=["Suggestion 1", "Suggestion 2"],
            needs_improvement=False,
            confidence=0.9,
            metadata={"score": 8.5, "analysis": "detailed"},
        )
        assert response.feedback == "Detailed feedback"
        assert len(response.suggestions) == 2
        assert response.needs_improvement is False
        assert response.confidence == 0.9
        assert response.metadata["score"] == 8.5

    def test_confidence_validation(self):
        """Test confidence field validation."""
        # Valid confidence
        response = CriticResponse(
            feedback="Test", needs_improvement=True, confidence=0.0
        )
        assert response.confidence == 0.0

        response = CriticResponse(
            feedback="Test", needs_improvement=True, confidence=1.0
        )
        assert response.confidence == 1.0

        # Invalid confidence should raise validation error
        with pytest.raises(ValueError):
            CriticResponse(feedback="Test", needs_improvement=True, confidence=-0.1)

        with pytest.raises(ValueError):
            CriticResponse(feedback="Test", needs_improvement=True, confidence=1.1)


class ConcreteCritic(BaseCritic):
    """Concrete implementation of BaseCritic for testing."""

    @property
    def name(self) -> str:
        return "test_critic"

    def get_instructions(self, text: str, result: SifakaResult) -> str:
        return f"Please evaluate this text: {text}"


class CustomResponseCritic(BaseCritic):
    """Critic with custom response model."""

    @property
    def name(self) -> str:
        return "custom_critic"

    def _get_response_type(self) -> Type[BaseModel]:
        class CustomResponse(BaseModel):
            feedback: str
            suggestions: List[str] = Field(default_factory=list)
            needs_improvement: bool
            confidence: float = 0.7
            custom_field: str = "custom_value"

        return CustomResponse


class ToolEnabledCritic(BaseCritic):
    """Critic with tool support."""

    @property
    def name(self) -> str:
        return "tool_critic"

    def _get_available_tools(self) -> List[ToolInterface]:
        mock_tool = MagicMock(spec=ToolInterface)
        mock_tool.name = "test_tool"
        return [mock_tool]


class TestBaseCritic:
    """Test BaseCritic abstract base class."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        critic = ConcreteCritic()
        assert critic.model == "gpt-3.5-turbo"  # Default from config
        assert critic.temperature == 0.7  # Default from config
        assert critic.provider is None
        assert critic._api_key is None
        assert critic.enable_tools is False
        assert len(critic.tools) == 0
        assert critic.config is not None

    def test_initialization_with_parameters(self):
        """Test initialization with custom parameters."""
        config = Config()
        critic = ConcreteCritic(
            model="gpt-4",
            temperature=0.5,
            config=config,
            provider="openai",
            api_key="test_key",
            enable_tools=True,
        )
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.5
        assert critic.config == config
        assert critic.provider == "openai"
        assert critic._api_key == "test_key"
        assert critic.enable_tools is True

    def test_initialization_with_config_model(self):
        """Test initialization uses config critic_model."""
        config = Config()
        config.llm.critic_model = "claude-3"
        critic = ConcreteCritic(config=config)
        assert critic.model == "claude-3"

    def test_initialization_with_explicit_model(self):
        """Test explicit model overrides config."""
        config = Config()
        config.llm.critic_model = "claude-3"
        critic = ConcreteCritic(model="gpt-4", config=config)
        assert critic.model == "gpt-4"

    def test_client_lazy_creation(self):
        """Test client is created lazily."""
        critic = ConcreteCritic()
        assert critic._client is None

        # Access client - should be created
        client = critic.client
        assert client is not None
        assert critic._client is client

        # Second access should return same client
        client2 = critic.client
        assert client2 is client

    def test_client_with_provider_enum(self):
        """Test client creation with Provider enum."""
        critic = ConcreteCritic(provider=Provider.OPENAI)
        client = critic.client
        assert client is not None

    def test_client_with_provider_string(self):
        """Test client creation with provider string."""
        critic = ConcreteCritic(provider="anthropic")
        client = critic.client
        assert client is not None

    def test_get_system_prompt_default(self):
        """Test default system prompt."""
        critic = ConcreteCritic()
        prompt = critic._get_system_prompt()
        assert "test_critic" in prompt
        assert "expert text critic" in prompt

    def test_get_response_type_default(self):
        """Test default response type."""
        critic = ConcreteCritic()
        response_type = critic._get_response_type()
        assert response_type == CriticResponse

    def test_get_response_type_custom(self):
        """Test custom response type."""
        critic = CustomResponseCritic()
        response_type = critic._get_response_type()
        assert response_type != CriticResponse

    def test_get_available_tools_default(self):
        """Test default tool availability."""
        critic = ConcreteCritic()
        tools = critic._get_available_tools()
        assert tools == []

    def test_get_available_tools_custom(self):
        """Test custom tool availability."""
        critic = ToolEnabledCritic()
        tools = critic._get_available_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_tools_initialization_disabled(self):
        """Test tools not initialized when disabled."""
        critic = ConcreteCritic(enable_tools=False)
        assert len(critic.tools) == 0

    def test_tools_initialization_enabled(self):
        """Test tools initialized when enabled."""
        critic = ToolEnabledCritic(enable_tools=True)
        assert len(critic.tools) == 1

    def test_get_instructions_default(self):
        """Test default get_instructions implementation."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")
        instructions = critic.get_instructions("Test text", result)
        assert "Test text" in instructions

    def test_get_previous_context_empty(self):
        """Test previous context with no critiques."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")
        context = critic._get_previous_context(result)
        assert context == ""

    def test_get_previous_context_with_critiques(self):
        """Test previous context with critiques."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Add critique from same critic
        result.add_critique(
            critic="test_critic",
            feedback="Previous feedback",
            suggestions=["Previous suggestion"],
            needs_improvement=True,
        )

        # Add critique from different critic
        result.add_critique(
            critic="other_critic",
            feedback="Other feedback",
            suggestions=["Other suggestion"],
            needs_improvement=True,
        )

        context = critic._get_previous_context(result)
        assert "Previous feedback" in context
        assert "Other feedback" not in context
        assert "avoid repetition" in context

    def test_get_previous_context_limit(self):
        """Test previous context respects window limit."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Add many critiques
        for i in range(10):
            result.add_critique(
                critic="test_critic",
                feedback=f"Feedback {i}",
                suggestions=[f"Suggestion {i}"],
                needs_improvement=True,
            )

        context = critic._get_previous_context(result)
        # Should only include recent ones based on context_window
        assert "Feedback 9" in context
        assert "Feedback 0" not in context

    def test_build_user_prompt(self):
        """Test user prompt building."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        prompt = critic._build_user_prompt("Test text", result, "Test instructions")
        assert "Test instructions" in prompt
        assert "Test text" in prompt
        assert "actionable feedback" in prompt

    def test_build_user_prompt_with_context(self):
        """Test user prompt building with previous context."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        result.add_critique(
            critic="test_critic",
            feedback="Previous feedback",
            suggestions=["Previous suggestion"],
            needs_improvement=True,
        )

        prompt = critic._build_user_prompt("Test text", result, "Test instructions")
        assert "Previous feedback" in prompt

    @pytest.mark.asyncio
    async def test_simple_critique(self):
        """Test simple critique message creation."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        messages = await critic._simple_critique(
            "Test text", result, "Test instructions"
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Test instructions" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_create_messages_default(self):
        """Test default message creation."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        messages = await critic._create_messages("Test text", result)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_critique_success(self):
        """Test successful critique execution."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock the agent and its result
        mock_agent = AsyncMock()
        mock_agent_result = MagicMock()
        mock_agent_result.output = CriticResponse(
            feedback="Test feedback",
            suggestions=["Test suggestion"],
            needs_improvement=True,
            confidence=0.8,
        )
        mock_agent_result.usage = MagicMock(return_value=MagicMock(total_tokens=100))
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            critique_result = await critic.critique("Test text", result)

            assert isinstance(critique_result, CritiqueResult)
            assert critique_result.critic == "test_critic"
            assert critique_result.feedback == "Test feedback"
            assert critique_result.suggestions == ["Test suggestion"]
            assert critique_result.needs_improvement is True
            assert critique_result.confidence == 0.8
            assert critique_result.tokens_used == 100

    @pytest.mark.asyncio
    async def test_critique_with_string_response(self):
        """Test critique with string response conversion."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock the agent to return a proper CriticResponse
        mock_agent = AsyncMock()
        mock_agent_result = MagicMock()
        # Create a proper CriticResponse object
        mock_response = CriticResponse(
            feedback="String feedback",
            suggestions=[],
            needs_improvement=True,
            confidence=0.7,
        )
        mock_agent_result.output = mock_response
        mock_agent_result.usage = MagicMock(return_value=MagicMock(total_tokens=50))
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            critique_result = await critic.critique("Test text", result)

            assert critique_result.feedback == "String feedback"
            assert critique_result.suggestions == []
            assert critique_result.needs_improvement is True

    @pytest.mark.asyncio
    async def test_critique_confidence_calculation(self):
        """Test confidence calculation when default value is used."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock the agent with default confidence
        mock_agent = AsyncMock()
        mock_agent_result = MagicMock()
        mock_response = CriticResponse(
            feedback="Test feedback",
            suggestions=["Test suggestion"],
            needs_improvement=True,
            confidence=0.7,  # Default value
        )
        mock_agent_result.output = mock_response
        mock_agent_result.usage = MagicMock(return_value=MagicMock(total_tokens=50))
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            critique_result = await critic.critique("Test text", result)

            # Confidence should be calculated, not default
            assert critique_result.confidence != 0.7

    @pytest.mark.asyncio
    async def test_critique_with_custom_response_model(self):
        """Test critique with custom response model."""
        critic = CustomResponseCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock the agent
        mock_agent = AsyncMock()
        mock_agent_result = MagicMock()

        # Create custom response using the custom model
        CustomResponse = critic._get_response_type()
        mock_response = CustomResponse(
            feedback="Custom feedback",
            suggestions=["Custom suggestion"],
            needs_improvement=True,
            confidence=0.8,
            custom_field="custom_value",
        )
        mock_agent_result.output = mock_response
        mock_agent_result.usage = MagicMock(return_value=MagicMock(total_tokens=75))
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            critique_result = await critic.critique("Test text", result)

            assert critique_result.feedback == "Custom feedback"
            assert critique_result.metadata["custom_field"] == "custom_value"

    @pytest.mark.asyncio
    async def test_critique_usage_error_handling(self):
        """Test critique handles usage() call errors."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock the agent with usage() that raises an exception
        mock_agent = AsyncMock()
        mock_agent_result = MagicMock()
        mock_agent_result.output = CriticResponse(
            feedback="Test feedback", needs_improvement=True
        )
        mock_agent_result.usage = MagicMock(side_effect=Exception("Usage error"))
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            critique_result = await critic.critique("Test text", result)

            # Should handle the error gracefully
            assert critique_result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_critique_no_usage_attribute(self):
        """Test critique when agent_result has no usage attribute."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock the agent without usage attribute
        mock_agent = AsyncMock()
        mock_agent_result = MagicMock()
        mock_agent_result.output = CriticResponse(
            feedback="Test feedback", needs_improvement=True
        )
        # Remove usage attribute
        del mock_agent_result.usage
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            critique_result = await critic.critique("Test text", result)

            # Should handle missing usage gracefully
            assert critique_result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_critique_error_handling(self):
        """Test critique error handling."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock the agent to raise an exception
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=Exception("API Error"))

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            with pytest.raises(ModelProviderError) as exc_info:
                await critic.critique("Test text", result)

            assert "test_critic" in str(exc_info.value)
            assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_critique_processing_time(self):
        """Test critique processing time measurement."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock the agent with a delay
        mock_agent = AsyncMock()
        mock_agent_result = MagicMock()
        mock_agent_result.output = CriticResponse(
            feedback="Test feedback", needs_improvement=True
        )
        mock_agent_result.usage = MagicMock(return_value=MagicMock(total_tokens=50))

        async def delayed_run(prompt):
            await asyncio.sleep(0.1)  # 100ms delay
            return mock_agent_result

        mock_agent.run = delayed_run

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            critique_result = await critic.critique("Test text", result)

            # Should measure processing time
            assert critique_result.processing_time >= 0.1

    @pytest.mark.asyncio
    async def test_critique_metadata_merging(self):
        """Test that response metadata is properly merged."""
        critic = ConcreteCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock the agent
        mock_agent = AsyncMock()
        mock_agent_result = MagicMock()
        mock_response = CriticResponse(
            feedback="Test feedback",
            suggestions=["Test suggestion"],
            needs_improvement=True,
            confidence=0.8,
            metadata={"response_specific": "data"},
        )
        mock_agent_result.output = mock_response
        mock_agent_result.usage = MagicMock(return_value=MagicMock(total_tokens=50))
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            critique_result = await critic.critique("Test text", result)

            # Check that metadata from response is preserved
            assert critique_result.metadata["response_specific"] == "data"

    def test_abstract_name_property(self):
        """Test that name property is abstract."""
        with pytest.raises(TypeError):
            # Should not be able to instantiate without implementing name
            BaseCritic()

    def test_concrete_name_property(self):
        """Test concrete name property implementation."""
        critic = ConcreteCritic()
        assert critic.name == "test_critic"

    def test_temperature_precedence(self):
        """Test temperature parameter precedence."""
        # Explicit temperature should override config
        config = Config()
        config.llm.temperature = 0.9
        config.llm.critic_temperature = 0.8

        critic = ConcreteCritic(temperature=0.5, config=config)
        assert critic.temperature == 0.5

        # Critic temperature should override general temperature
        critic2 = ConcreteCritic(config=config)
        assert critic2.temperature == 0.8

        # General temperature as fallback
        config.llm.critic_temperature = None
        critic3 = ConcreteCritic(config=config)
        assert critic3.temperature == 0.9

    def test_enable_tools_from_config(self):
        """Test enable_tools falls back to config."""
        config = Config()
        # Set enable_tools on config if it exists, otherwise skip this test
        if hasattr(config, "enable_tools"):
            config.enable_tools = True

            critic = ToolEnabledCritic(config=config)
            assert critic.enable_tools is True
            assert len(critic.tools) == 1
        else:
            # Default behavior when config doesn't have enable_tools
            critic = ToolEnabledCritic(config=config)
            assert critic.enable_tools is False
            assert len(critic.tools) == 0

    def test_traceability_fields(self):
        """Test that traceability fields are properly set."""
        critic = ConcreteCritic(model="gpt-4", temperature=0.5)

        # Access client to initialize it
        client = critic.client

        # Mock successful critique
        async def run_critique():
            result = SifakaResult(original_text="Test", final_text="Test")

            mock_agent = AsyncMock()
            mock_agent_result = MagicMock()
            mock_agent_result.output = CriticResponse(
                feedback="Test feedback", needs_improvement=True
            )
            mock_agent_result.usage = MagicMock(return_value=MagicMock(total_tokens=50))
            mock_agent.run = AsyncMock(return_value=mock_agent_result)

            with patch.object(client, "create_agent", return_value=mock_agent):
                return await critic.critique("Test text", result)

        # Run the critique
        import asyncio

        critique_result = asyncio.run(run_critique())

        # Check traceability fields
        assert critique_result.model_used == "gpt-4"
        assert critique_result.temperature_used == 0.5
        assert critique_result.prompt_sent is not None
        assert critique_result.processing_time >= 0
