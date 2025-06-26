"""Simplified base critic implementation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from ...core.models import CritiqueResult, SifakaResult
from ...core.config import Config
from ...core.interfaces import Critic
from ...core.llm_client import LLMClient, LLMManager, Provider
from ...tools.base import ToolInterface

from pydantic import BaseModel, Field
from typing import Any
from .confidence import ConfidenceCalculator


class CriticResponse(BaseModel):
    """Standardized response format for all critics."""

    feedback: str = Field(..., description="Main feedback about the text")
    suggestions: list[str] = Field(
        default_factory=list, description="Specific improvement suggestions"
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text needs improvement"
    )
    confidence: float = Field(
        0.7, ge=0.0, le=1.0, description="Confidence in the assessment"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional critic-specific data"
    )


class BaseCritic(Critic, ABC):
    """Simplified base critic with clear separation of concerns."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        config: Optional[Config] = None,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        enable_tools: Optional[bool] = None,
    ):
        """Initialize critic with configuration."""
        self.config = config or Config()
        self.model = model
        self.temperature = (
            temperature or self.config.critic_temperature or self.config.temperature
        )
        self.provider = provider
        self._api_key = api_key
        self._client: Optional[LLMClient] = None

        # Components
        self._confidence_calc = ConfidenceCalculator(self.config.critic_base_confidence)

        # Tool support
        self.enable_tools = (
            enable_tools
            if enable_tools is not None
            else getattr(self.config, "enable_tools", False)
        )
        self.tools: List[ToolInterface] = []
        if self.enable_tools:
            self.tools = self._get_available_tools()

    @property
    def client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._client is None:
            self._client = LLMManager.get_client(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                api_key=self._api_key,
            )
        return self._client

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the critic's name."""
        pass

    def _get_available_tools(self) -> List[ToolInterface]:
        """Override to specify which tools this critic can use."""
        return []

    def _get_system_prompt(self) -> str:
        """Get system prompt for PydanticAI agent. Override in subclasses for custom prompts."""
        return f"You are an expert text critic using the {self.name} technique for text improvement."

    def _get_response_type(self) -> type[BaseModel]:
        """Get response type for structured output. Override in subclasses for custom types."""
        return CriticResponse

    @abstractmethod
    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for the LLM.

        Args:
            text: Text to critique
            result: Current result with history

        Returns:
            List of messages for LLM
        """
        pass

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Critique the text.

        Args:
            text: Text to critique
            result: Current result with history

        Returns:
            CritiqueResult with feedback and suggestions
        """

        try:
            # Always use PydanticAI for structured outputs
            agent = self.client.create_agent(
                system_prompt=self._get_system_prompt(),
                result_type=self._get_response_type(),
            )

            # Get user prompt from messages
            messages = await self._create_messages(text, result)
            user_prompt = messages[-1]["content"] if messages else text

            # Run agent with structured output
            agent_result = await agent.run(user_prompt)
            critic_response = agent_result.output

            # Calculate confidence if not provided
            if critic_response.confidence == 0.7:  # Default value
                critic_response.confidence = self._confidence_calc.calculate(
                    feedback=critic_response.feedback,
                    suggestions=critic_response.suggestions,
                    response_length=len(str(critic_response)),
                    metadata=critic_response.metadata,
                )

            # Create result with all metadata
            # Convert the entire response to dict to preserve all fields
            response_dict = critic_response.model_dump()

            # Extract standard fields
            critique_result = CritiqueResult(
                critic=self.name,
                feedback=response_dict.pop("feedback"),
                suggestions=response_dict.pop("suggestions"),
                needs_improvement=response_dict.pop("needs_improvement"),
                confidence=response_dict.pop("confidence"),
                metadata=response_dict,  # Everything else goes in metadata
            )

            return critique_result

        except Exception as e:
            # Return error result
            return CritiqueResult(
                critic=self.name,
                feedback=f"Error during critique: {str(e)}",
                suggestions=["Please review the text manually"],
                needs_improvement=True,
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def _get_previous_context(self, result: SifakaResult) -> str:
        """Get context from previous critiques."""
        if not result.critiques:
            return ""

        # Get recent critiques from this critic
        recent = []
        for critique in list(result.critiques)[-self.config.critic_context_window :]:
            if critique.critic == self.name:
                recent.append(f"- {critique.feedback}")

        if not recent:
            return ""

        return (
            "\n\nPrevious feedback:\n"
            + "\n".join(recent)
            + "\n\nPlease provide NEW insights and avoid repetition."
        )

    def _build_user_prompt(
        self, text: str, result: SifakaResult, instructions: str
    ) -> str:
        """Build a standard user prompt with text and instructions."""
        previous_context = self._get_previous_context(result)

        return f"""{instructions}

Text to evaluate:
{text}
{previous_context}

Please provide specific, actionable feedback."""

    async def _simple_critique(
        self, text: str, result: SifakaResult, instructions: str
    ) -> List[Dict[str, str]]:
        """Simplified message creation for most critics."""
        return [
            {"role": "system", "content": self._get_system_prompt()},
            {
                "role": "user",
                "content": self._build_user_prompt(text, result, instructions),
            },
        ]
