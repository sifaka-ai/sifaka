"""Base implementation for all Sifaka critics.

This module provides the abstract base class that all critics inherit from.
It handles common functionality like LLM client management, confidence
calculation, and standardized response formatting.

Key features:
- Automatic LLM client creation and management
- Structured output using Pydantic models
- Tool support for critics that need external capabilities
- Confidence calculation based on various factors
- Comprehensive error handling and traceability"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from ...core.config import Config
from ...core.exceptions import ModelProviderError
from ...core.interfaces import Critic
from ...core.llm_client import LLMClient, Provider
from ...core.models import CritiqueResult, SifakaResult
from ...tools.base import ToolInterface
from .confidence import ConfidenceCalculator


class CriticResponse(BaseModel):
    """Standardized response format for all critics.

    This model defines the common structure that all critics must return,
    ensuring consistency across different critic implementations.

    Attributes:
        feedback: Main qualitative feedback explaining the text's strengths
            and weaknesses according to this critic's perspective
        suggestions: List of specific, actionable suggestions for improvement.
            Each suggestion should be concrete and implementable.
        needs_improvement: Boolean indicating whether this critic believes
            the text needs further refinement
        confidence: Float between 0.0 and 1.0 indicating the critic's
            confidence in its assessment. Higher values mean more certainty.
        metadata: Dictionary for critic-specific data like scores, analysis
            details, or other structured information
    """

    feedback: str = Field(..., description="Main feedback about the text")
    suggestions: List[str] = Field(
        default_factory=list, description="Specific improvement suggestions"
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text needs improvement"
    )
    confidence: float = Field(
        0.7, ge=0.0, le=1.0, description="Confidence in the assessment"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional critic-specific data"
    )


class BaseCritic(Critic, ABC):
    """Abstract base class for all Sifaka critics.

    This class provides the foundation for implementing critics that analyze
    and provide feedback on text. Subclasses must implement the abstract
    methods to define their specific critique logic.

    The base class handles:
    - LLM client initialization and management
    - Standardized request/response flow
    - Confidence calculation
    - Error handling and retry logic
    - Tool integration (if enabled)
    - Performance tracking and traceability

    Example:
        >>> class MyCritic(BaseCritic):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_critic"
        ...
        ...     def get_instructions(self, text: str, result: SifakaResult) -> str:
        ...         return "Analyze this text for clarity and conciseness..."
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        config: Optional[Config] = None,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        enable_tools: Optional[bool] = None,
    ):
        """Initialize the critic with configuration options.

        Args:
            model: Name of the LLM model to use (e.g., 'gpt-4', 'claude-3').
                Defaults to 'gpt-4o-mini' or the configured critic_model.
            temperature: Generation temperature (0.0-2.0). Lower values are
                more deterministic. If not provided, uses config settings.
            config: Full Sifaka configuration object. If provided, overrides
                individual parameters.
            provider: LLM provider ('openai', 'anthropic', etc.) or Provider
                enum value. Auto-detected from model name if not specified.
            api_key: API key for the LLM provider. If not provided, uses
                environment variables or config settings.
            enable_tools: Whether to enable tool usage for this critic.
                Some critics like self_rag can use tools for retrieval.
        """
        self.config = config or Config()
        # Use critic_model from config if not explicitly provided
        self.model = (
            model if model != "gpt-4o-mini" else (self.config.critic_model or model)
        )
        self.temperature = (
            temperature or self.config.critic_temperature or self.config.temperature
        )
        self.provider = provider
        self._api_key = api_key
        self._client: Optional[LLMClient] = None

        # Components
        self._confidence_calc: ConfidenceCalculator = ConfidenceCalculator(
            self.config.critic_base_confidence
        )

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
        """Get or lazily create the LLM client.

        The client is created on first access to avoid initialization
        overhead if the critic is never used.

        Returns:
            Configured LLMClient instance for making API calls
        """
        if self._client is None:
            # Convert string provider to Provider enum if needed
            provider_value = self.provider
            if isinstance(provider_value, str):
                provider_value = Provider(provider_value)

            self._client = LLMClient(
                model=self.model,
                temperature=self.temperature,
                provider=provider_value if provider_value else Provider.OPENAI,
                api_key=self._api_key,
            )
        return self._client

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name identifier for this critic.

        This name is used for registration, configuration, and tracking.
        It should be lowercase with underscores (e.g., 'self_refine').

        Returns:
            The critic's unique name identifier
        """
        pass

    def _get_available_tools(self) -> List[ToolInterface]:
        """Specify which tools this critic can use.

        Override in subclasses that need tool support (e.g., self_rag
        for retrieval tools).

        Returns:
            List of ToolInterface instances this critic can use
        """
        return []

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM.

        Override in subclasses to provide critic-specific instructions
        and context. The system prompt shapes the critic's behavior.

        Returns:
            System prompt string for the LLM
        """
        return f"You are an expert text critic using the {self.name} technique for text improvement."

    def _get_response_type(self) -> Type[BaseModel]:
        """Get the Pydantic model for structured output.

        Override in subclasses that need custom response formats beyond
        the standard CriticResponse.

        Returns:
            Pydantic model class for parsing LLM responses
        """
        return CriticResponse

    def get_instructions(self, text: str, result: SifakaResult) -> str:
        """Get the critique instructions for this critic.

        This is the main method subclasses should implement to define their
        critique logic. The instructions will be sent to the LLM along with
        the text to evaluate.

        Args:
            text: Current version of the text to evaluate
            result: Full result object with history and metadata

        Returns:
            Instructions string that will be sent to the LLM

        Example:
            >>> def get_instructions(self, text: str, result: SifakaResult) -> str:
            ...     return '''Analyze this text for clarity and coherence.
            ...     Focus on:
            ...     1. Clear main points
            ...     2. Logical flow
            ...     3. Concise language'''
        """
        return "Please analyze this text and provide specific, actionable feedback."

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for the LLM API call.

        This method builds the conversation history that will be sent to
        the LLM. The default implementation uses the get_instructions()
        method which is easier for subclasses to override.

        Subclasses can override this method for full control over message
        creation, but in most cases overriding get_instructions() is simpler.

        Args:
            text: Current version of the text to critique
            result: Complete result object with all history and metadata
                   including previous critiques and generations

        Returns:
            List of message dictionaries with 'role' and 'content' keys
            ready for the LLM API
        """
        instructions = self.get_instructions(text, result)
        return await self._simple_critique(text, result, instructions)

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Critique the given text and provide improvement suggestions.

        This is the main entry point called by the orchestrator. It handles
        the complete critique workflow including LLM interaction, response
        parsing, confidence calculation, and result formatting.

        Args:
            text: The current version of the text to critique. This is
                  typically the latest generation from the improvement process.
            result: The complete result object containing:
                    - original_text: Starting text before any improvements
                    - generations: All previous versions of the text
                    - critiques: All previous critique feedback
                    - validations: Results from quality validators
                    - metadata: Additional context and metrics

        Returns:
            CritiqueResult containing:
            - critic: Name of this critic
            - feedback: Natural language feedback about the text
            - suggestions: Specific improvement recommendations
            - needs_improvement: Whether further iteration is needed
            - confidence: How certain the critic is (0.0-1.0)
            - metadata: Additional critic-specific information
            - Traceability data (model, prompt, tokens, timing)

        Raises:
            ModelProviderError: If the LLM API call fails after retries
            ValueError: If the response cannot be parsed

        Note:
            This method is called by the orchestrator for each iteration.
            Critics should be stateless - use the result parameter for
            any historical context needed.
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

            # Run agent with structured output and capture usage
            start_time = time.time()
            agent_result = await agent.run(user_prompt)
            processing_time = time.time() - start_time
            critic_response = agent_result.output

            # Get actual usage data
            tokens_used = 0
            try:
                if hasattr(agent_result, "usage"):
                    usage = agent_result.usage()  # Call as function
                    if usage and hasattr(usage, "total_tokens"):
                        tokens_used = usage.total_tokens
            except Exception:
                # Fallback if usage() call fails
                tokens_used = 0

            # Calculate confidence if not provided
            if critic_response.confidence == 0.7:  # Default value
                # Pass needs_improvement in metadata for smarter confidence calculation
                # Handle response models that may not have a metadata field
                calc_metadata = {}
                if hasattr(critic_response, "metadata") and critic_response.metadata:
                    calc_metadata = critic_response.metadata.copy()
                calc_metadata["needs_improvement"] = critic_response.needs_improvement

                critic_response.confidence = self._confidence_calc.calculate(
                    feedback=critic_response.feedback,
                    suggestions=critic_response.suggestions,
                    response_length=len(str(critic_response)),
                    metadata=calc_metadata,
                )

            # Create result with all metadata
            # Convert the entire response to dict to preserve all fields
            response_dict = critic_response.model_dump()

            # Extract standard fields
            # Pop metadata if it exists to avoid nesting
            response_metadata = response_dict.pop("metadata", {})

            # Extract the standard fields first
            feedback = response_dict.pop("feedback")
            suggestions = response_dict.pop("suggestions")
            needs_improvement = response_dict.pop("needs_improvement")
            confidence = response_dict.pop("confidence")

            # Merge metadata from response (if any) with remaining fields
            final_metadata = {**response_metadata, **response_dict}

            critique_result = CritiqueResult(
                critic=self.name,
                feedback=feedback,
                suggestions=suggestions,
                needs_improvement=needs_improvement,
                confidence=confidence,
                metadata=final_metadata,
                # Add traceability with actual usage data
                model_used=(
                    self.client.model if hasattr(self.client, "model") else self.model
                ),
                temperature_used=(
                    self.client.temperature
                    if hasattr(self.client, "temperature")
                    else self.temperature
                ),
                prompt_sent=user_prompt,
                tokens_used=tokens_used,
                processing_time=processing_time,
            )

            return critique_result

        except Exception as e:
            # Wrap any exceptions with context
            raise ModelProviderError(
                f"Critic '{self.name}' failed to process text: {e}",
                provider=str(self.provider or "unknown"),
            )

    def _get_previous_context(self, result: SifakaResult) -> str:
        """Extract relevant context from previous critiques.

        Builds a summary of recent feedback from this critic to avoid
        repetition and track improvement trajectory.

        Args:
            result: The result object containing critique history

        Returns:
            Formatted string of previous feedback or empty string
        """
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
