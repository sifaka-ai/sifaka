"""Prompt-based critic for Sifaka v0.3.0+

This module implements a simple, customizable prompt-based critic using PydanticAI
agents with structured output. It allows users to define their own critique criteria
through custom prompts.

The PromptCritic provides a flexible foundation for creating domain-specific
critics without requiring complex implementations.
"""

from typing import List

from sifaka.core.thought import Thought
from sifaka.critics.base_pydantic import PydanticAICritic

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class PromptCritic(PydanticAICritic):
    """Modern customizable prompt-based critic using PydanticAI agents with structured output.

    This critic allows users to define their own critique criteria through
    custom prompts, making it easy to create domain-specific critics without
    complex implementations.

    Key features:
    - Structured output using CritiqueFeedback model
    - Customizable evaluation criteria
    - Flexible prompt templates
    - Domain-specific critique capabilities
    """

    def __init__(
        self,
        model_name: str,
        criteria: List[str] = None,
        custom_system_prompt: str = None,
        **agent_kwargs,
    ):
        """Initialize the Prompt critic.

        Args:
            model_name: The model name for the PydanticAI agent (e.g., "openai:gpt-4")
            criteria: List of specific criteria to evaluate.
            custom_system_prompt: Custom system prompt for the critic.
            **agent_kwargs: Additional arguments passed to the PydanticAI agent.
        """
        # Initialize parent with system prompt
        super().__init__(model_name=model_name, **agent_kwargs)

        self.criteria = criteria or [
            "Clarity and readability",
            "Accuracy and factual correctness",
            "Completeness and thoroughness",
            "Relevance to the task",
        ]

        self.custom_system_prompt = custom_system_prompt

        logger.info(f"Initialized PromptCritic with {len(self.criteria)} evaluation criteria")

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for prompt-based critique.

        Returns:
            The default system prompt string.
        """
        if self.custom_system_prompt:
            return self.custom_system_prompt

        criteria_text = "\n".join(f"- {criterion}" for criterion in self.criteria)

        return f"""You are an expert critic providing detailed, constructive feedback on text quality.

Your task is to evaluate text based on the following criteria:
{criteria_text}

You must return a CritiqueFeedback object with these REQUIRED fields:
- message: A clear summary of your evaluation (string)
- needs_improvement: Whether the text needs improvement (boolean)
- confidence: ConfidenceScore with overall confidence (object with 'overall' field as float 0.0-1.0)
- critic_name: Set this to "PromptCritic" (string)

And these OPTIONAL fields (can be empty lists or null):
- violations: List of ViolationReport objects for identified issues
- suggestions: List of ImprovementSuggestion objects for addressing issues
- processing_time_ms: Time taken in milliseconds (can be null)
- critic_version: Version string (can be null)
- metadata: Additional metadata dictionary (can be empty)

Focus on providing specific, constructive feedback based on the evaluation criteria."""

    async def _create_critique_prompt(self, thought: Thought) -> str:
        """Create the critique prompt for the given thought.

        Args:
            thought: The thought to critique.

        Returns:
            The formatted critique prompt.
        """
        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Get validation context if available
        validation_text = ""
        if hasattr(thought, "validation_results") and thought.validation_results:
            validation_text = f"\nValidation Context:\n{self._format_validation_context(thought.validation_results)}"

        # Create criteria text
        criteria_text = "\n".join(f"- {criterion}" for criterion in self.criteria)

        return f"""Evaluate the following text based on the specified criteria and provide structured feedback.

Original Task: {thought.prompt}

Text to Evaluate:
{thought.text}

Context:
{context}
{validation_text}

Evaluation Criteria:
{criteria_text}

Please provide a thorough critique that:

1. Evaluates the text against each criterion
2. Identifies specific issues or problems
3. Provides concrete suggestions for improvement
4. Determines whether improvement is needed
5. Assigns confidence based on the clarity of evaluation

Be specific and constructive in your feedback. Consider how well the text uses information from the retrieved context (if available)."""

    # Note: The old improve_async and improve_with_validation_context_async methods
    # are not needed in the PydanticAI approach since improvement is handled
    # by the chain/agent architecture, not individual critics.
