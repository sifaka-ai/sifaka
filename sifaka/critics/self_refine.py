"""Self-Refine critic for Sifaka.

This module implements the Self-Refine approach for critics, which enables language
models to iteratively critique and revise their own outputs without requiring
external feedback.

Based on "Self-Refine: Iterative Refinement with Self-Feedback":
https://arxiv.org/abs/2303.17651

@misc{madaan2023selfrefineiterativerefinementselffeedback,
      title={Self-Refine: Iterative Refinement with Self-Feedback},
      author={Aman Madaan and Niket Tandon and Prakhar Gupta and Skyler Hallinan and Luyu Gao and Sarah Wiegreffe and Uri Alon and Nouha Dziri and Shrimai Prabhumoye and Yiming Yang and Shashank Gupta and Bodhisattwa Prasad Majumder and Katherine Hermann and Sean Welleck and Amir Yazdanbakhsh and Peter Clark},
      year={2023},
      eprint={2303.17651},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2303.17651},
}

The SelfRefineCritic implements the core Self-Refine algorithm:
1. Iterative refinement through self-feedback
2. Multi-round critique and revision cycles
3. Self-generated improvement suggestions
4. Convergence detection for stopping criteria

Note: This implementation follows the original Self-Refine paper closely,
using a simple FEEDBACK → REFINE → FEEDBACK loop without additional
learning mechanisms that were not part of the original research.
"""

import time
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent

from sifaka.core.thought import Thought
from sifaka.critics.base_pydantic import PydanticAICritic
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class SelfRefineCritic(PydanticAICritic):
    """Critic that implements iterative self-refinement with validation awareness.

    This critic uses the Self-Refine approach to iteratively improve text through
    self-critique and revision. It uses the same language model to critique its
    own output and then revise it based on that critique.

    Enhanced with validation context awareness to prioritize validation constraints
    over conflicting critic suggestions.
    """

    def __init__(
        self,
        model_name: str,
        max_iterations: int = 3,
        improvement_criteria: Optional[List[str]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the Self-Refine critic.

        Args:
            model_name: The model name for the PydanticAI agent (e.g., "openai:gpt-4")
            max_iterations: Maximum number of refinement iterations.
            improvement_criteria: Specific criteria to focus on during improvement.
            **agent_kwargs: Additional arguments passed to the PydanticAI agent.
        """
        # Initialize parent with system prompt
        super().__init__(model_name=model_name, **agent_kwargs)

        self.max_iterations = max_iterations
        self.improvement_criteria = improvement_criteria or [
            "clarity",
            "accuracy",
            "completeness",
            "coherence",
        ]

        logger.info(
            f"Initialized SelfRefineCritic with max_iterations={max_iterations}, "
            f"criteria={self.improvement_criteria}"
        )

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for self-refine evaluation."""
        criteria_text = ", ".join(self.improvement_criteria)
        return f"""You are an expert self-refine critic that provides iterative feedback for text improvement. Your role is to critique text focusing on {criteria_text} and provide structured feedback.

You must return a CritiqueFeedback object with these REQUIRED fields:
- message: A clear summary of your self-refine evaluation (string)
- needs_improvement: Whether the text needs improvement based on self-critique (boolean)
- confidence: ConfidenceScore with overall confidence (object with 'overall' field as float 0.0-1.0)
- critic_name: Set this to "SelfRefineCritic" (string)

And these OPTIONAL fields (can be empty lists or null):
- violations: List of ViolationReport objects for identified issues
- suggestions: List of ImprovementSuggestion objects for addressing issues
- processing_time_ms: Time taken in milliseconds (can be null)
- critic_version: Version string (can be null)
- metadata: Additional metadata dictionary (can be empty)

IMPORTANT: Always provide the required fields. For confidence, use a simple object like {{"overall": 0.8}} where the number is between 0.0 and 1.0.

Focus on:
1. How well does the text address the original task?
2. Are there any factual errors or inconsistencies?
3. Is the text clear and well-structured?
4. What specific improvements could be made?
5. How well does the text use information from context?

Use the Self-Refine approach: provide detailed, constructive feedback for iterative improvement."""

    async def _create_critique_prompt(self, thought: Thought) -> str:
        """Create the critique prompt for self-refine evaluation.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            The formatted critique prompt.
        """
        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Get validation context if available
        validation_context = self._get_validation_context_dict(thought)
        validation_text = ""
        if validation_context:
            validation_text = f"\n\nValidation Context:\n{validation_context}"

        criteria_text = ", ".join(self.improvement_criteria)

        return f"""Please critique the following text focusing on {criteria_text}.

Original task: {thought.prompt}

Text to critique:
{thought.text}

Context:
{context}
{validation_text}

Please provide a detailed critique focusing on:
1. How well does the text address the original task?
2. Are there any factual errors or inconsistencies?
3. Is the text clear and well-structured?
4. What specific improvements could be made?
5. How well does the text use information from the retrieved context (if available)?

Self-Refine Parameters:
- Max Iterations: {self.max_iterations}
- Improvement Criteria: {criteria_text}

If the text is already excellent and needs no improvement, please state that clearly in your assessment."""

    async def improve_async(self, thought: Thought) -> str:
        """Improve text using iterative Self-Refine approach asynchronously.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text after iterative refinement.

        Raises:
            ImproverError: If the improvement fails.
        """
        start_time = time.time()

        with critic_context(
            critic_name="SelfRefineCritic",
            operation="improve",
            message_prefix="Failed to improve text with Self-Refine",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="SelfRefineCritic",
                    operation="improve",
                    suggestions=["Provide text to improve"],
                )

            current_text = thought.text

            # Prepare context once for all iterations (using mixin)
            context = self._prepare_context(thought)

            # Iterative refinement process following original Self-Refine algorithm
            for iteration in range(self.max_iterations):
                logger.debug(
                    f"SelfRefineCritic: Starting iteration {iteration + 1}/{self.max_iterations}"
                )

                # FEEDBACK: Generate critique for current text
                critique_result = await self._critique_current_text(thought, current_text)

                # Check if improvement is needed (stopping criteria)
                if not critique_result.feedback.needs_improvement:
                    logger.debug(
                        f"SelfRefineCritic: Stopping early at iteration {iteration + 1} - no improvement needed"
                    )
                    break

                # REFINE: Generate improved text
                current_text = await self._refine_text(
                    thought, current_text, critique_result.feedback.message, context
                )

                logger.debug(f"SelfRefineCritic: Completed iteration {iteration + 1}")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(
                f"SelfRefineCritic: Refinement completed in {processing_time:.2f}ms "
                f"after {iteration + 1} iterations"
            )

            return current_text

    async def _critique_current_text(self, thought: Thought, current_text: str):
        """Generate critique for current text iteration."""
        # Create a temporary thought with current text
        temp_thought = thought.model_copy(update={"text": current_text})
        return await self.critique_async(temp_thought)

    async def _refine_text(
        self, thought: Thought, current_text: str, critique: str, context: str
    ) -> str:
        """Generate improved text based on critique."""
        # Create improvement agent (returns string, not structured output)
        improvement_agent = Agent(
            model=self.model_name,
            output_type=str,
            system_prompt="You are an expert editor improving text based on self-critique feedback.",
        )

        # Create improvement prompt
        improve_prompt = f"""Please improve the following text based on the critique provided.

Original task: {thought.prompt}

Current text:
{current_text}

Context:
{context}

Critique:
{critique}

Please provide an improved version that addresses the issues identified in the critique while maintaining the core message and staying true to the original task. Better incorporate relevant information from the context if available.

Improved text:"""

        # Generate improved text
        result = await improvement_agent.run(improve_prompt)
        return result.output.strip()
