"""Reflexion critic for Sifaka v0.3.0+

This module implements a Reflexion-inspired approach for iterative improvement
through trial-and-error learning with verbal reinforcement using PydanticAI agents
with structured output.

Based on "Reflexion: Language Agents with Verbal Reinforcement Learning":
https://arxiv.org/abs/2303.11366

@misc{shinn2023reflexionlanguageagentsverbal,
      title={Reflexion: Language Agents with Verbal Reinforcement Learning},
      author={Noah Shinn and Federico Cassano and Edward Berman and Ashwin Gopinath and Karthik Narasimhan and Shunyu Yao},
      year={2023},
      eprint={2303.11366},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2303.11366},
}

The ReflexionCritic implements key Reflexion concepts:
1. Trial-based learning with episodic memory
2. Task performance feedback integration
3. Self-reflection on failures and successes
4. Verbal reinforcement for future attempts

IMPORTANT IMPLEMENTATION NOTES:
This implementation adapts the core Reflexion principle-based evaluation
for text critique using Sifaka's superior Thought infrastructure. The original
Reflexion paper focuses on multi-agent Actor/Evaluator/Self-Reflection architecture
for reinforcement learning from AI feedback.

Our implementation focuses on the critique and self-reflection aspects without the
full multi-agent training component, making it suitable for real-time text improvement
rather than model training. We leverage Sifaka's Thought system for episodic memory
instead of duplicating memory infrastructure.

Note: This is a simplified implementation that captures core Reflexion principles
without the full multi-agent Actor/Evaluator/Self-Reflection architecture.
"""

import time
from typing import Any, Dict, List, Optional

from sifaka.core.thought import Thought
from sifaka.critics.base_pydantic import PydanticAICritic
from sifaka.models.critic_results import (
    ConfidenceScore,
    ImprovementSuggestion,
    SeverityLevel,
    ViolationReport,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


# Task feedback can be stored in CriticFeedback.metadata
# Trial memory can be stored in Thought.history and Thought.metadata
# This leverages the existing Thought infrastructure instead of duplicating it


class ReflexionCritic(PydanticAICritic):
    """Modern Reflexion critic using PydanticAI agents with structured output.

    This critic implements the Reflexion approach for improving text through
    self-reflection and iterative learning from past attempts. It performs a critique,
    reflects on the critique to identify specific improvements, and then generates
    improved text.

    The process involves:
    1. Generating an initial critique of the text
    2. Reflecting on the critique to identify specific improvements
    3. Improving the text based on the reflection

    Enhanced with validation context awareness to prioritize validation constraints
    over conflicting reflection suggestions.

    Key features:
    - Structured output using CritiqueFeedback model
    - Self-reflection on critique quality
    - Trial-based learning with episodic memory
    - Task performance feedback integration
    - Validation context awareness
    - Enhanced episodic memory using Thought infrastructure
    """

    def __init__(
        self,
        model_name: str,
        max_trials: int = 3,
        reflection_depth: str = "deep",
        max_memory_size: int = 10,
        **agent_kwargs,
    ):
        """Initialize the Reflexion critic.

        Args:
            model_name: The model name for the PydanticAI agent (e.g., "openai:gpt-4")
            max_trials: Maximum number of reflection trials to consider.
            reflection_depth: Depth of reflection ("shallow", "medium", "deep").
            max_memory_size: Maximum number of reflections to keep in memory.
            **agent_kwargs: Additional arguments passed to the PydanticAI agent.
        """
        # Initialize parent with system prompt
        super().__init__(model_name=model_name, **agent_kwargs)

        self.max_trials = max_trials
        self.reflection_depth = reflection_depth
        self.max_memory_size = max_memory_size

        # Simple memory system for episodic learning (original Reflexion concept)
        self.memory_buffer: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized ReflexionCritic with max_trials={max_trials}, reflection_depth={reflection_depth}, max_memory_size={max_memory_size}"
        )

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for reflexion evaluation."""
        return """You are an expert reflexion critic that uses self-reflection to improve text quality. Your role is to critique text through iterative reflection and learning.

You must return a CritiqueFeedback object with these REQUIRED fields:
- message: A clear summary of your reflexive evaluation (string)
- needs_improvement: Whether the text needs improvement based on reflection (boolean)
- confidence: ConfidenceScore with overall confidence (object with 'overall' field as float 0.0-1.0)
- critic_name: Set this to "ReflexionCritic" (string)

And these OPTIONAL fields (can be empty lists or null):
- violations: List of ViolationReport objects for identified issues
- suggestions: List of ImprovementSuggestion objects for addressing issues
- processing_time_ms: Time taken in milliseconds (can be null)
- critic_version: Version string (can be null)
- metadata: Additional metadata dictionary (can be empty)

IMPORTANT: Always provide the required fields. For confidence, use a simple object like {"overall": 0.8} where the number is between 0.0 and 1.0.

Focus on:
1. Self-reflection on text quality and effectiveness
2. Learning from previous attempts and feedback
3. Iterative improvement through trial-and-error
4. Verbal reinforcement for successful patterns
5. Accurate confidence assessment based on reflection

Use the Reflexion approach: reflect deeply on what works and what doesn't."""

    async def _create_critique_prompt(self, thought: Thought) -> str:
        """Create the critique prompt for reflexion evaluation.

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

        # Extract trial history from thought metadata
        trial_history = self._extract_trial_history(thought)
        history_text = ""
        if trial_history:
            history_text = f"\n\nTrial History:\n{trial_history}"

        # Extract task feedback from thought metadata
        task_feedback = self._extract_task_feedback(thought)
        feedback_text = ""
        if task_feedback:
            feedback_text = f"\n\nTask Feedback:\n{task_feedback}"

        return f"""Perform a reflexive critique of the following text using self-reflection and learning from past attempts.

Original Task: {thought.prompt}

Text to Evaluate:
{thought.text}

Context:
{context}
{validation_text}
{history_text}
{feedback_text}

Reflexion Parameters:
- Max Trials: {self.max_trials}
- Reflection Depth: {self.reflection_depth}

Please provide a structured reflexive evaluation with:
1. Self-reflection on text quality and effectiveness
2. Analysis of what works well and what doesn't
3. Learning from previous attempts (if available)
4. Specific improvement suggestions based on reflection
5. Confidence assessment based on reflexive analysis

Use the Reflexion approach: reflect deeply on the text's strengths and weaknesses, learn from any available feedback or history, and provide actionable insights for improvement."""

    def _extract_trial_history(self, thought: Thought) -> str:
        """Extract trial history from thought metadata.

        Args:
            thought: The thought to extract history from.

        Returns:
            Formatted trial history string.
        """
        if not thought.history:
            return "No previous trials available."

        history_lines = []
        for i, ref in enumerate(thought.history[-self.max_trials :], 1):
            history_lines.append(f"Trial {i}: {ref.summary or 'No summary available'}")

        return "\n".join(history_lines)

    def _extract_task_feedback(self, thought: Thought) -> str:
        """Extract task feedback from thought metadata.

        Args:
            thought: The thought to extract feedback from.

        Returns:
            Formatted task feedback string.
        """
        if not thought.metadata:
            return "No task feedback available."

        feedback_items = []

        # Look for feedback in metadata
        if "task_feedback" in thought.metadata:
            feedback_items.append(f"Task Feedback: {thought.metadata['task_feedback']}")

        if "performance_score" in thought.metadata:
            feedback_items.append(f"Performance Score: {thought.metadata['performance_score']}")

        if "external_feedback" in thought.metadata:
            feedback_items.append(f"External Feedback: {thought.metadata['external_feedback']}")

        return "\n".join(feedback_items) if feedback_items else "No task feedback available."

    def add_task_feedback(
        self,
        thought: Thought,
        success: bool,
        score: Optional[float] = None,
        error_message: Optional[str] = None,
        external_feedback: Optional[str] = None,
    ) -> Thought:
        """Add external task performance feedback to a thought.

        This method allows external systems to provide task performance feedback
        that will be used in the Reflexion process.

        Args:
            thought: The Thought to add feedback to.
            success: Whether the task was completed successfully.
            score: Optional numeric score (0.0 to 1.0).
            error_message: Optional error message if task failed.
            external_feedback: Optional external feedback text.

        Returns:
            Updated thought with task feedback in metadata.
        """
        task_feedback = {
            "success": success,
            "score": score,
            "error_message": error_message,
            "external_feedback": external_feedback,
            "timestamp": time.time(),
        }

        updated_metadata = dict(thought.metadata or {})
        updated_metadata["task_feedback"] = task_feedback

        return thought.model_copy(update={"metadata": updated_metadata})

    def _add_to_memory(self, text: str, critique: str, reflection: str) -> None:
        """Add a reflection to memory for future learning.

        Args:
            text: The original text that was critiqued.
            critique: The critique that was generated.
            reflection: The reflection that was generated.
        """
        memory_entry = {
            "text": text[:200],  # Store first 200 chars to save memory
            "critique": critique[:300],  # Store first 300 chars
            "reflection": reflection[:300],  # Store first 300 chars
        }

        self.memory_buffer.append(memory_entry)

        # Keep only the most recent reflections
        if len(self.memory_buffer) > self.max_memory_size:
            self.memory_buffer.pop(0)

        logger.debug(f"Added reflection to memory. Buffer size: {len(self.memory_buffer)}")

    def _get_memory_context(self) -> str:
        """Get formatted memory context from past reflections.

        Returns:
            A formatted string with past reflections for context.
        """
        if not self.memory_buffer:
            return "No past reflections available."

        memory_parts = []
        for i, entry in enumerate(self.memory_buffer[-3:], 1):  # Use last 3 entries
            memory_part = f"Past Reflection {i}:\n"
            memory_part += f"Text: {entry['text']}...\n"
            memory_part += f"Critique: {entry['critique']}...\n"
            memory_part += f"Reflection: {entry['reflection']}...\n"
            memory_parts.append(memory_part)

        return "\n".join(memory_parts)
