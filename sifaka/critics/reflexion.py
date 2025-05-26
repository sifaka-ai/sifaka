"""Reflexion critic for Sifaka.

This module implements the Reflexion approach for text improvement, which uses
self-reflection to improve text quality through iterative refinement.

Based on "Reflexion: Language Agents with Verbal Reinforcement Learning":
https://arxiv.org/abs/2303.11366

The ReflexionCritic uses a language model to:
1. Generate an initial critique of the text
2. Reflect on the critique and identify specific improvements
3. Generate improved text based on the reflection
"""

import time
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ReflexionCritic(BaseCritic):
    """Critic that uses self-reflection to improve text quality.

    This critic implements the Reflexion approach for improving text through
    self-reflection. It performs a critique, reflects on the critique to identify
    specific improvements, and then generates improved text.

    The process involves:
    1. Generating an initial critique of the text
    2. Reflecting on the critique to identify specific improvements
    3. Improving the text based on the reflection
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        critique_prompt_template: Optional[str] = None,
        reflection_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        max_memory_size: int = 10,
        **model_kwargs: Any,
    ):
        """Initialize the Reflexion critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            critique_prompt_template: Template for the critique prompt.
            reflection_prompt_template: Template for the reflection prompt.
            improve_prompt_template: Template for the improvement prompt.
            max_memory_size: Maximum number of reflections to keep in memory.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        super().__init__(model=model, model_name=model_name, **model_kwargs)

        # Memory buffer for past reflections
        self.memory_buffer: List[Dict[str, str]] = []
        self.max_memory_size = max_memory_size

        # Set up prompt templates
        self.critique_prompt_template = critique_prompt_template or (
            "Please critique the following text and identify any issues or areas for improvement. "
            "Focus on clarity, coherence, accuracy, and relevance to the original prompt.\n\n"
            "Original prompt: {prompt}\n\n"
            "Text to critique:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Please provide your critique in the following format:\n"
            "Issues:\n- [List specific issues here]\n\n"
            "Suggestions:\n- [List specific suggestions here]\n\n"
            "Overall Assessment: [Brief overall assessment]\n\n"
            "Consider how well the text uses information from the retrieved context (if available)."
        )

        self.reflection_prompt_template = reflection_prompt_template or (
            "Based on the critique below, reflect on what specific improvements should be made. "
            "Think step-by-step about how to address each issue.\n\n"
            "Original prompt: {prompt}\n\n"
            "Original text:\n{text}\n\n"
            "Critique:\n{critique}\n\n"
            "Past reflections (for learning):\n{memory_context}\n\n"
            "Please provide a reflection in the following format:\n"
            "Key Issues to Address:\n- [List key issues]\n\n"
            "Improvement Strategy:\n- [List specific improvement steps]\n\n"
            "Expected Outcome: [What the improved text should achieve]\n\n"
            "Reflection: [Your meta-cognitive reflection on the improvement process]"
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Improve the following text based on the critique and reflection provided.\n\n"
            "Original prompt: {prompt}\n\n"
            "Original text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Critique:\n{critique}\n\n"
            "Reflection:\n{reflection}\n\n"
            "Please provide an improved version that:\n"
            "1. Addresses all the issues identified in the critique\n"
            "2. Follows the improvement strategy from the reflection\n"
            "3. Maintains the core message and purpose\n"
            "4. Better incorporates relevant information from the context (if available)\n\n"
            "Improved text:"
        )

    async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform the actual critique logic using Reflexion approach.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Step 1: Generate initial critique
        critique_result = await self._generate_critique_async(thought)

        # Step 2: Generate reflection on the critique
        reflection_result = await self._generate_reflection_async(
            thought, critique_result["critique"]
        )

        # Step 3: Store reflection in memory for future learning
        self._add_to_memory(
            thought.text, critique_result["critique"], reflection_result["reflection"]
        )

        logger.debug("ReflexionCritic: Critique and reflection completed")

        return {
            "needs_improvement": critique_result["needs_improvement"],
            "message": critique_result["critique"],
            "issues": critique_result["issues"],
            "suggestions": critique_result["suggestions"],
            "confidence": 0.8,  # Default confidence for Reflexion
            "metadata": {
                "reflection": reflection_result["reflection"],
                "improvement_strategy": reflection_result["improvement_strategy"],
                "memory_size": len(self.memory_buffer),
            },
        }

    def improve(self, thought: Thought) -> str:
        """Improve text based on critique and reflection.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text based on critique and reflection.

        Raises:
            ImproverError: If the improvement fails.
        """
        start_time = time.time()

        with critic_context(
            critic_name="ReflexionCritic",
            operation="improve",
            message_prefix="Failed to improve text with Reflexion approach",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="ReflexionCritic",
                    operation="improve",
                    suggestions=["Provide text to improve"],
                )

            # Get critique and reflection from thought
            critique_text = ""
            reflection_text = ""

            if thought.critic_feedback:
                for feedback in thought.critic_feedback:
                    if feedback.critic_name == "ReflexionCritic":
                        critique_text = feedback.feedback
                        reflection_text = feedback.metadata.get("reflection", "")
                        break

            # If no critique available, generate one
            if not critique_text:
                logger.debug("No critique found in thought, generating new critique")
                # Use sync version of critique generation
                import asyncio

                try:
                    asyncio.get_running_loop()
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, self._generate_critique_async(thought)
                        )
                        critique_result = future.result()
                except RuntimeError:
                    critique_result = asyncio.run(self._generate_critique_async(thought))

                critique_text = critique_result["critique"]

                # Generate reflection for the new critique
                try:
                    asyncio.get_running_loop()
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, self._generate_reflection_async(thought, critique_text)
                        )
                        reflection_result = future.result()
                except RuntimeError:
                    reflection_result = asyncio.run(
                        self._generate_reflection_async(thought, critique_text)
                    )

                reflection_text = reflection_result["reflection"]

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Create improvement prompt with context, critique, and reflection
            improve_prompt = self.improve_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                context=context,
                critique=critique_text,
                reflection=reflection_text,
            )

            # Generate improved text
            improved_text = self.model.generate(
                prompt=improve_prompt,
                system_prompt="You are an expert editor using reflection to improve text quality.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"ReflexionCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    async def _generate_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Generate initial critique of the text asynchronously.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results.
        """
        # Prepare context for critique (using mixin)
        context = self._prepare_context(thought)

        # Log context usage
        if self._has_context(thought):
            context_summary = self._get_context_summary(thought)
            logger.debug(f"ReflexionCritic using context for critique: {context_summary}")

        # Format the critique prompt with context
        critique_prompt = self.critique_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            context=context,
        )

        # Generate the critique (async)
        logger.debug("Generating initial critique")
        critique_text = await self.model._generate_async(
            prompt=critique_prompt,
            system_message="You are an expert critic providing detailed feedback on text quality.",
        )
        logger.debug(f"Generated critique of length {len(critique_text)}")

        # Parse the critique
        issues = []
        suggestions = []

        # Simple parsing logic - can be improved
        in_issues = False
        in_suggestions = False

        for line in critique_text.split("\n"):
            line = line.strip()
            if line.lower().startswith("issues:"):
                in_issues = True
                in_suggestions = False
                continue
            elif line.lower().startswith("suggestions:"):
                in_issues = False
                in_suggestions = True
                continue
            elif line.lower().startswith("overall assessment:"):
                in_issues = False
                in_suggestions = False
                continue
            elif not line or line.startswith("#"):
                continue

            if in_issues and line.startswith("-"):
                issues.append(line[1:].strip())
            elif in_suggestions and line.startswith("-"):
                suggestions.append(line[1:].strip())

        # Determine if improvement is needed
        needs_improvement = len(issues) > 0 or "improvement" in critique_text.lower()

        return {
            "critique": critique_text,
            "issues": issues,
            "suggestions": suggestions,
            "needs_improvement": needs_improvement,
        }

    async def _generate_reflection_async(self, thought: Thought, critique: str) -> Dict[str, Any]:
        """Generate reflection on the critique to identify specific improvements asynchronously.

        Args:
            thought: The Thought container with the text to reflect on.
            critique: The critique text to reflect on.

        Returns:
            A dictionary with reflection results.
        """
        # Prepare memory context from past reflections
        memory_context = self._get_memory_context()

        # Format the reflection prompt
        reflection_prompt = self.reflection_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            critique=critique,
            memory_context=memory_context,
        )

        # Generate the reflection (async)
        logger.debug("Generating reflection on critique")
        reflection_text = await self.model._generate_async(
            prompt=reflection_prompt,
            system_message="You are an expert reflecting on critique to identify specific improvements.",
        )
        logger.debug(f"Generated reflection of length {len(reflection_text)}")

        # Parse the reflection
        key_issues = []
        improvement_strategy = []

        # Simple parsing logic
        in_key_issues = False
        in_improvement_strategy = False

        for line in reflection_text.split("\n"):
            line = line.strip()
            if line.lower().startswith("key issues to address:"):
                in_key_issues = True
                in_improvement_strategy = False
                continue
            elif line.lower().startswith("improvement strategy:"):
                in_key_issues = False
                in_improvement_strategy = True
                continue
            elif line.lower().startswith("expected outcome:") or line.lower().startswith(
                "reflection:"
            ):
                in_key_issues = False
                in_improvement_strategy = False
                continue
            elif not line or line.startswith("#"):
                continue

            if in_key_issues and line.startswith("-"):
                key_issues.append(line[1:].strip())
            elif in_improvement_strategy and line.startswith("-"):
                improvement_strategy.append(line[1:].strip())

        return {
            "reflection": reflection_text,
            "key_issues": key_issues,
            "improvement_strategy": improvement_strategy,
        }

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
