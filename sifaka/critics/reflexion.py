"""Reflexion critic for Sifaka.

This module implements a Reflexion-inspired approach for iterative improvement
through trial-and-error learning with verbal reinforcement.

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

Note: This is a simplified implementation that captures core Reflexion principles
without the full multi-agent Actor/Evaluator/Self-Reflection architecture.
"""

import time
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic
from sifaka.critics.mixins.validation_aware import ValidationAwareMixin
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger
from sifaka.validators.validation_context import create_validation_context

logger = get_logger(__name__)


# Task feedback can be stored in CriticFeedback.metadata
# Trial memory can be stored in Thought.history and Thought.metadata
# This leverages the existing Thought infrastructure instead of duplicating it


class ReflexionCritic(BaseCritic, ValidationAwareMixin):
    """Critic that uses self-reflection to improve text quality with validation awareness.

    This critic implements the Reflexion approach for improving text through
    self-reflection. It performs a critique, reflects on the critique to identify
    specific improvements, and then generates improved text.

    The process involves:
    1. Generating an initial critique of the text
    2. Reflecting on the critique to identify specific improvements
    3. Improving the text based on the reflection

    Enhanced with validation context awareness to prioritize validation constraints
    over conflicting reflection suggestions.
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

        # Simple memory system for episodic learning (original Reflexion concept)
        self.max_memory_size = max_memory_size
        self.memory_buffer: List[Dict[str, Any]] = []

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
            "Based on the critique below, reflect deeply on what specific improvements should be made. "
            "Use your episodic memory of past trials to inform your reflection.\n\n"
            "Original prompt: {prompt}\n\n"
            "Original text:\n{text}\n\n"
            "Critique:\n{critique}\n\n"
            "Trial Context:\n"
            "- Current trial: {trial_number}\n"
            "- Previous attempts: {previous_attempts}\n"
            "- Success patterns: {success_patterns}\n"
            "- Failure patterns: {failure_patterns}\n"
            "- External feedback: {external_feedback}\n\n"
            "Episodic Memory:\n{episodic_memory}\n\n"
            "Please provide a multi-layered reflection:\n\n"
            "1. CRITIQUE QUALITY ASSESSMENT:\n"
            "- How accurate and useful was this critique?\n"
            "- What did the critique miss or get wrong?\n\n"
            "2. TASK PERFORMANCE REFLECTION:\n"
            "- Why might this text have these issues?\n"
            "- What patterns do I see from past similar tasks?\n"
            "- What worked well in previous successful attempts?\n\n"
            "3. IMPROVEMENT STRATEGY:\n"
            "- Specific steps to address each issue\n"
            "- How to avoid repeating past failures\n"
            "- How to leverage successful patterns\n\n"
            "4. META-REFLECTION:\n"
            "- What am I learning about this type of task?\n"
            "- How can I improve my critique process?\n"
            "- What should I remember for future attempts?\n\n"
            "Expected Outcome: [What the improved text should achieve based on learning]"
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
        """Perform the actual critique logic using Reflexion approach (async version).

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Check for external task feedback in thought metadata
        task_feedback = self._extract_task_feedback(thought)

        # Step 1: Generate initial critique (considering task feedback if available)
        critique_result = await self._generate_critique_async(thought, task_feedback)

        # Step 2: Generate self-reflection on the critique and task performance
        reflection_result = await self._generate_reflection_async(
            thought, critique_result["critique"], task_feedback
        )

        # Step 3: Store reflection in memory for future learning (legacy)
        self._add_to_memory(
            thought.text or "", critique_result["critique"], reflection_result["reflection"]
        )

        # Store trial outcome in thought metadata for enhanced episodic learning
        self._store_trial_outcome(thought, reflection_result)

        logger.debug("ReflexionCritic: Async critique and reflection completed")

        # Calculate dynamic confidence based on Reflexion-specific factors
        confidence = self._calculate_reflexion_confidence(
            thought, critique_result, reflection_result, task_feedback
        )

        return {
            "needs_improvement": critique_result["needs_improvement"],
            "message": critique_result["critique"],
            "issues": critique_result["issues"],
            "suggestions": critique_result["suggestions"],
            "confidence": confidence,
            "metadata": {
                "reflection": reflection_result["reflection"],
                "improvement_strategy": reflection_result["improvement_strategy"],
                "memory_size": len(self.memory_buffer),
                "task_feedback": task_feedback,
                "trial_number": thought.iteration,
            },
        }

    async def improve_async(self, thought: Thought) -> str:
        """Improve text based on critique and reflection asynchronously.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text based on critique and reflection.

        Raises:
            ImproverError: If the improvement fails.
        """
        # Use the enhanced method with validation context from thought
        validation_context = create_validation_context(getattr(thought, "validation_results", None))
        return await self.improve_with_validation_context_async(thought, validation_context)

    async def improve_with_validation_context_async(
        self, thought: Thought, validation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Improve text with validation context awareness.

        Args:
            thought: The Thought container with the text to improve and critique.
            validation_context: Optional validation context for constraint awareness.

        Returns:
            The improved text that prioritizes validation constraints.

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

            # If no critique available, generate one using async methods
            if not critique_text:
                logger.debug("No critique found in thought, generating new critique")
                critique_result = await self._generate_critique_async(thought, None)
                critique_text = critique_result["critique"]

                # Generate reflection for the new critique
                reflection_result = await self._generate_reflection_async(
                    thought, critique_text, None
                )
                reflection_text = reflection_result["reflection"]

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Create improvement prompt with validation awareness
            if validation_context:
                # Create a critique string for the enhanced prompt
                critique = f"Critique:\n{critique_text}\n\nReflection:\n{reflection_text}"

                # Use enhanced prompt with validation awareness
                improve_prompt = self._create_enhanced_improvement_prompt(
                    prompt=thought.prompt,
                    text=thought.text,
                    critique=critique,
                    context=context,
                    validation_context=validation_context,
                    critic_suggestions=[],  # ReflexionCritic doesn't have structured suggestions
                )
            else:
                # Use original prompt template
                improve_prompt = self.improve_prompt_template.format(
                    prompt=thought.prompt,
                    text=thought.text,
                    context=context,
                    critique=critique_text,
                    reflection=reflection_text,
                )

            # Generate improved text (async only)
            improved_text = await self.model._generate_async(
                prompt=improve_prompt,
                system_prompt="You are an expert editor using reflection to improve text quality.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"ReflexionCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    async def _generate_critique_async(
        self, thought: Thought, task_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate initial critique of the text (async version).

        Args:
            thought: The Thought container with the text to critique.
            task_feedback: Optional external task feedback.

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

        # Generate the critique using async model generation
        logger.debug("Generating initial critique (async)")
        critique_text = await self.model._generate_async(
            prompt=critique_prompt,
            system_prompt="You are an expert critic providing detailed feedback on text quality.",
        )
        logger.debug(f"Generated critique of length {len(critique_text)}")

        # Parse the critique (same logic as sync version)
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

    async def _generate_reflection_async(
        self, thought: Thought, critique: str, task_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate reflection on the critique to identify specific improvements (async version).

        Args:
            thought: The Thought container with the text to reflect on.
            critique: The critique text to reflect on.
            task_feedback: Optional external task feedback.

        Returns:
            A dictionary with reflection results.
        """
        # Extract trial context from thought
        trial_context = self._extract_trial_context(thought)

        # Get episodic memory from thought history and metadata
        episodic_memory = self._build_episodic_memory(thought)

        # Extract patterns from past experiences
        success_patterns = self._extract_success_patterns(thought)
        failure_patterns = self._extract_failure_patterns(thought)

        # Get external feedback
        external_feedback = task_feedback or self._extract_task_feedback(thought)

        # Format the enhanced reflection prompt
        reflection_prompt = self.reflection_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            critique=critique,
            trial_number=trial_context.get("trial_number", 1),
            previous_attempts=trial_context.get("previous_attempts", "None"),
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
            external_feedback=external_feedback or "None available",
            episodic_memory=episodic_memory,
        )

        # Generate the reflection using async model generation
        logger.debug("Generating reflection on critique (async)")
        reflection_text = await self.model._generate_async(
            prompt=reflection_prompt,
            system_prompt="You are an expert reflecting on critique to identify specific improvements.",
        )
        logger.debug(f"Generated reflection of length {len(reflection_text)}")

        # Parse the reflection (same logic as sync version)
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

    def _extract_task_feedback(self, thought: Thought) -> Optional[Dict[str, Any]]:
        """Extract task performance feedback from thought metadata.

        Args:
            thought: The Thought container to extract feedback from.

        Returns:
            Task feedback dictionary if available, None otherwise.
        """
        # Check thought metadata for task feedback
        if thought.metadata and "task_feedback" in thought.metadata:
            return thought.metadata["task_feedback"]

        # Check previous iterations for task feedback
        if thought.history:
            for _ref in thought.history[:3]:  # Check last 3 iterations
                # In a real implementation, you'd load the thought from storage
                # For now, just check if there's feedback info in the reference
                pass

        return None

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

    def _extract_trial_context(self, thought: Thought) -> Dict[str, Any]:
        """Extract trial context from thought for episodic learning.

        Args:
            thought: The Thought to extract context from.

        Returns:
            Dictionary with trial context information.
        """
        trial_context = {
            "trial_number": thought.iteration,
            "chain_id": thought.chain_id,
            "previous_attempts": len(thought.history) if thought.history else 0,
        }

        # Extract previous attempt summaries from history
        if thought.history:
            attempts = []
            for i, ref in enumerate(thought.history[-3:]):  # Last 3 attempts
                attempts.append(f"Attempt {i + 1}: iteration {ref.iteration}")
            trial_context["previous_attempts"] = "; ".join(attempts)

        return trial_context

    def _build_episodic_memory(self, thought: Thought) -> str:
        """Build episodic memory from thought history and metadata.

        Args:
            thought: The Thought to build memory from.

        Returns:
            Formatted episodic memory string.
        """
        memory_parts = []

        # Extract from thought metadata
        if thought.metadata:
            reflexion_data = thought.metadata.get("reflexion_memory", {})
            if reflexion_data:
                memory_parts.append("Previous Reflexion Sessions:")
                for session in reflexion_data.get("sessions", [])[-3:]:  # Last 3 sessions
                    memory_parts.append(f"- {session.get('summary', 'No summary')}")

        # Extract from thought history
        if thought.history:
            memory_parts.append("\nRecent Trial History:")
            for ref in thought.history[-3:]:  # Last 3 trials
                memory_parts.append(f"- Trial {ref.iteration}: {ref.summary or 'No summary'}")

        # Extract from critic feedback history
        if thought.critic_feedback:
            reflexion_feedback = [
                f for f in thought.critic_feedback if f.critic_name == "ReflexionCritic"
            ]
            if reflexion_feedback:
                memory_parts.append("\nPrevious Reflexion Feedback:")
                for feedback in reflexion_feedback[-2:]:  # Last 2 feedback instances
                    reflection = feedback.metadata.get("reflection", "")
                    if reflection:
                        memory_parts.append(f"- {reflection[:150]}...")

        return "\n".join(memory_parts) if memory_parts else "No episodic memory available."

    def _extract_success_patterns(self, thought: Thought) -> str:
        """Extract success patterns from thought history.

        Args:
            thought: The Thought to extract patterns from.

        Returns:
            Formatted success patterns string.
        """
        patterns = []

        # Look for successful patterns in metadata
        if thought.metadata:
            success_data = thought.metadata.get("reflexion_success_patterns", [])
            patterns.extend(success_data[-3:])  # Last 3 success patterns

        # Analyze critic feedback for successful outcomes
        if thought.critic_feedback:
            for feedback in thought.critic_feedback:
                if feedback.critic_name == "ReflexionCritic":
                    if not feedback.metadata.get("needs_improvement", True):
                        patterns.append("Previous attempt was successful - no improvement needed")

        return "; ".join(patterns) if patterns else "No clear success patterns identified yet."

    def _extract_failure_patterns(self, thought: Thought) -> str:
        """Extract failure patterns from thought history.

        Args:
            thought: The Thought to extract patterns from.

        Returns:
            Formatted failure patterns string.
        """
        patterns = []

        # Look for failure patterns in metadata
        if thought.metadata:
            failure_data = thought.metadata.get("reflexion_failure_patterns", [])
            patterns.extend(failure_data[-3:])  # Last 3 failure patterns

        # Analyze critic feedback for recurring issues
        if thought.critic_feedback:
            issue_counts = {}
            for feedback in thought.critic_feedback:
                if feedback.critic_name == "ReflexionCritic":
                    for issue in feedback.violations:
                        issue_counts[issue] = issue_counts.get(issue, 0) + 1

            # Identify recurring issues (appeared more than once)
            recurring = [issue for issue, count in issue_counts.items() if count > 1]
            if recurring:
                patterns.extend([f"Recurring issue: {issue}" for issue in recurring[:3]])

        return "; ".join(patterns) if patterns else "No clear failure patterns identified yet."

    def _calculate_reflexion_confidence(
        self,
        thought: Thought,
        critique_result: Dict[str, Any],
        reflection_result: Dict[str, Any],
        task_feedback: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate confidence score based on Reflexion-specific factors.

        Args:
            thought: The Thought being analyzed.
            critique_result: Results from the critique generation.
            reflection_result: Results from the reflection generation.
            task_feedback: Optional external task feedback.

        Returns:
            Confidence score between 0.1 and 1.0.
        """
        # Start with base confidence
        confidence = 0.5

        # Factor 1: Critique quality (based on specificity and detail)
        critique_text = critique_result.get("critique", "")
        critique_quality = self._assess_critique_quality(critique_text)
        confidence += critique_quality * 0.2

        # Factor 2: Reflection depth and actionability
        reflection_text = reflection_result.get("reflection", "")
        reflection_quality = self._assess_reflection_quality(reflection_text)
        confidence += reflection_quality * 0.2

        # Factor 3: Memory coherence (consistency with past experiences)
        memory_coherence = self._assess_memory_coherence(thought, reflection_text)
        confidence += memory_coherence * 0.15

        # Factor 4: Trial progression (confidence increases with experience)
        trial_progression = self._assess_trial_progression(thought)
        confidence += trial_progression * 0.1

        # Factor 5: External validation (if task feedback is available)
        if task_feedback:
            external_validation = self._assess_external_validation(task_feedback)
            confidence += external_validation * 0.15

        # Factor 6: Consistency with previous critiques (if available)
        critique_consistency = self._assess_critique_consistency(thought, critique_result)
        confidence += critique_consistency * 0.1

        # Ensure confidence is within valid range
        return max(0.1, min(1.0, confidence))

    def _assess_critique_quality(self, critique_text: str) -> float:
        """Assess the quality of the critique based on specificity and structure.

        Args:
            critique_text: The critique text to assess.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        if not critique_text:
            return 0.0

        quality_score = 0.0

        # Check for structured feedback (numbered points, sections)
        if any(marker in critique_text for marker in ["1.", "2.", "**", "###", "- "]):
            quality_score += 0.3

        # Check for specific examples and suggestions
        if any(word in critique_text.lower() for word in ["example", "specific", "instead"]):
            quality_score += 0.2

        # Check for actionable language
        if any(
            word in critique_text.lower() for word in ["should", "could", "consider", "improve"]
        ):
            quality_score += 0.2

        # Check for detailed analysis (longer critiques tend to be more thorough)
        if len(critique_text) > 500:
            quality_score += 0.2
        elif len(critique_text) > 200:
            quality_score += 0.1

        # Check for balanced assessment (both strengths and weaknesses)
        has_strengths = any(
            word in critique_text.lower() for word in ["good", "strong", "excellent", "effective"]
        )
        has_weaknesses = any(
            word in critique_text.lower() for word in ["weak", "improve", "issue", "problem"]
        )
        if has_strengths and has_weaknesses:
            quality_score += 0.1

        return min(1.0, quality_score)

    def _assess_reflection_quality(self, reflection_text: str) -> float:
        """Assess the quality of the self-reflection.

        Args:
            reflection_text: The reflection text to assess.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        if not reflection_text:
            return 0.0

        quality_score = 0.0

        # Check for self-awareness indicators
        if any(
            phrase in reflection_text.lower()
            for phrase in ["my critique", "i missed", "i should", "my analysis"]
        ):
            quality_score += 0.3

        # Check for specific improvement strategies
        if any(
            word in reflection_text.lower()
            for word in ["strategy", "approach", "method", "technique"]
        ):
            quality_score += 0.2

        # Check for learning from past experiences
        if any(
            phrase in reflection_text.lower()
            for phrase in ["previous", "past", "learned", "experience"]
        ):
            quality_score += 0.2

        # Check for actionable insights
        if any(
            word in reflection_text.lower()
            for word in ["next time", "future", "going forward", "will"]
        ):
            quality_score += 0.2

        # Check for depth (longer reflections tend to be more thoughtful)
        if len(reflection_text) > 300:
            quality_score += 0.1

        return min(1.0, quality_score)

    def _assess_memory_coherence(self, thought: Thought, reflection_text: str) -> float:
        """Assess how well the current reflection aligns with past experiences.

        Args:
            thought: The current thought.
            reflection_text: The current reflection text.

        Returns:
            Coherence score between 0.0 and 1.0.
        """
        # If no memory, return neutral score
        if len(self.memory_buffer) == 0:
            return 0.5

        coherence_score = 0.5  # Start neutral

        # Check if reflection references past experiences
        if any(
            word in reflection_text.lower() for word in ["previous", "past", "before", "earlier"]
        ):
            coherence_score += 0.3

        # Check for consistency with memory patterns
        success_patterns = self._extract_success_patterns(thought)
        failure_patterns = self._extract_failure_patterns(thought)

        # If reflection acknowledges known patterns, increase confidence
        if success_patterns != "No clear success patterns identified yet.":
            for pattern in success_patterns.split(";"):
                if any(word in reflection_text.lower() for word in pattern.lower().split()[:3]):
                    coherence_score += 0.1
                    break

        if failure_patterns != "No clear failure patterns identified yet.":
            for pattern in failure_patterns.split(";"):
                if any(word in reflection_text.lower() for word in pattern.lower().split()[:3]):
                    coherence_score += 0.1
                    break

        return min(1.0, coherence_score)

    def _assess_trial_progression(self, thought: Thought) -> float:
        """Assess confidence based on trial number and learning progression.

        Args:
            thought: The current thought.

        Returns:
            Progression score between 0.0 and 1.0.
        """
        trial_number = thought.iteration
        memory_size = len(self.memory_buffer)

        # Confidence increases with experience, but with diminishing returns
        experience_factor = min(0.8, (trial_number + memory_size) * 0.1)

        # Bonus for having substantial memory
        if memory_size >= 3:
            experience_factor += 0.1

        return min(1.0, experience_factor)

    def _assess_external_validation(self, task_feedback: Dict[str, Any]) -> float:
        """Assess confidence based on external task feedback.

        Args:
            task_feedback: External feedback about task performance.

        Returns:
            Validation score between 0.0 and 1.0.
        """
        if not task_feedback:
            return 0.0

        validation_score = 0.5  # Start neutral

        # Check for positive feedback indicators
        feedback_text = str(task_feedback).lower()
        if any(word in feedback_text for word in ["good", "correct", "success", "improved"]):
            validation_score += 0.3

        # Check for negative feedback indicators
        if any(word in feedback_text for word in ["poor", "incorrect", "failed", "worse"]):
            validation_score -= 0.3

        return max(0.0, min(1.0, validation_score))

    def _assess_critique_consistency(
        self, thought: Thought, critique_result: Dict[str, Any]
    ) -> float:
        """Assess consistency with previous critiques.

        Args:
            thought: The current thought.
            critique_result: Current critique results.

        Returns:
            Consistency score between 0.0 and 1.0.
        """
        if not thought.critic_feedback:
            return 0.5  # Neutral if no previous critiques

        # Look for previous ReflexionCritic feedback
        previous_critiques = [
            feedback
            for feedback in thought.critic_feedback
            if feedback.critic_name == "ReflexionCritic"
        ]

        if not previous_critiques:
            return 0.5

        consistency_score = 0.5
        current_issues = set(critique_result.get("issues", []))

        # Check for consistency in identified issues
        for prev_critique in previous_critiques[-2:]:  # Check last 2 critiques
            prev_issues = set(prev_critique.violations)
            if current_issues and prev_issues:
                overlap = len(current_issues.intersection(prev_issues))
                total_unique = len(current_issues.union(prev_issues))
                if total_unique > 0:
                    consistency_ratio = overlap / total_unique
                    consistency_score += consistency_ratio * 0.2

        return min(1.0, consistency_score)

    def _store_trial_outcome(self, thought: Thought, reflection_result: Dict[str, Any]) -> None:
        """Store trial outcome in thought metadata for future learning.

        Args:
            thought: The Thought to store outcome in.
            reflection_result: The reflection result to store.
        """
        if not thought.metadata:
            thought.metadata = {}

        # Initialize reflexion memory if not exists
        if "reflexion_memory" not in thought.metadata:
            thought.metadata["reflexion_memory"] = {"sessions": []}

        # Store this session
        session_data = {
            "trial_number": thought.iteration,
            "timestamp": time.time(),
            "reflection": reflection_result.get("reflection", ""),
            "improvement_strategy": reflection_result.get("improvement_strategy", []),
            "summary": f"Trial {thought.iteration}: {len(reflection_result.get('improvement_strategy', []))} strategies identified",
        }

        thought.metadata["reflexion_memory"]["sessions"].append(session_data)

        # Keep only last 10 sessions
        if len(thought.metadata["reflexion_memory"]["sessions"]) > 10:
            thought.metadata["reflexion_memory"]["sessions"] = thought.metadata["reflexion_memory"][
                "sessions"
            ][-10:]
