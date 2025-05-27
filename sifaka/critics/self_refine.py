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

The SelfRefineCritic implements key Self-Refine concepts:
1. Iterative refinement through self-feedback
2. Multi-round critique and revision cycles
3. Self-generated improvement suggestions
4. Convergence detection for stopping criteria
5. Learning from refinement patterns and strategies (enhanced)
6. Adaptive refinement based on past success/failure patterns (enhanced)

Note: This implementation captures core Self-Refine principles with enhanced
learning capabilities through integration with the Sifaka thoughts system.
The critic learns from past refinement attempts to improve future performance.
"""

import time
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class SelfRefineCritic(BaseCritic):
    """Critic that implements iterative self-refinement.

    This critic uses the Self-Refine approach to iteratively improve text through
    self-critique and revision. It uses the same language model to critique its
    own output and then revise it based on that critique.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 3,
        improvement_criteria: Optional[List[str]] = None,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Self-Refine critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            max_iterations: Maximum number of refinement iterations.
            improvement_criteria: Specific criteria to focus on during improvement.
            critique_prompt_template: Template for the critique prompt.
            improve_prompt_template: Template for the improvement prompt.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        super().__init__(model=model, model_name=model_name, **model_kwargs)

        self.max_iterations = max_iterations
        self.improvement_criteria = improvement_criteria or [
            "clarity",
            "accuracy",
            "completeness",
            "coherence",
        ]

        # Set up prompt templates
        criteria_text = ", ".join(self.improvement_criteria)

        self.critique_prompt_template = critique_prompt_template or (
            f"Please critique the following text focusing on {criteria_text}.\n\n"
            "Original task: {prompt}\n\n"
            "Text to critique:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Please provide a detailed critique focusing on:\n"
            "1. How well does the text address the original task?\n"
            "2. Are there any factual errors or inconsistencies?\n"
            "3. Is the text clear and well-structured?\n"
            "4. What specific improvements could be made?\n"
            "5. How well does the text use information from the retrieved context (if available)?\n\n"
            "Format your response as:\n"
            "Issues:\n- [List specific issues here]\n\n"
            "Suggestions:\n- [List specific suggestions here]\n\n"
            "Overall Assessment: [Brief assessment]\n\n"
            "If the text is already excellent and needs no improvement, please state that clearly."
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Please improve the following text based on the critique provided.\n\n"
            "Original task: {prompt}\n\n"
            "Current text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Critique:\n{critique}\n\n"
            "Please provide an improved version that addresses the issues identified "
            "in the critique while maintaining the core message and staying true to "
            "the original task. Better incorporate relevant information from the context if available.\n\n"
            "Improved text:"
        )

    async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform the actual critique logic using Self-Refine approach.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Create critique prompt with context
        critique_prompt = self.critique_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            context=context,
        )

        # Generate critique
        critique_response = await self.model._generate_async(
            prompt=critique_prompt,
            system_message="You are an expert critic providing detailed, constructive feedback.",
        )

        # Parse the critique
        issues, suggestions = self._parse_critique(critique_response)

        # Determine if improvement is needed based on critique content
        needs_improvement = self._needs_improvement(critique_response)

        logger.debug("SelfRefineCritic: Critique completed")

        return {
            "needs_improvement": needs_improvement,
            "message": critique_response,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": 0.8,  # Default confidence for Self-Refine
            "metadata": {
                "max_iterations": self.max_iterations,
                "improvement_criteria": self.improvement_criteria,
            },
        }

    def improve(self, thought: Thought) -> str:
        """Improve text using iterative Self-Refine approach.

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
            iteration_history = []

            # Prepare context once for all iterations (using mixin)
            context = self._prepare_context(thought)

            # Extract learning context from thought for enhanced refinement
            learning_context = self._extract_learning_context(thought)

            # Iterative refinement process with learning
            for iteration in range(self.max_iterations):
                logger.debug(
                    f"SelfRefineCritic: Starting iteration {iteration + 1}/{self.max_iterations}"
                )

                # Generate critique for current text with context and learning
                critique_prompt = self._build_enhanced_critique_prompt(
                    thought.prompt, current_text, context, learning_context, iteration
                )

                critique = self.model.generate(
                    prompt=critique_prompt,
                    system_prompt="You are an expert critic providing detailed, constructive feedback with learning from past refinement patterns.",
                )

                # Check if improvement is needed
                if not self._needs_improvement(critique):
                    logger.debug(
                        f"SelfRefineCritic: Stopping early at iteration {iteration + 1} - no improvement needed"
                    )
                    break

                # Generate improved text with context and learning
                improve_prompt = self._build_enhanced_improve_prompt(
                    thought.prompt, current_text, critique, context, learning_context
                )

                improved_text = self.model.generate(
                    prompt=improve_prompt,
                    system_prompt="You are an expert editor improving text based on critique and learned refinement strategies.",
                )

                # Evaluate improvement quality for learning
                improvement_quality = self._evaluate_improvement_quality(
                    current_text, improved_text.strip(), critique
                )

                # Store iteration history with learning data
                iteration_data = {
                    "iteration": iteration + 1,
                    "critique": critique,
                    "text": current_text,
                    "improved_text": improved_text.strip(),
                    "improvement_quality": improvement_quality,
                    "learning_applied": bool(learning_context.get("strategies")),
                }
                iteration_history.append(iteration_data)

                # Update current text for next iteration
                current_text = improved_text.strip()

                logger.debug(f"SelfRefineCritic: Completed iteration {iteration + 1}")

            # Store learning outcomes in thought metadata for future refinements
            self._store_refinement_outcomes(thought, iteration_history, learning_context)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(
                f"SelfRefineCritic: Refinement completed in {processing_time:.2f}ms "
                f"after {len(iteration_history)} iterations with learning integration"
            )

            return current_text

    def _parse_critique(self, critique: str) -> tuple[List[str], List[str]]:
        """Parse critique text to extract issues and suggestions.

        Args:
            critique: The critique text to parse.

        Returns:
            A tuple of (issues, suggestions) lists.
        """
        issues = []
        suggestions = []

        # Simple parsing logic
        in_issues = False
        in_suggestions = False

        for line in critique.split("\n"):
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

        # If no structured format found, extract from general content
        if not issues and not suggestions:
            critique_lower = critique.lower()
            if any(word in critique_lower for word in ["issue", "problem", "error", "unclear"]):
                issues.append("General issues identified in critique")
            if any(word in critique_lower for word in ["improve", "suggest", "consider", "should"]):
                suggestions.append("See critique for improvement suggestions")

        return issues, suggestions

    def _needs_improvement(self, critique: str) -> bool:
        """Determine if text needs improvement based on critique content.

        Args:
            critique: The critique text to analyze.

        Returns:
            True if improvement is needed, False otherwise.
        """
        # Simple heuristic based on common phrases in critiques
        no_improvement_phrases = [
            "no issues",
            "looks good",
            "well written",
            "excellent",
            "great job",
            "perfect",
            "no improvement needed",
            "already excellent",
            "no changes needed",
            "well-structured",
            "clear and concise",
            "high quality",
        ]

        improvement_phrases = [
            "could be improved",
            "needs improvement",
            "issues",
            "problems",
            "unclear",
            "confusing",
            "missing",
            "incorrect",
            "should be",
            "consider",
            "suggest",
            "recommend",
            "enhance",
            "revise",
        ]

        critique_lower = critique.lower()

        # Check for explicit "no improvement" indicators
        for phrase in no_improvement_phrases:
            if phrase in critique_lower:
                return False

        # Check for improvement indicators
        for phrase in improvement_phrases:
            if phrase in critique_lower:
                return True

        # Default to needing improvement if unclear
        return True

    def _extract_learning_context(self, thought: Thought) -> Dict[str, Any]:
        """Extract learning context from thought for enhanced refinement.

        Args:
            thought: The Thought to extract learning context from.

        Returns:
            Dictionary with learning context including strategies and patterns.
        """
        learning_context = {
            "refinement_sessions": 0,
            "successful_strategies": [],
            "failed_strategies": [],
            "improvement_patterns": {},
            "task_type": self._classify_task_type(thought.prompt),
        }

        # Extract from thought metadata
        if thought.metadata:
            self_refine_data = thought.metadata.get("self_refine_memory", {})
            if self_refine_data:
                learning_context["refinement_sessions"] = len(self_refine_data.get("sessions", []))
                learning_context["successful_strategies"] = self_refine_data.get(
                    "successful_strategies", []
                )[
                    -5:
                ]  # Last 5
                learning_context["failed_strategies"] = self_refine_data.get(
                    "failed_strategies", []
                )[
                    -5:
                ]  # Last 5
                learning_context["improvement_patterns"] = self_refine_data.get(
                    "improvement_patterns", {}
                )

        # Extract from thought history
        if thought.history:
            learning_context["previous_attempts"] = len(thought.history)
            # Analyze previous refinement attempts
            for ref in thought.history[-3:]:  # Last 3 attempts
                # In a real implementation, you'd load the thought from storage
                # For now, just note that we have history
                pass

        # Extract from critic feedback history
        if thought.critic_feedback:
            self_refine_feedback = [
                f for f in thought.critic_feedback if f.critic_name == "SelfRefineCritic"
            ]
            if self_refine_feedback:
                learning_context["previous_feedback_count"] = len(self_refine_feedback)
                # Analyze patterns in previous feedback
                for feedback in self_refine_feedback[-2:]:  # Last 2 feedback instances
                    if feedback.metadata and "improvement_patterns" in feedback.metadata:
                        patterns = feedback.metadata["improvement_patterns"]
                        for pattern, effectiveness in patterns.items():
                            if pattern not in learning_context["improvement_patterns"]:
                                learning_context["improvement_patterns"][pattern] = []
                            learning_context["improvement_patterns"][pattern].append(effectiveness)

        return learning_context

    def _classify_task_type(self, prompt: str) -> str:
        """Classify the task type based on the prompt for targeted learning.

        Args:
            prompt: The task prompt to classify.

        Returns:
            String representing the task type.
        """
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["analyze", "analysis", "examine", "evaluate"]):
            return "analysis"
        elif any(word in prompt_lower for word in ["explain", "describe", "define", "what is"]):
            return "explanation"
        elif any(word in prompt_lower for word in ["compare", "contrast", "versus", "vs"]):
            return "comparison"
        elif any(word in prompt_lower for word in ["summarize", "summary", "brief", "overview"]):
            return "summary"
        elif any(word in prompt_lower for word in ["argue", "persuade", "convince", "opinion"]):
            return "persuasive"
        elif any(word in prompt_lower for word in ["create", "generate", "write", "compose"]):
            return "creative"
        else:
            return "general"

    def _build_enhanced_critique_prompt(
        self, prompt: str, text: str, context: str, learning_context: Dict[str, Any], iteration: int
    ) -> str:
        """Build enhanced critique prompt with learning context.

        Args:
            prompt: The original task prompt.
            text: The text to critique.
            context: Retrieved context.
            learning_context: Learning context from past refinements.
            iteration: Current iteration number.

        Returns:
            Enhanced critique prompt string.
        """
        base_prompt = self.critique_prompt_template.format(
            prompt=prompt,
            text=text,
            context=context,
        )

        # Add learning enhancements if available
        if learning_context.get("successful_strategies") or learning_context.get(
            "failed_strategies"
        ):
            learning_section = "\n\nLearning Context:\n"

            if learning_context.get("successful_strategies"):
                learning_section += "Previously successful refinement strategies:\n"
                for strategy in learning_context["successful_strategies"][-3:]:  # Last 3
                    learning_section += f"- {strategy}\n"

            if learning_context.get("failed_strategies"):
                learning_section += "Previously ineffective strategies to avoid:\n"
                for strategy in learning_context["failed_strategies"][-3:]:  # Last 3
                    learning_section += f"- {strategy}\n"

            if learning_context.get("improvement_patterns"):
                learning_section += "Improvement patterns for this task type:\n"
                task_type = learning_context.get("task_type", "general")
                patterns = learning_context["improvement_patterns"]
                for pattern, effectiveness_scores in patterns.items():
                    avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
                    if avg_effectiveness > 0.6:  # Only show effective patterns
                        learning_section += (
                            f"- {pattern} (effectiveness: {avg_effectiveness:.1f})\n"
                        )

            learning_section += f"\nThis is iteration {iteration + 1}. Focus your critique on areas that have shown improvement potential in past refinements.\n"

            base_prompt += learning_section

        return base_prompt

    def _build_enhanced_improve_prompt(
        self, prompt: str, text: str, critique: str, context: str, learning_context: Dict[str, Any]
    ) -> str:
        """Build enhanced improvement prompt with learning context.

        Args:
            prompt: The original task prompt.
            text: The text to improve.
            critique: The critique to address.
            context: Retrieved context.
            learning_context: Learning context from past refinements.

        Returns:
            Enhanced improvement prompt string.
        """
        base_prompt = self.improve_prompt_template.format(
            prompt=prompt,
            text=text,
            critique=critique,
            context=context,
        )

        # Add learning-based improvement strategies
        if learning_context.get("successful_strategies"):
            strategy_section = "\n\nApply these proven successful strategies:\n"
            for strategy in learning_context["successful_strategies"][-3:]:  # Last 3
                strategy_section += f"- {strategy}\n"

            strategy_section += "\nAvoid these previously ineffective approaches:\n"
            for strategy in learning_context.get("failed_strategies", [])[-3:]:  # Last 3
                strategy_section += f"- {strategy}\n"

            base_prompt += strategy_section

        return base_prompt

    def _evaluate_improvement_quality(
        self, original_text: str, improved_text: str, critique: str
    ) -> float:
        """Evaluate the quality of improvement for learning purposes.

        Args:
            original_text: The original text before improvement.
            improved_text: The improved text.
            critique: The critique that guided the improvement.

        Returns:
            Float between 0.0 and 1.0 representing improvement quality.
        """
        # Simple heuristic-based evaluation
        quality_score = 0.5  # Base score

        # Length-based improvements (reasonable length changes)
        length_ratio = len(improved_text) / max(len(original_text), 1)
        if 0.8 <= length_ratio <= 1.5:  # Reasonable length change
            quality_score += 0.1

        # Content diversity (improved text should be different but not completely different)
        words_original = set(original_text.lower().split())
        words_improved = set(improved_text.lower().split())
        overlap = len(words_original & words_improved) / max(len(words_original), 1)
        if 0.3 <= overlap <= 0.8:  # Good balance of change and preservation
            quality_score += 0.2

        # Critique addressing (check if improved text addresses critique issues)
        critique_lower = critique.lower()
        improvement_indicators = [
            "clear",
            "specific",
            "detailed",
            "accurate",
            "complete",
            "coherent",
            "structured",
            "organized",
            "relevant",
            "comprehensive",
        ]

        addressed_issues = 0
        for indicator in improvement_indicators:
            if indicator in critique_lower and indicator in improved_text.lower():
                addressed_issues += 1

        if addressed_issues > 0:
            quality_score += min(0.3, addressed_issues * 0.1)

        return min(1.0, quality_score)

    def _store_refinement_outcomes(
        self,
        thought: Thought,
        iteration_history: List[Dict[str, Any]],
        learning_context: Dict[str, Any],
    ) -> None:
        """Store refinement outcomes in thought metadata for future learning.

        Args:
            thought: The Thought to store outcomes in.
            iteration_history: History of refinement iterations.
            learning_context: The learning context used.
        """
        if not thought.metadata:
            thought.metadata = {}

        # Initialize self-refine memory if not exists
        if "self_refine_memory" not in thought.metadata:
            thought.metadata["self_refine_memory"] = {
                "sessions": [],
                "successful_strategies": [],
                "failed_strategies": [],
                "improvement_patterns": {},
            }

        # Analyze this refinement session
        session_data = {
            "session_id": f"session_{int(time.time())}",
            "task_type": learning_context.get("task_type", "general"),
            "iterations": len(iteration_history),
            "final_quality": (
                iteration_history[-1]["improvement_quality"] if iteration_history else 0.0
            ),
            "strategies_applied": learning_context.get("successful_strategies", []),
            "timestamp": time.time(),
        }

        # Extract successful and failed strategies from this session
        successful_strategies = []
        failed_strategies = []

        for iteration_data in iteration_history:
            quality = iteration_data["improvement_quality"]
            critique = iteration_data["critique"]

            # Extract strategies from critique and improvement
            strategies = self._extract_strategies_from_critique(critique)

            if quality > 0.7:  # High quality improvement
                successful_strategies.extend(strategies)
            elif quality < 0.4:  # Low quality improvement
                failed_strategies.extend(strategies)

        # Update successful strategies (keep unique, recent ones)
        current_successful = thought.metadata["self_refine_memory"]["successful_strategies"]
        updated_successful = list(set(current_successful + successful_strategies))[
            -20:
        ]  # Keep last 20
        thought.metadata["self_refine_memory"]["successful_strategies"] = updated_successful

        # Update failed strategies (keep unique, recent ones)
        current_failed = thought.metadata["self_refine_memory"]["failed_strategies"]
        updated_failed = list(set(current_failed + failed_strategies))[-20:]  # Keep last 20
        thought.metadata["self_refine_memory"]["failed_strategies"] = updated_failed

        # Update improvement patterns for this task type
        task_type = learning_context.get("task_type", "general")
        if task_type not in thought.metadata["self_refine_memory"]["improvement_patterns"]:
            thought.metadata["self_refine_memory"]["improvement_patterns"][task_type] = {}

        patterns = thought.metadata["self_refine_memory"]["improvement_patterns"][task_type]
        for strategy in successful_strategies:
            if strategy not in patterns:
                patterns[strategy] = []
            patterns[strategy].append(0.8)  # High effectiveness

        for strategy in failed_strategies:
            if strategy not in patterns:
                patterns[strategy] = []
            patterns[strategy].append(0.2)  # Low effectiveness

        # Store this session
        thought.metadata["self_refine_memory"]["sessions"].append(session_data)

        # Keep only last 10 sessions
        if len(thought.metadata["self_refine_memory"]["sessions"]) > 10:
            thought.metadata["self_refine_memory"]["sessions"] = thought.metadata[
                "self_refine_memory"
            ]["sessions"][-10:]

    def _extract_strategies_from_critique(self, critique: str) -> List[str]:
        """Extract refinement strategies from critique text.

        Args:
            critique: The critique text to analyze.

        Returns:
            List of identified strategies.
        """
        strategies = []
        critique_lower = critique.lower()

        # Common refinement strategies
        strategy_patterns = {
            "add_examples": ["example", "instance", "illustration", "case"],
            "improve_clarity": ["clear", "clarity", "confusing", "unclear"],
            "add_details": ["detail", "specific", "elaborate", "expand"],
            "improve_structure": ["structure", "organize", "order", "flow"],
            "fix_accuracy": ["accurate", "correct", "error", "mistake"],
            "enhance_completeness": ["complete", "missing", "incomplete", "add"],
            "improve_coherence": ["coherent", "logical", "consistent", "connect"],
            "simplify_language": ["simple", "complex", "jargon", "accessible"],
        }

        for strategy, keywords in strategy_patterns.items():
            if any(keyword in critique_lower for keyword in keywords):
                strategies.append(strategy)

        return strategies
