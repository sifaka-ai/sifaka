"""Self-Consistency critic for Sifaka.

This module implements a Self-Consistency approach for text critique and improvement,
where multiple critiques are generated for the same text and consensus is used to
determine the most reliable feedback.

Based on "Self-Consistency Improves Chain of Thought Reasoning in Language Models":
https://arxiv.org/abs/2203.11171

@misc{wang2022selfconsistency,
      title={Self-Consistency Improves Chain of Thought Reasoning in Language Models},
      author={Xuezhi Wang and Jason Wei and Dale Schuurmans and Quoc Le and Ed Chi and Sharan Narang and Aakanksha Chowdhery and Denny Zhou},
      year={2022},
      eprint={2203.11171},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2203.11171},
}

The SelfConsistencyCritic implements key Self-Consistency concepts:
1. Multiple critique generation for the same text
2. Chain-of-thought reasoning in each critique
3. Consensus-based aggregation of feedback
4. Confidence scoring based on agreement level
5. Majority voting for final recommendations
6. Learning from consistency patterns and reliability prediction (enhanced)
7. Adaptive consensus mechanisms based on task types and past performance (enhanced)

Note: This is an adaptation of the Self-Consistency approach from the original paper
with enhanced learning capabilities through integration with the Sifaka thoughts system.
The critic learns when consensus is reliable vs. unreliable and adapts its approach
based on past consistency patterns for different types of tasks.
"""

import asyncio
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class SelfConsistencyCritic(BaseCritic):
    """Critic that implements Self-Consistency with multiple critique generation.

    This critic uses the Self-Consistency approach which generates multiple critiques
    of the same text and uses consensus to determine the most reliable feedback.
    This improves critique reliability by reducing the impact of single inconsistent
    or low-quality critiques.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        base_critic: Optional[BaseCritic] = None,
        num_iterations: int = 5,
        consensus_threshold: float = 0.6,
        aggregation_method: str = "majority_vote",
        use_chain_of_thought: bool = True,
        similarity_threshold: float = 0.7,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Self-Consistency critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            base_critic: Optional base critic to use for generating individual critiques.
            num_iterations: Number of critique iterations to generate (default: 5).
            consensus_threshold: Minimum agreement ratio for high confidence (default: 0.6).
            aggregation_method: Method for aggregating critiques ("majority_vote", "weighted").
            use_chain_of_thought: Whether to use chain-of-thought prompting.
            similarity_threshold: Threshold for considering issues/suggestions similar.
            critique_prompt_template: Template for the critique prompt.
            improve_prompt_template: Template for the improvement prompt.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        super().__init__(model=model, model_name=model_name, **model_kwargs)

        # Set up base critic (if provided) or use self-critique
        self.base_critic = base_critic

        # Configuration parameters
        self.num_iterations = max(3, num_iterations)  # Minimum 3 for meaningful consensus
        self.consensus_threshold = max(0.5, min(1.0, consensus_threshold))
        self.aggregation_method = aggregation_method
        self.use_chain_of_thought = use_chain_of_thought
        self.similarity_threshold = similarity_threshold

        # Set up prompt templates
        self.critique_prompt_template = (
            critique_prompt_template or self._default_critique_template()
        )
        self.improve_prompt_template = improve_prompt_template or self._default_improve_template()

    def _default_critique_template(self) -> str:
        """Default template for individual critique generation."""
        cot_instruction = ""
        if self.use_chain_of_thought:
            cot_instruction = " Think step-by-step and provide detailed reasoning."

        return (
            "Evaluate the following text for quality, accuracy, and areas for improvement.\n\n"
            "Original task: {prompt}\n\n"
            "Text to evaluate:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            f"Please provide a thorough critique.{cot_instruction}\n\n"
            "Structure your response as follows:\n\n"
            "Reasoning: [Your step-by-step analysis]\n\n"
            "Strengths:\n- [List specific strengths]\n\n"
            "Issues:\n- [List specific issues or problems]\n\n"
            "Suggestions:\n- [List specific improvement suggestions]\n\n"
            "Overall Assessment: [Summary of your evaluation]\n\n"
            "Needs Improvement: [Yes/No - whether the text needs improvement]"
        )

    def _default_improve_template(self) -> str:
        """Default template for improvement."""
        return (
            "Improve the following text based on the consensus feedback from multiple evaluations.\n\n"
            "Original task: {prompt}\n\n"
            "Current text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Consensus Issues (found in {consensus_count}/{total_iterations} evaluations):\n{consensus_issues}\n\n"
            "Consensus Suggestions (found in {consensus_count}/{total_iterations} evaluations):\n{consensus_suggestions}\n\n"
            "Confidence Level: {confidence:.1%} (based on evaluator agreement)\n\n"
            "Please provide an improved version that:\n"
            "1. Addresses the most commonly identified issues\n"
            "2. Incorporates the most frequently suggested improvements\n"
            "3. Maintains the original intent and style\n"
            "4. Focuses on changes supported by multiple evaluations\n\n"
            "Improved text:"
        )

    async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform the actual critique logic using Self-Consistency approach.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Extract learning context from thought for enhanced consistency evaluation
        learning_context = self._extract_consistency_learning_context(thought)

        # Generate multiple critiques with learning-informed approach
        critiques = await self._generate_multiple_critiques_with_learning_async(
            thought, learning_context
        )

        # Aggregate critiques using enhanced consensus with learning
        aggregated_result = self._aggregate_critiques_with_learning(critiques, learning_context)

        # Calculate confidence based on agreement and learned patterns
        confidence = self._calculate_confidence_with_learning(
            critiques, aggregated_result, learning_context
        )

        # Determine if improvement is needed based on consensus and learning
        needs_improvement = self._determine_improvement_need_with_learning(
            critiques, aggregated_result, learning_context
        )

        # Create comprehensive feedback message
        combined_message = self._format_consensus_message(critiques, aggregated_result, confidence)

        # Store consistency learning outcomes for future evaluations
        self._store_consistency_outcomes(
            thought, learning_context, critiques, aggregated_result, confidence
        )

        logger.debug(
            f"SelfConsistencyCritic: Completed {len(critiques)} critique iterations with learning integration"
        )

        return {
            "needs_improvement": needs_improvement,
            "message": combined_message,
            "issues": aggregated_result["consensus_issues"],
            "suggestions": aggregated_result["consensus_suggestions"],
            "confidence": confidence,
            "metadata": {
                "num_iterations": len(critiques),
                "consensus_threshold": self.consensus_threshold,
                "aggregation_method": self.aggregation_method,
                "individual_critiques": critiques,
                "consensus_stats": aggregated_result["stats"],
                "base_critic_used": self.base_critic is not None,
                "use_chain_of_thought": self.use_chain_of_thought,
                "learning_applied": bool(learning_context.get("consistency_patterns")),
                "task_type": learning_context.get("task_type", "general"),
                "predicted_reliability": learning_context.get("predicted_reliability", 0.5),
            },
        }

    async def _generate_multiple_critiques_async(self, thought: Thought) -> List[Dict[str, Any]]:
        """Generate multiple critiques of the same text.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            List of critique results from multiple iterations.
        """
        critiques = []

        if self.base_critic:
            # Use base critic for multiple iterations
            tasks = [self.base_critic._critique_async(thought) for _ in range(self.num_iterations)]
            critique_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(critique_results):
                if isinstance(result, Exception):
                    logger.warning(f"Critique iteration {i+1} failed: {result}")
                    continue
                critiques.append(result)
        else:
            # Generate critiques using our own model
            tasks = [
                self._generate_single_critique_async(thought) for _ in range(self.num_iterations)
            ]
            critique_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(critique_results):
                if isinstance(result, Exception):
                    logger.warning(f"Critique iteration {i+1} failed: {result}")
                    continue
                critiques.append(result)

        if not critiques:
            raise ImproverError(
                message="All critique iterations failed",
                component="SelfConsistencyCritic",
                operation="generate_multiple_critiques",
                suggestions=["Check model availability", "Verify base critic configuration"],
            )

        logger.debug(
            f"Generated {len(critiques)} successful critiques out of {self.num_iterations} attempts"
        )
        return critiques

    async def _generate_single_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Generate a single critique using our own model.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            Single critique result dictionary.
        """
        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Create critique prompt
        critique_prompt = self.critique_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            context=context,
        )

        # Generate critique
        critique_response = await self.model._generate_async(
            prompt=critique_prompt,
            system_message="You are an expert evaluator providing detailed, constructive feedback on text quality.",
        )

        # Parse the critique response
        parsed_critique = self._parse_single_critique(critique_response)

        return {
            "message": critique_response,
            "issues": parsed_critique["issues"],
            "suggestions": parsed_critique["suggestions"],
            "needs_improvement": parsed_critique["needs_improvement"],
            "reasoning": parsed_critique["reasoning"],
            "strengths": parsed_critique["strengths"],
        }

    def _parse_single_critique(self, critique_text: str) -> Dict[str, Any]:
        """Parse a single critique response into structured components.

        Args:
            critique_text: The raw critique response text.

        Returns:
            Dictionary with parsed critique components.
        """
        issues = []
        suggestions = []
        reasoning = ""
        strengths = []
        needs_improvement = False

        # Simple parsing logic for structured feedback
        in_reasoning = False
        in_strengths = False
        in_issues = False
        in_suggestions = False

        for line in critique_text.split("\n"):
            line = line.strip()

            # Section headers
            if line.lower().startswith("reasoning:"):
                in_reasoning = True
                in_strengths = False
                in_issues = False
                in_suggestions = False
                reasoning = line[10:].strip()  # Remove "Reasoning:" prefix
                continue
            elif line.lower().startswith("strengths:"):
                in_reasoning = False
                in_strengths = True
                in_issues = False
                in_suggestions = False
                continue
            elif line.lower().startswith("issues:"):
                in_reasoning = False
                in_strengths = False
                in_issues = True
                in_suggestions = False
                continue
            elif line.lower().startswith("suggestions:"):
                in_reasoning = False
                in_strengths = False
                in_issues = False
                in_suggestions = True
                continue
            elif line.lower().startswith(("overall assessment:", "needs improvement:")):
                in_reasoning = False
                in_strengths = False
                in_issues = False
                in_suggestions = False
                # Check for improvement need
                if "yes" in line.lower() or "needs improvement" in line.lower():
                    needs_improvement = True
                continue
            elif not line or line.startswith("#"):
                continue

            # Extract content from sections
            if in_reasoning and line:
                reasoning += " " + line if reasoning else line
            elif in_strengths and line.startswith("-"):
                strengths.append(line[1:].strip())
            elif in_issues and line.startswith("-"):
                issues.append(line[1:].strip())
            elif in_suggestions and line.startswith("-"):
                suggestions.append(line[1:].strip())

        # Fallback: extract from general content if no structured format found
        if not issues and not suggestions:
            critique_lower = critique_text.lower()
            if any(word in critique_lower for word in ["issue", "problem", "error", "incorrect"]):
                issues.append("Issues identified in evaluation")
            if any(word in critique_lower for word in ["suggest", "improve", "should", "could"]):
                suggestions.append("See evaluation for improvement suggestions")
            if any(word in critique_lower for word in ["poor", "weak", "needs", "improve"]):
                needs_improvement = True

        return {
            "issues": issues,
            "suggestions": suggestions,
            "reasoning": reasoning,
            "strengths": strengths,
            "needs_improvement": needs_improvement,
        }

    def _aggregate_critiques(self, critiques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple critiques using consensus-based approach.

        Args:
            critiques: List of individual critique results.

        Returns:
            Aggregated critique result with consensus information.
        """
        if not critiques:
            return {
                "consensus_issues": [],
                "consensus_suggestions": [],
                "stats": {"total_critiques": 0, "consensus_items": 0},
            }

        # Collect all issues and suggestions
        all_issues = []
        all_suggestions = []
        improvement_votes = []

        for critique in critiques:
            all_issues.extend(critique.get("issues", []))
            all_suggestions.extend(critique.get("suggestions", []))
            improvement_votes.append(critique.get("needs_improvement", False))

        # Find consensus items using similarity-based grouping
        consensus_issues = self._find_consensus_items(all_issues)
        consensus_suggestions = self._find_consensus_items(all_suggestions)

        # Calculate statistics
        total_critiques = len(critiques)
        consensus_items = len(consensus_issues) + len(consensus_suggestions)

        return {
            "consensus_issues": consensus_issues,
            "consensus_suggestions": consensus_suggestions,
            "improvement_votes": improvement_votes,
            "stats": {
                "total_critiques": total_critiques,
                "consensus_items": consensus_items,
                "agreement_ratio": (
                    sum(improvement_votes) / total_critiques if total_critiques > 0 else 0
                ),
            },
        }

    def _find_consensus_items(self, items: List[str]) -> List[Dict[str, Any]]:
        """Find consensus items from a list using frequency and similarity.

        Args:
            items: List of issues or suggestions.

        Returns:
            List of consensus items with frequency information.
        """
        if not items:
            return []

        # Simple frequency-based consensus (can be enhanced with semantic similarity)
        item_counts = Counter(items)
        consensus_items = []

        # Items that appear in multiple critiques
        min_frequency = max(2, int(self.num_iterations * self.consensus_threshold))

        for item, count in item_counts.items():
            if count >= min_frequency:
                consensus_items.append(
                    {
                        "text": item,
                        "frequency": count,
                        "confidence": count / self.num_iterations,
                    }
                )

        # Sort by frequency (most common first)
        consensus_items.sort(key=lambda x: x["frequency"], reverse=True)

        return consensus_items

    def _calculate_confidence(
        self, critiques: List[Dict[str, Any]], aggregated_result: Dict[str, Any]
    ) -> float:
        """Calculate confidence based on agreement between critiques.

        Args:
            critiques: List of individual critique results.
            aggregated_result: Aggregated critique result.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not critiques:
            return 0.0

        # Base confidence from agreement ratio
        agreement_ratio = aggregated_result["stats"]["agreement_ratio"]

        # Adjust based on consensus items
        consensus_items = len(aggregated_result["consensus_issues"]) + len(
            aggregated_result["consensus_suggestions"]
        )
        total_items = sum(
            len(c.get("issues", [])) + len(c.get("suggestions", [])) for c in critiques
        )

        if total_items > 0:
            consensus_ratio = consensus_items / (total_items / len(critiques))
            confidence = (agreement_ratio + consensus_ratio) / 2
        else:
            confidence = agreement_ratio

        # Boost confidence if we have many iterations
        if len(critiques) >= 5:
            confidence = min(1.0, confidence * 1.1)

        return max(0.1, min(1.0, confidence))

    def _determine_improvement_need(
        self, critiques: List[Dict[str, Any]], aggregated_result: Dict[str, Any]
    ) -> bool:
        """Determine if improvement is needed based on consensus.

        Args:
            critiques: List of individual critique results.
            aggregated_result: Aggregated critique result.

        Returns:
            True if improvement is needed based on consensus.
        """
        if not critiques:
            return False

        # Check majority vote for improvement need
        improvement_votes = aggregated_result["improvement_votes"]
        improvement_ratio = sum(improvement_votes) / len(improvement_votes)

        # Need improvement if majority agrees
        if improvement_ratio >= self.consensus_threshold:
            return True

        # Also check if we have consensus issues/suggestions
        consensus_items = len(aggregated_result["consensus_issues"]) + len(
            aggregated_result["consensus_suggestions"]
        )
        if consensus_items > 0:
            return True

        return False

    def _format_consensus_message(
        self, critiques: List[Dict[str, Any]], aggregated_result: Dict[str, Any], confidence: float
    ) -> str:
        """Format the consensus message from multiple critiques.

        Args:
            critiques: List of individual critique results.
            aggregated_result: Aggregated critique result.
            confidence: Confidence score.

        Returns:
            Formatted consensus message.
        """
        num_critiques = len(critiques)
        consensus_issues = aggregated_result["consensus_issues"]
        consensus_suggestions = aggregated_result["consensus_suggestions"]

        message = f"=== Self-Consistency Evaluation ({num_critiques} iterations) ===\n\n"
        message += f"Confidence Level: {confidence:.1%}\n"
        message += f"Agreement Threshold: {self.consensus_threshold:.1%}\n\n"

        if consensus_issues:
            message += "CONSENSUS ISSUES (found in multiple evaluations):\n"
            for issue in consensus_issues:
                freq_pct = (issue["frequency"] / num_critiques) * 100
                message += f"• {issue['text']} (found in {issue['frequency']}/{num_critiques} evaluations, {freq_pct:.0f}%)\n"
            message += "\n"

        if consensus_suggestions:
            message += "CONSENSUS SUGGESTIONS (found in multiple evaluations):\n"
            for suggestion in consensus_suggestions:
                freq_pct = (suggestion["frequency"] / num_critiques) * 100
                message += f"• {suggestion['text']} (found in {suggestion['frequency']}/{num_critiques} evaluations, {freq_pct:.0f}%)\n"
            message += "\n"

        # Add summary statistics
        stats = aggregated_result["stats"]
        message += f"EVALUATION SUMMARY:\n"
        message += f"• Total evaluations: {stats['total_critiques']}\n"
        message += f"• Consensus items: {stats['consensus_items']}\n"
        message += f"• Agreement on improvement need: {stats['agreement_ratio']:.1%}\n"

        message += "\n=== End Self-Consistency Evaluation ==="

        return message

    def improve(self, thought: Thought) -> str:
        """Improve text based on self-consistency critique.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text that addresses consensus feedback.

        Raises:
            ImproverError: If the improvement fails.
        """
        start_time = time.time()

        with critic_context(
            critic_name="SelfConsistencyCritic",
            operation="improve",
            message_prefix="Failed to improve text with Self-Consistency approach",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="SelfConsistencyCritic",
                    operation="improve",
                    suggestions=["Provide text to improve"],
                )

            # Get critique from thought
            consensus_issues = []
            consensus_suggestions = []
            confidence = 0.0
            total_iterations = 0

            if thought.critic_feedback:
                for feedback in thought.critic_feedback:
                    if feedback.critic_name == "SelfConsistencyCritic":
                        metadata = feedback.metadata or {}
                        consensus_issues = metadata.get("consensus_stats", {}).get(
                            "consensus_issues", []
                        )
                        consensus_suggestions = metadata.get("consensus_stats", {}).get(
                            "consensus_suggestions", []
                        )
                        confidence = feedback.confidence
                        total_iterations = metadata.get("num_iterations", 0)
                        break

            # If no critique available, generate one
            if not consensus_issues and not consensus_suggestions:
                logger.debug(
                    "No self-consistency critique found in thought, generating new critique"
                )
                import asyncio

                try:
                    asyncio.get_running_loop()
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._perform_critique_async(thought))
                        critique_result = future.result()
                except RuntimeError:
                    critique_result = asyncio.run(self._perform_critique_async(thought))

                metadata = critique_result["metadata"]
                consensus_issues = critique_result["issues"]
                consensus_suggestions = critique_result["suggestions"]
                confidence = critique_result["confidence"]
                total_iterations = metadata["num_iterations"]

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Format consensus feedback for improvement prompt
            issues_text = (
                "\n".join([f"• {issue['text']}" for issue in consensus_issues])
                if consensus_issues
                else "None identified"
            )
            suggestions_text = (
                "\n".join([f"• {suggestion['text']}" for suggestion in consensus_suggestions])
                if consensus_suggestions
                else "None provided"
            )

            consensus_count = max(len(consensus_issues), len(consensus_suggestions))

            # Create improvement prompt with consensus feedback
            improve_prompt = self.improve_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                context=context,
                consensus_issues=issues_text,
                consensus_suggestions=suggestions_text,
                consensus_count=consensus_count,
                total_iterations=total_iterations,
                confidence=confidence,
            )

            # Generate improved text
            improved_text = self.model.generate(
                prompt=improve_prompt,
                system_prompt="You are an expert editor using consensus feedback from multiple evaluations to improve text quality.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"SelfConsistencyCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    def _extract_consistency_learning_context(self, thought: Thought) -> Dict[str, Any]:
        """Extract learning context from thought for enhanced consistency evaluation.

        Args:
            thought: The Thought to extract learning context from.

        Returns:
            Dictionary with consistency learning context.
        """
        learning_context = {
            "consistency_sessions": 0,
            "reliable_consensus": [],
            "unreliable_consensus": [],
            "consistency_patterns": {},
            "task_type": self._classify_consistency_task_type(thought.prompt),
            "predicted_reliability": 0.5,
        }

        # Extract from thought metadata
        if thought.metadata:
            consistency_data = thought.metadata.get("self_consistency_memory", {})
            if consistency_data:
                learning_context["consistency_sessions"] = len(consistency_data.get("sessions", []))
                learning_context["reliable_consensus"] = consistency_data.get(
                    "reliable_consensus", []
                )[
                    -10:
                ]  # Last 10
                learning_context["unreliable_consensus"] = consistency_data.get(
                    "unreliable_consensus", []
                )[
                    -10:
                ]  # Last 10
                learning_context["consistency_patterns"] = consistency_data.get(
                    "consistency_patterns", {}
                )

        # Extract from thought history
        if thought.history:
            learning_context["previous_attempts"] = len(thought.history)

        # Extract from critic feedback history
        if thought.critic_feedback:
            consistency_feedback = [
                f for f in thought.critic_feedback if f.critic_name == "SelfConsistencyCritic"
            ]
            if consistency_feedback:
                learning_context["previous_feedback_count"] = len(consistency_feedback)
                # Analyze consistency reliability from previous feedback
                reliable_count = 0
                total_count = 0
                for feedback in consistency_feedback[-5:]:  # Last 5 feedback instances
                    if feedback.metadata:
                        confidence = feedback.metadata.get("confidence", 0.5)
                        consensus_stats = feedback.metadata.get("consensus_stats", {})
                        agreement_ratio = consensus_stats.get("agreement_ratio", 0.5)

                        total_count += 1
                        if confidence > 0.7 and agreement_ratio > 0.6:
                            reliable_count += 1
                            learning_context["reliable_consensus"].append(
                                {
                                    "task_type": learning_context["task_type"],
                                    "confidence": confidence,
                                    "agreement_ratio": agreement_ratio,
                                }
                            )
                        elif confidence < 0.4 or agreement_ratio < 0.3:
                            learning_context["unreliable_consensus"].append(
                                {
                                    "task_type": learning_context["task_type"],
                                    "confidence": confidence,
                                    "agreement_ratio": agreement_ratio,
                                }
                            )

                # Predict reliability for this task type
                if total_count > 0:
                    learning_context["predicted_reliability"] = reliable_count / total_count

        return learning_context

    def _classify_consistency_task_type(self, prompt: str) -> str:
        """Classify the task type for consistency learning purposes.

        Args:
            prompt: The task prompt to classify.

        Returns:
            String representing the consistency task type.
        """
        prompt_lower = prompt.lower()

        # Tasks where consistency is typically high/low
        if any(word in prompt_lower for word in ["objective", "fact", "data", "statistic"]):
            return "objective"
        elif any(
            word in prompt_lower for word in ["subjective", "opinion", "creative", "artistic"]
        ):
            return "subjective"
        elif any(
            word in prompt_lower for word in ["technical", "code", "programming", "specification"]
        ):
            return "technical"
        elif any(word in prompt_lower for word in ["analysis", "evaluate", "assess", "critique"]):
            return "analytical"
        elif any(word in prompt_lower for word in ["summary", "summarize", "brief", "overview"]):
            return "summary"
        elif any(word in prompt_lower for word in ["explain", "describe", "define", "what is"]):
            return "explanatory"
        elif any(word in prompt_lower for word in ["compare", "contrast", "versus", "vs"]):
            return "comparative"
        else:
            return "general"

    async def _generate_multiple_critiques_with_learning_async(
        self, thought: Thought, learning_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate multiple critiques with learning-informed approach.

        Args:
            thought: The Thought container with the text to critique.
            learning_context: Learning context from past consistency evaluations.

        Returns:
            List of critique results from multiple iterations.
        """
        # Adjust number of iterations based on predicted reliability
        predicted_reliability = learning_context.get("predicted_reliability", 0.5)
        task_type = learning_context.get("task_type", "general")

        # Adaptive iteration count based on learning
        if predicted_reliability > 0.8:
            # High reliability expected, fewer iterations needed
            adaptive_iterations = max(3, self.num_iterations - 1)
            logger.debug(
                f"High reliability predicted for {task_type}, using {adaptive_iterations} iterations"
            )
        elif predicted_reliability < 0.3:
            # Low reliability expected, more iterations needed
            adaptive_iterations = min(10, self.num_iterations + 2)
            logger.debug(
                f"Low reliability predicted for {task_type}, using {adaptive_iterations} iterations"
            )
        else:
            adaptive_iterations = self.num_iterations

        # Use the original method with adaptive iteration count
        original_iterations = self.num_iterations
        self.num_iterations = adaptive_iterations

        try:
            critiques = await self._generate_multiple_critiques_async(thought)
        finally:
            # Restore original iteration count
            self.num_iterations = original_iterations

        return critiques

    def _aggregate_critiques_with_learning(
        self, critiques: List[Dict[str, Any]], learning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate critiques using enhanced consensus with learning.

        Args:
            critiques: List of individual critique results.
            learning_context: Learning context from past consistency evaluations.

        Returns:
            Aggregated critique result with enhanced consensus information.
        """
        # Start with base aggregation
        base_result = self._aggregate_critiques(critiques)

        # Enhance with learning-based adjustments
        task_type = learning_context.get("task_type", "general")
        consistency_patterns = learning_context.get("consistency_patterns", {})

        # Adjust consensus threshold based on task type patterns
        if task_type in consistency_patterns:
            pattern = consistency_patterns[task_type]
            avg_reliability = pattern.get("avg_reliability", 0.5)

            if avg_reliability > 0.7:
                # High reliability task type, can be more lenient with consensus
                adjusted_threshold = max(0.4, self.consensus_threshold - 0.1)
            elif avg_reliability < 0.3:
                # Low reliability task type, need stricter consensus
                adjusted_threshold = min(0.8, self.consensus_threshold + 0.1)
            else:
                adjusted_threshold = self.consensus_threshold

            logger.debug(f"Adjusted consensus threshold for {task_type}: {adjusted_threshold:.2f}")

            # Re-find consensus items with adjusted threshold
            original_threshold = self.consensus_threshold
            self.consensus_threshold = adjusted_threshold

            try:
                # Recalculate consensus with adjusted threshold
                all_issues = []
                all_suggestions = []
                for critique in critiques:
                    all_issues.extend(critique.get("issues", []))
                    all_suggestions.extend(critique.get("suggestions", []))

                enhanced_consensus_issues = self._find_consensus_items(all_issues)
                enhanced_consensus_suggestions = self._find_consensus_items(all_suggestions)

                base_result["consensus_issues"] = enhanced_consensus_issues
                base_result["consensus_suggestions"] = enhanced_consensus_suggestions
                base_result["stats"]["adjusted_threshold"] = adjusted_threshold

            finally:
                # Restore original threshold
                self.consensus_threshold = original_threshold

        return base_result

    def _calculate_confidence_with_learning(
        self,
        critiques: List[Dict[str, Any]],
        aggregated_result: Dict[str, Any],
        learning_context: Dict[str, Any],
    ) -> float:
        """Calculate confidence with learning from past patterns.

        Args:
            critiques: List of individual critique results.
            aggregated_result: Aggregated critique result.
            learning_context: Learning context from past evaluations.

        Returns:
            Enhanced confidence score between 0.0 and 1.0.
        """
        # Start with base confidence
        base_confidence = self._calculate_confidence(critiques, aggregated_result)

        # Adjust based on learning patterns
        task_type = learning_context.get("task_type", "general")
        predicted_reliability = learning_context.get("predicted_reliability", 0.5)

        # Adjust confidence based on predicted reliability for this task type
        if predicted_reliability > 0.7:
            # This task type typically has reliable consensus
            enhanced_confidence = min(1.0, base_confidence * 1.1)
        elif predicted_reliability < 0.3:
            # This task type typically has unreliable consensus
            enhanced_confidence = max(0.1, base_confidence * 0.9)
        else:
            enhanced_confidence = base_confidence

        # Further adjust based on consistency patterns
        consistency_patterns = learning_context.get("consistency_patterns", {})
        if task_type in consistency_patterns:
            pattern = consistency_patterns[task_type]
            avg_confidence = pattern.get("avg_confidence", 0.5)

            # If this task type typically has different confidence levels, adjust accordingly
            confidence_adjustment = (avg_confidence - 0.5) * 0.2  # Small adjustment
            enhanced_confidence = max(0.1, min(1.0, enhanced_confidence + confidence_adjustment))

        logger.debug(
            f"Enhanced confidence for {task_type}: {enhanced_confidence:.3f} (base: {base_confidence:.3f})"
        )

        return enhanced_confidence

    def _determine_improvement_need_with_learning(
        self,
        critiques: List[Dict[str, Any]],
        aggregated_result: Dict[str, Any],
        learning_context: Dict[str, Any],
    ) -> bool:
        """Determine improvement need with learning from past patterns.

        Args:
            critiques: List of individual critique results.
            aggregated_result: Aggregated critique result.
            learning_context: Learning context from past evaluations.

        Returns:
            Enhanced determination of whether improvement is needed.
        """
        # Start with base determination
        base_need = self._determine_improvement_need(critiques, aggregated_result)

        # Adjust based on learning patterns
        task_type = learning_context.get("task_type", "general")
        predicted_reliability = learning_context.get("predicted_reliability", 0.5)

        # If reliability is very low, be more conservative about improvement recommendations
        if predicted_reliability < 0.3:
            # Low reliability consensus, only recommend improvement if very strong agreement
            improvement_votes = aggregated_result["improvement_votes"]
            strong_agreement_ratio = (
                sum(improvement_votes) / len(improvement_votes) if improvement_votes else 0
            )

            if strong_agreement_ratio < 0.8:  # Need very strong agreement
                logger.debug(
                    f"Low reliability predicted for {task_type}, requiring stronger consensus for improvement"
                )
                return False

        return base_need

    def _store_consistency_outcomes(
        self,
        thought: Thought,
        learning_context: Dict[str, Any],
        critiques: List[Dict[str, Any]],
        aggregated_result: Dict[str, Any],
        confidence: float,
    ) -> None:
        """Store consistency outcomes in thought metadata for future learning.

        Args:
            thought: The Thought to store outcomes in.
            learning_context: The learning context used.
            critiques: The critiques generated.
            aggregated_result: The aggregated result.
            confidence: The final confidence score.
        """
        if not thought.metadata:
            thought.metadata = {}

        # Initialize self-consistency memory if not exists
        if "self_consistency_memory" not in thought.metadata:
            thought.metadata["self_consistency_memory"] = {
                "sessions": [],
                "reliable_consensus": [],
                "unreliable_consensus": [],
                "consistency_patterns": {},
            }

        # Analyze this consistency session
        task_type = learning_context.get("task_type", "general")
        agreement_ratio = aggregated_result["stats"]["agreement_ratio"]
        consensus_items = len(aggregated_result["consensus_issues"]) + len(
            aggregated_result["consensus_suggestions"]
        )

        session_data = {
            "session_id": f"consistency_session_{int(time.time())}",
            "task_type": task_type,
            "num_critiques": len(critiques),
            "confidence": confidence,
            "agreement_ratio": agreement_ratio,
            "consensus_items": consensus_items,
            "predicted_reliability": learning_context.get("predicted_reliability", 0.5),
            "timestamp": time.time(),
        }

        # Determine if this was a reliable or unreliable consensus
        is_reliable = confidence > 0.7 and agreement_ratio > 0.6

        if is_reliable:
            thought.metadata["self_consistency_memory"]["reliable_consensus"].append(
                {
                    "task_type": task_type,
                    "confidence": confidence,
                    "agreement_ratio": agreement_ratio,
                    "consensus_items": consensus_items,
                }
            )
        else:
            thought.metadata["self_consistency_memory"]["unreliable_consensus"].append(
                {
                    "task_type": task_type,
                    "confidence": confidence,
                    "agreement_ratio": agreement_ratio,
                    "consensus_items": consensus_items,
                }
            )

        # Update consistency patterns for this task type
        if task_type not in thought.metadata["self_consistency_memory"]["consistency_patterns"]:
            thought.metadata["self_consistency_memory"]["consistency_patterns"][task_type] = {
                "sessions": 0,
                "reliable_sessions": 0,
                "total_confidence": 0.0,
                "total_agreement": 0.0,
            }

        patterns = thought.metadata["self_consistency_memory"]["consistency_patterns"][task_type]
        patterns["sessions"] += 1
        patterns["total_confidence"] += confidence
        patterns["total_agreement"] += agreement_ratio

        if is_reliable:
            patterns["reliable_sessions"] += 1

        # Calculate averages
        patterns["avg_reliability"] = patterns["reliable_sessions"] / patterns["sessions"]
        patterns["avg_confidence"] = patterns["total_confidence"] / patterns["sessions"]
        patterns["avg_agreement"] = patterns["total_agreement"] / patterns["sessions"]

        # Store this session
        thought.metadata["self_consistency_memory"]["sessions"].append(session_data)

        # Keep only last 20 sessions
        if len(thought.metadata["self_consistency_memory"]["sessions"]) > 20:
            thought.metadata["self_consistency_memory"]["sessions"] = thought.metadata[
                "self_consistency_memory"
            ]["sessions"][-20:]

        # Keep only last 30 reliable/unreliable consensus records
        if len(thought.metadata["self_consistency_memory"]["reliable_consensus"]) > 30:
            thought.metadata["self_consistency_memory"]["reliable_consensus"] = thought.metadata[
                "self_consistency_memory"
            ]["reliable_consensus"][-30:]

        if len(thought.metadata["self_consistency_memory"]["unreliable_consensus"]) > 30:
            thought.metadata["self_consistency_memory"]["unreliable_consensus"] = thought.metadata[
                "self_consistency_memory"
            ]["unreliable_consensus"][-30:]
