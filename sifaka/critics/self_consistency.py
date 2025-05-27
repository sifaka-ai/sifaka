"""Self-Consistency critic for Sifaka.

This module implements a Self-Consistency approach for text critique and improvement,
where multiple critiques are generated for the same text and majority voting is used to
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

The SelfConsistencyCritic implements the core Self-Consistency algorithm:
1. Multiple critique generation for the same text using chain-of-thought reasoning
2. Majority voting to select the most consistent feedback
3. Confidence scoring based on agreement level

Note: This implementation follows the original Self-Consistency paper closely,
using simple majority voting over multiple reasoning paths without additional
learning mechanisms that were not part of the original research.
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
        num_iterations: int = 5,
        use_chain_of_thought: bool = True,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Self-Consistency critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            num_iterations: Number of critique iterations to generate (default: 5).
            use_chain_of_thought: Whether to use chain-of-thought prompting.
            critique_prompt_template: Template for the critique prompt.
            improve_prompt_template: Template for the improvement prompt.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        super().__init__(model=model, model_name=model_name, **model_kwargs)

        # Configuration parameters
        self.num_iterations = max(3, num_iterations)  # Minimum 3 for meaningful consensus
        self.use_chain_of_thought = use_chain_of_thought

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
        # Generate multiple critiques following original Self-Consistency algorithm
        critiques = await self._generate_multiple_critiques_async(thought)

        # Aggregate critiques using majority voting
        aggregated_result = self._aggregate_critiques(critiques)

        # Calculate confidence based on agreement
        confidence = self._calculate_confidence(critiques, aggregated_result)

        # Determine if improvement is needed based on majority vote
        needs_improvement = self._determine_improvement_need(critiques, aggregated_result)

        # Create comprehensive feedback message
        combined_message = self._format_consensus_message(critiques, aggregated_result, confidence)

        logger.debug(f"SelfConsistencyCritic: Completed {len(critiques)} critique iterations")

        return {
            "needs_improvement": needs_improvement,
            "message": combined_message,
            "issues": aggregated_result["consensus_issues"],
            "suggestions": aggregated_result["consensus_suggestions"],
            "confidence": confidence,
            "metadata": {
                "num_iterations": len(critiques),
                "individual_critiques": critiques,
                "consensus_stats": aggregated_result["stats"],
                "use_chain_of_thought": self.use_chain_of_thought,
            },
        }

    async def _generate_multiple_critiques_async(self, thought: Thought) -> List[Dict[str, Any]]:
        """Generate multiple critiques of the same text.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            List of critique results from multiple iterations.
        """
        # Generate critiques using our own model (following original Self-Consistency)
        tasks = [self._generate_single_critique_async(thought) for _ in range(self.num_iterations)]
        critique_results = await asyncio.gather(*tasks, return_exceptions=True)

        critiques = []
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
                suggestions=["Check model availability"],
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
        """Aggregate multiple critiques using majority voting (original Self-Consistency).

        Args:
            critiques: List of individual critique results.

        Returns:
            Aggregated critique result with majority vote information.
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

        # Find majority consensus items using simple frequency counting
        consensus_issues = self._find_majority_items(all_issues)
        consensus_suggestions = self._find_majority_items(all_suggestions)

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

    def _find_majority_items(self, items: List[str]) -> List[Dict[str, Any]]:
        """Find majority items from a list using frequency counting.

        Args:
            items: List of issues or suggestions.

        Returns:
            List of majority items with frequency information.
        """
        if not items:
            return []

        # Simple frequency-based majority voting (original Self-Consistency approach)
        item_counts = Counter(items)
        majority_items = []

        # Items that appear in majority of critiques (more than half)
        min_frequency = max(2, (self.num_iterations + 1) // 2)

        for item, count in item_counts.items():
            if count >= min_frequency:
                majority_items.append(
                    {
                        "text": item,
                        "frequency": count,
                        "confidence": count / self.num_iterations,
                    }
                )

        # Sort by frequency (most common first)
        majority_items.sort(key=lambda x: x["frequency"], reverse=True)

        return majority_items

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
        """Determine if improvement is needed based on majority vote.

        Args:
            critiques: List of individual critique results.
            aggregated_result: Aggregated critique result.

        Returns:
            True if improvement is needed based on majority vote.
        """
        if not critiques:
            return False

        # Simple majority vote for improvement need (original Self-Consistency)
        improvement_votes = aggregated_result["improvement_votes"]
        improvement_ratio = sum(improvement_votes) / len(improvement_votes)

        # Need improvement if majority agrees (more than half)
        return improvement_ratio > 0.5

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
        message += f"Majority Threshold: >50% (original Self-Consistency)\n\n"

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
