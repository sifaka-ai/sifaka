"""Meta-Rewarding critic for Sifaka.

This module implements a Meta-Rewarding approach for text critique and improvement,
where the model judges its own responses and then judges its own judgments to
improve both response quality and judgment capabilities.

Based on "Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge":
https://arxiv.org/abs/2407.19594

@misc{wu2024metarewardinglanguagemodels,
      title={Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge},
      author={Tianhao Wu and Weizhe Yuan and Olga Golovneva and Jing Xu and Yuandong Tian and Jiantao Jiao and Jason E Weston and Sainbayar Sukhbaatar},
      year={2024},
      eprint={2407.19594},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.19594},
}

The MetaRewardingCritic implements core Meta-Rewarding concepts:
1. Two-stage judgment process (judge responses, then judge judgments)
2. Self-improving alignment through meta-evaluation
3. Feedback loop for improving both responses and judgment capabilities
4. Unsupervised self-improvement without human supervision

Note: This implementation adapts the core Meta-Rewarding two-stage judgment process
for text critique. The original Meta-Rewarding paper focuses on training methodology
for improving model alignment, while this implementation applies the judgment
concepts to single-text improvement scenarios.
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


class MetaRewardingCritic(BaseCritic, ValidationAwareMixin):
    """Critic that implements Meta-Rewarding with two-stage judgment process and validation awareness.

    This critic uses the Meta-Rewarding approach which first judges the response
    quality, then judges the quality of that judgment itself (meta-judgment).
    This creates a feedback loop that can improve both response quality and
    judgment capabilities.

    Enhanced with validation context awareness to prioritize validation constraints
    over conflicting meta-rewarding suggestions.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        base_critic: Optional[BaseCritic] = None,
        meta_judge_model: Optional[Model] = None,
        meta_judge_model_name: Optional[str] = None,
        judgment_criteria: Optional[List[str]] = None,
        meta_judgment_criteria: Optional[List[str]] = None,
        use_scoring: bool = True,
        score_range: tuple[int, int] = (1, 10),
        critique_prompt_template: Optional[str] = None,
        meta_critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Meta-Rewarding critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            base_critic: Optional base critic to use for initial judgment.
            meta_judge_model: Optional separate model for meta-judgment.
            meta_judge_model_name: Name of meta-judge model if meta_judge_model not provided.
            judgment_criteria: Criteria for judging response quality.
            meta_judgment_criteria: Criteria for judging judgment quality.
            use_scoring: Whether to use numerical scoring in judgments.
            score_range: Range for numerical scores (min, max).
            critique_prompt_template: Template for the initial critique prompt.
            meta_critique_prompt_template: Template for the meta-critique prompt.
            improve_prompt_template: Template for the improvement prompt.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        super().__init__(model=model, model_name=model_name, **model_kwargs)

        # Set up base critic (if provided) or use self-judgment
        self.base_critic = base_critic

        # Set up meta-judge model (can be same as main model or different)
        if meta_judge_model is not None:
            self.meta_judge_model = meta_judge_model
        elif meta_judge_model_name is not None:
            from sifaka.models.base import create_model

            self.meta_judge_model = create_model(meta_judge_model_name, **model_kwargs)
        else:
            self.meta_judge_model = self.model  # Use same model for meta-judgment

        # Set up judgment criteria
        self.judgment_criteria = judgment_criteria or [
            "Accuracy and factual correctness",
            "Helpfulness and relevance to the task",
            "Clarity and coherence of expression",
            "Completeness of the response",
            "Appropriateness of tone and style",
        ]

        self.meta_judgment_criteria = meta_judgment_criteria or [
            "Quality and thoroughness of the evaluation",
            "Accuracy of identified strengths and weaknesses",
            "Constructiveness and specificity of feedback",
            "Consistency with stated evaluation criteria",
            "Fairness and objectivity of the judgment",
        ]

        self.use_scoring = use_scoring
        self.score_range = score_range

        # Set up prompt templates
        self.critique_prompt_template = (
            critique_prompt_template or self._default_critique_template()
        )
        self.meta_critique_prompt_template = (
            meta_critique_prompt_template or self._default_meta_critique_template()
        )
        self.improve_prompt_template = improve_prompt_template or self._default_improve_template()

    def _default_critique_template(self) -> str:
        """Default template for initial judgment."""
        scoring_instruction = ""
        if self.use_scoring:
            min_score, max_score = self.score_range
            scoring_instruction = (
                f"\nScore: [Provide a numerical score from {min_score} to {max_score}]"
            )

        return (
            "Evaluate the following response against the specified criteria.\n\n"
            "Original task: {prompt}\n\n"
            "Response to evaluate:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Evaluation Criteria:\n{criteria}\n\n"
            "Please provide a thorough evaluation including:\n\n"
            "Strengths:\n- [List specific strengths of the response]\n\n"
            "Weaknesses:\n- [List specific areas for improvement]\n\n"
            "Overall Assessment: [Detailed assessment of response quality]"
            f"{scoring_instruction}\n\n"
            "Be specific, constructive, and fair in your evaluation."
        )

    def _default_meta_critique_template(self) -> str:
        """Default template for meta-judgment."""
        return (
            "Now evaluate the quality of the judgment itself. You are acting as a meta-judge, "
            "assessing how well the previous evaluation was conducted.\n\n"
            "Original task: {prompt}\n\n"
            "Original response: {text}\n\n"
            "Previous judgment: {initial_judgment}\n\n"
            "Meta-evaluation Criteria:\n{meta_criteria}\n\n"
            "Please assess the judgment quality:\n\n"
            "Judgment Strengths:\n- [What the judgment did well]\n\n"
            "Judgment Weaknesses:\n- [How the judgment could be improved]\n\n"
            "Missing Aspects:\n- [Important aspects the judgment overlooked]\n\n"
            "Revised Assessment: [Your improved evaluation of the original response]\n\n"
            "Meta-judgment: [Overall assessment of the judgment quality]\n\n"
            "Be critical and thorough in evaluating the judgment process itself."
        )

    def _default_improve_template(self) -> str:
        """Default template for improvement."""
        return (
            "Improve the following text based on the comprehensive feedback provided.\n\n"
            "Original task: {prompt}\n\n"
            "Current text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Initial judgment:\n{initial_judgment}\n\n"
            "Meta-judgment (evaluation of the judgment):\n{meta_judgment}\n\n"
            "Please provide an improved version that:\n"
            "1. Addresses the issues identified in both the initial judgment and meta-judgment\n"
            "2. Builds on the strengths noted in the evaluations\n"
            "3. Incorporates insights from the meta-evaluation process\n"
            "4. Maintains relevance to the original task\n"
            "5. Uses available context information effectively\n\n"
            "Improved text:"
        )

    async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform the actual critique logic using Meta-Rewarding approach (async).

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Stage 1: Initial judgment (either using base critic or self-judgment)
        if self.base_critic:
            # Use provided base critic for initial judgment
            initial_judgment_result = await self.base_critic.critique_async(thought)
            initial_judgment = initial_judgment_result["message"]
        else:
            # Generate initial judgment using our own model
            initial_judgment = await self._generate_initial_judgment_async(thought)

        # Stage 2: Meta-judgment (judge the quality of the initial judgment)
        meta_judgment = await self._generate_meta_judgment_async(thought, initial_judgment)

        # Stage 3: Combine judgments into final assessment
        combined_result = self._combine_judgments(initial_judgment, meta_judgment)

        logger.debug("MetaRewardingCritic: Completed async two-stage judgment process")

        return combined_result

    async def _generate_initial_judgment_async(self, thought: Thought) -> str:
        """Generate the initial judgment of the response (async).

        Args:
            thought: The Thought container with the text to judge.

        Returns:
            The initial judgment text.
        """
        # Format criteria for the prompt
        criteria_text = "\n".join(
            f"{i + 1}. {criterion}" for i, criterion in enumerate(self.judgment_criteria)
        )

        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Create initial judgment prompt
        judgment_prompt = self.critique_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            context=context,
            criteria=criteria_text,
        )

        # Generate initial judgment (async only)
        judgment_response = await self.model._generate_async(
            prompt=judgment_prompt,
            system_prompt="You are an expert evaluator providing detailed, fair assessment of text quality.",
        )

        return judgment_response

    async def _generate_meta_judgment_async(self, thought: Thought, initial_judgment: str) -> str:
        """Generate the meta-judgment (judgment of the judgment) (async).

        Args:
            thought: The Thought container with the original text.
            initial_judgment: The initial judgment to evaluate.

        Returns:
            The meta-judgment text.
        """
        # Format meta-criteria for the prompt
        meta_criteria_text = "\n".join(
            f"{i + 1}. {criterion}" for i, criterion in enumerate(self.meta_judgment_criteria)
        )

        # Create meta-judgment prompt
        meta_judgment_prompt = self.meta_critique_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            initial_judgment=initial_judgment,
            meta_criteria=meta_criteria_text,
        )

        # Generate meta-judgment using meta-judge model (async only)
        meta_judgment_response = await self.meta_judge_model._generate_async(
            prompt=meta_judgment_prompt,
            system_prompt="You are a meta-judge evaluating the quality of evaluations. Be critical and thorough.",
        )

        return meta_judgment_response

    def _combine_judgments(self, initial_judgment: str, meta_judgment: str) -> Dict[str, Any]:
        """Combine initial judgment and meta-judgment into final assessment.

        Args:
            initial_judgment: The initial judgment text.
            meta_judgment: The meta-judgment text.

        Returns:
            Combined critique result dictionary.
        """
        # Extract issues and suggestions primarily from initial judgment
        # The meta judgment evaluates the judgment itself, not the original text
        initial_issues, initial_suggestions = self._parse_judgment(initial_judgment)

        # Use initial judgment issues/suggestions as the primary feedback
        all_issues = initial_issues
        all_suggestions = initial_suggestions

        # Extract scores if scoring is enabled
        initial_score = self._extract_score(initial_judgment) if self.use_scoring else None

        # Determine if improvement is needed based on various factors
        needs_improvement = self._determine_improvement_need(
            initial_judgment, meta_judgment, initial_score, all_issues
        )

        # Calculate confidence based on consistency between judgments
        confidence = self._calculate_confidence(initial_judgment, meta_judgment)

        # Create comprehensive feedback message
        combined_message = self._format_combined_message(initial_judgment, meta_judgment)

        return {
            "needs_improvement": needs_improvement,
            "message": combined_message,
            "issues": all_issues,
            "suggestions": all_suggestions,
            "confidence": confidence,
            "metadata": {
                "initial_judgment": initial_judgment,
                "meta_judgment": meta_judgment,
                "initial_score": initial_score,
                "judgment_criteria": self.judgment_criteria,
                "meta_judgment_criteria": self.meta_judgment_criteria,
                "base_critic_used": self.base_critic is not None,
                "meta_judge_model": self.meta_judge_model.__class__.__name__,
            },
        }

    async def improve_async(self, thought: Thought) -> str:
        """Improve text based on meta-rewarding critique asynchronously.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text that addresses both initial and meta-judgment feedback.

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
            critic_name="MetaRewardingCritic",
            operation="improve",
            message_prefix="Failed to improve text with Meta-Rewarding approach",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="MetaRewardingCritic",
                    operation="improve",
                    suggestions=["Provide text to improve"],
                )

            # Get critique from thought
            initial_judgment = ""
            meta_judgment = ""
            if thought.critic_feedback:
                for feedback in thought.critic_feedback:
                    if feedback.critic_name == "MetaRewardingCritic":
                        metadata = feedback.metadata or {}
                        initial_judgment = metadata.get("initial_judgment", "")
                        meta_judgment = metadata.get("meta_judgment", "")
                        break

            # If no critique available, generate one using async method
            if not initial_judgment or not meta_judgment:
                logger.debug("No meta-rewarding critique found in thought, generating new critique")
                critique_result = await self._perform_critique_async(thought)
                metadata = critique_result["metadata"]
                initial_judgment = metadata["initial_judgment"]
                meta_judgment = metadata["meta_judgment"]

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Create improvement prompt with validation awareness
            if validation_context:
                # Create a critique string for the enhanced prompt
                critique = (
                    f"Initial Judgment:\n{initial_judgment}\n\nMeta-Judgment:\n{meta_judgment}"
                )

                # Use enhanced prompt with validation awareness
                improve_prompt = self._create_enhanced_improvement_prompt(
                    prompt=thought.prompt,
                    text=thought.text,
                    critique=critique,
                    context=context,
                    validation_context=validation_context,
                    critic_suggestions=[],  # MetaRewardingCritic doesn't have structured suggestions
                )
            else:
                # Use original prompt template
                improve_prompt = self.improve_prompt_template.format(
                    prompt=thought.prompt,
                    text=thought.text,
                    context=context,
                    initial_judgment=initial_judgment,
                    meta_judgment=meta_judgment,
                )

            # Generate improved text (async only)
            improved_text = await self.model._generate_async(
                prompt=improve_prompt,
                system_prompt="You are an expert editor using meta-rewarding feedback to improve text quality.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"MetaRewardingCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    def _parse_judgment(self, judgment: str) -> tuple[List[str], List[str]]:
        """Parse judgment text to extract issues and suggestions.

        Args:
            judgment: The judgment text to parse.

        Returns:
            A tuple of (issues, suggestions) lists.
        """
        issues = []
        suggestions = []

        # Simple parsing logic for structured feedback
        in_weaknesses = False
        in_missing = False

        for line in judgment.split("\n"):
            line = line.strip()

            # Section headers
            if line.lower().startswith(("strengths:", "judgment strengths:")):
                in_weaknesses = False
                in_missing = False
                continue
            elif line.lower().startswith(("weaknesses:", "judgment weaknesses:")):
                in_weaknesses = True
                in_missing = False
                continue
            elif line.lower().startswith(
                ("missing aspects:", "to improve:", "recommendations:", "suggestions:")
            ):
                in_weaknesses = False
                in_missing = True
                continue
            elif line.lower().startswith(
                ("overall assessment:", "revised assessment:", "meta-judgment:")
            ):
                in_weaknesses = False
                in_missing = False
                continue
            elif not line or line.startswith("#"):
                continue

            # Extract content from sections
            if in_weaknesses and (line.startswith("-") or line.startswith("*")):
                # Extract the issue text, handling both "- text" and "* **Title:** text" formats
                if line.startswith("* **") and ":**" in line:
                    # Format: "* **Title:** description"
                    issue_text = line.split(":**", 1)[1].strip()
                    if issue_text:
                        issues.append(issue_text)
                else:
                    # Format: "- text" or "* text"
                    issue_text = line[1:].strip()
                    if issue_text:
                        issues.append(issue_text)
            elif (in_weaknesses or in_missing) and (line.startswith("-") or line.startswith("*")):
                # Extract suggestions similarly
                if line.startswith("* **") and ":**" in line:
                    suggestion_text = line.split(":**", 1)[1].strip()
                    if suggestion_text:
                        suggestions.append(f"Address: {suggestion_text}")
                else:
                    suggestion_text = line[1:].strip()
                    if suggestion_text:
                        suggestions.append(f"Address: {suggestion_text}")

        # Fallback: extract actual content from unstructured judgment
        if not issues and not suggestions:
            # Split judgment into sentences for better parsing
            sentences = [s.strip() for s in judgment.replace("\n", ". ").split(".") if s.strip()]

            for sentence in sentences:
                sentence_lower = sentence.lower()

                # Extract issues from sentences containing negative indicators
                if any(
                    word in sentence_lower
                    for word in [
                        "weakness",
                        "issue",
                        "problem",
                        "lacking",
                        "insufficient",
                        "unclear",
                        "confusing",
                    ]
                ):
                    if len(sentence) > 10:  # Avoid very short fragments
                        issues.append(sentence.strip())

                # Extract suggestions from sentences containing improvement indicators
                elif any(
                    word in sentence_lower
                    for word in [
                        "should",
                        "could",
                        "recommend",
                        "suggest",
                        "consider",
                        "improve",
                        "enhance",
                        "add",
                        "include",
                    ]
                ):
                    if len(sentence) > 10:  # Avoid very short fragments
                        suggestions.append(sentence.strip())

            # If still no specific feedback found, use the first few sentences as general feedback
            if not issues and not suggestions and sentences:
                # Take the first 2-3 meaningful sentences as suggestions
                meaningful_sentences = [s for s in sentences[:3] if len(s) > 20]
                if meaningful_sentences:
                    suggestions.extend(meaningful_sentences)

        return issues, suggestions

    def _extract_score(self, judgment: str) -> Optional[float]:
        """Extract numerical score from judgment text.

        Args:
            judgment: The judgment text to parse.

        Returns:
            The extracted score or None if not found.
        """
        if not self.use_scoring:
            return None

        import re

        # Look for score patterns like "Score: 7" or "Score: 7/10"
        score_patterns = [
            r"score:\s*(\d+(?:\.\d+)?)",
            r"score\s*=\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*/\s*\d+",
            r"rating:\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in score_patterns:
            match = re.search(pattern, judgment.lower())
            if match:
                try:
                    score = float(match.group(1))
                    min_score, max_score = self.score_range
                    # Clamp score to valid range
                    return max(min_score, min(max_score, score))
                except (ValueError, IndexError):
                    continue

        return None

    def _determine_improvement_need(
        self,
        initial_judgment: str,
        meta_judgment: str,
        initial_score: Optional[float],
        all_issues: List[str],
    ) -> bool:
        """Determine if improvement is needed based on judgments.

        Args:
            initial_judgment: The initial judgment text.
            meta_judgment: The meta-judgment text.
            initial_score: The extracted score from initial judgment.
            all_issues: Combined list of issues from both judgments.

        Returns:
            True if improvement is needed, False otherwise.
        """
        # Check if there are explicit issues identified
        if all_issues:
            return True

        # Check score-based criteria
        if initial_score is not None:
            min_score, max_score = self.score_range
            threshold = min_score + (max_score - min_score) * 0.7  # 70% threshold
            if initial_score < threshold:
                return True

        # Check for improvement keywords in judgments
        improvement_keywords = [
            "improve",
            "weakness",
            "issue",
            "problem",
            "lacking",
            "insufficient",
        ]
        combined_text = (initial_judgment + " " + meta_judgment).lower()

        if any(keyword in combined_text for keyword in improvement_keywords):
            return True

        # Check if meta-judgment suggests the initial judgment was inadequate
        meta_keywords = ["inadequate", "insufficient", "missed", "overlooked", "incomplete"]
        if any(keyword in meta_judgment.lower() for keyword in meta_keywords):
            return True

        return False

    def _calculate_confidence(self, initial_judgment: str, meta_judgment: str) -> float:
        """Calculate confidence based on consistency between judgments.

        Args:
            initial_judgment: The initial judgment text.
            meta_judgment: The meta-judgment text.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Base confidence
        confidence = 0.7

        # Check for consistency indicators in meta-judgment
        positive_meta_indicators = ["thorough", "accurate", "comprehensive", "well-evaluated"]
        negative_meta_indicators = ["missed", "overlooked", "inadequate", "incomplete", "biased"]

        meta_lower = meta_judgment.lower()
        initial_lower = initial_judgment.lower()

        # Increase confidence if meta-judgment is positive about initial judgment
        positive_count = sum(1 for indicator in positive_meta_indicators if indicator in meta_lower)
        negative_count = sum(1 for indicator in negative_meta_indicators if indicator in meta_lower)

        # Check for consistency in tone between judgments
        if any(word in initial_lower for word in ["good", "excellent", "strong"]) and any(
            word in meta_lower for word in ["accurate", "thorough"]
        ):
            confidence += 0.1
        elif any(word in initial_lower for word in ["poor", "weak", "inadequate"]) and any(
            word in meta_lower for word in ["missed", "incomplete"]
        ):
            confidence += 0.05

        # Adjust confidence based on meta-judgment assessment
        confidence += positive_count * 0.1
        confidence -= negative_count * 0.15

        # Ensure confidence is within valid range
        return max(0.1, min(1.0, confidence))

    def _format_combined_message(self, initial_judgment: str, meta_judgment: str) -> str:
        """Format the combined message from both judgments.

        Args:
            initial_judgment: The initial judgment text.
            meta_judgment: The meta-judgment text.

        Returns:
            Formatted combined message.
        """
        return (
            "=== Meta-Rewarding Evaluation ===\n\n"
            "INITIAL JUDGMENT:\n"
            f"{initial_judgment}\n\n"
            "META-JUDGMENT (Evaluation of the Judgment):\n"
            f"{meta_judgment}\n\n"
            "=== End Meta-Rewarding Evaluation ==="
        )
