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

The MetaRewardingCritic implements key Meta-Rewarding concepts:
1. Two-stage judgment process (judge responses, then judge judgments)
2. Self-improving alignment through meta-evaluation
3. Feedback loop for improving both responses and judgment capabilities
4. Unsupervised self-improvement without human supervision
5. Learning from meta-judgment accuracy patterns and effectiveness (enhanced)
6. Adaptive meta-judgment strategies based on past performance (enhanced)

Note: This implementation captures core Meta-Rewarding principles with enhanced
learning capabilities through integration with the Sifaka thoughts system.
The critic learns when meta-judgments improve vs. worsen initial judgments and
adapts its meta-evaluation strategies accordingly.
"""

import time
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class MetaRewardingCritic(BaseCritic):
    """Critic that implements Meta-Rewarding with two-stage judgment process.

    This critic uses the Meta-Rewarding approach which first judges the response
    quality, then judges the quality of that judgment itself (meta-judgment).
    This creates a feedback loop that can improve both response quality and
    judgment capabilities.
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
        """Perform the actual critique logic using Meta-Rewarding approach.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Extract learning context from thought for enhanced meta-judgment
        learning_context = self._extract_meta_learning_context(thought)

        # Stage 1: Initial judgment (either using base critic or self-judgment)
        if self.base_critic:
            # Use provided base critic for initial judgment
            initial_judgment_result = await self.base_critic._critique_async(thought)
            initial_judgment = initial_judgment_result["message"]
        else:
            # Generate initial judgment using our own model with learning context
            initial_judgment = await self._generate_initial_judgment_with_learning_async(
                thought, learning_context
            )

        # Stage 2: Enhanced meta-judgment with learning from past patterns
        meta_judgment = await self._generate_meta_judgment_with_learning_async(
            thought, initial_judgment, learning_context
        )

        # Stage 3: Combine judgments with learning-enhanced assessment
        combined_result = self._combine_judgments_with_learning(
            thought, initial_judgment, meta_judgment, learning_context
        )

        # Store meta-learning outcomes for future improvements
        self._store_meta_learning_outcomes(
            thought, learning_context, initial_judgment, meta_judgment, combined_result
        )

        logger.debug(
            "MetaRewardingCritic: Completed two-stage judgment process with learning integration"
        )

        return combined_result

    async def _generate_initial_judgment_async(self, thought: Thought) -> str:
        """Generate the initial judgment of the response.

        Args:
            thought: The Thought container with the text to judge.

        Returns:
            The initial judgment text.
        """
        # Format criteria for the prompt
        criteria_text = "\n".join(
            f"{i+1}. {criterion}" for i, criterion in enumerate(self.judgment_criteria)
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

        # Generate initial judgment
        judgment_response = await self.model._generate_async(
            prompt=judgment_prompt,
            system_message="You are an expert evaluator providing detailed, fair assessment of text quality.",
        )

        return judgment_response

    async def _generate_meta_judgment_async(self, thought: Thought, initial_judgment: str) -> str:
        """Generate the meta-judgment (judgment of the judgment).

        Args:
            thought: The Thought container with the original text.
            initial_judgment: The initial judgment to evaluate.

        Returns:
            The meta-judgment text.
        """
        # Format meta-criteria for the prompt
        meta_criteria_text = "\n".join(
            f"{i+1}. {criterion}" for i, criterion in enumerate(self.meta_judgment_criteria)
        )

        # Create meta-judgment prompt
        meta_judgment_prompt = self.meta_critique_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            initial_judgment=initial_judgment,
            meta_criteria=meta_criteria_text,
        )

        # Generate meta-judgment using meta-judge model
        meta_judgment_response = await self.meta_judge_model._generate_async(
            prompt=meta_judgment_prompt,
            system_message="You are a meta-judge evaluating the quality of evaluations. Be critical and thorough.",
        )

        return meta_judgment_response

    def _combine_judgments(
        self, thought: Thought, initial_judgment: str, meta_judgment: str
    ) -> Dict[str, Any]:
        """Combine initial judgment and meta-judgment into final assessment.

        Args:
            thought: The original thought.
            initial_judgment: The initial judgment text.
            meta_judgment: The meta-judgment text.

        Returns:
            Combined critique result dictionary.
        """
        # Extract issues and suggestions from both judgments
        initial_issues, initial_suggestions = self._parse_judgment(initial_judgment)
        meta_issues, meta_suggestions = self._parse_judgment(meta_judgment)

        # Combine issues and suggestions
        all_issues = initial_issues + meta_issues
        all_suggestions = initial_suggestions + meta_suggestions

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

    def improve(self, thought: Thought) -> str:
        """Improve text based on meta-rewarding critique.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text that addresses both initial and meta-judgment feedback.

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

            # If no critique available, generate one
            if not initial_judgment or not meta_judgment:
                logger.debug("No meta-rewarding critique found in thought, generating new critique")
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
                initial_judgment = metadata["initial_judgment"]
                meta_judgment = metadata["meta_judgment"]

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Create improvement prompt with both judgments
            improve_prompt = self.improve_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                context=context,
                initial_judgment=initial_judgment,
                meta_judgment=meta_judgment,
            )

            # Generate improved text
            improved_text = self.model.generate(
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
            elif line.lower().startswith("missing aspects:"):
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
            if in_weaknesses and line.startswith("-"):
                issues.append(line[1:].strip())
            elif (in_weaknesses or in_missing) and line.startswith("-"):
                suggestions.append(f"Address: {line[1:].strip()}")

        # Fallback: extract from general content if no structured format found
        if not issues and not suggestions:
            judgment_lower = judgment.lower()
            if any(word in judgment_lower for word in ["weakness", "issue", "problem", "improve"]):
                issues.append("Issues identified in meta-rewarding evaluation")
            if any(word in judgment_lower for word in ["suggest", "should", "could", "recommend"]):
                suggestions.append("See meta-rewarding feedback for improvement suggestions")

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

    def _extract_meta_learning_context(self, thought: Thought) -> Dict[str, Any]:
        """Extract learning context from thought for enhanced meta-judgment.

        Args:
            thought: The Thought to extract learning context from.

        Returns:
            Dictionary with meta-learning context.
        """
        learning_context = {
            "meta_sessions": 0,
            "effective_meta_judgments": [],
            "ineffective_meta_judgments": [],
            "meta_patterns": {},
            "task_type": self._classify_meta_task_type(thought.prompt),
            "predicted_meta_effectiveness": 0.5,
        }

        # Extract from thought metadata
        if thought.metadata:
            meta_data = thought.metadata.get("meta_rewarding_memory", {})
            if meta_data:
                learning_context["meta_sessions"] = len(meta_data.get("sessions", []))
                learning_context["effective_meta_judgments"] = meta_data.get(
                    "effective_meta_judgments", []
                )[
                    -10:
                ]  # Last 10
                learning_context["ineffective_meta_judgments"] = meta_data.get(
                    "ineffective_meta_judgments", []
                )[
                    -10:
                ]  # Last 10
                learning_context["meta_patterns"] = meta_data.get("meta_patterns", {})

        # Extract from thought history
        if thought.history:
            learning_context["previous_attempts"] = len(thought.history)

        # Extract from critic feedback history
        if thought.critic_feedback:
            meta_feedback = [
                f for f in thought.critic_feedback if f.critic_name == "MetaRewardingCritic"
            ]
            if meta_feedback:
                learning_context["previous_feedback_count"] = len(meta_feedback)
                # Analyze meta-judgment effectiveness from previous feedback
                effective_count = 0
                total_count = 0
                for feedback in meta_feedback[-5:]:  # Last 5 feedback instances
                    if feedback.metadata:
                        confidence = feedback.metadata.get("confidence", 0.5)
                        initial_score = feedback.metadata.get("initial_score")

                        total_count += 1
                        # Consider meta-judgment effective if it resulted in high confidence
                        if confidence > 0.7:
                            effective_count += 1
                            learning_context["effective_meta_judgments"].append(
                                {
                                    "task_type": learning_context["task_type"],
                                    "confidence": confidence,
                                    "initial_score": initial_score,
                                }
                            )
                        elif confidence < 0.4:
                            learning_context["ineffective_meta_judgments"].append(
                                {
                                    "task_type": learning_context["task_type"],
                                    "confidence": confidence,
                                    "initial_score": initial_score,
                                }
                            )

                # Predict meta-judgment effectiveness for this task type
                if total_count > 0:
                    learning_context["predicted_meta_effectiveness"] = effective_count / total_count

        return learning_context

    def _classify_meta_task_type(self, prompt: str) -> str:
        """Classify the task type for meta-judgment learning purposes.

        Args:
            prompt: The task prompt to classify.

        Returns:
            String representing the meta-judgment task type.
        """
        prompt_lower = prompt.lower()

        # Tasks where meta-judgment is typically more/less effective
        if any(
            word in prompt_lower for word in ["complex", "nuanced", "sophisticated", "advanced"]
        ):
            return "complex"
        elif any(word in prompt_lower for word in ["simple", "basic", "straightforward", "clear"]):
            return "simple"
        elif any(
            word in prompt_lower for word in ["creative", "artistic", "imaginative", "original"]
        ):
            return "creative"
        elif any(word in prompt_lower for word in ["analytical", "critical", "evaluate", "assess"]):
            return "analytical"
        elif any(
            word in prompt_lower for word in ["technical", "code", "programming", "specification"]
        ):
            return "technical"
        elif any(
            word in prompt_lower for word in ["subjective", "opinion", "personal", "preference"]
        ):
            return "subjective"
        elif any(word in prompt_lower for word in ["objective", "fact", "data", "evidence"]):
            return "objective"
        else:
            return "general"

    async def _generate_initial_judgment_with_learning_async(
        self, thought: Thought, learning_context: Dict[str, Any]
    ) -> str:
        """Generate initial judgment with learning-informed approach.

        Args:
            thought: The Thought container with the text to judge.
            learning_context: Learning context from past meta-judgments.

        Returns:
            The enhanced initial judgment text.
        """
        # Use the original method but with enhanced criteria based on learning
        task_type = learning_context.get("task_type", "general")
        predicted_effectiveness = learning_context.get("predicted_meta_effectiveness", 0.5)

        # Adjust judgment criteria based on what works for this task type
        enhanced_criteria = self.judgment_criteria.copy()

        # Add task-specific criteria based on learning
        if task_type == "complex" and predicted_effectiveness > 0.7:
            enhanced_criteria.append("Depth and sophistication of analysis")
        elif task_type == "creative" and predicted_effectiveness > 0.6:
            enhanced_criteria.append("Originality and creative expression")
        elif task_type == "technical" and predicted_effectiveness > 0.6:
            enhanced_criteria.append("Technical accuracy and precision")

        # Format enhanced criteria for the prompt
        criteria_text = "\n".join(
            f"{i+1}. {criterion}" for i, criterion in enumerate(enhanced_criteria)
        )

        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Create enhanced judgment prompt
        judgment_prompt = self.critique_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            context=context,
            criteria=criteria_text,
        )

        # Add learning context to the prompt if available
        if learning_context.get("meta_sessions", 0) > 3:
            learning_note = f"\n\nNote: This is a {task_type} task. Based on past experience, focus on criteria that typically benefit from meta-evaluation."
            judgment_prompt += learning_note

        # Generate initial judgment
        judgment_response = await self.model._generate_async(
            prompt=judgment_prompt,
            system_message="You are an expert evaluator providing detailed, fair assessment with awareness of meta-evaluation patterns.",
        )

        return judgment_response

    async def _generate_meta_judgment_with_learning_async(
        self, thought: Thought, initial_judgment: str, learning_context: Dict[str, Any]
    ) -> str:
        """Generate meta-judgment with learning from past patterns.

        Args:
            thought: The Thought container with the original text.
            initial_judgment: The initial judgment to evaluate.
            learning_context: Learning context from past meta-judgments.

        Returns:
            The enhanced meta-judgment text.
        """
        task_type = learning_context.get("task_type", "general")
        predicted_effectiveness = learning_context.get("predicted_meta_effectiveness", 0.5)

        # Adjust meta-judgment approach based on predicted effectiveness
        enhanced_meta_criteria = self.meta_judgment_criteria.copy()

        if predicted_effectiveness < 0.3:
            # Meta-judgment typically not effective for this task type, be more conservative
            enhanced_meta_criteria = [
                "Basic accuracy of the evaluation",
                "Presence of major evaluation errors",
                "Overall reasonableness of the judgment",
            ]
            logger.debug(
                f"Using conservative meta-judgment for {task_type} (low predicted effectiveness)"
            )
        elif predicted_effectiveness > 0.7:
            # Meta-judgment typically very effective, be more thorough
            enhanced_meta_criteria.extend(
                [
                    "Nuanced understanding of evaluation quality",
                    "Identification of subtle judgment biases",
                    "Sophisticated meta-cognitive assessment",
                ]
            )
            logger.debug(
                f"Using thorough meta-judgment for {task_type} (high predicted effectiveness)"
            )

        # Format enhanced meta-criteria for the prompt
        meta_criteria_text = "\n".join(
            f"{i+1}. {criterion}" for i, criterion in enumerate(enhanced_meta_criteria)
        )

        # Create enhanced meta-judgment prompt
        meta_judgment_prompt = self.meta_critique_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            initial_judgment=initial_judgment,
            meta_criteria=meta_criteria_text,
        )

        # Add learning context to the prompt
        if learning_context.get("meta_sessions", 0) > 2:
            effectiveness_note = f"\n\nLearning Context: Meta-judgment for {task_type} tasks has {predicted_effectiveness:.1%} effectiveness rate. "
            if predicted_effectiveness > 0.6:
                effectiveness_note += (
                    "Focus on detailed meta-evaluation as it typically improves judgment quality."
                )
            else:
                effectiveness_note += "Be conservative as meta-judgment may not significantly improve initial judgment."
            meta_judgment_prompt += effectiveness_note

        # Generate meta-judgment using meta-judge model
        meta_judgment_response = await self.meta_judge_model._generate_async(
            prompt=meta_judgment_prompt,
            system_message="You are a meta-judge with learning from past meta-evaluation patterns. Adapt your approach based on task effectiveness.",
        )

        return meta_judgment_response

    def _combine_judgments_with_learning(
        self,
        thought: Thought,
        initial_judgment: str,
        meta_judgment: str,
        learning_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine judgments with learning-enhanced assessment.

        Args:
            thought: The original thought.
            initial_judgment: The initial judgment text.
            meta_judgment: The meta-judgment text.
            learning_context: Learning context from past meta-judgments.

        Returns:
            Enhanced combined critique result dictionary.
        """
        # Start with base combination
        base_result = self._combine_judgments(thought, initial_judgment, meta_judgment)

        # Enhance with learning-based adjustments
        task_type = learning_context.get("task_type", "general")
        predicted_effectiveness = learning_context.get("predicted_meta_effectiveness", 0.5)

        # Adjust confidence based on predicted meta-judgment effectiveness
        base_confidence = base_result["confidence"]
        if predicted_effectiveness > 0.7:
            # Meta-judgment typically effective, boost confidence
            enhanced_confidence = min(1.0, base_confidence * 1.1)
        elif predicted_effectiveness < 0.3:
            # Meta-judgment typically not effective, reduce confidence boost
            enhanced_confidence = max(0.1, base_confidence * 0.9)
        else:
            enhanced_confidence = base_confidence

        # Add learning metadata
        base_result["confidence"] = enhanced_confidence
        base_result["metadata"]["learning_applied"] = bool(learning_context.get("meta_patterns"))
        base_result["metadata"]["task_type"] = task_type
        base_result["metadata"]["predicted_meta_effectiveness"] = predicted_effectiveness

        logger.debug(
            f"Enhanced meta-rewarding confidence for {task_type}: {enhanced_confidence:.3f} (base: {base_confidence:.3f})"
        )

        return base_result

    def _store_meta_learning_outcomes(
        self,
        thought: Thought,
        learning_context: Dict[str, Any],
        initial_judgment: str,
        meta_judgment: str,
        combined_result: Dict[str, Any],
    ) -> None:
        """Store meta-learning outcomes in thought metadata for future improvements.

        Args:
            thought: The Thought to store outcomes in.
            learning_context: The learning context used.
            initial_judgment: The initial judgment generated.
            meta_judgment: The meta-judgment generated.
            combined_result: The combined result.
        """
        if not thought.metadata:
            thought.metadata = {}

        # Initialize meta-rewarding memory if not exists
        if "meta_rewarding_memory" not in thought.metadata:
            thought.metadata["meta_rewarding_memory"] = {
                "sessions": [],
                "effective_meta_judgments": [],
                "ineffective_meta_judgments": [],
                "meta_patterns": {},
            }

        # Analyze this meta-rewarding session
        task_type = learning_context.get("task_type", "general")
        confidence = combined_result["confidence"]
        initial_score = combined_result["metadata"].get("initial_score")

        session_data = {
            "session_id": f"meta_session_{int(time.time())}",
            "task_type": task_type,
            "confidence": confidence,
            "initial_score": initial_score,
            "predicted_effectiveness": learning_context.get("predicted_meta_effectiveness", 0.5),
            "timestamp": time.time(),
        }

        # Determine if meta-judgment was effective
        # Consider effective if confidence is high and meta-judgment added value
        meta_added_value = (
            len(meta_judgment) > len(initial_judgment) * 0.3
        )  # Meta-judgment is substantial
        is_effective = confidence > 0.7 and meta_added_value

        if is_effective:
            thought.metadata["meta_rewarding_memory"]["effective_meta_judgments"].append(
                {
                    "task_type": task_type,
                    "confidence": confidence,
                    "initial_score": initial_score,
                    "meta_length_ratio": len(meta_judgment) / max(len(initial_judgment), 1),
                }
            )
        else:
            thought.metadata["meta_rewarding_memory"]["ineffective_meta_judgments"].append(
                {
                    "task_type": task_type,
                    "confidence": confidence,
                    "initial_score": initial_score,
                    "reason": "low_confidence" if confidence < 0.5 else "minimal_meta_value",
                }
            )

        # Update meta-patterns for this task type
        if task_type not in thought.metadata["meta_rewarding_memory"]["meta_patterns"]:
            thought.metadata["meta_rewarding_memory"]["meta_patterns"][task_type] = {
                "sessions": 0,
                "effective_sessions": 0,
                "total_confidence": 0.0,
                "total_meta_value": 0.0,
            }

        patterns = thought.metadata["meta_rewarding_memory"]["meta_patterns"][task_type]
        patterns["sessions"] += 1
        patterns["total_confidence"] += confidence

        if is_effective:
            patterns["effective_sessions"] += 1

        # Calculate meta-value (how much meta-judgment added)
        meta_value = len(meta_judgment) / max(len(initial_judgment), 1)
        patterns["total_meta_value"] += meta_value

        # Calculate averages
        patterns["avg_effectiveness"] = patterns["effective_sessions"] / patterns["sessions"]
        patterns["avg_confidence"] = patterns["total_confidence"] / patterns["sessions"]
        patterns["avg_meta_value"] = patterns["total_meta_value"] / patterns["sessions"]

        # Store this session
        thought.metadata["meta_rewarding_memory"]["sessions"].append(session_data)

        # Keep only last 20 sessions
        if len(thought.metadata["meta_rewarding_memory"]["sessions"]) > 20:
            thought.metadata["meta_rewarding_memory"]["sessions"] = thought.metadata[
                "meta_rewarding_memory"
            ]["sessions"][-20:]

        # Keep only last 30 effective/ineffective meta-judgments
        if len(thought.metadata["meta_rewarding_memory"]["effective_meta_judgments"]) > 30:
            thought.metadata["meta_rewarding_memory"]["effective_meta_judgments"] = (
                thought.metadata["meta_rewarding_memory"]["effective_meta_judgments"][-30:]
            )

        if len(thought.metadata["meta_rewarding_memory"]["ineffective_meta_judgments"]) > 30:
            thought.metadata["meta_rewarding_memory"]["ineffective_meta_judgments"] = (
                thought.metadata["meta_rewarding_memory"]["ineffective_meta_judgments"][-30:]
            )
