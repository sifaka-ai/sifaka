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
from typing import Any, List, Optional

from pydantic_ai import Agent

from sifaka.core.thought import Thought
from sifaka.critics.base_pydantic import PydanticAICritic
from sifaka.utils.error_handling import critic_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class MetaRewardingCritic(PydanticAICritic):
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
        model_name: str,
        meta_judge_model_name: Optional[str] = None,
        judgment_criteria: Optional[List[str]] = None,
        meta_judgment_criteria: Optional[List[str]] = None,
        use_scoring: bool = True,
        score_range: tuple[int, int] = (1, 10),
        **agent_kwargs: Any,
    ):
        """Initialize the Meta-Rewarding critic.

        Args:
            model_name: The model name for the PydanticAI agent (e.g., "openai:gpt-4")
            meta_judge_model_name: Optional separate model for meta-judgment.
            judgment_criteria: Criteria for judging response quality.
            meta_judgment_criteria: Criteria for judging judgment quality.
            use_scoring: Whether to use numerical scoring in judgments.
            score_range: Range for numerical scores (min, max).
            **agent_kwargs: Additional arguments passed to the PydanticAI agent.
        """
        # Initialize parent with system prompt
        super().__init__(model_name=model_name, **agent_kwargs)

        # Set up meta-judge model (can be same as main model or different)
        self.meta_judge_model_name = meta_judge_model_name or model_name

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

        # Create meta-judge agent with structured output
        from sifaka.models.critic_results import CritiqueFeedback

        self.meta_judge_agent = Agent(
            model=self.meta_judge_model_name,
            output_type=CritiqueFeedback,  # Meta-judgment now returns structured output
            system_prompt=self._get_meta_judge_system_prompt(),
        )

        logger.info(
            f"Initialized MetaRewardingCritic with model={model_name}, "
            f"meta_judge={self.meta_judge_model_name}, scoring={use_scoring}"
        )

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for initial judgment."""
        scoring_instruction = ""
        if self.use_scoring:
            min_score, max_score = self.score_range
            scoring_instruction = f" Include a numerical score from {min_score} to {max_score}."

        return f"""You are an expert evaluator providing detailed, fair assessment of text quality using Meta-Rewarding principles. Your role is to provide structured feedback for the initial judgment stage.

You must return a CritiqueFeedback object with these REQUIRED fields:
- message: A clear summary of your meta-rewarding evaluation (string)
- needs_improvement: Whether the text needs improvement based on initial judgment (boolean)
- confidence: ConfidenceScore with overall confidence (object with 'overall' field as float 0.0-1.0)
- critic_name: Set this to "MetaRewardingCritic" (string)

And these OPTIONAL fields (can be empty lists or null):
- violations: List of ViolationReport objects for identified issues
- suggestions: List of ImprovementSuggestion objects for addressing issues
- processing_time_ms: Time taken in milliseconds (can be null)
- critic_version: Version string (can be null)
- metadata: Additional metadata dictionary (can be empty)

IMPORTANT: Always provide the required fields. For confidence, use a simple object like {{"overall": 0.8}} where the number is between 0.0 and 1.0.

Focus on Meta-Rewarding principles:
1. Thorough evaluation against specified criteria
2. Identification of specific strengths and weaknesses
3. Constructive feedback for improvement
4. Fair and objective assessment{scoring_instruction}

This is the initial judgment stage - be thorough and specific in your evaluation."""

    def _get_meta_judge_system_prompt(self) -> str:
        """Get the system prompt for the meta-judge."""
        return """You are a meta-judge evaluating the quality of evaluations. Your role is to assess how well the previous judgment was conducted and provide improved assessment.

You must return a CritiqueFeedback object with these REQUIRED fields:
- message: A clear summary of your meta-judgment evaluation (string)
- needs_improvement: Whether the original text needs improvement based on meta-analysis (boolean)
- confidence: ConfidenceScore with overall confidence (object with 'overall' field as float 0.0-1.0)
- critic_name: Set this to "MetaRewardingCritic" (string)

And these OPTIONAL fields (can be empty lists or null):
- violations: List of ViolationReport objects for identified issues in the original text
- suggestions: List of ImprovementSuggestion objects for addressing issues in the original text
- processing_time_ms: Time taken in milliseconds (can be null)
- critic_version: Version string (can be null)
- metadata: Additional metadata dictionary (can be empty)

IMPORTANT: Always provide the required fields. For confidence, use a simple object like {{"overall": 0.8}} where the number is between 0.0 and 1.0.

Focus on:
1. Quality and thoroughness of the evaluation
2. Accuracy of identified strengths and weaknesses
3. Constructiveness and specificity of feedback
4. Consistency with stated evaluation criteria
5. Fairness and objectivity of the judgment

Be critical and thorough in evaluating the judgment process itself. Identify what the judgment did well and how it could be improved. Your assessment should focus on the original text quality, not just the judgment quality."""

    async def _create_critique_prompt(self, thought: Thought) -> str:
        """Create the critique prompt for Meta-Rewarding evaluation.

        This implements the two-stage judgment process:
        1. Initial judgment of the text
        2. Meta-judgment of the judgment quality

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            The formatted critique prompt for initial judgment.
        """
        # Format criteria for the prompt
        criteria_text = "\n".join(
            f"{i + 1}. {criterion}" for i, criterion in enumerate(self.judgment_criteria)
        )

        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Get validation context if available
        validation_context = self._get_validation_context_dict(thought)
        validation_text = ""
        if validation_context:
            validation_text = f"\n\nValidation Context:\n{validation_context}"

        scoring_instruction = ""
        if self.use_scoring:
            min_score, max_score = self.score_range
            scoring_instruction = (
                f"\nScore: [Provide a numerical score from {min_score} to {max_score}]"
            )

        return f"""Evaluate the following response against the specified criteria using Meta-Rewarding principles.

Original task: {thought.prompt}

Response to evaluate:
{thought.text}

Retrieved context:
{context}
{validation_text}

Evaluation Criteria:
{criteria_text}

Please provide a thorough evaluation including:

Strengths:
- [List specific strengths of the response]

Weaknesses:
- [List specific areas for improvement]

Overall Assessment: [Detailed assessment of response quality]{scoring_instruction}

Meta-Rewarding Stage: This is the initial judgment stage. Be specific, constructive, and fair in your evaluation. Your judgment will be evaluated by a meta-judge."""

    async def critique_async(self, thought: Thought):
        """Perform Meta-Rewarding critique with two-stage judgment process.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A CriticResult with structured feedback from both judgment stages.
        """
        start_time = time.time()

        with critic_context(
            critic_name=self.__class__.__name__,
            operation="critique_async",
            message_prefix=f"Failed to critique text with {self.__class__.__name__}",
        ):
            try:
                # Stage 1: Initial judgment using the main agent
                initial_result = await super().critique_async(thought)
                initial_judgment = initial_result.feedback.message

                # Stage 2: Meta-judgment (judge the quality of the initial judgment)
                meta_judgment_feedback = await self._generate_meta_judgment(
                    thought, initial_judgment
                )

                # Stage 3: Combine judgments into final assessment
                combined_feedback = self._combine_judgments(
                    initial_result.feedback, initial_judgment, meta_judgment_feedback
                )

                # Calculate total processing time
                processing_time = (time.time() - start_time) * 1000

                # Create final result with combined feedback
                from sifaka.models.critic_results import CriticResult

                return CriticResult(
                    feedback=combined_feedback,
                    operation_type="critique_async",
                    success=True,
                    total_processing_time_ms=processing_time,
                    model_calls=2,  # Initial judgment + meta-judgment
                    input_text_length=len(thought.text or ""),
                    validation_context=self._get_validation_context_dict(thought),
                )

            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                logger.error(f"MetaRewardingCritic critique failed: {e}")

                # Return error result
                from sifaka.models.critic_results import CriticResult, CritiqueFeedback

                error_feedback = CritiqueFeedback(
                    message=f"Critique failed: {str(e)}",
                    needs_improvement=True,
                    confidence={"overall": 0.0},
                    critic_name=self.__class__.__name__,
                )

                return CriticResult(
                    feedback=error_feedback,
                    operation_type="critique_async",
                    success=False,
                    error_message=str(e),
                    error_type=type(e).__name__,
                    total_processing_time_ms=processing_time,
                    model_calls=0,
                    input_text_length=len(thought.text or ""),
                )

    async def _generate_meta_judgment(self, thought: Thought, initial_judgment: str):
        """Generate the meta-judgment (judgment of the judgment).

        Args:
            thought: The Thought container with the original text.
            initial_judgment: The initial judgment to evaluate.

        Returns:
            The meta-judgment CritiqueFeedback object.
        """
        # Format meta-criteria for the prompt
        meta_criteria_text = "\n".join(
            f"{i + 1}. {criterion}" for i, criterion in enumerate(self.meta_judgment_criteria)
        )

        # Create meta-judgment prompt
        meta_judgment_prompt = f"""Now evaluate the quality of the judgment itself. You are acting as a meta-judge, assessing how well the previous evaluation was conducted.

Original task: {thought.prompt}

Original response: {thought.text}

Previous judgment: {initial_judgment}

Meta-evaluation Criteria:
{meta_criteria_text}

Please provide a structured meta-judgment that includes:

1. Assessment of the judgment quality (what it did well and how it could be improved)
2. Missing aspects the judgment overlooked
3. Your improved evaluation of the original response
4. Overall assessment of whether the original text needs improvement

Focus on providing a comprehensive evaluation of the original text quality based on your meta-analysis of the judgment process."""

        # Generate meta-judgment using meta-judge agent with structured output
        result = await self.meta_judge_agent.run(meta_judgment_prompt)
        return result.output

    def _combine_judgments(self, initial_feedback, initial_judgment: str, meta_judgment_feedback):
        """Combine initial judgment and meta-judgment into final assessment.

        Args:
            initial_feedback: The initial CritiqueFeedback object.
            initial_judgment: The initial judgment text.
            meta_judgment_feedback: The meta-judgment CritiqueFeedback object.

        Returns:
            Combined CritiqueFeedback object.
        """
        # Create comprehensive feedback message
        combined_message = f"""Meta-Rewarding Assessment:

Initial Judgment:
{initial_judgment}

Meta-Judgment (Evaluation of the Judgment):
{meta_judgment_feedback.message}

This assessment uses the Meta-Rewarding approach with two-stage judgment process for improved evaluation quality."""

        # Use meta-judgment confidence as it's based on deeper analysis
        confidence = (
            meta_judgment_feedback.confidence.overall if meta_judgment_feedback.confidence else 0.5
        )

        # Create enhanced metadata
        enhanced_metadata = dict(initial_feedback.metadata or {})
        enhanced_metadata.update(
            {
                "initial_judgment": initial_judgment,
                "meta_judgment": meta_judgment_feedback.message,
                "meta_judgment_confidence": (
                    meta_judgment_feedback.confidence.overall
                    if meta_judgment_feedback.confidence
                    else None
                ),
                "judgment_criteria": self.judgment_criteria,
                "meta_judgment_criteria": self.meta_judgment_criteria,
                "meta_judge_model": self.meta_judge_model_name,
                "two_stage_process": True,
            }
        )

        # Create new feedback with combined information
        from sifaka.models.critic_results import CritiqueFeedback

        # Combine violations and suggestions from both judgments
        combined_violations = list(initial_feedback.violations or [])
        if meta_judgment_feedback.violations:
            combined_violations.extend(meta_judgment_feedback.violations)

        combined_suggestions = list(initial_feedback.suggestions or [])
        if meta_judgment_feedback.suggestions:
            combined_suggestions.extend(meta_judgment_feedback.suggestions)

        # Use meta-judgment's needs_improvement as it's based on deeper analysis
        needs_improvement = meta_judgment_feedback.needs_improvement

        return CritiqueFeedback(
            message=combined_message,
            needs_improvement=needs_improvement,
            violations=combined_violations,
            suggestions=combined_suggestions,
            confidence={"overall": confidence},
            critic_name=self.__class__.__name__,
            critic_version=initial_feedback.critic_version,
            processing_time_ms=initial_feedback.processing_time_ms,
            timestamp=initial_feedback.timestamp,
            metadata=enhanced_metadata,
        )
