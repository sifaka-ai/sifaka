"""N-Critics ensemble critic for Sifaka.

This module implements the N-Critics ensemble approach for multi-perspective
text evaluation and improvement through diverse critical viewpoints.

Based on "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics":
https://arxiv.org/abs/2310.18679

@misc{tian2023ncriticsselfrefinelargelanguage,
      title={N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics},
      author={Xiaoyu Tian and Xiang Chen and Zhigang Kan and Ningyu Zhang and Huajun Chen},
      year={2023},
      eprint={2310.18679},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.18679},
}

The NCriticsCritic implements the N-Critics ensemble methodology:
1. Multi-perspective evaluation from diverse critical viewpoints
2. Ensemble-based critique aggregation
3. Specialized focus areas for comprehensive coverage
4. Consensus-building for reliable feedback

IMPORTANT IMPLEMENTATION NOTES AND CAVEATS:

This implementation adapts the N-Critics ensemble approach for single-model
critique by simulating multiple critical perspectives within a single agent.
The original paper uses multiple separate critic models, but we adapt this
to work within Sifaka's architecture using perspective-based prompting.

The original N-Critics paper demonstrates that using an ensemble of critics
with different specializations leads to more comprehensive and reliable
text evaluation compared to single-critic approaches. Each critic focuses
on specific aspects (clarity, accuracy, completeness, style) and their
feedback is aggregated for final assessment.

CAVEATS AND LIMITATIONS:
1. This is a single-model simulation of the multi-model ensemble approach
   described in the original paper, which may not capture the full diversity
   of perspectives that separate models would provide.
2. The perspective-based prompting may not fully replicate the specialized
   knowledge and focus that dedicated critic models would have.
3. Consensus building is simplified compared to the sophisticated aggregation
   methods described in the original research.
4. The approach may be more computationally expensive than single-perspective
   critics due to the need to consider multiple viewpoints.
5. Quality depends on the underlying model's ability to adopt and maintain
   different critical perspectives consistently.

ENSEMBLE PERSPECTIVES:
This implementation uses 4 core critical perspectives:
1. Clarity Critic: Focus on readability, structure, and comprehensibility
2. Accuracy Critic: Focus on factual correctness and logical consistency
3. Completeness Critic: Focus on thoroughness and adequate coverage
4. Style Critic: Focus on tone, voice, and appropriateness for audience

RETRIEVAL AUGMENTATION:
This critic supports optional retrieval augmentation to enhance each critical
perspective with external context, examples, or domain-specific knowledge
during the ensemble evaluation process.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.thought import SifakaThought
from sifaka.critics.base import BaseCritic, CritiqueFeedback


class NCriticsCritic(BaseCritic):
    """N-Critics ensemble critic implementing Tian et al. 2023 methodology.

    This critic can operate in two modes:
    1. Single-model mode: Simulates multiple perspectives within one model (default)
    2. Multi-model mode: Uses separate models for each critical perspective (true ensemble)

    Each perspective focuses on specific aspects while contributing to an
    overall assessment through ensemble aggregation.

    Enhanced with validation context awareness to ensure all critical perspectives
    consider validation requirements in their specialized evaluations.
    """

    def __init__(
        self,
        model_name: str = "groq:llama-3.1-8b-instant",
        critic_perspectives: Optional[List[str]] = None,
        perspective_models: Optional[Dict[str, str]] = None,
        retrieval_tools: Optional[List[Any]] = None,
        auto_discover_tools: bool = False,
        tool_categories: Optional[List[str]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the N-Critics ensemble critic.

        Args:
            model_name: The model name for the PydanticAI agent (used in single-model mode)
            critic_perspectives: Custom critic perspectives (uses defaults if None)
            perspective_models: Dict mapping perspective names to model names for multi-model mode.
                              If provided, enables true multi-model ensemble. Format:
                              {"Clarity": "openai:gpt-4o-mini", "Accuracy": "anthropic:claude-3-5-haiku-latest", ...}
            retrieval_tools: Optional list of retrieval tools for RAG support
            auto_discover_tools: If True, automatically discover and use all available tools
            tool_categories: Optional list of tool categories to include when auto-discovering
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        self.critic_perspectives = critic_perspectives or self._get_default_perspectives()
        self.perspective_models = perspective_models
        self.is_multi_model = perspective_models is not None

        # Store individual agents for multi-model mode
        self.perspective_agents = {}

        if self.is_multi_model:
            # Multi-model mode: create separate agents for each perspective
            self._create_perspective_agents(
                perspective_models,
                retrieval_tools,
                auto_discover_tools,
                tool_categories,
                **agent_kwargs,
            )
            # Use first model as primary for base class
            primary_model = next(iter(perspective_models.values()))
            system_prompt = self._create_aggregation_prompt()
        else:
            # Single-model mode: use traditional approach
            system_prompt = self._create_system_prompt()

        paper_reference = (
            "Tian, X., Chen, X., Kan, Z., Zhang, N., & Chen, H. (2023). "
            "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics. "
            "arXiv preprint arXiv:2310.18679. https://arxiv.org/abs/2310.18679"
        )

        mode_description = (
            "true multi-model ensemble" if self.is_multi_model else "single-model simulation"
        )
        methodology = (
            f"N-Critics ensemble methodology: Multi-perspective evaluation through diverse critical viewpoints. "
            f"Implemented as {mode_description} with specialized critics for comprehensive text assessment."
        )

        super().__init__(
            model_name=primary_model if self.is_multi_model else model_name,
            system_prompt=system_prompt,
            paper_reference=paper_reference,
            methodology=methodology,
            retrieval_tools=retrieval_tools,
            auto_discover_tools=auto_discover_tools,
            tool_categories=tool_categories,
            **agent_kwargs,
        )

    def _create_perspective_agents(
        self,
        perspective_models: Dict[str, str],
        retrieval_tools,
        auto_discover_tools,
        tool_categories,
        **agent_kwargs,
    ):
        """Create individual agents for each perspective in multi-model mode."""
        from pydantic_ai import Agent

        # Handle tool discovery
        if auto_discover_tools:
            from sifaka.tools import discover_all_tools

            discovered_tools = discover_all_tools(categories=tool_categories)
            tools = (retrieval_tools or []) + discovered_tools
        else:
            tools = retrieval_tools or []

        # Create agent for each perspective
        for perspective_name, model_name in perspective_models.items():
            system_prompt = self._create_perspective_prompt(perspective_name)
            self.perspective_agents[perspective_name] = Agent(
                model=model_name,
                output_type=CritiqueFeedback,
                system_prompt=system_prompt,
                tools=tools,
                **agent_kwargs,
            )

    def _create_perspective_prompt(self, perspective_name: str) -> str:
        """Create a specialized system prompt for a specific perspective."""
        perspective_descriptions = {
            "Clarity": "You are a Clarity Critic specializing in readability, structure, organization, and comprehensibility. Focus on how well the text communicates its message to the target audience.",
            "Accuracy": "You are an Accuracy Critic specializing in factual correctness, logical consistency, and evidence-based claims. Focus on the truthfulness and logical soundness of the content.",
            "Completeness": "You are a Completeness Critic specializing in thoroughness and adequate coverage. Focus on whether the text adequately addresses the topic and identifies missing information.",
            "Style": "You are a Style Critic specializing in tone, voice, writing style, and appropriateness. Focus on whether the writing style matches the intended purpose and audience.",
        }

        base_description = perspective_descriptions.get(
            perspective_name, f"You are a {perspective_name} Critic."
        )

        return f"""{base_description}

Your role is to evaluate text from your specialized perspective and provide focused,
expert feedback within your area of expertise.

RESPONSE FORMAT:
- needs_improvement: boolean based on your specialized assessment
- message: detailed analysis from your perspective
- suggestions: 1-3 specific suggestions within your area of expertise
- confidence: float 0.0-1.0 based on your certainty in this specialized assessment
- reasoning: explanation of your specialized evaluation process

CONFIDENCE SCORING:
- Use the FULL range 0.0-1.0
- High confidence (0.9-1.0): Very certain about your specialized assessment
- Medium confidence (0.6-0.8): Moderately certain with some uncertainty
- Low confidence (0.3-0.5): Uncertain or mixed signals in your area
- Very low confidence (0.0-0.2): Cannot make reliable assessment

Focus exclusively on your area of expertise while providing thorough,
specialized evaluation within that domain."""

    def _create_aggregation_prompt(self) -> str:
        """Create system prompt for aggregating multi-model perspectives."""
        return """You are an N-Critics ensemble aggregator implementing the methodology from Tian et al. 2023.

Your role is to synthesize feedback from multiple specialized critic perspectives
into a comprehensive, unified assessment.

You will receive individual critiques from specialized critics and must:
1. Analyze consensus and disagreement across perspectives
2. Prioritize issues identified by multiple critics
3. Include unique insights from individual perspectives
4. Build overall confidence based on consensus strength
5. Provide comprehensive, actionable feedback

RESPONSE FORMAT:
- needs_improvement: boolean based on ensemble consensus
- message: comprehensive synthesis of all perspectives
- suggestions: 1-3 suggestions addressing multiple perspectives
- confidence: float 0.0-1.0 based on consensus strength
- reasoning: explanation of how perspectives were synthesized

CONFIDENCE SCORING:
- High confidence (0.9-1.0): Strong consensus across perspectives
- Medium confidence (0.6-0.8): Moderate consensus with some disagreement
- Low confidence (0.3-0.5): Mixed perspectives with significant disagreement
- Very low confidence (0.0-0.2): No consensus or conflicting assessments

Synthesize the specialized feedback into comprehensive, balanced assessment."""

    def _get_default_perspectives(self) -> List[str]:
        """Get the default critic perspectives for the ensemble."""
        return [
            "Clarity Critic: Evaluate readability, structure, organization, and comprehensibility for the target audience",
            "Accuracy Critic: Assess factual correctness, logical consistency, and evidence-based claims",
            "Completeness Critic: Examine thoroughness, adequate coverage of the topic, and missing information",
            "Style Critic: Analyze tone, voice, writing style, and appropriateness for the intended purpose and audience",
        ]

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the N-Critics ensemble."""
        perspectives_text = "\n".join(
            [f"{i+1}. {perspective}" for i, perspective in enumerate(self.critic_perspectives)]
        )

        return f"""You are an N-Critics ensemble critic implementing the methodology from Tian et al. 2023.

Your role is to evaluate text from multiple critical perspectives and provide
comprehensive feedback by considering diverse viewpoints and building consensus
among different specialized critical approaches.

ENSEMBLE CRITIC PERSPECTIVES:
{perspectives_text}

N-CRITICS METHODOLOGY:
1. Evaluate the text from each critical perspective independently
2. Identify strengths and weaknesses from each viewpoint
3. Look for consensus areas where multiple perspectives agree
4. Highlight perspective-specific concerns that others might miss
5. Synthesize feedback into comprehensive, actionable suggestions

RESPONSE FORMAT:
- needs_improvement: boolean based on ensemble consensus
- message: comprehensive analysis incorporating all critical perspectives
- suggestions: 1-3 suggestions that address concerns from multiple perspectives
- confidence: float 0.0-1.0 based on consensus strength across perspectives
- reasoning: explanation of how different perspectives contributed to the assessment

CONFIDENCE SCORING REQUIREMENTS:
- Use the FULL range 0.0-1.0, not just 0.8!
- High confidence (0.9-1.0): Strong consensus across all perspectives
- Medium confidence (0.6-0.8): Moderate consensus with some disagreement
- Low confidence (0.3-0.5): Mixed perspectives with significant disagreement
- Very low confidence (0.0-0.2): No consensus or conflicting assessments

ENSEMBLE EVALUATION PROCESS:
- Consider each perspective's specialized focus area
- Look for agreement and disagreement between perspectives
- Prioritize issues identified by multiple perspectives
- Include perspective-specific insights that add unique value
- Build consensus while respecting specialized expertise

Provide a balanced assessment that leverages the strengths of multiple critical
viewpoints for comprehensive text evaluation."""

    async def critique_async(self, thought: SifakaThought) -> None:
        """Override base critique to handle multi-model ensemble."""
        if self.is_multi_model:
            await self._multi_model_critique(thought)
        else:
            # Use base class single-model approach
            await super().critique_async(thought)

    async def _multi_model_critique(self, thought: SifakaThought) -> None:
        """Perform multi-model ensemble critique."""
        import time

        start_time = time.time()
        all_tools_used = []
        all_retrieval_context = {}

        try:
            # Get individual critiques from each perspective
            perspective_critiques = {}

            for perspective_name, agent in self.perspective_agents.items():
                try:
                    prompt = await self._build_perspective_prompt(thought, perspective_name)
                    result = await agent.run(prompt)
                    perspective_critiques[perspective_name] = {
                        "feedback": result.output,
                        "tool_calls": getattr(result, "tool_calls", []),
                    }

                    # Track tools used
                    if hasattr(result, "tool_calls") and result.tool_calls:
                        tools_used = [call.tool_name for call in result.tool_calls]
                        all_tools_used.extend(tools_used)
                        all_retrieval_context[perspective_name] = self._extract_retrieval_context(
                            result.tool_calls
                        )

                except Exception as e:
                    # Handle individual perspective failure
                    perspective_critiques[perspective_name] = {
                        "feedback": CritiqueFeedback(
                            needs_improvement=False,
                            message=f"Perspective critique failed: {str(e)}",
                            suggestions=[],
                            confidence=0.0,
                            reasoning=f"Error in {perspective_name} perspective: {str(e)}",
                        ),
                        "tool_calls": [],
                    }

            # Aggregate the perspectives
            aggregated_feedback = await self._aggregate_perspectives(thought, perspective_critiques)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Apply validation filtering
            filtered_feedback = self._apply_validation_filtering(thought, aggregated_feedback)

            # Add critique to thought
            thought.add_critique(
                critic=self.__class__.__name__,
                feedback=filtered_feedback.message,
                suggestions=filtered_feedback.suggestions,
                confidence=filtered_feedback.confidence,
                reasoning=filtered_feedback.reasoning,
                needs_improvement=filtered_feedback.needs_improvement,
                critic_metadata=self._get_multi_model_metadata(
                    perspective_critiques, filtered_feedback
                ),
                processing_time_ms=processing_time_ms,
                model_name=f"ensemble({len(self.perspective_agents)} models)",
                paper_reference=self.paper_reference,
                methodology=self.methodology,
                tools_used=list(set(all_tools_used)),
                retrieval_context=all_retrieval_context if all_retrieval_context else None,
            )

        except Exception as e:
            # Handle complete failure
            processing_time_ms = (time.time() - start_time) * 1000

            thought.add_critique(
                critic=self.__class__.__name__,
                feedback=f"Multi-model ensemble critique failed: {str(e)}",
                suggestions=[],
                confidence=0.0,
                reasoning=f"Error in ensemble critique: {str(e)}",
                needs_improvement=False,
                critic_metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "mode": "multi_model",
                },
                processing_time_ms=processing_time_ms,
                model_name=f"ensemble({len(self.perspective_agents)} models)",
                paper_reference=self.paper_reference,
                methodology=self.methodology,
                tools_used=list(set(all_tools_used)),
                retrieval_context=all_retrieval_context if all_retrieval_context else None,
            )

    async def _build_critique_prompt(self, thought: SifakaThought) -> str:
        """Build the critique prompt for N-Critics ensemble methodology (single-model mode)."""
        if not thought.current_text:
            return "No text available for ensemble critique."

        prompt_parts = [
            "N-CRITICS ENSEMBLE CRITIQUE REQUEST",
            "=" * 50,
            "",
            f"Original Task: {thought.prompt}",
            f"Current Iteration: {thought.iteration}",
            "",
            "TEXT TO EVALUATE:",
            thought.current_text,
            "",
            "ENSEMBLE CRITIC PERSPECTIVES:",
            "=" * 35,
        ]

        # Add each critic perspective with detailed instructions
        for i, perspective in enumerate(self.critic_perspectives, 1):
            prompt_parts.extend(
                [
                    f"{i}. {perspective}",
                    "",
                ]
            )

        # Add validation context
        validation_context = self._get_validation_context(thought)
        if validation_context:
            prompt_parts.extend(
                [
                    "VALIDATION REQUIREMENTS:",
                    "=" * 25,
                    validation_context,
                    "",
                    "NOTE: Each critical perspective should consider validation requirements",
                    "within their specialized evaluation focus.",
                    "",
                ]
            )

        # Add previous ensemble feedback if available
        if thought.iteration > 0:
            prev_ensemble_critiques = [
                c
                for c in thought.critiques
                if c.iteration == thought.iteration - 1 and c.critic == "NCriticsCritic"
            ]
            if prev_ensemble_critiques:
                prompt_parts.extend(
                    [
                        "PREVIOUS ENSEMBLE FEEDBACK:",
                        "=" * 30,
                    ]
                )
                for critique in prev_ensemble_critiques[-1:]:  # Last ensemble critique
                    prompt_parts.extend(
                        [
                            f"Previous Consensus: {critique.feedback[:150]}{'...' if len(critique.feedback) > 150 else ''}",
                            f"Previous Suggestions: {', '.join(critique.suggestions)}",
                            f"Previous Confidence: {critique.confidence if hasattr(critique, 'confidence') else 'N/A'}",
                            "",
                        ]
                    )

        # Add ensemble evaluation instructions
        prompt_parts.extend(
            [
                "ENSEMBLE EVALUATION INSTRUCTIONS:",
                "=" * 40,
                "1. Evaluate the text from EACH critical perspective independently",
                "2. For each perspective, identify specific strengths and concerns",
                "3. Look for consensus areas where multiple perspectives agree",
                "4. Note unique insights that only specific perspectives would identify",
                "5. Synthesize findings into comprehensive, actionable feedback",
                "6. Build confidence based on consensus strength across perspectives",
                "",
                "EVALUATION PROCESS:",
                "- Start with Clarity Critic perspective",
                "- Move through Accuracy, Completeness, and Style perspectives",
                "- Identify overlapping concerns and unique perspective insights",
                "- Build consensus while respecting specialized expertise",
                "- Provide comprehensive feedback that leverages ensemble strengths",
            ]
        )

        return "\n".join(prompt_parts)

    async def _build_perspective_prompt(self, thought: SifakaThought, perspective_name: str) -> str:
        """Build prompt for individual perspective in multi-model mode."""
        if not thought.current_text:
            return f"No text available for {perspective_name} critique."

        prompt_parts = [
            f"{perspective_name.upper()} PERSPECTIVE CRITIQUE",
            "=" * 50,
            "",
            f"Original Task: {thought.prompt}",
            f"Current Iteration: {thought.iteration}",
            "",
            "TEXT TO EVALUATE:",
            thought.current_text,
            "",
        ]

        # Add validation context
        validation_context = self._get_validation_context(thought)
        if validation_context:
            prompt_parts.extend(
                [
                    "VALIDATION REQUIREMENTS:",
                    "=" * 25,
                    validation_context,
                    "",
                    f"NOTE: Evaluate from your {perspective_name} perspective while considering validation requirements.",
                    "",
                ]
            )

        # Add previous feedback from this perspective if available
        if thought.iteration > 0:
            prev_critiques = [
                c
                for c in thought.critiques
                if c.iteration == thought.iteration - 1
                and perspective_name.lower() in c.critic.lower()
            ]
            if prev_critiques:
                prompt_parts.extend(
                    [
                        f"PREVIOUS {perspective_name.upper()} FEEDBACK:",
                        "=" * 30,
                        f"Previous Assessment: {prev_critiques[-1].feedback[:150]}{'...' if len(prev_critiques[-1].feedback) > 150 else ''}",
                        "",
                    ]
                )

        prompt_parts.extend(
            [
                f"SPECIALIZED {perspective_name.upper()} EVALUATION:",
                f"Evaluate the text exclusively from your {perspective_name} perspective.",
                f"Focus on your area of expertise and provide detailed, specialized feedback.",
            ]
        )

        return "\n".join(prompt_parts)

    async def _aggregate_perspectives(
        self, thought: SifakaThought, perspective_critiques: Dict[str, Any]
    ) -> CritiqueFeedback:
        """Aggregate individual perspective critiques into ensemble feedback."""
        # Build aggregation prompt
        aggregation_parts = [
            "ENSEMBLE AGGREGATION REQUEST",
            "=" * 40,
            "",
            f"Original Task: {thought.prompt}",
            f"Text Evaluated: {thought.current_text[:200]}{'...' if len(thought.current_text) > 200 else ''}",
            "",
            "INDIVIDUAL PERSPECTIVE CRITIQUES:",
            "=" * 40,
        ]

        for perspective_name, critique_data in perspective_critiques.items():
            feedback = critique_data["feedback"]
            aggregation_parts.extend(
                [
                    f"\n{perspective_name.upper()} PERSPECTIVE:",
                    f"Needs Improvement: {feedback.needs_improvement}",
                    f"Confidence: {feedback.confidence}",
                    f"Assessment: {feedback.message}",
                    f"Suggestions: {', '.join(feedback.suggestions)}",
                    f"Reasoning: {feedback.reasoning}",
                    "",
                ]
            )

        aggregation_parts.extend(
            [
                "AGGREGATION INSTRUCTIONS:",
                "=" * 25,
                "Synthesize the above perspectives into a comprehensive ensemble assessment.",
                "Consider consensus areas, unique insights, and overall confidence based on agreement.",
                "Provide unified feedback that leverages the strengths of all perspectives.",
            ]
        )

        # Run aggregation through the main agent
        aggregation_prompt = "\n".join(aggregation_parts)
        result = await self.agent.run(aggregation_prompt)
        return result.output

    def _get_multi_model_metadata(
        self, perspective_critiques: Dict[str, Any], final_feedback: CritiqueFeedback
    ) -> Dict[str, Any]:
        """Extract metadata for multi-model ensemble critique."""
        base_metadata = super()._get_critic_specific_metadata(final_feedback)

        # Calculate perspective agreement
        perspective_needs_improvement = [
            critique_data["feedback"].needs_improvement
            for critique_data in perspective_critiques.values()
        ]
        perspective_confidences = [
            critique_data["feedback"].confidence for critique_data in perspective_critiques.values()
        ]

        agreement_ratio = (
            sum(perspective_needs_improvement) / len(perspective_needs_improvement)
            if perspective_needs_improvement
            else 0
        )
        avg_confidence = (
            sum(perspective_confidences) / len(perspective_confidences)
            if perspective_confidences
            else 0
        )

        multi_model_metadata = {
            "methodology": "n_critics_true_ensemble",
            "mode": "multi_model",
            "num_perspectives": len(perspective_critiques),
            "perspective_models": {
                name: str(agent.model) for name, agent in self.perspective_agents.items()
            },
            "perspective_agreement": agreement_ratio,
            "average_perspective_confidence": avg_confidence,
            "consensus_strength": (
                "strong"
                if agreement_ratio >= 0.8 or agreement_ratio <= 0.2
                else "moderate" if 0.3 <= agreement_ratio <= 0.7 else "weak"
            ),
            "ensemble_consensus": final_feedback.confidence > 0.6,
            "perspective_coverage": list(perspective_critiques.keys()),
            "true_multi_model_ensemble": True,
        }

        base_metadata.update(multi_model_metadata)
        return base_metadata

    def _get_critic_specific_metadata(self, feedback) -> Dict[str, Any]:
        """Extract N-Critics ensemble-specific metadata."""
        base_metadata = super()._get_critic_specific_metadata(feedback)

        # Add N-Critics-specific metadata
        n_critics_metadata = {
            "methodology": "n_critics_ensemble",
            "mode": "single_model" if not self.is_multi_model else "multi_model",
            "num_perspectives": len(self.critic_perspectives),
            "ensemble_consensus": feedback.confidence > 0.6,  # High confidence indicates consensus
            "perspective_coverage": [
                p.split(":")[0] for p in self.critic_perspectives
            ],  # Extract perspective names
            "consensus_strength": (
                "strong"
                if feedback.confidence > 0.8
                else "moderate" if feedback.confidence > 0.5 else "weak"
            ),
            "multi_perspective_analysis": True,
            "is_true_ensemble": self.is_multi_model,
        }

        base_metadata.update(n_critics_metadata)
        return base_metadata
