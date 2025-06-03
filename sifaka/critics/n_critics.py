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
from sifaka.critics.base import BaseCritic


class NCriticsCritic(BaseCritic):
    """N-Critics ensemble critic implementing Tian et al. 2023 methodology.

    This critic simulates an ensemble of critics with different specializations
    to provide comprehensive, multi-perspective evaluation of text quality.
    Each perspective focuses on specific aspects while contributing to an
    overall assessment.

    Enhanced with validation context awareness to ensure all critical perspectives
    consider validation requirements in their specialized evaluations.
    """

    def __init__(
        self,
        model_name: str = "groq:llama-3.1-8b-instant",
        critic_perspectives: Optional[List[str]] = None,
        retrieval_tools: Optional[List[Any]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the N-Critics ensemble critic.

        Args:
            model_name: The model name for the PydanticAI agent
            critic_perspectives: Custom critic perspectives (uses defaults if None)
            retrieval_tools: Optional list of retrieval tools for RAG support
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        self.critic_perspectives = critic_perspectives or self._get_default_perspectives()
        
        system_prompt = self._create_system_prompt()
        paper_reference = (
            "Tian, X., Chen, X., Kan, Z., Zhang, N., & Chen, H. (2023). "
            "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics. "
            "arXiv preprint arXiv:2310.18679. https://arxiv.org/abs/2310.18679"
        )
        methodology = (
            "N-Critics ensemble methodology: Multi-perspective evaluation through diverse critical viewpoints. "
            "Simulates ensemble of specialized critics for comprehensive text assessment. "
            "Adapted for single-model implementation with perspective-based prompting."
        )

        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            paper_reference=paper_reference,
            methodology=methodology,
            retrieval_tools=retrieval_tools,
            **agent_kwargs,
        )

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
        perspectives_text = "\n".join([f"{i+1}. {perspective}" for i, perspective in enumerate(self.critic_perspectives)])
        
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

ENSEMBLE EVALUATION PROCESS:
- Consider each perspective's specialized focus area
- Look for agreement and disagreement between perspectives
- Prioritize issues identified by multiple perspectives
- Include perspective-specific insights that add unique value
- Build consensus while respecting specialized expertise

Provide a balanced assessment that leverages the strengths of multiple critical
viewpoints for comprehensive text evaluation."""

    async def _build_critique_prompt(self, thought: SifakaThought) -> str:
        """Build the critique prompt for N-Critics ensemble methodology."""
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
            prompt_parts.extend([
                f"{i}. {perspective}",
                "",
            ])

        # Add validation context
        validation_context = self._get_validation_context(thought)
        if validation_context:
            prompt_parts.extend([
                "VALIDATION REQUIREMENTS:",
                "=" * 25,
                validation_context,
                "",
                "NOTE: Each critical perspective should consider validation requirements",
                "within their specialized evaluation focus.",
                "",
            ])

        # Add previous ensemble feedback if available
        if thought.iteration > 0:
            prev_ensemble_critiques = [
                c for c in thought.critiques 
                if c.iteration == thought.iteration - 1 and c.critic == "NCriticsCritic"
            ]
            if prev_ensemble_critiques:
                prompt_parts.extend([
                    "PREVIOUS ENSEMBLE FEEDBACK:",
                    "=" * 30,
                ])
                for critique in prev_ensemble_critiques[-1:]:  # Last ensemble critique
                    prompt_parts.extend([
                        f"Previous Consensus: {critique.feedback[:150]}{'...' if len(critique.feedback) > 150 else ''}",
                        f"Previous Suggestions: {', '.join(critique.suggestions)}",
                        f"Previous Confidence: {critique.confidence if hasattr(critique, 'confidence') else 'N/A'}",
                        "",
                    ])

        # Add ensemble evaluation instructions
        prompt_parts.extend([
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
        ])

        return "\n".join(prompt_parts)

    def _get_critic_specific_metadata(self, feedback) -> Dict[str, Any]:
        """Extract N-Critics ensemble-specific metadata."""
        base_metadata = super()._get_critic_specific_metadata(feedback)
        
        # Add N-Critics-specific metadata
        n_critics_metadata = {
            "methodology": "n_critics_ensemble",
            "num_perspectives": len(self.critic_perspectives),
            "ensemble_consensus": feedback.confidence > 0.6,  # High confidence indicates consensus
            "perspective_coverage": [p.split(":")[0] for p in self.critic_perspectives],  # Extract perspective names
            "consensus_strength": "strong" if feedback.confidence > 0.8 else "moderate" if feedback.confidence > 0.5 else "weak",
            "multi_perspective_analysis": True,
        }
        
        base_metadata.update(n_critics_metadata)
        return base_metadata
