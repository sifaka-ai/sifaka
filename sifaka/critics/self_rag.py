"""Self-RAG critic for Sifaka.

This module implements the Self-RAG (Self-Reflective Retrieval-Augmented Generation)
approach for evaluating text quality and determining when additional retrieval
would improve content accuracy and completeness.

Based on "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection":
https://arxiv.org/abs/2310.11511

@misc{asai2023selfraglearningretrievegenerate,
      title={Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
      author={Akari Asai and Zeqiu Wu and Yizhong Wang and Avirup Sil and Hannaneh Hajishirzi},
      year={2023},
      eprint={2310.11511},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.11511},
}

The SelfRAGCritic implements key Self-RAG concepts:
1. Self-reflective evaluation of content quality and factuality
2. Retrieval need assessment for knowledge gaps
3. Evidence-based critique with source evaluation
4. Adaptive retrieval recommendations

IMPORTANT IMPLEMENTATION NOTES AND CAVEATS:

This implementation adapts the Self-RAG approach for text critique and retrieval
guidance. The original Self-RAG paper focuses on training models to learn when
to retrieve, what to retrieve, and how to use retrieved information during
generation. Our implementation focuses on the critique and retrieval assessment
aspects without the full training pipeline.

The original Self-RAG introduces special tokens ([Retrieve], [IsRel], [IsSup], [IsUse])
to control retrieval and generation behavior. Our implementation translates these
concepts into natural language critique and suggestions.

CAVEATS AND LIMITATIONS:
1. This is a critique-only implementation that provides retrieval recommendations
   without the full Self-RAG training and generation pipeline.
2. We simulate the retrieval need assessment through prompting rather than
   learned retrieval triggers as in the original paper.
3. The factuality assessment is based on the model's internal knowledge rather
   than the sophisticated evidence evaluation in the original work.
4. We don't implement the special token system for retrieval control, instead
   providing natural language guidance.
5. Performance depends on the underlying model's ability to assess its own
   knowledge limitations and identify retrieval needs.
6. The approach may be conservative in retrieval recommendations, potentially
   missing subtle knowledge gaps that would benefit from external information.

RETRIEVAL ASSESSMENT FOCUS:
This implementation evaluates:
1. Factual accuracy and potential knowledge gaps
2. Currency of information and need for updates
3. Completeness of coverage for the given topic
4. Evidence quality and source reliability needs
5. Domain-specific knowledge requirements

RETRIEVAL AUGMENTATION:
This critic is designed to work with retrieval tools and can provide specific
guidance on what types of information should be retrieved to improve content
quality and factual accuracy.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.thought import SifakaThought
from sifaka.critics.base import BaseCritic


class SelfRAGCritic(BaseCritic):
    """Self-RAG critic implementing Asai et al. 2023 methodology.

    This critic evaluates text from a retrieval-augmented perspective,
    assessing factual accuracy, identifying knowledge gaps, and providing
    guidance on when and what to retrieve for improved content quality.

    Enhanced with validation context awareness to ensure retrieval recommendations
    align with validation requirements and task objectives.
    """

    def __init__(
        self,
        model_name: str = "groq:mixtral-8x7b-32768",
        retrieval_focus_areas: Optional[List[str]] = None,
        retrieval_tools: Optional[List[Any]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the Self-RAG critic.

        Args:
            model_name: The model name for the PydanticAI agent
            retrieval_focus_areas: Specific areas to assess for retrieval needs (uses defaults if None)
            retrieval_tools: Optional list of retrieval tools for RAG support
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        self.retrieval_focus_areas = retrieval_focus_areas or self._get_default_focus_areas()
        
        system_prompt = self._create_system_prompt()
        paper_reference = (
            "Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). "
            "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. "
            "arXiv preprint arXiv:2310.11511. https://arxiv.org/abs/2310.11511"
        )
        methodology = (
            "Self-RAG methodology: Self-reflective evaluation with retrieval need assessment. "
            "Evaluates factual accuracy, identifies knowledge gaps, and provides retrieval guidance. "
            "Adapted for critique without full training pipeline."
        )

        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            paper_reference=paper_reference,
            methodology=methodology,
            retrieval_tools=retrieval_tools,
            **agent_kwargs,
        )

    def _get_default_focus_areas(self) -> List[str]:
        """Get the default retrieval assessment focus areas."""
        return [
            "Factual Accuracy: Verify claims, statistics, and factual statements for correctness",
            "Currency: Assess if information is up-to-date and current for time-sensitive topics",
            "Completeness: Identify missing information or knowledge gaps that external sources could fill",
            "Evidence Quality: Evaluate the strength of evidence and need for authoritative sources",
            "Domain Expertise: Assess need for specialized knowledge from domain-specific sources",
        ]

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the Self-RAG critic."""
        focus_text = "\n".join([f"- {area}" for area in self.retrieval_focus_areas])
        
        return f"""You are a Self-RAG critic implementing the methodology from Asai et al. 2023.

Your role is to evaluate text from a retrieval-augmented perspective, assessing
factual accuracy, identifying knowledge gaps, and determining when additional
retrieval would improve content quality and reliability.

SELF-RAG METHODOLOGY:
1. Self-reflect on the content's factual accuracy and completeness
2. Assess knowledge gaps that could benefit from external retrieval
3. Evaluate evidence quality and source reliability needs
4. Determine retrieval necessity and specify what should be retrieved
5. Provide confidence based on knowledge certainty and gap identification

RETRIEVAL ASSESSMENT FOCUS:
{focus_text}

RESPONSE FORMAT:
- needs_improvement: boolean indicating if retrieval would improve content
- message: detailed analysis of factual accuracy and retrieval needs
- suggestions: 1-3 specific retrieval recommendations or content improvements
- confidence: float 0.0-1.0 based on knowledge certainty and gap assessment
- reasoning: explanation of retrieval need assessment and knowledge gap analysis

SELF-REFLECTION PROCESS:
- Evaluate factual claims against internal knowledge
- Identify areas of uncertainty or potential knowledge gaps
- Assess currency and completeness of information
- Determine specific retrieval needs and sources
- Consider evidence quality and authoritative source requirements

If the content is factually sound and complete without retrieval needs,
set needs_improvement to false and explain the adequacy of current information."""

    async def _build_critique_prompt(self, thought: SifakaThought) -> str:
        """Build the critique prompt for Self-RAG methodology."""
        if not thought.current_text:
            return "No text available for Self-RAG critique."

        prompt_parts = [
            "SELF-RAG CRITIQUE REQUEST",
            "=" * 50,
            "",
            f"Original Task: {thought.prompt}",
            f"Current Iteration: {thought.iteration}",
            "",
            "TEXT TO EVALUATE:",
            thought.current_text,
            "",
        ]

        # Add retrieval context if tools are available
        if self.retrieval_tools:
            prompt_parts.extend([
                "AVAILABLE RETRIEVAL TOOLS:",
                "=" * 30,
                f"Number of retrieval tools available: {len(self.retrieval_tools)}",
                "These tools can be used to gather additional information if needed.",
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
                "NOTE: Retrieval recommendations should prioritize addressing validation failures",
                "and improving factual accuracy to meet validation requirements.",
                "",
            ])

        # Add previous retrieval assessments
        if thought.iteration > 0:
            prev_rag_critiques = [
                c for c in thought.critiques 
                if c.iteration == thought.iteration - 1 and c.critic == "SelfRAGCritic"
            ]
            if prev_rag_critiques:
                prompt_parts.extend([
                    "PREVIOUS RETRIEVAL ASSESSMENT:",
                    "=" * 35,
                ])
                for critique in prev_rag_critiques[-1:]:  # Last RAG critique
                    prompt_parts.extend([
                        f"Previous Assessment: {critique.feedback[:150]}{'...' if len(critique.feedback) > 150 else ''}",
                        f"Previous Retrieval Needs: {', '.join(critique.suggestions)}",
                        "",
                    ])

        # Add current focus areas
        prompt_parts.extend([
            "RETRIEVAL ASSESSMENT FOCUS:",
            "=" * 35,
        ])
        for area in self.retrieval_focus_areas:
            prompt_parts.append(f"- {area}")
        
        prompt_parts.extend([
            "",
            "SELF-RAG EVALUATION INSTRUCTIONS:",
            "=" * 40,
            "1. Self-reflect on factual accuracy and knowledge certainty",
            "2. Identify specific claims that may need verification",
            "3. Assess completeness and potential knowledge gaps",
            "4. Evaluate currency of information for time-sensitive topics",
            "5. Determine specific retrieval needs and recommended sources",
            "6. Consider evidence quality and authoritative source requirements",
            "",
            "RETRIEVAL DECISION CRITERIA:",
            "- Are there factual claims that need verification?",
            "- Is the information current and up-to-date?",
            "- Are there knowledge gaps that external sources could fill?",
            "- Would additional evidence strengthen the content?",
            "- Are authoritative sources needed for credibility?",
            "",
            "Provide specific, actionable retrieval recommendations if needed.",
        ])

        return "\n".join(prompt_parts)

    def _get_critic_specific_metadata(self, feedback) -> Dict[str, Any]:
        """Extract Self-RAG-specific metadata."""
        base_metadata = super()._get_critic_specific_metadata(feedback)
        
        # Add Self-RAG-specific metadata
        self_rag_metadata = {
            "methodology": "self_rag_retrieval_assessment",
            "retrieval_recommended": feedback.needs_improvement,
            "knowledge_certainty": feedback.confidence,
            "retrieval_focus_areas": len(self.retrieval_focus_areas),
            "factual_assessment": "uncertain" if feedback.needs_improvement else "confident",
            "retrieval_tools_available": len(self.retrieval_tools) > 0,
            "gap_identification": feedback.needs_improvement and len(feedback.suggestions) > 0,
        }
        
        base_metadata.update(self_rag_metadata)
        return base_metadata
