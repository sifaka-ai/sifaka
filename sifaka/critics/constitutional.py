"""Constitutional AI critic for Sifaka.

This module implements a Constitutional AI approach for evaluating and improving
text based on a set of constitutional principles focused on helpfulness,
harmlessness, and honesty.

Based on "Constitutional AI: Harmlessness from AI Feedback":
https://arxiv.org/abs/2212.08073

@misc{bai2022constitutionalai,
      title={Constitutional AI: Harmlessness from AI Feedback},
      author={Yuntao Bai and Andy Jones and Kamal Ndousse and Amanda Askell and Anna Chen and Nova DasSarma and Dawn Drain and Stanislav Fort and Deep Ganguli and Tom Henighan and Nicholas Joseph and Saurav Kadavath and Jackson Kernion and Tom Conerly and Sheer El-Showk and Nelson Elhage and Zac Hatfield-Dodds and Danny Hernandez and Tristan Hume and Scott Johnston and Shauna Kravec and Liane Lovitt and Neel Nanda and Catherine Olsson and Dario Amodei and Tom Brown and Jack Clark and Sam McCandlish and Chris Olah and Ben Mann and Jared Kaplan},
      year={2022},
      eprint={2212.08073},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2212.08073},
}

The ConstitutionalCritic implements core Constitutional AI concepts:
1. Principle-based evaluation against a written constitution
2. Natural language feedback on principle violations
3. Iterative improvement through constitutional critique
4. Harmlessness assessment through AI feedback

IMPORTANT IMPLEMENTATION NOTES AND CAVEATS:

This implementation adapts the core Constitutional AI principle-based evaluation
for text critique. The original Constitutional AI paper focuses on training methodology
for creating more aligned models through two phases:
1. Supervised Learning: Generate → Critique → Revise → Train on revisions
2. RL from AI Feedback: Generate multiple responses → AI evaluation → RL training

Our implementation focuses on the critique and revision guidance aspects without
the model training component, making it suitable for real-time text improvement
rather than model alignment training.

CAVEATS AND LIMITATIONS:
1. This is an evaluation-only implementation that provides critique and suggestions
   without the full Constitutional AI training pipeline.
2. The constitutional principles are simplified and focused on general text quality
   rather than the comprehensive safety principles in the original paper.
3. We use a fixed set of principles rather than the iterative principle refinement
   process described in the original work.
4. The harmlessness assessment is adapted for general text rather than the
   specific safety concerns addressed in the original research.
5. Performance depends heavily on the underlying model's ability to understand
   and apply constitutional principles consistently.

CONSTITUTIONAL PRINCIPLES:
This implementation uses 5 core principles focused on:
1. Helpfulness: Content should be useful and informative
2. Harmlessness: Content should avoid harmful or misleading information
3. Honesty: Content should be accurate and truthful
4. Clarity: Content should be clear and understandable
5. Completeness: Content should adequately address the given task

RETRIEVAL AUGMENTATION:
This critic supports optional retrieval augmentation to enhance constitutional
evaluation by providing external context, fact-checking resources, or
domain-specific guidelines during the critique process.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.thought import SifakaThought
from sifaka.critics.base import BaseCritic


class ConstitutionalCritic(BaseCritic):
    """Constitutional AI critic implementing Bai et al. 2022 methodology.

    This critic evaluates text against constitutional principles and provides
    structured feedback with violations, suggestions, and confidence scores.

    Key features:
    - Structured output using CritiqueFeedback model
    - Constitutional principle evaluation
    - Detailed violation reporting
    - Improvement suggestions with priorities
    - Validation context awareness
    """

    def __init__(
        self,
        model_name: str = "anthropic:claude-3-5-haiku-20241022",
        constitutional_principles: Optional[List[str]] = None,
        retrieval_tools: Optional[List[Any]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the Constitutional AI critic.

        Args:
            model_name: The model name for the PydanticAI agent
            constitutional_principles: Custom constitutional principles (uses defaults if None)
            retrieval_tools: Optional list of retrieval tools for RAG support
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        self.constitutional_principles = constitutional_principles or self._get_default_principles()
        
        system_prompt = self._create_system_prompt()
        paper_reference = (
            "Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., ... & Kaplan, J. (2022). "
            "Constitutional AI: Harmlessness from AI Feedback. "
            "arXiv preprint arXiv:2212.08073. https://arxiv.org/abs/2212.08073"
        )
        methodology = (
            "Constitutional AI methodology: Principle-based evaluation against written constitution. "
            "Focuses on helpfulness, harmlessness, and honesty through structured critique. "
            "Adapted for text evaluation without full training pipeline."
        )

        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            paper_reference=paper_reference,
            methodology=methodology,
            retrieval_tools=retrieval_tools,
            **agent_kwargs,
        )

    def _get_default_principles(self) -> List[str]:
        """Get the default constitutional principles."""
        return [
            "Helpfulness: Content should be useful, informative, and directly address the user's needs and questions.",
            "Harmlessness: Content should avoid harmful, misleading, or dangerous information that could cause harm to individuals or society.",
            "Honesty: Content should be accurate, truthful, and acknowledge uncertainty when appropriate rather than making false claims.",
            "Clarity: Content should be clear, well-structured, and understandable to the intended audience.",
            "Completeness: Content should adequately address the given task and provide sufficient detail without being unnecessarily verbose.",
        ]

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the Constitutional AI critic."""
        principles_text = "\n".join([f"{i+1}. {principle}" for i, principle in enumerate(self.constitutional_principles)])
        
        return f"""You are a Constitutional AI critic implementing the methodology from Bai et al. 2022.

Your role is to evaluate text against constitutional principles and provide structured feedback
on principle violations and improvements needed.

CONSTITUTIONAL PRINCIPLES:
{principles_text}

EVALUATION METHODOLOGY:
1. Assess the text against each constitutional principle
2. Identify specific violations or areas of concern
3. Evaluate the severity and impact of any violations
4. Generate specific improvement suggestions to address violations
5. Provide confidence assessment based on principle clarity and violation severity

RESPONSE FORMAT:
- needs_improvement: boolean indicating if constitutional violations exist
- message: detailed analysis of principle adherence and violations
- suggestions: 1-3 specific suggestions to address constitutional concerns
- confidence: float 0.0-1.0 based on violation clarity and assessment certainty
- reasoning: explanation of constitutional analysis and principle application

FOCUS AREAS:
- Adherence to each constitutional principle
- Severity and impact of any violations
- Specific actionable improvements to enhance constitutional compliance
- Balance between principles when they may conflict

Be thorough in constitutional analysis. If the text adheres to all principles,
set needs_improvement to false and explain the constitutional compliance."""

    async def _build_critique_prompt(self, thought: SifakaThought) -> str:
        """Build the critique prompt for Constitutional AI methodology."""
        if not thought.current_text:
            return "No text available for constitutional critique."

        prompt_parts = [
            "CONSTITUTIONAL AI CRITIQUE REQUEST",
            "=" * 50,
            "",
            f"Original Task: {thought.prompt}",
            f"Current Iteration: {thought.iteration}",
            "",
            "TEXT TO EVALUATE:",
            thought.current_text,
            "",
            "CONSTITUTIONAL PRINCIPLES TO EVALUATE AGAINST:",
            "=" * 45,
        ]

        # Add constitutional principles
        for i, principle in enumerate(self.constitutional_principles, 1):
            prompt_parts.append(f"{i}. {principle}")
        
        prompt_parts.append("")

        # Add validation context if available
        validation_context = self._get_validation_context(thought)
        if validation_context:
            prompt_parts.extend([
                "VALIDATION CONTEXT:",
                "=" * 20,
                validation_context,
                "",
                "NOTE: Constitutional evaluation should consider validation requirements",
                "and prioritize suggestions that address both constitutional and validation concerns.",
                "",
            ])

        # Add previous constitutional feedback if available
        if thought.iteration > 0:
            prev_constitutional_critiques = [
                c for c in thought.critiques 
                if c.iteration == thought.iteration - 1 and c.critic == "ConstitutionalCritic"
            ]
            if prev_constitutional_critiques:
                prompt_parts.extend([
                    "PREVIOUS CONSTITUTIONAL FEEDBACK:",
                    "=" * 35,
                ])
                for critique in prev_constitutional_critiques[-1:]:  # Last constitutional critique
                    prompt_parts.extend([
                        f"Previous Assessment: {critique.feedback}",
                        f"Previous Suggestions: {', '.join(critique.suggestions)}",
                        "",
                    ])

        # Add evaluation instructions
        prompt_parts.extend([
            "CONSTITUTIONAL EVALUATION INSTRUCTIONS:",
            "=" * 40,
            "1. Evaluate the text against each constitutional principle",
            "2. Identify specific violations or areas of concern",
            "3. Assess the severity and potential impact of violations",
            "4. Generate targeted suggestions to improve constitutional compliance",
            "5. Consider the balance between principles if conflicts exist",
            "",
            "Provide a thorough constitutional analysis with specific, actionable feedback.",
        ])

        return "\n".join(prompt_parts)

    def _get_critic_specific_metadata(self, feedback) -> Dict[str, Any]:
        """Extract Constitutional AI-specific metadata."""
        base_metadata = super()._get_critic_specific_metadata(feedback)
        
        # Add Constitutional AI-specific metadata
        constitutional_metadata = {
            "methodology": "constitutional_ai_principles",
            "num_principles": len(self.constitutional_principles),
            "constitutional_compliance": not feedback.needs_improvement,
            "violation_severity": "high" if feedback.needs_improvement and feedback.confidence > 0.8 else "medium" if feedback.needs_improvement else "none",
            "principles_evaluated": [p.split(":")[0] for p in self.constitutional_principles],  # Extract principle names
        }
        
        base_metadata.update(constitutional_metadata)
        return base_metadata
