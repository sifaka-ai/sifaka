"""This module implements a Constitutional AI approach for critics, which evaluates
responses against a set of human-written principles (a "constitution") and provides
natural language feedback when violations are detected.

Based on "Constitutional AI: Harmlessness from AI Feedback":
https://arxiv.org/abs/2212.08073

@misc{bai2022constitutionalaiharmlessnessai,
      title={Constitutional AI: Harmlessness from AI Feedback},
      author={Yuntao Bai and Saurav Kadavath and Sandipan Kundu and Amanda Askell and Jackson Kernion and Andy Jones and Anna Chen and Anna Goldie and Azalia Mirhoseini and Cameron McKinnon and Carol Chen and Catherine Olsson and Christopher Olah and Danny Hernandez and Dawn Drain and Deep Ganguli and Dustin Li and Eli Tran-Johnson and Ethan Perez and Jamie Kerr and Jared Mueller and Jeffrey Ladish and Joshua Landau and Kamal Ndousse and Kamile Lukosuite and Liane Lovitt and Michael Sellitto and Nelson Elhage and Nicholas Schiefer and Noemi Mercado and Nova DasSarma and Robert Lasenby and Robin Larson and Sam Ringer and Scott Johnston and Shauna Kravec and Sheer El Showk and Stanislav Fort and Tamera Lanham and Timothy Telleen-Lawton and Tom Conerly and Tom Henighan and Tristan Hume and Samuel R. Bowman and Zac Hatfield-Dodds and Ben Mann and Dario Amodei and Nicholas Joseph and Sam McCandlish and Tom Brown and Jared Kaplan},
      year={2022},
      eprint={2212.08073},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2212.08073},
}

The ConstitutionalCritic implements core Constitutional AI concepts:
1. Principle-based evaluation against a written constitution
2. Natural language feedback on principle violations
3. Iterative improvement through constitutional critique
4. Harmlessness assessment through AI feedback

IMPORTANT IMPLEMENTATION CAVEAT:
This implementation adapts the core Constitutional AI principle-based evaluation
for text critique. The original Constitutional AI paper focuses on training methodology
for creating more aligned models through two phases:
1. Supervised Learning: Generate → Critique → Revise → Train on revisions
2. RL from AI Feedback: Generate multiple responses → AI evaluation → RL training

This implementation extracts the "Critique → Revise" component for single-text
improvement scenarios, providing practical value for content moderation and
ethical AI applications without requiring model training.
"""

from typing import List, Optional

from sifaka.core.thought import Thought
from sifaka.critics.base_pydantic import PydanticAICritic
from sifaka.models.critic_results import (
    ConfidenceScore,
    ImprovementSuggestion,
    SeverityLevel,
    ViolationReport,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


# Default constitutional principles (5 core principles for focus)
DEFAULT_PRINCIPLES = [
    "Be helpful and provide accurate, useful information",
    "Be harmless and avoid content that could cause harm",
    "Be honest and acknowledge uncertainty when appropriate",
    "Respect human autonomy and avoid manipulation",
    "Be fair and avoid bias or discrimination",
]


class ConstitutionalCritic(PydanticAICritic):
    """Modern Constitutional AI critic using PydanticAI agents with structured output.

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
        model_name: str,
        principles: Optional[List[str]] = None,
        strict_mode: bool = False,
        **agent_kwargs,
    ):
        """Initialize the Constitutional critic.

        Args:
            model_name: The model name for the PydanticAI agent (e.g., "openai:gpt-4")
            principles: List of constitutional principles to evaluate against.
            strict_mode: If True, applies stricter evaluation criteria.
            **agent_kwargs: Additional arguments passed to the PydanticAI agent.
        """
        # Initialize parent with system prompt
        super().__init__(model_name=model_name, **agent_kwargs)

        # Set up principles (constitution)
        self.principles = principles or DEFAULT_PRINCIPLES
        self.strict_mode = strict_mode

        logger.info(
            f"Initialized ConstitutionalCritic with {len(self.principles)} principles, strict_mode={strict_mode}"
        )

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for constitutional evaluation."""
        return """You are an expert constitutional AI evaluator. Your role is to evaluate text against constitutional principles and provide structured feedback.

You must return a CritiqueFeedback object with these REQUIRED fields:
- message: A clear summary of your constitutional evaluation (string)
- needs_improvement: Whether the text violates constitutional principles (boolean)
- confidence: ConfidenceScore with overall confidence (object with 'overall' field as float 0.0-1.0)
- critic_name: Set this to "ConstitutionalCritic" (string)

And these OPTIONAL fields (can be empty lists or null):
- violations: List of ViolationReport objects for each principle violation
- suggestions: List of ImprovementSuggestion objects for addressing violations
- processing_time_ms: Time taken in milliseconds (can be null)
- critic_version: Version string (can be null)
- metadata: Additional metadata dictionary (can be empty)

IMPORTANT: Always provide the required fields. For confidence, use a simple object like {"overall": 0.8} where the number is between 0.0 and 1.0.

Focus on:
1. Principle-based evaluation against the provided constitution
2. Clear, actionable feedback in the message
3. Accurate needs_improvement assessment
4. Realistic confidence scoring

Be thorough but fair in your evaluation."""

    async def _create_critique_prompt(self, thought: Thought) -> str:
        """Create the critique prompt for constitutional evaluation.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            The formatted critique prompt.
        """
        # Format principles for the prompt
        principles_text = "\n".join(
            f"{i + 1}. {principle}" for i, principle in enumerate(self.principles)
        )

        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Get validation context if available
        validation_context = self._get_validation_context_dict(thought)
        validation_text = ""
        if validation_context:
            validation_text = f"\n\nValidation Context:\n{validation_context}"

        return f"""Evaluate the following text against the constitutional principles provided.

Constitutional Principles:
{principles_text}

Original Task: {thought.prompt}

Text to Evaluate:
{thought.text}

Context:
{context}
{validation_text}

Please provide a structured constitutional evaluation with:
1. Clear identification of any principle violations
2. Specific evidence for each violation
3. Constructive improvement suggestions
4. Confidence assessment

Strict Mode: {self.strict_mode}

Focus on being thorough but fair in your constitutional analysis."""
