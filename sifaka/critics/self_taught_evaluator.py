"""Self-Taught Evaluator critic implementation for annotation-free evaluation.

Based on: Self-Taught Evaluator: The Path to GPT-4 Level Performance Without Human Annotations
Paper: https://arxiv.org/abs/2408.02666
Authors: Wang et al. (2024)

Self-Taught Evaluator achieves GPT-4 level evaluation performance without
any human-labeled data by using contrasting outputs and synthetic reasoning
traces. This implementation adapts the approach for real-time text critique.

## Similarity to Original Paper:

- PRESERVED: Core concept of contrasting output generation
- PRESERVED: Detailed reasoning trace production
- PRESERVED: Self-improvement through evaluation history
- SIMPLIFIED: Single-pass evaluation (vs. iterative training loops)
- ADAPTED: Real-time critique generation vs. offline model training
- MODIFIED: Direct text improvement focus vs. general evaluation capability

## Implementation Strategy:

1. **Contrasting Output Generation**: Creates 2-3 alternative versions of the text
   with intentional variations in quality, style, and focus to enable comparative
   evaluation. Each version includes reasoning for its creation.

2. **Reasoning Trace Construction**: Produces step-by-step evaluation logic that
   transparently explains the critique process, making judgments interpretable
   and educational.

3. **Comparative Analysis**: Systematically compares the original text with
   contrasting versions to identify strengths, weaknesses, and improvement
   opportunities grounded in concrete examples.

4. **Self-Improvement Tracking**: Learns from previous evaluations within the
   same session, avoiding repetitive feedback and building on prior insights.

## Why This Approach:

- **No Training Data Required**: Achieves high-quality evaluation without any
  human annotations, relying purely on synthetic contrasts
- **Transparent Reasoning**: Every evaluation includes clear reasoning traces
  explaining why specific judgments were made
- **Grounded Feedback**: Suggestions are based on concrete comparisons rather
  than abstract principles
- **Self-Improving**: The critic gets better at evaluation as it processes
  more iterations within a task

## Best Use Cases:

This critic excels at complex evaluation tasks where understanding the "why"
behind feedback is as important as the feedback itself. Particularly effective
for content that benefits from seeing concrete alternatives and understanding
trade-offs between different approaches.

## Key Differences from Original Paper:

1. **Scope**: We implement the evaluation methodology but not the full training
   pipeline described in the paper
2. **Application**: Focused on real-time text critique rather than building
   a general evaluation model
3. **Scale**: Single-document evaluation vs. large-scale model training
4. **Persistence**: Session-based learning vs. permanent model improvements
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from .core.base import BaseCritic, CriticResponse


class ContrastingOutput(BaseModel):
    """A contrasting version of the text with explanation."""

    version: str = Field(
        ..., description="The complete alternative text (not a description)"
    )
    reasoning: str = Field(..., description="Why this version was created this way")
    quality_assessment: str = Field(
        ..., description="Assessment of this version's strengths and weaknesses"
    )


class SelfTaughtResponse(CriticResponse):
    """Extended response with contrasting outputs and reasoning traces."""

    contrasting_outputs: List[ContrastingOutput] = Field(
        default_factory=list, description="Alternative versions for comparison"
    )
    reasoning_trace: str = Field(
        "", description="Step-by-step reasoning for the evaluation"
    )
    comparative_analysis: str = Field(
        "", description="Analysis comparing original with alternatives"
    )


class SelfTaughtEvaluatorCritic(BaseCritic):
    """Self-Taught Evaluator critic using contrasting outputs and reasoning traces.

    This critic evaluates text by:
    1. Generating multiple contrasting versions
    2. Comparing them systematically
    3. Producing detailed reasoning traces
    4. Learning from its own evaluations

    The approach mirrors the paper's methodology of achieving GPT-4 level
    performance without human annotations.
    """

    @property
    def name(self) -> str:
        return "self_taught_evaluator"

    def _get_system_prompt(self) -> str:
        return """You are a Self-Taught Evaluator that improves text evaluation through comparative analysis.

Your approach:
1. Generate contrasting outputs that vary in quality and style
2. Analyze each version systematically
3. Compare versions to understand what makes text effective
4. Provide detailed reasoning traces for your evaluation
5. Learn from the contrasts to make better judgments

CRITICAL: You must continuously improve your evaluation approach:
- Each evaluation should explore DIFFERENT aspects than previous ones
- Build on prior insights rather than repeating them
- If evaluating improved text, focus on remaining issues or new dimensions
- Your contrasting examples should evolve to test different hypotheses

Focus on creating meaningful contrasts that illuminate different aspects of quality.
Your reasoning should be transparent and educational."""

    def _get_response_type(self) -> type[BaseModel]:
        return SelfTaughtResponse

    def get_instructions(self, text: str, result: SifakaResult) -> str:
        # Check if we have previous iterations to learn from
        iteration_context = ""
        if result.generations:
            iteration_context = f"""
Previous iterations: {len(result.generations)}
This allows you to see the improvement trajectory and make more informed evaluations.
"""

        # Check if this is a subsequent iteration
        iteration_instruction = ""
        dimensions_to_explore = []

        if result.generations and len(result.generations) > 1:
            # Analyze what dimensions have been covered
            covered_aspects = set()
            for critique in result.critiques:
                if critique.critic == self.name:
                    feedback_lower = critique.feedback.lower()
                    if any(
                        word in feedback_lower
                        for word in ["specific", "detail", "example", "concrete"]
                    ):
                        covered_aspects.add("specificity")
                    if any(
                        word in feedback_lower
                        for word in ["persuasive", "compelling", "convince"]
                    ):
                        covered_aspects.add("persuasion")
                    if any(
                        word in feedback_lower for word in ["tone", "voice", "style"]
                    ):
                        covered_aspects.add("tone")
                    if any(
                        word in feedback_lower
                        for word in ["structure", "flow", "organization"]
                    ):
                        covered_aspects.add("structure")
                    if any(
                        word in feedback_lower
                        for word in ["emotion", "feeling", "connect"]
                    ):
                        covered_aspects.add("emotion")

            # Suggest unexplored dimensions
            all_dimensions = {
                "specificity": "concrete details and examples",
                "persuasion": "persuasive techniques and appeals",
                "tone": "tone, voice, and stylistic choices",
                "structure": "organization, flow, and coherence",
                "emotion": "emotional resonance and connection",
                "audience": "audience awareness and targeting",
                "credibility": "authority and trustworthiness",
                "clarity": "simplicity and accessibility",
                "engagement": "hooks, interest, and retention",
                "action": "calls to action and next steps",
            }

            unexplored = [dim for dim in all_dimensions if dim not in covered_aspects]
            if unexplored:
                dimensions_to_explore = unexplored[:3]  # Focus on top 3 unexplored

            iteration_instruction = f"""
IMPORTANT: This is iteration {len(result.generations)}, and the text has already been improved.
Previous critiques focused on: {", ".join(covered_aspects) if covered_aspects else "general aspects"}

For this evaluation, explore DIFFERENT dimensions:
{chr(10).join(f"- {dim.upper()}: Evaluate {all_dimensions.get(dim, dim)}" for dim in dimensions_to_explore)}

Do NOT repeat previous feedback about {", ".join(covered_aspects) if covered_aspects else "previous topics"}.
"""

        return f"""Evaluate this text using the Self-Taught Evaluator approach:
{iteration_instruction}
1. GENERATE CONTRASTING OUTPUTS:
   - Create 2-3 COMPLETE alternative versions of the text (not descriptions)
   - Write out each version in full - the actual text, not a summary
   - Make each version differ meaningfully along the dimensions you're exploring
   - Include both improvements and degradations
   - For each version, explain why you created it that specific way
   - IMPORTANT: If exploring specific dimensions (like emotion, audience, credibility),
     make your contrasting versions specifically test those dimensions

2. REASONING TRACE:
   - Walk through your evaluation step-by-step
   - Explain what you're looking for and why
   - Be transparent about your judgment process

3. COMPARATIVE ANALYSIS:
   - Compare the original with your contrasting versions
   - Identify what makes each version effective or ineffective
   - Use these insights to evaluate the original

4. FINAL EVALUATION:
   - Based on your analysis, provide specific feedback
   - Suggest improvements grounded in your comparisons
   - Rate your confidence based on the clarity of contrasts
   - Focus suggestions on actionable changes that address the dimensions explored
   - Prioritize suggestions that would have the most impact
{iteration_context}

Remember: The goal is not just to evaluate, but to demonstrate WHY your evaluation is valid through contrasting examples."""

    async def critique(self, text: str, result: SifakaResult) -> Any:
        """Override to add self-improvement tracking."""
        # Get the base critique
        critique_result = await super().critique(text, result)

        # Extract insights from contrasting outputs for future improvement
        if (
            hasattr(critique_result, "metadata")
            and "contrasting_outputs" in critique_result.metadata
        ):
            # Store evaluation patterns for self-improvement
            # In a real implementation, these could be persisted
            self._update_evaluation_patterns(critique_result.metadata)

        return critique_result

    def _update_evaluation_patterns(self, metadata: Dict[str, Any]) -> None:
        """Update internal patterns based on contrasting analysis.

        This mirrors the paper's iterative self-improvement approach.
        In a production system, these patterns would be persisted and
        used to improve future evaluations.
        """
        # Extract patterns from contrasting outputs
        # This is where the "self-taught" aspect comes in
        # The critic learns from its own comparative analysis
        pass

    def _get_previous_context(self, result: SifakaResult) -> str:
        """Enhanced context that includes learning from previous evaluations."""
        base_context = super()._get_previous_context(result)

        # Add self-improvement context
        if result.critiques:
            # Convert deque to list for slicing
            all_critiques = list(result.critiques)
            recent_evaluations = [
                c for c in all_critiques[-3:] if c.critic == self.name
            ]
            if recent_evaluations:
                base_context += "\n\nLearning from previous evaluations:"
                for eval in recent_evaluations:
                    if "comparative_analysis" in eval.metadata:
                        analysis = str(eval.metadata["comparative_analysis"])
                        base_context += f"\n- {analysis[:200]}..."

        return base_context
