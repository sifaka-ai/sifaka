"""Meta-Rewarding critic implementation.

Based on: Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge
Paper: https://arxiv.org/abs/2407.19594
Authors: Wu et al. (2024)

Meta-Rewarding uses a two-stage judgment process where the model evaluates
its own evaluations to improve critique quality.

## Similarity to Original Paper:
- PRESERVED: Two-stage evaluation process
- PRESERVED: Meta-judgment of evaluation quality
- SIMPLIFIED: Single model vs separate judge model
- ADAPTED: Text critique vs preference ranking

## Implementation Choices:
1. First stage: Generate initial critique
2. Second stage: Evaluate the critique quality
3. Adjust confidence based on meta-evaluation
4. Focus on critique consistency and depth

## Why This Approach:
- Meta-evaluation improves critique reliability
- Self-reflection on judgment quality
- More nuanced confidence scoring
- Reduces poor quality critiques
"""

from typing import Optional, Union, List, Dict

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config


class MetaRewardingCritic(BaseCritic):
    """Implements Meta-Rewarding two-stage critique.

    ## When to Use This Critic:

    ✅ When to use:
    - High-stakes content requiring thorough review
    - Documents where critique quality itself matters
    - Complex evaluations needing self-reflection
    - Final quality assurance before publication

    ❌ When to avoid:
    - Quick iterative improvements
    - Simple text corrections
    - When single-pass evaluation is sufficient

    🎯 Best for:
    - Executive communications
    - Legal or compliance documents
    - Published articles or white papers
    - Critical business proposals
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        # Initialize with custom config
        if config is None:
            config = Config()
        super().__init__(model, temperature, config, provider, api_key)

    @property
    def name(self) -> str:
        return "meta_rewarding"

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for two-stage meta-rewarding critique."""
        # Get previous context
        previous_context = self._get_previous_context(result)

        user_prompt = f"""You are a Meta-Rewarding critic that performs two-stage evaluation.

Text to evaluate:
{text}
{previous_context}

STAGE 1 - Initial Critique:
Provide a thorough critique considering:
- Content quality and accuracy
- Structure and organization
- Clarity and readability
- Completeness and depth
- Engagement and impact

STAGE 2 - Meta-Evaluation:
After your initial critique, evaluate your own judgment:
- Is your critique fair and balanced?
- Did you identify the most important issues?
- Are your suggestions practical and helpful?
- How confident are you in this assessment?

Integrate both stages into a refined critique that reflects your meta-evaluation insights. Adjust your confidence based on the meta-evaluation."""

        return [
            {
                "role": "system",
                "content": "You are a Meta-Rewarding critic that uses two-stage evaluation: first critiquing the text, then evaluating your own critique quality to provide refined feedback.",
            },
            {"role": "user", "content": user_prompt},
        ]
