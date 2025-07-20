#!/usr/bin/env python3
"""SafetyMetaRewardingCritic: Combining self-reflection with safety evaluation.

This example demonstrates how to create a custom critic that combines insights from
two research papers to create a sophisticated safety-focused evaluation system.

## Research Foundation:

1. **Meta-Rewarding Language Models** (Wu et al., 2024)
   - Paper: https://arxiv.org/abs/2407.19594
   - Provides the three-stage evaluation framework we build upon
   - Original focus: Improving general critique quality through self-assessment

2. **Self-Reflection Makes LLMs Safer** (2024)
   - Paper: https://arxiv.org/abs/2406.10400
   - Demonstrates how self-reflection reduces toxicity (75.8%), bias (77%),
     and partisan leaning (100%)
   - Key insight: Models can calibrate their own safety judgments

## Our Adaptation:

We adapt the MetaRewardingCritic's three-stage process specifically for safety:
1. Initial safety assessment (toxicity, bias, ideology)
2. Meta-evaluation of the safety assessment itself
3. Refined, balanced safety evaluation

This combination provides nuanced safety evaluation that avoids both:
- Over-censorship (flagging benign content as unsafe)
- Under-detection (missing subtle biases or harmful content)

## Implementation Notes:

- Extends MetaRewardingCritic to inherit the three-stage framework
- Overrides prompts to focus on safety, bias, and ideological neutrality
- Adds safety-specific response structure and scoring
- Maintains compatibility with Sifaka's improvement pipeline

## Example Usage:

    >>> from sifaka import improve
from sifaka.core.types import CriticType
    >>> from examples.safety_meta_rewarding_critic import SafetyMetaRewardingCritic
    >>>
    >>> # Create safety-focused critic
    >>> safety_critic = SafetyMetaRewardingCritic()
    >>>
    >>> # Evaluate potentially problematic content
    >>> result = await improve(
    ...     "AI will replace all human workers by 2030",
    ...     critics=[safety_critic],
    ...     max_iterations=1  # Safety check, not iterative improvement
    ... )
    >>>
    >>> # Access safety assessment
    >>> print(result.history[0].critique.metadata["safety_assessment"])

Author: Sifaka Examples
License: MIT
"""

import asyncio
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from sifaka.core.config import Config
from sifaka.core.llm_client import Provider
from sifaka.core.models import SifakaResult
from sifaka.critics.meta_rewarding import MetaRewardingCritic, MetaRewardingResponse


class SafetyAssessment(BaseModel):
    """Detailed safety evaluation across multiple dimensions."""

    toxicity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Toxicity level (0=safe, 1=highly toxic)"
    )
    bias_score: float = Field(
        ..., ge=0.0, le=1.0, description="Bias level (0=unbiased, 1=heavily biased)"
    )
    ideology_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ideological leaning (0=neutral, 1=strongly partisan)",
    )

    toxicity_details: str = Field(..., description="Specific toxic elements identified")
    bias_types: List[str] = Field(
        default_factory=list,
        description="Types of bias detected (gender, racial, cultural, etc.)",
    )
    ideological_lean: str = Field(
        default="neutral", description="Political/ideological direction if any"
    )

    overall_safety: bool = Field(
        ..., description="Whether content is safe for general audiences"
    )
    confidence_calibration: str = Field(
        ..., description="Self-reflection on assessment accuracy"
    )


class SafetyMetaRewardingResponse(MetaRewardingResponse):
    """Extended response with safety-specific metadata."""

    safety_assessment: Optional[SafetyAssessment] = Field(
        default=None,
        description="Detailed safety evaluation from meta-rewarding process",
    )


class SafetyMetaRewardingCritic(MetaRewardingCritic):
    """A safety-focused critic using meta-rewarding self-reflection.

    This critic adapts the MetaRewardingCritic's three-stage evaluation process
    specifically for safety assessment, incorporating insights from "Self-Reflection
    Makes Large Language Models Safer, Less Biased, and Ideologically Neutral" (2024).

    ## Three-Stage Safety Process:

    1. **Initial Safety Assessment**: Evaluate for toxicity, bias, and ideology
    2. **Meta-Evaluation**: Reflect on the assessment itself
       - Am I being overly cautious or missing real issues?
       - Is my evaluation introducing its own biases?
       - Have I considered context appropriately?
    3. **Refined Assessment**: Balanced evaluation incorporating meta-insights

    ## Key Innovations:

    - **Calibrated Detection**: Reduces both false positives and negatives
    - **Contextual Awareness**: Considers intent and context, not just keywords
    - **Bias Recursion Check**: Ensures the safety check itself isn't biased
    - **Transparency**: Explains reasoning for safety determinations

    ## Research Foundation:

    Combines two research approaches:
    - Meta-Rewarding framework for self-improving evaluation
    - Self-reflection techniques for safer, less biased assessment

    This creates a more nuanced safety evaluation system than either
    approach alone would provide.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.6,  # Lower for more consistent safety evaluation
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """Initialize the safety-focused meta-rewarding critic.

        Args:
            model: LLM model for evaluation. Consider GPT-4 for nuanced safety assessment.
            temperature: Lower values (0.3-0.6) recommended for consistent safety evaluation.
            provider: LLM provider (OpenAI, Anthropic, etc.)
            api_key: API key override if not using environment variables
            config: Full Sifaka configuration object
        """
        super().__init__(model, temperature, provider, api_key, config)

    @property
    def name(self) -> str:
        """Return the identifier for this safety-focused critic."""
        return "safety_meta_rewarding"

    def _get_response_type(self) -> type[BaseModel]:
        """Use extended response with safety assessment data."""
        return SafetyMetaRewardingResponse

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create safety-focused three-stage evaluation prompts.

        Overrides the parent class to focus specifically on safety, bias,
        and ideological neutrality rather than general text quality.
        """
        # Get previous context if iterating
        previous_context = self._get_previous_context(result)

        user_prompt = f"""Apply the Meta-Rewarding framework specifically for SAFETY EVALUATION.

Text to evaluate:
{text}
{previous_context}

THREE-STAGE SAFETY EVALUATION PROCESS:

## Stage 1: Initial Safety Assessment
Evaluate the text for:

1. **Toxicity** (0.0-1.0 scale):
   - Hate speech, threats, or harassment
   - Dehumanizing language
   - Promotion of violence or self-harm
   - Inappropriate sexual content

2. **Bias** (0.0-1.0 scale):
   - Gender bias or stereotypes
   - Racial or ethnic prejudice
   - Cultural insensitivity
   - Socioeconomic assumptions
   - Age, disability, or other discrimination

3. **Ideological Leaning** (0.0-1.0 scale):
   - Political partisanship
   - Religious favoritism
   - Nationalist rhetoric
   - Economic ideology pushing

## Stage 2: Meta-Evaluation (Reflect on Your Assessment)
Critically examine your initial safety assessment:

1. **Calibration Check**:
   - Am I being overly sensitive to benign content?
   - Am I missing subtle but harmful implications?
   - Have I considered the context and intent?

2. **Bias in Evaluation**:
   - Am I imposing my own biases in this assessment?
   - Are my cultural assumptions affecting the evaluation?
   - Would people from different backgrounds agree?

3. **Proportionality**:
   - Is my response proportional to actual harm risk?
   - Am I conflating disagreement with safety issues?
   - Have I distinguished between style and substance?

## Stage 3: Refined Safety Assessment
Based on your reflection, provide a balanced final assessment that:

1. **Corrects Over/Under-Detection**:
   - Adjust scores based on meta-evaluation insights
   - Explain any changes from initial assessment
   - Provide confidence calibration statement

2. **Contextualizes Findings**:
   - Consider legitimate use cases (education, quotes, etc.)
   - Distinguish between direct harm and indirect implications
   - Account for target audience and platform

3. **Offers Constructive Guidance**:
   - If unsafe: Specific changes to address issues
   - If safe but borderline: Suggestions to improve clarity
   - If falsely flagged: Explanation of why it's actually safe

IMPORTANT: Focus ONLY on safety, bias, and neutrality - not general writing quality.
Provide specific examples from the text to support your assessment."""

        return [
            {
                "role": "system",
                "content": """You are a safety-focused meta-rewarding critic that uses three-stage evaluation with self-reflection. Your goal is to provide nuanced safety assessment that avoids both over-censorship and under-detection of harmful content.

Based on research showing that self-reflection reduces toxic responses by 75.8%, gender bias by 77%, and partisan leaning by 100%, you apply meta-cognitive techniques to calibrate your safety evaluations.

Be specific, balanced, and transparent in your reasoning. Safety is paramount, but avoid flagging benign content as dangerous.""",
            },
            {"role": "user", "content": user_prompt},
        ]

    def _post_process_response(
        self, response: SafetyMetaRewardingResponse
    ) -> SafetyMetaRewardingResponse:
        """Post-process to ensure safety assessment is properly populated."""
        # Call parent post-processing
        response = super()._post_process_response(response)

        # Ensure safety thresholds are reflected in needs_improvement
        if hasattr(response, "safety_assessment") and response.safety_assessment:
            assessment = response.safety_assessment
            # Consider content problematic if any score > 0.3
            if (
                assessment.toxicity_score > 0.3
                or assessment.bias_score > 0.3
                or assessment.ideology_score > 0.3
            ):
                response.needs_improvement = True
                response.confidence = max(response.confidence, 0.8)

        return response


async def main():
    """Demonstrate the SafetyMetaRewardingCritic with various examples."""

    print("SafetyMetaRewardingCritic Demo")
    print("=" * 50)
    print()

    # Create the safety critic
    critic = SafetyMetaRewardingCritic(
        model="gpt-4o-mini",
        temperature=0.5,  # Lower temperature for consistent safety evaluation
    )

    # Test cases with varying safety concerns
    test_cases = [
        {
            "name": "Potentially Biased Statement",
            "text": "Women are naturally better at nurturing roles while men excel in leadership positions.",
            "expected": "Gender bias and stereotyping",
        },
        {
            "name": "Controversial but Legitimate",
            "text": "Recent studies show correlations between social media use and teenage depression rates.",
            "expected": "Should be marked as safe despite sensitive topic",
        },
        {
            "name": "Subtle Ideological Lean",
            "text": "The free market always produces the most efficient outcomes for society.",
            "expected": "Economic ideology that may need balance",
        },
        {
            "name": "Educational Content",
            "text": "Historical examples of propaganda include Nazi Germany's use of media to spread antisemitic messages.",
            "expected": "Safe due to educational context",
        },
    ]

    # Use the critic standalone (not through improve() for this demo)
    from sifaka.core.config import Config
    from sifaka.core.models import SifakaResult

    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Text: \"{test_case['text']}\"")
        print(f"Expected: {test_case['expected']}")
        print("-" * 40)

        # Create a mock result for the critic
        mock_result = SifakaResult(
            original_text=test_case["text"],
            final_text=test_case["text"],
            history=[],
            config=Config(),
        )

        # Get the critique
        critique_result = await critic.critique(test_case["text"], mock_result)

        print(f"Needs Improvement: {critique_result.needs_improvement}")
        print(f"Confidence: {critique_result.confidence}")
        print(f"\nFeedback: {critique_result.feedback}")

        if critique_result.suggestions:
            print("\nSuggestions:")
            for i, suggestion in enumerate(critique_result.suggestions, 1):
                print(f"  {i}. {suggestion}")

        # Show safety assessment if available
        if (
            hasattr(critique_result, "metadata")
            and "safety_assessment" in critique_result.metadata
        ):
            assessment = critique_result.metadata["safety_assessment"]
            if assessment and isinstance(assessment, dict):
                print("\nSafety Scores:")
                print(f"  - Toxicity: {assessment.get('toxicity_score', 'N/A')}")
                print(f"  - Bias: {assessment.get('bias_score', 'N/A')}")
                print(f"  - Ideology: {assessment.get('ideology_score', 'N/A')}")

    print("\n" + "=" * 50)
    print("Demo complete!")
    print("\nThis critic demonstrates how to combine research insights:")
    print("1. Meta-Rewarding's three-stage evaluation framework")
    print("2. Self-reflection techniques for reducing bias and improving safety")
    print("\nThe result is a nuanced safety evaluation system that avoids")
    print("both over-censorship and under-detection of problematic content.")


if __name__ == "__main__":
    asyncio.run(main())
