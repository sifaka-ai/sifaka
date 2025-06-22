"""Meta-Rewarding critic implementation.

Based on: Meta-Rewarding: Learning to Judge Judges with Self-Generated Meta-Judgments
Paper: https://arxiv.org/abs/2407.19594
Authors: Wu et al. (2024)

Meta-Rewarding uses a two-stage judgment process where the model first
evaluates content, then evaluates its own evaluation for reliability.

## Similarity to Original Paper:
- PRESERVED: Two-stage judgment process (evaluate, then meta-evaluate)
- PRESERVED: Self-assessment of judgment quality
- PRESERVED: Reliability scoring of evaluations
- SIMPLIFIED: No reward model training; uses prompting

## Implementation Choices:
1. Stage 1: Standard 5-dimension evaluation (clarity, accuracy, etc.)
2. Stage 2: Meta-assessment of the initial judgment
3. Meta checks: comprehensiveness, accuracy, biases, reliability, blind spots
4. Reliability score (0.0-1.0) indicates judgment confidence
5. Uses older Critic interface for backward compatibility

## Why This Approach:
- Two-stage process catches evaluation errors and biases
- More reliable for high-stakes text evaluation
- Self-awareness of judgment limitations
- Higher computational cost justified by better reliability
- Useful when evaluation quality is as important as the evaluation itself

"""

from typing import List, Optional, Union, Dict, Any
import re

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .base import BaseCritic, CriticConfig, create_prompt_with_format, CriticResponse


class MetaRewardingCritic(BaseCritic):
    """Implements Meta-Rewarding two-stage judgment approach."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[CriticConfig] = None,
    ):
        # Initialize with custom config for meta-rewarding
        if config is None:
            config = CriticConfig(
                response_format="json",
                base_confidence=0.6,
                context_weight=0.15,
                depth_weight=0.25,
            )
        super().__init__(model, temperature, config, provider, api_key)

    @property
    def name(self) -> str:
        return "meta_rewarding"

    async def _generate_critique(self, text: str, result: SifakaResult) -> str:
        """Generate two-stage meta-rewarding critique."""
        # Stage 1: Initial judgment
        initial_judgment = await self._get_initial_judgment(text)
        
        # Stage 2: Meta-judgment of the initial judgment
        meta_judgment = await self._get_meta_judgment(text, initial_judgment)
        
        # Combine both judgments into a structured response
        if self.config.response_format == "json":
            return self._format_as_json(initial_judgment, meta_judgment)
        else:
            return self._format_as_text(initial_judgment, meta_judgment)

    async def _get_initial_judgment(self, text: str) -> str:
        """Get initial judgment of the text."""
        prompt = f"""Evaluate this text for quality across multiple dimensions:

Text to evaluate:
{text}

Provide a comprehensive assessment covering:
1. Clarity and readability
2. Accuracy and factual correctness
3. Completeness and thoroughness  
4. Structure and organization
5. Engagement and effectiveness

Rate each dimension (1-5) and provide overall assessment.

Format as:
CLARITY: [score] - [brief explanation]
ACCURACY: [score] - [brief explanation]  
COMPLETENESS: [score] - [brief explanation]
STRUCTURE: [score] - [brief explanation]
ENGAGEMENT: [score] - [brief explanation]
OVERALL: [summary and suggestions]"""

        response = await self.client.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are a thorough text evaluator providing detailed assessments.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        return response.content

    async def _get_meta_judgment(self, text: str, initial_judgment: str) -> str:
        """Get meta-judgment evaluating the quality of the initial judgment."""
        prompt = f"""Now evaluate the quality and reliability of this text evaluation:

ORIGINAL TEXT:
{text}

INITIAL JUDGMENT:
{initial_judgment}

As a meta-judge, assess:
1. Is the evaluation comprehensive and fair?
2. Are the scores and ratings accurate?
3. Are there any biases or blind spots?
4. How reliable is this judgment?
5. What aspects might have been missed?

Provide a meta-evaluation and confidence score (0.0-1.0) for the initial judgment.

Format as:
META_ASSESSMENT: [evaluation of the judgment quality]
RELIABILITY: [confidence score 0.0-1.0]
CORRECTIONS: [any needed corrections or additions]"""

        response = await self.client.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are a meta-judge evaluating the quality of text evaluations.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        return response.content

    def _format_as_json(self, initial: str, meta: str) -> str:
        """Format two-stage judgment as JSON."""
        # Parse initial judgment
        initial_data = self._parse_initial_judgment(initial)
        
        # Parse meta judgment
        meta_data = self._parse_meta_judgment(meta)
        
        # Combine into final response
        import json
        return json.dumps({
            "feedback": f"{initial_data['overall']} Meta-assessment: {meta_data['assessment']}",
            "suggestions": initial_data['suggestions'] + meta_data['corrections'],
            "needs_improvement": meta_data['reliability'] < 0.7 or initial_data['avg_score'] < 3.5,
            "confidence": meta_data['reliability'],
            "metadata": {
                "initial_scores": initial_data['scores'],
                "meta_reliability": meta_data['reliability'],
                "dimensions_evaluated": 5
            }
        })
    
    def _format_as_text(self, initial: str, meta: str) -> str:
        """Format two-stage judgment as text."""
        return f"""INITIAL JUDGMENT:
{initial}

META-JUDGMENT:
{meta}"""
    
    def _parse_initial_judgment(self, initial: str) -> Dict[str, Any]:
        """Parse the initial judgment."""
        scores = {}
        suggestions = []
        overall = ""
        
        # Extract scores from initial judgment
        for line in initial.split("\n"):
            for dim in ["CLARITY", "ACCURACY", "COMPLETENESS", "STRUCTURE", "ENGAGEMENT"]:
                if dim in line.upper():
                    score_match = re.search(r"(\d+)", line)
                    if score_match:
                        scores[dim.lower()] = int(score_match.group(1))
            
            if "OVERALL:" in line.upper():
                overall = line.split("OVERALL:")[-1].strip()
        
        # Extract suggestions from overall section
        if overall:
            # Look for suggestions in the overall text
            if "suggest" in overall.lower() or "improve" in overall.lower():
                suggestions.append(overall)
        
        avg_score = sum(scores.values()) / len(scores) if scores else 3.0
        
        return {
            "scores": scores,
            "suggestions": suggestions,
            "overall": overall or "Quality assessment completed",
            "avg_score": avg_score
        }
    
    def _parse_meta_judgment(self, meta: str) -> Dict[str, Any]:
        """Parse the meta judgment."""
        assessment = ""
        reliability = 0.7
        corrections = []
        
        for line in meta.split("\n"):
            if "META_ASSESSMENT:" in line.upper():
                assessment = line.split("META_ASSESSMENT:")[-1].strip()
            elif "RELIABILITY:" in line.upper():
                conf_match = re.search(r"(\d*\.?\d+)", line)
                if conf_match:
                    try:
                        reliability = float(conf_match.group(1))
                        reliability = max(0.0, min(1.0, reliability))
                    except ValueError:
                        pass
            elif "CORRECTIONS:" in line.upper():
                correction_text = line.split("CORRECTIONS:")[-1].strip()
                if correction_text and correction_text != "[any needed corrections or additions]":
                    corrections.append(correction_text)
        
        return {
            "assessment": assessment or "Meta-evaluation completed",
            "reliability": reliability,
            "corrections": corrections
        }
