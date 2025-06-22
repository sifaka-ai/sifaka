"""Self-Consistency critic implementation.

Based on: Self-Consistency Improves Chain of Thought Reasoning in Language Models
Paper: https://arxiv.org/abs/2203.11171
Authors: Wang et al. (2022)

Self-Consistency generates multiple reasoning paths and uses majority
consensus to improve reliability and identify inconsistencies.

## Similarity to Original Paper:
- PRESERVED: Multiple independent evaluation paths
- PRESERVED: Consensus through agreement/voting
- ADAPTED: Applied to critique rather than problem-solving
- PRESERVED: Higher temperature for evaluation diversity

## Implementation Choices:
1. 3 independent evaluations by default (configurable)
2. Temperature 0.8 for diversity (vs typical 0.3-0.5)
3. Consistency metrics: score variance and priority agreement
4. Common themes extracted across evaluations
5. Confidence based on consistency level (variance-based)

## Why This Approach:
- Multiple evaluations catch different aspects/issues
- Consensus provides more robust feedback
- Higher temperature ensures diverse critical perspectives
- Variance metric naturally measures agreement
- Trades computation cost for evaluation reliability

"""

from typing import List, Optional, Union, Dict, Any
from collections import Counter
import re
import asyncio
import json

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .base import BaseCritic, CriticConfig, create_prompt_with_format, CriticResponse


class SelfConsistencyCritic(BaseCritic):
    """Implements Self-Consistency approach for consensus-based critique."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.8,
        num_samples: int = 3,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[CriticConfig] = None,
    ):
        # Initialize with custom config for self-consistency
        if config is None:
            config = CriticConfig(
                response_format="json",
                base_confidence=0.7,
                context_weight=0.2,
                depth_weight=0.1,
            )
        # Higher temperature for diversity
        super().__init__(model, temperature, config, provider, api_key)
        self.num_samples = num_samples  # Number of independent evaluations

    @property
    def name(self) -> str:
        return "self_consistency"

    async def _generate_critique(self, text: str, result: SifakaResult) -> str:
        """Generate critique using multiple independent evaluations."""
        # Generate multiple evaluations in parallel (async batching)
        evaluation_tasks = [
            self._get_independent_evaluation(text, i + 1)
            for i in range(self.num_samples)
        ]
        evaluations = await asyncio.gather(*evaluation_tasks)
        
        # Analyze consistency and build consensus
        consensus_data = self._analyze_consistency(evaluations)
        
        # Format response based on config
        if self.config.response_format == "json":
            return json.dumps({
                "feedback": consensus_data["feedback"],
                "suggestions": consensus_data["suggestions"],
                "needs_improvement": consensus_data["consistency_score"] < 0.7,
                "confidence": consensus_data["consistency_score"],
                "metadata": {
                    "num_samples": self.num_samples,
                    "consistency_score": consensus_data["consistency_score"],
                    "common_themes": consensus_data["common_themes"],
                    "score_variance": consensus_data["score_variance"]
                }
            })
        else:
            return self._format_as_text(consensus_data)

    async def _get_independent_evaluation(self, text: str, sample_num: int) -> str:
        """Get one independent evaluation of the text."""
        prompt = f"""Independently evaluate this text for quality and identify any issues:

Text to evaluate:
{text}

Provide your assessment considering:
1. Overall quality (rate 1-5)
2. Key strengths
3. Main weaknesses  
4. Specific improvement areas
5. Priority level for changes (HIGH/MEDIUM/LOW)

Be thorough and specific in your evaluation.

Format as:
QUALITY_SCORE: [1-5]
STRENGTHS: [key strengths]
WEAKNESSES: [main issues]
IMPROVEMENTS: [specific suggestions]
PRIORITY: [HIGH/MEDIUM/LOW]"""

        response = await self.client.complete(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an independent text evaluator (Assessment #{sample_num}). Provide an honest, thorough evaluation.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        return response.content

    def _analyze_consistency(self, evaluations: List[str]) -> Dict[str, Any]:
        """Analyze consistency across multiple evaluations."""
        # Extract key data from evaluations
        scores = []
        priorities = []
        strengths = []
        weaknesses = []
        improvements = []

        for eval_text in evaluations:
            lines = eval_text.split("\n")
            for line in lines:
                line = line.strip()

                # Extract quality scores
                if line.startswith("QUALITY_SCORE:"):
                    score_match = re.search(r"(\d+)", line)
                    if score_match:
                        scores.append(int(score_match.group(1)))

                # Extract priorities
                elif line.startswith("PRIORITY:"):
                    priority = line.replace("PRIORITY:", "").strip()
                    if priority in ["HIGH", "MEDIUM", "LOW"]:
                        priorities.append(priority)

                # Collect other feedback
                elif line.startswith("STRENGTHS:"):
                    strengths.append(line.replace("STRENGTHS:", "").strip())
                elif line.startswith("WEAKNESSES:"):
                    weaknesses.append(line.replace("WEAKNESSES:", "").strip())
                elif line.startswith("IMPROVEMENTS:"):
                    improvements.append(line.replace("IMPROVEMENTS:", "").strip())

        # Calculate consistency metrics
        score_consistency = self._calculate_score_consistency(scores)
        priority_consistency = self._calculate_priority_consistency(priorities)

        # Overall consistency (average of metrics)
        consistency_score = (score_consistency + priority_consistency) / 2

        # Build consensus feedback
        feedback = self._build_consensus_feedback(
            scores, priorities, strengths, weaknesses, consistency_score
        )

        # Build consensus suggestions
        suggestions = self._build_consensus_suggestions(improvements, consistency_score)
        
        # Extract common themes
        common_themes = self._extract_common_themes(weaknesses)
        
        # Calculate score variance
        avg_score = sum(scores) / len(scores) if scores else 3.0
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores) if scores else 0.0
        
        return {
            "feedback": feedback,
            "suggestions": suggestions,
            "consistency_score": consistency_score,
            "common_themes": common_themes,
            "score_variance": score_variance
        }
    
    def _format_as_text(self, consensus_data: Dict[str, Any]) -> str:
        """Format consensus data as structured text."""
        return f"""CONSENSUS EVALUATION:
{consensus_data['feedback']}

SUGGESTIONS:
{chr(10).join('- ' + s for s in consensus_data['suggestions'])}

CONSISTENCY: {consensus_data['consistency_score']:.2f}"""
    
    def _build_consensus_suggestions(self, improvements: List[str], consistency_score: float) -> List[str]:
        """Build consensus suggestions from multiple evaluations."""
        suggestions = []
        
        # Add consistency-based suggestions
        if consistency_score < 0.5:
            suggestions.append("High evaluation inconsistency detected - consider fundamental revision")
        elif consistency_score < 0.7:
            suggestions.append("Moderate evaluation inconsistency - review conflicting areas")
        
        # Extract common improvement themes
        if improvements:
            improvement_words = []
            for improvement in improvements:
                improvement_words.extend(improvement.lower().split())
            
            common_words = [
                word for word, count in Counter(improvement_words).most_common(3)
                if count > 1 and len(word) > 3
            ]
            
            if common_words:
                suggestions.append(f"Common improvement themes: {', '.join(common_words)}")
        
        # Fallback
        if not suggestions:
            suggestions = ["Multiple independent evaluations completed"]
        
        return suggestions
    
    def _extract_common_themes(self, weaknesses: List[str]) -> List[str]:
        """Extract common themes from weaknesses."""
        if not weaknesses:
            return []
        
        # Simple theme extraction - can be enhanced
        all_words = []
        for weakness in weaknesses:
            all_words.extend(weakness.lower().split())
        
        # Get words that appear multiple times
        word_counts = Counter(all_words)
        common_themes = [
            word for word, count in word_counts.most_common(5)
            if count > 1 and len(word) > 4 and word not in ["text", "that", "this", "with", "from"]
        ]
        
        return common_themes

    def _calculate_score_consistency(self, scores: List[int]) -> float:
        """Calculate consistency of quality scores."""
        if len(scores) < 2:
            return 1.0

        # Calculate variance in scores
        avg_score = sum(scores) / len(scores)
        variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)

        # Convert variance to consistency (0-1, higher is more consistent)
        # Max variance for 1-5 scale is 4, so we normalize
        consistency = max(0.0, 1.0 - (variance / 4.0))
        return consistency

    def _calculate_priority_consistency(self, priorities: List[str]) -> float:
        """Calculate consistency of priority assessments."""
        if len(priorities) < 2:
            return 1.0

        # Count most common priority
        priority_counts = Counter(priorities)
        most_common_count = priority_counts.most_common(1)[0][1]

        # Consistency is the proportion of majority agreement
        consistency = most_common_count / len(priorities)
        return consistency

    def _build_consensus_feedback(
        self,
        scores: List[int],
        priorities: List[str],
        strengths: List[str],
        weaknesses: List[str],
        consistency: float,
    ) -> str:
        """Build consensus feedback from multiple evaluations."""
        feedback_parts = []

        # Score consensus
        if scores:
            avg_score = sum(scores) / len(scores)
            score_range = (
                f"{min(scores)}-{max(scores)}"
                if min(scores) != max(scores)
                else str(scores[0])
            )
            feedback_parts.append(
                f"Quality consensus: Average {avg_score:.1f}/5 (range: {score_range})"
            )

        # Priority consensus
        if priorities:
            priority_counts = Counter(priorities)
            most_common_priority = priority_counts.most_common(1)[0][0]
            feedback_parts.append(f"Priority consensus: {most_common_priority}")

        # Consistency assessment
        feedback_parts.append(f"Evaluation consistency: {consistency:.2f}")

        # Common themes
        if strengths:
            feedback_parts.append("Identified strengths across evaluations")
        if weaknesses:
            feedback_parts.append("Identified weaknesses across evaluations")

        return " ".join(feedback_parts)

