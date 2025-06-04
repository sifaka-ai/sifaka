#!/usr/bin/env python3
"""
Meta-Rewarding Specific Analyzer

This script provides specialized analysis for MetaRewardingCritic data,
focusing on the dual-layer evaluation structure and score progression.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MetaRewardingAnalysis:
    """Analysis of a single meta-rewarding evaluation"""

    iteration: int
    initial_score: float
    meta_score: Optional[str]
    judgment_criteria: List[str]
    meta_judgment_criteria: List[str]
    strengths: List[str]
    weaknesses: List[str]
    base_critic_used: bool
    meta_judge_model: str
    confidence: float
    needs_improvement: bool


class MetaRewardingAnalyzer:
    """Specialized analyzer for MetaRewardingCritic data"""

    def __init__(self, json_file: str):
        self.json_file = Path(json_file)
        self.data = self._load_data()
        self.analyses = self._parse_meta_rewarding_data()

    def _load_data(self) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {e}")

    def _parse_meta_rewarding_data(self) -> List[MetaRewardingAnalysis]:
        """Parse meta-rewarding specific data from thoughts"""
        analyses = []

        # Sort thoughts by iteration
        thoughts = []
        for thought_id, thought_data in self.data.items():
            thought_data["id"] = thought_id
            thoughts.append(thought_data)
        thoughts.sort(key=lambda t: t.get("iteration", 0))

        for thought in thoughts:
            iteration = thought.get("iteration", 0)
            critic_feedback = thought.get("critic_feedback", [])

            for critic in critic_feedback:
                if critic.get("critic_name") == "MetaRewardingCritic":
                    analysis = self._extract_meta_rewarding_analysis(critic, iteration)
                    if analysis:
                        analyses.append(analysis)

        return analyses

    def _extract_meta_rewarding_analysis(
        self, critic: dict, iteration: int
    ) -> Optional[MetaRewardingAnalysis]:
        """Extract structured analysis from meta-rewarding critic data"""
        metadata = critic.get("metadata", {})

        # Extract initial score
        initial_score = metadata.get("initial_score", 0.0)

        # Extract meta-judgment score
        meta_judgment = metadata.get("meta_judgment", "")
        meta_score = self._extract_meta_score(meta_judgment)

        # Extract criteria
        judgment_criteria = metadata.get("judgment_criteria", [])
        meta_judgment_criteria = metadata.get("meta_judgment_criteria", [])

        # Extract strengths and weaknesses
        initial_judgment = metadata.get("initial_judgment", "")
        strengths, weaknesses = self._extract_strengths_weaknesses(initial_judgment)

        return MetaRewardingAnalysis(
            iteration=iteration,
            initial_score=initial_score,
            meta_score=meta_score,
            judgment_criteria=judgment_criteria,
            meta_judgment_criteria=meta_judgment_criteria,
            strengths=strengths,
            weaknesses=weaknesses,
            base_critic_used=metadata.get("base_critic_used", False),
            meta_judge_model=metadata.get("meta_judge_model", "Unknown"),
            confidence=critic.get("confidence", 0.0),
            needs_improvement=critic.get("needs_improvement", True),
        )

    def _extract_meta_score(self, meta_judgment: str) -> Optional[str]:
        """Extract meta-judgment score from text"""
        if not meta_judgment:
            return None

        # Try multiple patterns
        patterns = [
            r"judgment quality is rated as (\d+(?:\.\d+)?/10)",
            r"Thus, the judgment quality is rated as (\d+(?:\.\d+)?/10)",
            r"(\d+(?:\.\d+)?)/10",
        ]

        for pattern in patterns:
            match = re.search(pattern, meta_judgment)
            if match:
                return match.group(1) if "/10" in match.group(1) else f"{match.group(1)}/10"

        return None

    def _extract_strengths_weaknesses(self, initial_judgment: str) -> tuple[List[str], List[str]]:
        """Extract strengths and weaknesses from initial judgment"""
        strengths = []
        weaknesses = []

        if "**Strengths:**" in initial_judgment:
            strengths_section = initial_judgment.split("**Strengths:**")[1].split(
                "**Weaknesses:**"
            )[0]
            strength_points = [
                line.strip()
                for line in strengths_section.split("\n")
                if line.strip() and line.strip().startswith(("1.", "2.", "3.", "4.", "5."))
            ]
            strengths = [
                point.split("**")[1] if "**" in point else point for point in strength_points
            ]

        if "**Weaknesses:**" in initial_judgment:
            weaknesses_section = initial_judgment.split("**Weaknesses:**")[1].split(
                "**Overall Assessment**"
            )[0]
            weakness_points = [
                line.strip()
                for line in weaknesses_section.split("\n")
                if line.strip() and line.strip().startswith(("1.", "2.", "3.", "4.", "5."))
            ]
            weaknesses = [
                point.split("**")[1] if "**" in point else point for point in weakness_points
            ]

        return strengths, weaknesses

    def print_overview(self):
        """Print overview of meta-rewarding analysis"""
        print("=" * 80)
        print("META-REWARDING ANALYSIS OVERVIEW")
        print("=" * 80)

        print(f"📁 File: {self.json_file}")
        print(f"🎯 Meta-Rewarding Evaluations: {len(self.analyses)}")

        if self.analyses:
            initial_scores = [a.initial_score for a in self.analyses]
            print(f"📊 Score Range: {min(initial_scores):.1f} - {max(initial_scores):.1f}")
            print(f"📈 Average Score: {sum(initial_scores) / len(initial_scores):.1f}")

            # Show score progression
            iterations = sorted(set(a.iteration for a in self.analyses))
            print(f"🔄 Iterations: {min(iterations)} - {max(iterations)}")

            # Show models used
            models = set(a.meta_judge_model for a in self.analyses)
            print(f"🤖 Meta Judge Models: {', '.join(models)}")

        print()

    def print_score_progression(self):
        """Print score progression across iterations"""
        print("=" * 80)
        print("SCORE PROGRESSION")
        print("=" * 80)

        # Group by iteration
        by_iteration = {}
        for analysis in self.analyses:
            if analysis.iteration not in by_iteration:
                by_iteration[analysis.iteration] = []
            by_iteration[analysis.iteration].append(analysis)

        for iteration in sorted(by_iteration.keys()):
            analyses = by_iteration[iteration]
            print(f"\n🔄 Iteration {iteration}")
            print("-" * 40)

            for i, analysis in enumerate(analyses):
                print(f"   Evaluation {i+1}:")
                print(f"      📊 Initial Score: {analysis.initial_score}/10")
                if analysis.meta_score:
                    print(f"      🔄 Meta Score: {analysis.meta_score}")
                print(f"      🎯 Confidence: {analysis.confidence:.1f}")
                print(
                    f"      ✅ Status: {'Needs Improvement' if analysis.needs_improvement else 'Approved'}"
                )
                print()

        print()
