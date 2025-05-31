#!/usr/bin/env python3
"""
Thought Data Analyzer and Visualizer

This script analyzes and visualizes thought data from Sifaka's JSON files,
providing insights into critic feedback, model performance, and iteration patterns.
"""

import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import textwrap


@dataclass
class CriticAnalysis:
    """Analysis of a single critic's feedback"""

    name: str
    needs_improvement: bool
    confidence: float
    violations: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]
    processing_time_ms: Optional[int]


@dataclass
class ThoughtSummary:
    """Summary of a single thought iteration"""

    id: str
    iteration: int
    timestamp: str
    model_name: str
    prompt: str
    text_length: int
    has_validation_results: bool
    critic_count: int
    critics: List[CriticAnalysis]
    chain_id: str
    parent_id: Optional[str]


class ThoughtAnalyzer:
    """Analyzes and visualizes thought data"""

    def __init__(self, json_file: str):
        self.json_file = Path(json_file)
        self.data = self._load_data()
        self.thoughts = self._parse_thoughts()

    def _load_data(self) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {e}")

    def _parse_thoughts(self) -> List[ThoughtSummary]:
        """Parse thoughts from JSON data"""
        thoughts = []

        for thought_id, thought_data in self.data.items():
            # Parse critic feedback
            critics = []
            if thought_data.get("critic_feedback"):
                for critic_data in thought_data["critic_feedback"]:
                    critic = CriticAnalysis(
                        name=critic_data.get("critic_name", "Unknown"),
                        needs_improvement=critic_data.get("needs_improvement", False),
                        confidence=critic_data.get("confidence", 0.0),
                        violations=critic_data.get("violations", []),
                        suggestions=critic_data.get("suggestions", []),
                        metadata=critic_data.get("metadata", {}),
                        processing_time_ms=critic_data.get("processing_time_ms"),
                    )
                    critics.append(critic)

            # Create thought summary
            thought = ThoughtSummary(
                id=thought_id,
                iteration=thought_data.get("iteration", 0),
                timestamp=thought_data.get("timestamp", ""),
                model_name=thought_data.get("model_name", "Unknown"),
                prompt=thought_data.get("prompt", ""),
                text_length=len(thought_data.get("text", "")),
                has_validation_results=thought_data.get("validation_results") is not None,
                critic_count=len(critics),
                critics=critics,
                chain_id=thought_data.get("chain_id", ""),
                parent_id=thought_data.get("parent_id"),
            )
            thoughts.append(thought)

        # Sort by iteration
        thoughts.sort(key=lambda t: t.iteration)
        return thoughts

    def print_overview(self):
        """Print high-level overview of the thought data"""
        print("=" * 80)
        print("THOUGHT DATA OVERVIEW")
        print("=" * 80)

        print(f"📁 File: {self.json_file}")
        print(f"🧠 Total Thoughts: {len(self.thoughts)}")

        if self.thoughts:
            print(
                f"🔄 Iterations: {min(t.iteration for t in self.thoughts)} - {max(t.iteration for t in self.thoughts)}"
            )
            print(f"🤖 Models Used: {', '.join(set(t.model_name for t in self.thoughts))}")
            print(f"⏰ Time Range: {self.thoughts[0].timestamp} to {self.thoughts[-1].timestamp}")

        # Critic statistics
        all_critics = [critic.name for thought in self.thoughts for critic in thought.critics]
        unique_critics = set(all_critics)
        print(f"🔍 Critics Used: {', '.join(unique_critics)}")
        print(f"📊 Total Critic Evaluations: {len(all_critics)}")

        print()

    def print_iteration_summary(self):
        """Print summary of each iteration"""
        print("=" * 80)
        print("ITERATION SUMMARY")
        print("=" * 80)

        for thought in self.thoughts:
            print(f"\n🔄 Iteration {thought.iteration}")
            print(f"   ID: {thought.id[:8]}...")
            print(f"   Model: {thought.model_name}")
            print(f"   Response Length: {thought.text_length:,} characters")
            print(f"   Critics: {thought.critic_count}")

            if thought.critics:
                for critic in thought.critics:
                    status = "❌ Needs Improvement" if critic.needs_improvement else "✅ Approved"
                    print(f"     • {critic.name}: {status} (confidence: {critic.confidence:.1f})")

        print()

    def print_critic_analysis(self):
        """Print detailed analysis of critic feedback"""
        print("=" * 80)
        print("CRITIC ANALYSIS")
        print("=" * 80)

        for thought in self.thoughts:
            if not thought.critics:
                continue

            print(f"\n🔄 Iteration {thought.iteration} - Critic Feedback")
            print("-" * 60)

            for critic in thought.critics:
                print(f"\n🔍 {critic.name}")
                print(f"   Needs Improvement: {'Yes' if critic.needs_improvement else 'No'}")
                print(f"   Confidence: {critic.confidence:.1f}")

                if critic.violations:
                    print("   ⚠️  Violations:")
                    for violation in critic.violations:
                        wrapped = textwrap.fill(
                            violation,
                            width=70,
                            initial_indent="      • ",
                            subsequent_indent="        ",
                        )
                        print(wrapped)

                if critic.suggestions:
                    print("   💡 Suggestions:")
                    for suggestion in critic.suggestions:
                        wrapped = textwrap.fill(
                            suggestion,
                            width=70,
                            initial_indent="      • ",
                            subsequent_indent="        ",
                        )
                        print(wrapped)

                # Show metadata insights
                if critic.metadata:
                    self._print_metadata_insights(critic.metadata)

                # Special handling for MetaRewardingCritic
                if critic.name == "MetaRewardingCritic":
                    self._print_meta_rewarding_details(critic)

        print()

    def _print_meta_rewarding_details(self, critic: CriticAnalysis):
        """Print detailed analysis of MetaRewardingCritic feedback"""
        print("   🎯 Meta-Rewarding Analysis:")

        # Extract initial judgment and meta-judgment from feedback
        feedback = critic.metadata.get("initial_judgment", "")
        meta_judgment = critic.metadata.get("meta_judgment", "")

        if feedback:
            # Extract score from initial judgment
            if "Score:" in feedback:
                score_line = [line for line in feedback.split("\n") if "Score:" in line]
                if score_line:
                    print(f"      Initial Score: {score_line[0].split('Score:')[-1].strip()}")

            # Extract key strengths and weaknesses
            if "**Strengths:**" in feedback:
                strengths_section = feedback.split("**Strengths:**")[1].split("**Weaknesses:**")[0]
                strength_points = [
                    line.strip()
                    for line in strengths_section.split("\n")
                    if line.strip() and line.strip().startswith(("1.", "2.", "3.", "4.", "5."))
                ]
                if strength_points:
                    print("      ✅ Key Strengths:")
                    for point in strength_points[:3]:  # Show top 3
                        title = point.split("**")[1] if "**" in point else point
                        print(f"         • {title}")

            if "**Weaknesses:**" in feedback:
                weaknesses_section = feedback.split("**Weaknesses:**")[1].split(
                    "**Overall Assessment**"
                )[0]
                weakness_points = [
                    line.strip()
                    for line in weaknesses_section.split("\n")
                    if line.strip() and line.strip().startswith(("1.", "2.", "3.", "4.", "5."))
                ]
                if weakness_points:
                    print("      ❌ Key Weaknesses:")
                    for point in weakness_points[:3]:  # Show top 3
                        title = point.split("**")[1] if "**" in point else point
                        print(f"         • {title}")

        if meta_judgment:
            # Extract meta-judgment score
            if "judgment quality is rated as" in meta_judgment:
                meta_score = (
                    meta_judgment.split("judgment quality is rated as")[-1].split(".")[0].strip()
                )
                print(f"      Meta-Judgment Score: {meta_score}")
            elif "Thus, the judgment quality is rated as" in meta_judgment:
                meta_score = (
                    meta_judgment.split("Thus, the judgment quality is rated as")[-1]
                    .split(".")[0]
                    .strip()
                )
                print(f"      Meta-Judgment Score: {meta_score}")

    def _print_metadata_insights(self, metadata: Dict[str, Any]):
        """Print insights from critic metadata"""
        print("   📋 Metadata Insights:")

        # Show scores if available
        if "initial_score" in metadata:
            print(f"      Initial Score: {metadata['initial_score']}")

        # Show criteria used
        if "judgment_criteria" in metadata:
            criteria = metadata["judgment_criteria"]
            print(f"      Judgment Criteria: {', '.join(criteria)}")

        if "meta_judgment_criteria" in metadata:
            meta_criteria = metadata["meta_judgment_criteria"]
            print(f"      Meta-Judgment Criteria: {', '.join(meta_criteria)}")

        # Show if base critic was used
        if "base_critic_used" in metadata:
            print(f"      Base Critic Used: {metadata['base_critic_used']}")

        # Show meta judge model
        if "meta_judge_model" in metadata:
            print(f"      Meta Judge Model: {metadata['meta_judge_model']}")

    def print_prompt_evolution(self):
        """Show how prompts evolved across iterations"""
        print("=" * 80)
        print("PROMPT EVOLUTION")
        print("=" * 80)

        for thought in self.thoughts:
            print(f"\n🔄 Iteration {thought.iteration}")
            print("-" * 60)

            # Show original prompt
            if thought.prompt:
                wrapped_prompt = textwrap.fill(
                    thought.prompt,
                    width=70,
                    initial_indent="📝 Original: ",
                    subsequent_indent="          ",
                )
                print(wrapped_prompt)

            # Show model prompt if different and available
            model_prompt = self.data[thought.id].get("model_prompt", "")
            if model_prompt and model_prompt != thought.prompt:
                # Extract just the user part for brevity
                if "User:" in model_prompt:
                    user_part = model_prompt.split("User:")[-1].strip()
                    if len(user_part) > 200:
                        user_part = user_part[:200] + "..."
                    wrapped_model = textwrap.fill(
                        user_part,
                        width=70,
                        initial_indent="🤖 Model:    ",
                        subsequent_indent="          ",
                    )
                    print(wrapped_model)

        print()

    def print_visual_timeline(self):
        """Print a visual timeline of the thought process"""
        print("=" * 80)
        print("VISUAL TIMELINE")
        print("=" * 80)

        for i, thought in enumerate(self.thoughts):
            # Timeline connector
            if i == 0:
                print("┌─ START")
            else:
                print("│")
                print("├─ ITERATION", thought.iteration)

            # Thought details
            print(f"│  🤖 Model: {thought.model_name}")
            print(f"│  📝 Response: {thought.text_length:,} chars")

            # Critic results
            if thought.critics:
                for critic in thought.critics:
                    status_icon = "❌" if critic.needs_improvement else "✅"
                    print(f"│  {status_icon} {critic.name} (conf: {critic.confidence:.1f})")

                    # Show key metadata for MetaRewardingCritic
                    if critic.name == "MetaRewardingCritic" and critic.metadata:
                        initial_score = critic.metadata.get("initial_score")
                        if initial_score:
                            print(f"│     📊 Score: {initial_score}/10")
            else:
                print("│  🔍 No critics applied")

            print("│")

        print("└─ END")
        print()

    def export_summary(self, output_file: str):
        """Export analysis summary to a file"""
        with open(output_file, "w", encoding="utf-8") as f:
            # Redirect print to file
            import sys

            original_stdout = sys.stdout
            sys.stdout = f

            try:
                self.print_overview()
                self.print_visual_timeline()
                self.print_iteration_summary()
                self.print_critic_analysis()
                self.print_prompt_evolution()
            finally:
                sys.stdout = original_stdout

        print(f"📄 Analysis exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize Sifaka thought data")
    parser.add_argument("json_file", help="Path to the thought JSON file")
    parser.add_argument("--export", "-e", help="Export analysis to file")
    parser.add_argument("--overview-only", "-o", action="store_true", help="Show only overview")

    args = parser.parse_args()

    try:
        analyzer = ThoughtAnalyzer(args.json_file)

        if args.overview_only:
            analyzer.print_overview()
        else:
            analyzer.print_overview()
            analyzer.print_visual_timeline()
            analyzer.print_iteration_summary()
            analyzer.print_critic_analysis()
            analyzer.print_prompt_evolution()

        if args.export:
            analyzer.export_summary(args.export)

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
