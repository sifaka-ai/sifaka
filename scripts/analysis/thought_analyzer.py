#!/usr/bin/env python3
"""
Thought Data Analyzer and Visualizer

This script analyzes and visualizes thought data from Sifaka's JSON files,
providing insights into critic feedback, model performance, and iteration patterns.
"""

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        """Parse thoughts from JSON data and show all critic feedback for each iteration"""
        thoughts = []

        for thought_id, thought_data in self.data.items():
            # Parse critic feedback - show ALL feedback for this iteration
            critics = []
            if thought_data.get("critic_feedback"):
                # Show all critic feedback for this iteration
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
                text_length=len(thought_data.get("text") or ""),
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

        # Group thoughts by chain_id
        chains = {}
        for thought in self.thoughts:
            chain_id = thought.chain_id
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(thought)

        print(f"🔗 Chains: {len(chains)}")

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

        # Show chain details
        print()
        for i, (chain_id, chain_thoughts) in enumerate(chains.items(), 1):
            chain_thoughts.sort(key=lambda t: t.iteration)
            iterations = [t.iteration for t in chain_thoughts]
            timestamps = [t.timestamp for t in chain_thoughts if t.timestamp]
            print(
                f"🔗 Chain {i} ({chain_id[:8]}...): {len(chain_thoughts)} thoughts, iterations {min(iterations)}-{max(iterations)}"
            )
            if timestamps:
                print(f"   ⏰ {min(timestamps)} to {max(timestamps)}")

        print()

    def print_iteration_summary(self):
        """Print summary of each iteration grouped by chain"""
        print("=" * 80)
        print("ITERATION SUMMARY")
        print("=" * 80)

        # Group thoughts by chain_id
        chains = {}
        for thought in self.thoughts:
            chain_id = thought.chain_id
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(thought)

        for i, (chain_id, chain_thoughts) in enumerate(chains.items(), 1):
            print(f"\n🔗 Chain {i} ({chain_id[:8]}...):")
            print("-" * 40)

            chain_thoughts.sort(key=lambda t: t.iteration)
            for thought in chain_thoughts:
                print(f"\n🔄 Iteration {thought.iteration}")
                print(f"   ID: {thought.id[:8]}...")
                print(f"   Model: {thought.model_name}")
                print(f"   Response Length: {thought.text_length:,} characters")
                print(f"   Critics: {thought.critic_count}")

                if thought.critics:
                    for critic in thought.critics:
                        status = (
                            "❌ Needs Improvement" if critic.needs_improvement else "✅ Approved"
                        )
                        print(
                            f"     • {critic.name}: {status} (confidence: {critic.confidence:.1f})"
                        )

        print()

    def print_critic_analysis(self):
        """Print detailed analysis of critic feedback grouped by chain"""
        print("=" * 80)
        print("CRITIC ANALYSIS")
        print("=" * 80)

        # Group thoughts by chain_id
        chains = {}
        for thought in self.thoughts:
            chain_id = thought.chain_id
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(thought)

        for i, (chain_id, chain_thoughts) in enumerate(chains.items(), 1):
            print(f"\n🔗 Chain {i} ({chain_id[:8]}...):")
            print("=" * 40)

            chain_thoughts.sort(key=lambda t: t.iteration)
            for thought in chain_thoughts:
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

                    # Use modular critic analysis based on type
                    if critic.name == "SelfConsistencyCritic":
                        self._print_self_consistency_details(critic)
                    elif critic.name == "ReflexionCritic":
                        self._print_reflexion_details(critic)
                    elif critic.name == "MetaRewardingCritic":
                        self._print_meta_rewarding_details(critic)
                    elif critic.name == "ConstitutionalCritic":
                        self._print_constitutional_details(critic)
                    elif critic.name == "NCriticsCritic":
                        self._print_n_critics_details(critic)
                    elif critic.name == "PromptCritic":
                        self._print_prompt_critic_details(critic)
                    elif critic.name == "SelfRAGCritic":
                        self._print_self_rag_details(critic)
                    else:
                        self._print_generic_critic_details(critic)

        print()

    def _print_self_consistency_details(self, critic: CriticAnalysis):
        """Print detailed analysis of SelfConsistencyCritic feedback"""
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

        # Show consensus statistics
        if critic.metadata.get("consensus_stats"):
            stats = critic.metadata["consensus_stats"]["stats"]
            print("   📊 Consensus Analysis:")
            print(f"      Total Critiques: {stats.get('total_critiques', 'N/A')}")
            print(f"      Consensus Items: {stats.get('consensus_items', 'N/A')}")
            print(f"      Agreement Ratio: {stats.get('agreement_ratio', 0):.1%}")

    def _print_reflexion_details(self, critic: CriticAnalysis):
        """Print detailed analysis of ReflexionCritic feedback"""
        metadata = critic.metadata

        print("   🔄 Reflexion Analysis:")
        print(f"      Trial Number: {metadata.get('trial_number', 'N/A')}")
        print(f"      Memory Size: {metadata.get('memory_size', 'N/A')}")

        # Show reflection if available
        reflection = metadata.get("reflection", "")
        if reflection:
            print("   🧠 Self-Reflection:")
            # Show first few lines of reflection
            reflection_lines = reflection.split("\n")[:3]
            for line in reflection_lines:
                if line.strip():
                    wrapped = textwrap.fill(
                        line.strip(),
                        width=70,
                        initial_indent="      ",
                        subsequent_indent="      ",
                    )
                    print(wrapped)
            if len(reflection.split("\n")) > 3:
                print("      ...")

        # Show memory sessions
        reflexion_memory = metadata.get("reflexion_memory", {})
        sessions = reflexion_memory.get("sessions", [])
        if sessions:
            print("   📚 Memory Sessions:")
            for session in sessions[-2:]:  # Show last 2 sessions
                trial_num = session.get("trial_number", "Unknown")
                summary = session.get("summary", "No summary")
                print(f"      Trial {trial_num}: {summary}")

    def _print_generic_critic_details(self, critic: CriticAnalysis):
        """Print generic critic details for unknown critic types"""
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

    def _print_meta_rewarding_details(self, critic: CriticAnalysis):
        """Print detailed analysis of MetaRewardingCritic feedback"""
        print("   🎯 Meta-Rewarding Analysis:")

        # Show initial score prominently
        initial_score = critic.metadata.get("initial_score", "N/A")
        print(f"      📊 Initial Score: {initial_score}/10")

        # Show judgment criteria
        judgment_criteria = critic.metadata.get("judgment_criteria", [])
        if judgment_criteria:
            print("      📋 Judgment Criteria:")
            for criterion in judgment_criteria[:5]:  # Show first 5
                print(f"         • {criterion}")

        # Extract initial judgment and meta-judgment from feedback
        initial_judgment = critic.metadata.get("initial_judgment", "")
        meta_judgment = critic.metadata.get("meta_judgment", "")

        if initial_judgment:
            # Extract key strengths and weaknesses from structured initial judgment
            if "**Strengths:**" in initial_judgment:
                strengths_section = initial_judgment.split("**Strengths:**")[1].split(
                    "**Weaknesses:**"
                )[0]
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

            if "**Weaknesses:**" in initial_judgment:
                weaknesses_section = initial_judgment.split("**Weaknesses:**")[1].split(
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
            # Extract meta-judgment score with multiple patterns
            meta_score = None
            if "judgment quality is rated as" in meta_judgment:
                meta_score = (
                    meta_judgment.split("judgment quality is rated as")[-1].split(".")[0].strip()
                )
            elif "Thus, the judgment quality is rated as" in meta_judgment:
                meta_score = (
                    meta_judgment.split("Thus, the judgment quality is rated as")[-1]
                    .split(".")[0]
                    .strip()
                )
            elif "Meta-judgment:" in meta_judgment and "/10" in meta_judgment:
                # Look for patterns like "8/10" in meta-judgment
                import re

                score_match = re.search(r"(\d+(?:\.\d+)?)/10", meta_judgment)
                if score_match:
                    meta_score = score_match.group(1) + "/10"

            if meta_score:
                print(f"      🔄 Meta-Judgment Score: {meta_score}")

        # Show technical details
        base_critic_used = critic.metadata.get("base_critic_used", "N/A")
        meta_judge_model = critic.metadata.get("meta_judge_model", "N/A")
        print(f"      🔧 Base Critic Used: {base_critic_used}")
        print(f"      🤖 Meta Judge Model: {meta_judge_model}")

    def _print_n_critics_details(self, critic: CriticAnalysis):
        """Print detailed analysis of NCriticsCritic feedback"""
        print("   👥 N Critics Ensemble Analysis:")

        # Show ensemble score prominently
        aggregated_score = critic.metadata.get("aggregated_score", "N/A")
        improvement_threshold = critic.metadata.get("improvement_threshold", "N/A")
        num_critics = critic.metadata.get("num_critics", "N/A")

        print(f"      📊 Ensemble Score: {aggregated_score:.1f}/10")
        print(f"      🎯 Improvement Threshold: {improvement_threshold}")
        print(f"      👥 Number of Critics: {num_critics}")

        # Show individual critic feedback
        critic_feedback = critic.metadata.get("critic_feedback", [])
        if critic_feedback:
            print(f"      📋 Individual Critics ({len(critic_feedback)}):")
            for i, individual_critic in enumerate(critic_feedback):
                role = individual_critic.get("role", "Unknown Critic")
                score = individual_critic.get("score", "N/A")
                needs_improvement = individual_critic.get("needs_improvement", False)
                issues = individual_critic.get("issues", [])
                suggestions = individual_critic.get("suggestions", [])

                status_icon = "❌" if needs_improvement else "✅"
                print(f"         {i+1}. {status_icon} {role}")
                print(f"            Score: {score}/10")

                if issues:
                    print(f"            Issues ({len(issues)}):")
                    for issue in issues[:2]:  # Show first 2 issues
                        wrapped = textwrap.fill(
                            issue,
                            width=60,
                            initial_indent="              • ",
                            subsequent_indent="                ",
                        )
                        print(wrapped)
                    if len(issues) > 2:
                        print(f"              ... and {len(issues) - 2} more")

                if suggestions:
                    print(f"            Suggestions ({len(suggestions)}):")
                    for suggestion in suggestions[:2]:  # Show first 2 suggestions
                        wrapped = textwrap.fill(
                            suggestion,
                            width=60,
                            initial_indent="              • ",
                            subsequent_indent="                ",
                        )
                        print(wrapped)
                    if len(suggestions) > 2:
                        print(f"              ... and {len(suggestions) - 2} more")

    def _print_constitutional_details(self, critic: CriticAnalysis):
        """Print detailed analysis of ConstitutionalCritic feedback"""
        print("   ⚖️ Constitutional Analysis:")

        # Show principle violations
        principle_violations = critic.metadata.get("principle_violations", [])
        if principle_violations:
            print(f"      📜 Principle Violations ({len(principle_violations)}):")
            for violation in principle_violations[:5]:  # Show first 5
                description = violation.get("description", "No description")
                severity = violation.get("severity", "medium")

                wrapped = textwrap.fill(
                    description,
                    width=70,
                    initial_indent=f"         • [{severity.upper()}] ",
                    subsequent_indent="           ",
                )
                print(wrapped)

        # Show principles evaluated
        principles_evaluated = critic.metadata.get("principles_evaluated", 0)
        if principles_evaluated:
            print(f"      📊 Principles Evaluated: {principles_evaluated}")

        # Show strict mode
        strict_mode = critic.metadata.get("strict_mode", False)
        print(f"      🔒 Strict Mode: {'Enabled' if strict_mode else 'Disabled'}")

        # Show constitutional compliance rate
        if principle_violations and principles_evaluated:
            compliance_rate = (
                principles_evaluated - len(principle_violations)
            ) / principles_evaluated
            print(f"      ✅ Compliance Rate: {compliance_rate:.1%}")

        # Show issues and suggestions if available
        if critic.violations:
            print("      ⚠️ Constitutional Issues:")
            for violation in critic.violations[:3]:
                wrapped = textwrap.fill(
                    violation,
                    width=70,
                    initial_indent="         • ",
                    subsequent_indent="           ",
                )
                print(wrapped)

        if critic.suggestions:
            print("      💡 Constitutional Suggestions:")
            for suggestion in critic.suggestions[:3]:
                wrapped = textwrap.fill(
                    suggestion,
                    width=70,
                    initial_indent="         • ",
                    subsequent_indent="           ",
                )
                print(wrapped)

    def _print_prompt_critic_details(self, critic: CriticAnalysis):
        """Print detailed analysis of PromptCritic feedback"""
        print("   📝 Prompt Critic Analysis:")

        # Show evaluation criteria
        criteria = critic.metadata.get("criteria", [])
        if criteria:
            print(f"      🎯 Evaluation Criteria ({len(criteria)}):")
            for criterion in criteria:
                wrapped = textwrap.fill(
                    criterion,
                    width=70,
                    initial_indent="         • ",
                    subsequent_indent="           ",
                )
                print(wrapped)

        # Show issues and suggestions
        if critic.violations:
            print("      ⚠️ Issues Identified:")
            for violation in critic.violations[:3]:
                wrapped = textwrap.fill(
                    violation,
                    width=70,
                    initial_indent="         • ",
                    subsequent_indent="           ",
                )
                print(wrapped)

        if critic.suggestions:
            print("      💡 Improvement Suggestions:")
            for suggestion in critic.suggestions[:3]:
                wrapped = textwrap.fill(
                    suggestion,
                    width=70,
                    initial_indent="         • ",
                    subsequent_indent="           ",
                )
                print(wrapped)

        # Show system prompt (truncated)
        system_prompt = critic.metadata.get("system_prompt", "")
        if system_prompt:
            print("      🤖 System Prompt:")
            truncated_prompt = (
                system_prompt[:150] + "..." if len(system_prompt) > 150 else system_prompt
            )
            wrapped = textwrap.fill(
                truncated_prompt,
                width=70,
                initial_indent="         ",
                subsequent_indent="         ",
            )
            print(wrapped)

    def _print_self_rag_details(self, critic: CriticAnalysis):
        """Print detailed analysis of SelfRAGCritic feedback"""
        print("   🔍 Self-RAG Critic Analysis:")

        # Show Self-RAG reflection tokens
        rag_assessments = critic.metadata.get("rag_assessments", {})
        if rag_assessments:
            print("      🎯 Self-RAG Reflection Tokens:")
            for token_type, token_value in rag_assessments.items():
                print(f"         • {token_type.title()}: {token_value}")

        # Show utility score with color coding
        utility_score = critic.metadata.get("utility_score")
        if utility_score is not None:
            score_emoji = "🟢" if utility_score >= 4 else "🟡" if utility_score >= 3 else "🔴"
            print(f"      ⭐ Utility Score: {score_emoji} {utility_score}/5")

        # Show retrieval information
        retrieval_needed = critic.metadata.get("retrieval_needed")
        retrieved_docs_count = critic.metadata.get("retrieved_docs_count", 0)
        retriever_used = critic.metadata.get("retriever_used")
        context_length = critic.metadata.get("context_length", 0)

        if retrieval_needed is not None:
            print("      📚 Retrieval Information:")
            print(f"         • Retrieval Needed: {'✅ Yes' if retrieval_needed else '❌ No'}")
            print(f"         • Documents Retrieved: {retrieved_docs_count}")
            print(f"         • Context Length: {context_length} characters")
            if retriever_used:
                print(f"         • Retriever Used: {retriever_used}")

        # Show issues and suggestions
        if critic.violations:
            print("      ⚠️ Issues Identified:")
            for violation in critic.violations[:3]:
                wrapped = textwrap.fill(
                    violation,
                    width=70,
                    initial_indent="         • ",
                    subsequent_indent="           ",
                )
                print(wrapped)

        if critic.suggestions:
            print("      💡 Improvement Suggestions:")
            for suggestion in critic.suggestions[:3]:
                wrapped = textwrap.fill(
                    suggestion,
                    width=70,
                    initial_indent="         • ",
                    subsequent_indent="           ",
                )
                print(wrapped)

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

    def export_summary(self, output_file: str = None):
        """Export analysis summary to a file"""
        if output_file is None:
            # Create default output file in /analysis/reports directory
            analysis_dir = Path("analysis")
            reports_dir = analysis_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename based on input file
            base_name = self.json_file.stem
            output_file = reports_dir / f"{base_name}_analysis_summary.txt"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
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

        print(f"📄 Analysis exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize Sifaka thought data")
    parser.add_argument("json_file", help="Path to the thought JSON file")
    parser.add_argument(
        "--export",
        "-e",
        help="Export analysis to file (default: analysis/reports/{filename}_analysis_summary.txt)",
        default=None,
    )
    parser.add_argument("--overview-only", "-o", action="store_true", help="Show only overview")
    parser.add_argument(
        "--auto-export",
        "-a",
        action="store_true",
        help="Automatically export analysis to default location",
    )

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

        if args.export is not None or args.auto_export:
            analyzer.export_summary(args.export)

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
