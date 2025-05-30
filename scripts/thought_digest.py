#!/usr/bin/env python3
"""
Thought Digest Tool

A tool to convert Sifaka thought JSON files into a more readable format.
Presents model prompts followed by text in chronological order, with
validation summaries in between.

Usage:
    python scripts/thought_digest.py <thought_file.json>
    python scripts/thought_digest.py thoughts/pydantic_ai_basic_example.json
    python scripts/thought_digest.py thoughts/*.json  # Process all JSON files
    python scripts/thought_digest.py thoughts/file.json --stdout  # Print to console
    python scripts/thought_digest.py thoughts/file.json -o custom.txt  # Custom output file
"""

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def format_validation_summary(validation_results: Optional[Dict[str, Any]]) -> str:
    """Format validation results into a brief summary."""
    if not validation_results:
        return "✅ No validation issues"

    passed = []
    failed = []

    for validator_name, result in validation_results.items():
        if result.get("passed", False):
            passed.append(validator_name)
        else:
            failed.append(f"{validator_name}: {result.get('message', 'Failed')}")

    summary_parts = []
    if passed:
        summary_parts.append(f"✅ Passed: {', '.join(passed)}")
    if failed:
        summary_parts.append(f"❌ Failed: {', '.join(failed)}")

    return " | ".join(summary_parts) if summary_parts else "✅ No validation issues"


def format_critic_summary(critic_feedback: Optional[List[Dict[str, Any]]]) -> str:
    """Format critic feedback into a brief summary."""
    if not critic_feedback:
        return "💭 No critic feedback"

    needs_improvement = []
    satisfied = []
    has_suggestions = []

    for feedback in critic_feedback:
        critic_name = feedback.get("critic_name", "Unknown")
        if feedback.get("needs_improvement", False):
            needs_improvement.append(critic_name)
        else:
            satisfied.append(critic_name)
            # Check if satisfied critic still has useful suggestions
            suggestions = feedback.get("suggestions", [])
            if suggestions and any(
                s.strip() and s != "See critique for improvement suggestions" for s in suggestions
            ):
                has_suggestions.append(critic_name)

    summary_parts = []
    if satisfied:
        satisfied_text = f"✅ Satisfied: {', '.join(satisfied)}"
        if has_suggestions:
            satisfied_text += f" (with suggestions: {', '.join(has_suggestions)})"
        summary_parts.append(satisfied_text)
    if needs_improvement:
        summary_parts.append(f"🔄 Needs improvement: {', '.join(needs_improvement)}")

    return " | ".join(summary_parts) if summary_parts else "💭 No critic feedback"


def extract_feedback_from_prompt(model_prompt: str) -> Optional[str]:
    """Extract the feedback section from a model prompt."""
    if not model_prompt or "Feedback:" not in model_prompt:
        return None

    try:
        # Find the feedback section
        feedback_start = model_prompt.find("Feedback:")
        if feedback_start == -1:
            return None

        # Find the end of feedback (usually before "Please provide" or "Improved text:")
        feedback_section = model_prompt[feedback_start:]
        end_markers = ["Please provide", "Improved text:", "\n\nImproved text"]

        for marker in end_markers:
            if marker in feedback_section:
                feedback_section = feedback_section[: feedback_section.find(marker)]
                break

        # Clean up the feedback section
        feedback_section = feedback_section.replace("Feedback:", "").strip()
        return feedback_section if feedback_section else None

    except Exception:
        return None


def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display."""
    try:
        # Handle both string and datetime formats
        if isinstance(timestamp_str, str):
            # Try parsing ISO format
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        else:
            dt = timestamp_str
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, AttributeError):
        return str(timestamp_str)


def extract_thoughts_in_order(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract thoughts and sort them by iteration order."""
    thoughts = []

    for key, thought_data in data.items():
        if key.startswith("thought_"):
            thoughts.append(thought_data)

    # Sort by iteration number
    thoughts.sort(key=lambda x: x.get("iteration", 0))
    return thoughts


def check_validators_critics_activated(thought: Dict[str, Any]) -> str:
    """Check if validators and/or critics were activated before this iteration."""
    validation_results = thought.get("validation_results")
    critic_feedback = thought.get("critic_feedback")

    activated = []

    # Check if validators were activated (have results)
    if validation_results:
        activated.append("Validators")

    # Check if critics were activated (have feedback)
    if critic_feedback:
        activated.append("Critics")

    if not activated:
        return "N/A"

    return " + ".join(activated) + " activated"


def digest_thought_file(file_path: Path) -> str:
    """Convert a thought JSON file into a readable digest."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"❌ Error reading file: {e}"

    thoughts = extract_thoughts_in_order(data)

    if not thoughts:
        return "❌ No thoughts found in file"

    # Get chain info from first thought
    first_thought = thoughts[0]
    chain_id = first_thought.get("chain_id", "Unknown")

    digest_lines = [
        "=" * 80,
        f"🧠 THOUGHT DIGEST: {file_path.name}",
        "=" * 80,
        f"Chain ID: {chain_id}",
        f"Total Iterations: {len(thoughts)}",
        "",
    ]

    for i, thought in enumerate(thoughts, 1):
        iteration = thought.get("iteration", i)

        # Check if validators/critics were activated before this iteration
        if iteration == 1:
            activation_status = "N/A"
        else:
            activation_status = check_validators_critics_activated(thought)

        digest_lines.extend(
            [
                f"ITERATION {iteration}",
                f"Validators/Critics before generation: {activation_status}",
                "",
            ]
        )

        # Model prompt
        model_prompt = thought.get("model_prompt", thought.get("prompt", "No prompt available"))
        digest_lines.extend(
            [
                "MODEL PROMPT:",
                f"{model_prompt}",
                "",
            ]
        )

        # Generated text
        text = thought.get("text", "No text generated")
        digest_lines.extend(
            [
                "TEXT:",
                f"{text}",
                "",
                "=" * 40,
                "",
            ]
        )

    return "\n".join(digest_lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert Sifaka thought JSON files into readable digests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/thought_digest.py thoughts/example.json
  python scripts/thought_digest.py thoughts/*.json
  python scripts/thought_digest.py thoughts/example.json --stdout
  python scripts/thought_digest.py thoughts/example.json -o custom.txt
        """,
    )

    parser.add_argument("files", nargs="+", help="One or more thought JSON files to process")
    parser.add_argument(
        "-o", "--output", help="Output file (default: auto-generate in digests/ folder)"
    )
    parser.add_argument(
        "--stdout", action="store_true", help="Print to stdout instead of saving to file"
    )

    args = parser.parse_args()

    # Expand glob patterns
    all_files = []
    for pattern in args.files:
        if "*" in pattern or "?" in pattern:
            all_files.extend(glob.glob(pattern))
        else:
            all_files.append(pattern)

    if not all_files:
        print("❌ No files found matching the patterns")
        sys.exit(1)

    # Process all files
    all_digests = []
    for file_str in all_files:
        file_path = Path(file_str)

        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue

        if not file_path.suffix == ".json":
            print(f"❌ Skipping non-JSON file: {file_path}")
            continue

        digest = digest_thought_file(file_path)
        all_digests.append(digest)

    # Output results
    final_output = "\n\n".join(all_digests)

    if args.stdout:
        # Print to stdout
        print(final_output)
    else:
        # Save to file
        if args.output:
            output_path = Path(args.output)
        else:
            # Auto-generate filename in digests folder
            digests_dir = Path("digests")
            digests_dir.mkdir(exist_ok=True)

            if len(all_files) == 1:
                # Single file: use its name
                input_name = Path(all_files[0]).stem
                output_path = digests_dir / f"{input_name}_digest.txt"
            else:
                # Multiple files: use timestamp
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = digests_dir / f"digest_{timestamp}.txt"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_output)
        print(f"✅ Digest saved to: {output_path}")


if __name__ == "__main__":
    main()
