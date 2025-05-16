#!/usr/bin/env python
"""
Text Simplifier Example

This example demonstrates how to use Sifaka to build a text simplification tool
that can make complex text more accessible.
"""

import sys
import os
import argparse
import re
from typing import Dict, Any, List, Optional, Tuple

# Add the project root to the path so we can import sifaka
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sifaka.chain import Chain
from sifaka.validators import Critic
from sifaka.results import ValidationResult, ImprovementResult
from sifaka.interfaces import Model
from sifaka.factories import create_model_from_string


class SimplificationCritic(Critic):
    """Critic that evaluates and improves text simplicity."""

    def validate(self, text: str) -> ValidationResult:
        """Validate text simplicity.

        Args:
            text: Text to validate.

        Returns:
            ValidationResult indicating whether the text is simple enough.
        """
        prompt = f"""
        Evaluate the simplicity of the following text.
        Consider factors such as:
        - Use of simple language
        - Short sentences
        - Clear explanations without jargon
        - Readability for a general audience

        Text to evaluate:
        ---
        {text}
        ---

        First, provide a score from 1-10 where:
        1-3: Very complex
        4-6: Moderately complex
        7-10: Very simple

        Then, explain your reasoning in detail.

        Format your response as:
        SCORE: [your score]
        REASONING: [your detailed explanation]
        PASSED: [YES if score >= 7, NO if score < 7]
        """

        response = self.model.generate(prompt, **self.model_options)

        # Extract score and passed status
        score_match = re.search(r"SCORE:\s*(\d+)", response)
        passed_match = re.search(r"PASSED:\s*(YES|NO)", response)
        reasoning_match = re.search(r"REASONING:\s*(.*?)(?=PASSED:|$)", response, re.DOTALL)

        if not score_match or not passed_match:
            return ValidationResult(
                passed=False,
                message="Could not determine validation result from critic response",
                details={"response": response},
            )

        score = int(score_match.group(1))
        passed = passed_match.group(1) == "YES"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return ValidationResult(
            passed=passed,
            message=f"Simplicity score: {score}/10. {reasoning}",
            details={"score": score, "reasoning": reasoning, "response": response},
        )

    def improve(self, text: str) -> Tuple[str, ImprovementResult]:
        """Simplify text.

        Args:
            text: Text to simplify.

        Returns:
            Tuple of (simplified_text, ImprovementResult).
        """
        prompt = f"""
        Simplify the following text to make it more accessible.
        Focus on:
        - Using simpler language
        - Shortening sentences
        - Removing jargon and technical terms
        - Explaining complex concepts clearly
        - Maintaining the original meaning

        Text to simplify:
        ---
        {text}
        ---

        First, rewrite the text to make it simpler.
        Then, explain the changes you made.

        Format your response as:
        SIMPLIFIED TEXT:
        [your simplified version of the text]

        EXPLANATION:
        [explanation of changes made]
        """

        response = self.model.generate(prompt, **self.model_options)

        # Extract simplified text and explanation
        text_match = re.search(r"SIMPLIFIED TEXT:\s*(.*?)(?=EXPLANATION:|$)", response, re.DOTALL)
        explanation_match = re.search(r"EXPLANATION:\s*(.*?)$", response, re.DOTALL)

        if not text_match:
            return text, ImprovementResult(
                original_text=text,
                improved_text=text,
                changes_made=False,
                message="Could not extract simplified text from critic response",
                details={"response": response},
            )

        simplified_text = text_match.group(1).strip()
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        # Check if the text was actually changed
        changes_made = simplified_text != text

        return simplified_text, ImprovementResult(
            original_text=text,
            improved_text=simplified_text,
            changes_made=changes_made,
            message=f"Simplification improvements: {explanation}",
            details={"explanation": explanation, "response": response},
        )


class TextSimplifier:
    """A text simplification tool that makes complex text more accessible."""

    def __init__(self, model: str = "openai:gpt-4"):
        """Initialize the text simplifier.

        Args:
            model: The model to use for simplification.
        """
        self.model = model
        # Convert string model specification to model instance if needed
        if isinstance(model, str):
            model_instance = create_model_from_string(model)
            self._critic = SimplificationCritic(model_instance)
        else:
            self._critic = SimplificationCritic(model)

    def simplify(self, text: str, target_grade_level: Optional[int] = None) -> Dict[str, Any]:
        """Simplify text to make it more accessible.

        Args:
            text: The text to simplify.
            target_grade_level: Optional target grade level for simplification.

        Returns:
            A dictionary containing the simplified text and metadata.
        """
        # If a target grade level is specified, add it to the model options
        model_options = {}
        if target_grade_level is not None:
            model_options["system_message"] = (
                f"You are an expert at simplifying text to a {target_grade_level}th grade reading level. "
                f"Focus on making the text accessible to readers at that level while preserving the meaning."
            )

        # Create a critic with the specified options
        if isinstance(self.model, str):
            model_instance = create_model_from_string(self.model)
            critic = SimplificationCritic(model_instance, **model_options)
        else:
            critic = SimplificationCritic(self.model, **model_options)

        # Validate the original text
        validation_result = critic.validate(text)

        # Simplify the text
        simplified_text, improvement_result = critic.improve(text)

        # Validate the simplified text
        simplified_validation_result = critic.validate(simplified_text)

        # Calculate readability metrics
        original_metrics = self._calculate_readability_metrics(text)
        simplified_metrics = self._calculate_readability_metrics(simplified_text)

        # Return the result
        return {
            "original_text": text,
            "simplified_text": simplified_text,
            "changes_made": improvement_result.changes_made,
            "explanation": improvement_result.message,
            "original_simplicity_score": validation_result.details.get("score", 0),
            "simplified_simplicity_score": simplified_validation_result.details.get("score", 0),
            "original_metrics": original_metrics,
            "simplified_metrics": simplified_metrics,
            "target_grade_level": target_grade_level,
        }

    def _calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics for text.

        Args:
            text: The text to analyze.

        Returns:
            A dictionary containing readability metrics.
        """
        # Count words, sentences, and syllables
        words = text.split()
        word_count = len(words)
        sentence_count = max(1, len(re.split(r"[.!?]+", text)) - 1)

        # Simple syllable counter (not perfect but good enough for demonstration)
        def count_syllables(word: str) -> int:
            word = word.lower()
            if len(word) <= 3:
                return 1
            word = re.sub(r"e$", "", word)  # Remove trailing e
            vowels = "aeiouy"
            count = 0
            prev_is_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
            return max(1, count)

        syllable_count = sum(count_syllables(word) for word in words)

        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)

        # Calculate average sentence length
        avg_sentence_length = word_count / max(1, sentence_count)

        # Calculate Flesch-Kincaid Grade Level
        if word_count == 0 or sentence_count == 0:
            fk_grade = 0
        else:
            fk_grade = (
                0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
            )

        # Calculate Flesch Reading Ease
        if word_count == 0 or sentence_count == 0:
            flesch_ease = 0
        else:
            flesch_ease = (
                206.835
                - 1.015 * (word_count / sentence_count)
                - 84.6 * (syllable_count / word_count)
            )
            flesch_ease = max(0, min(100, flesch_ease))  # Clamp to 0-100

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "syllable_count": syllable_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "flesch_kincaid_grade": fk_grade,
            "flesch_reading_ease": flesch_ease,
        }


def main():
    """Run the text simplifier example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simplify text using Sifaka")
    parser.add_argument("--model", default="openai:gpt-4", help="Model to use")
    parser.add_argument("--text", help="Text to simplify")
    parser.add_argument("--file", help="File containing text to simplify")
    parser.add_argument("--grade-level", type=int, help="Target grade level for simplification")
    parser.add_argument("--output", help="Output file to write simplified text to")
    args = parser.parse_args()

    try:
        # Ensure at least one input is provided
        if not args.text and not args.file:
            parser.error("At least one of --text or --file must be provided")

        # Read text from file if specified
        if args.file:
            with open(args.file, "r") as f:
                args.text = f.read()

        # Create a text simplifier
        simplifier = TextSimplifier(model=args.model)

        # Simplify the text
        print(f"Simplifying text...")
        if args.grade_level:
            print(f"Target grade level: {args.grade_level}")
        result = simplifier.simplify(args.text, args.grade_level)

        # Print the result
        print("\nSimplification Result:")
        print("=" * 40)
        print("Original Text:")
        print(result["original_text"])
        print("\nSimplified Text:")
        print(result["simplified_text"])
        print("=" * 40)

        print(f"\nChanges Made: {result['changes_made']}")
        print(f"Explanation: {result['explanation']}")

        print("\nReadability Metrics:")
        print(f"Original Simplicity Score: {result['original_simplicity_score']}/10")
        print(f"Simplified Simplicity Score: {result['simplified_simplicity_score']}/10")

        print("\nOriginal Text Metrics:")
        print(f"Word Count: {result['original_metrics']['word_count']}")
        print(f"Sentence Count: {result['original_metrics']['sentence_count']}")
        print(
            f"Average Word Length: {result['original_metrics']['avg_word_length']:.2f} characters"
        )
        print(
            f"Average Sentence Length: {result['original_metrics']['avg_sentence_length']:.2f} words"
        )
        print(
            f"Flesch-Kincaid Grade Level: {result['original_metrics']['flesch_kincaid_grade']:.1f}"
        )
        print(f"Flesch Reading Ease: {result['original_metrics']['flesch_reading_ease']:.1f}/100")

        print("\nSimplified Text Metrics:")
        print(f"Word Count: {result['simplified_metrics']['word_count']}")
        print(f"Sentence Count: {result['simplified_metrics']['sentence_count']}")
        print(
            f"Average Word Length: {result['simplified_metrics']['avg_word_length']:.2f} characters"
        )
        print(
            f"Average Sentence Length: {result['simplified_metrics']['avg_sentence_length']:.2f} words"
        )
        print(
            f"Flesch-Kincaid Grade Level: {result['simplified_metrics']['flesch_kincaid_grade']:.1f}"
        )
        print(f"Flesch Reading Ease: {result['simplified_metrics']['flesch_reading_ease']:.1f}/100")

        # Write to file if specified
        if args.output:
            with open(args.output, "w") as f:
                f.write(result["simplified_text"])
            print(f"\nSimplified text written to {args.output}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
