"""Metrics for tracking text improvement quality.

This module contains only objective, measurable metrics.
No pseudo-intelligent analysis - just facts.
"""

import re
from typing import Any, Dict, List


def analyze_suggestion_implementation(
    suggestions: List[str], old_text: str, new_text: str
) -> Dict[str, Any]:
    """Track objective metrics about text changes.

    Only reports measurable facts, no interpretation.

    Args:
        suggestions: List of suggestions from critics
        old_text: Text before improvement
        new_text: Text after improvement

    Returns:
        Objective metrics about the changes
    """
    old_words = old_text.split()
    new_words = new_text.split()

    # Objective counts
    metrics = {
        # Required for backward compatibility
        "suggestions_given": suggestions,
        "suggestions_implemented": [],  # Can't objectively determine
        "suggestions_not_implemented": suggestions,  # Can't objectively determine
        "implementation_rate": 0.0,  # Can't objectively determine
        "implementation_count": 0,  # Can't objectively determine
        # Objective metrics
        "old_text_length": len(old_text),
        "new_text_length": len(new_text),
        "length_change": len(new_text) - len(old_text),
        "length_change_ratio": (
            len(new_text) / len(old_text) if old_text else float("inf")
        ),
        "old_word_count": len(old_words),
        "new_word_count": len(new_words),
        "word_count_change": len(new_words) - len(old_words),
        "word_count_ratio": (
            len(new_words) / len(old_words) if old_words else float("inf")
        ),
        "old_sentence_count": len(re.split(r"[.!?]+", old_text.strip())) - 1,
        "new_sentence_count": len(re.split(r"[.!?]+", new_text.strip())) - 1,
        "old_paragraph_count": len(re.split(r"\n\n+", old_text.strip())),
        "new_paragraph_count": len(re.split(r"\n\n+", new_text.strip())),
        "suggestion_count": len(suggestions),
        "avg_suggestion_length": (
            sum(len(s) for s in suggestions) / len(suggestions) if suggestions else 0
        ),
        # Text similarity (objective measure)
        "text_similarity": _calculate_similarity(old_text, new_text),
        # Specific content changes
        "numbers_added": _count_numbers(new_text) - _count_numbers(old_text),
        "quotes_added": _count_quotes(new_text) - _count_quotes(old_text),
        "questions_added": _count_questions(new_text) - _count_questions(old_text),
    }

    return metrics


def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts based on words."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 and not words2:
        return 1.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if union else 0.0


def _count_numbers(text: str) -> int:
    """Count numeric references in text."""
    # Matches numbers, percentages, years, etc.
    patterns = [
        r"\b\d+\b",  # Plain numbers
        r"\d+%",  # Percentages
        r"\$[\d,]+",  # Money
        r"\d{4}",  # Years
    ]
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def _count_quotes(text: str) -> int:
    """Count quoted sections in text."""
    return len(re.findall(r'"[^"]+"|\'[^\']+\'', text))


def _count_questions(text: str) -> int:
    """Count questions in text."""
    return len(re.findall(r"[.!?]\s*[A-Z][^.!?]*\?", " " + text)) + (
        1 if text.strip().endswith("?") else 0
    )
