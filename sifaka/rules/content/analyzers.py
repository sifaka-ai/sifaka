"""
Content analysis implementations.

This module provides concrete implementations of content analyzers for
different types of content analysis (tone, readability, etc.).
"""

from typing import Dict, List, Optional, Set
import re

from sifaka.rules.base import RuleConfig
from .base import BaseContentAnalyzer


class ToneAnalyzer(BaseContentAnalyzer[Dict[str, float]]):
    """Analyzer for text tone."""

    def analyze(self, content: str) -> Dict[str, float]:
        """Analyze tone of content."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(content)

        return {
            "compound": scores["compound"],
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
        }


class ReadabilityAnalyzer(BaseContentAnalyzer[Dict[str, float]]):
    """Analyzer for text readability."""

    def analyze(self, content: str) -> Dict[str, float]:
        """Analyze readability of content."""
        words = content.split()
        sentences = re.split(r"[.!?]+", content)
        syllables = sum(self._count_syllables(word) for word in words)

        # Calculate basic metrics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        avg_syllables_per_word = syllables / max(word_count, 1)

        # Calculate Flesch Reading Ease
        flesch = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        flesch = max(0.0, min(100.0, flesch))

        return {
            "flesch_score": flesch,
            "avg_words_per_sentence": avg_words_per_sentence,
            "avg_syllables_per_word": avg_syllables_per_word,
            "word_count": word_count,
            "sentence_count": sentence_count,
        }

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using a basic heuristic."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        if word.endswith("e"):
            count -= 1
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1
        return max(1, count)


class ProhibitedContentAnalyzer(BaseContentAnalyzer[Dict[str, List[str]]]):
    """Analyzer for prohibited content."""

    def __init__(self, config: Optional[RuleConfig] = None) -> None:
        """Initialize analyzer with configuration."""
        super().__init__(config)
        self._prohibited_terms: Set[str] = set()
        if config and "prohibited_terms" in config.params:
            self._prohibited_terms.update(config.params["prohibited_terms"])

    def analyze(self, content: str) -> Dict[str, List[str]]:
        """Analyze content for prohibited terms."""
        found_terms = []
        content_lower = content.lower()

        for term in self._prohibited_terms:
            if term.lower() in content_lower:
                found_terms.append(term)

        return {
            "found_terms": found_terms,
            "total_matches": len(found_terms),
        }
