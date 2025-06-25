"""Composable validator system for more flexible and simple validation."""

from typing import Callable, List, Optional
from dataclasses import dataclass
import re

from ..core.models import SifakaResult
from .base import BaseValidator


@dataclass
class ValidationRule:
    """A single validation rule."""

    name: str
    check: Callable[[str], bool]
    score_func: Callable[[str], float]
    detail_func: Callable[[str], str]


class ComposableValidator(BaseValidator):
    """A validator that can be composed with others using operators."""

    def __init__(self, name: str, rules: Optional[List[ValidationRule]] = None):
        """Initialize composable validator.

        Args:
            name: Validator name
            rules: List of validation rules
        """
        super().__init__()
        self._name = name
        self.rules = rules or []

    @property
    def name(self) -> str:
        """Return the validator name."""
        return self._name

    async def _perform_validation(
        self, text: str, result: SifakaResult
    ) -> tuple[bool, float, str]:
        """Perform validation using all rules."""
        if not self.rules:
            return True, 1.0, "No validation rules"

        passed_rules = 0
        total_score = 0.0
        details = []

        for rule in self.rules:
            try:
                rule_passed = rule.check(text)
                rule_score = rule.score_func(text) if rule_passed else 0.0
                rule_detail = rule.detail_func(text)

                if rule_passed:
                    passed_rules += 1
                    total_score += rule_score

                details.append(f"{rule.name}: {rule_detail}")

            except Exception as e:
                details.append(f"{rule.name}: Error - {str(e)}")

        # Calculate overall results
        all_passed = passed_rules == len(self.rules)
        avg_score = total_score / len(self.rules) if self.rules else 0.0
        detail_text = "\n".join(details)

        return all_passed, avg_score, detail_text

    def __and__(self, other: "ComposableValidator") -> "ComposableValidator":
        """Combine validators with AND logic (both must pass)."""
        combined_name = f"({self.name} AND {other.name})"
        combined_rules = self.rules + other.rules
        return ComposableValidator(combined_name, combined_rules)

    def __or__(self, other: "ComposableValidator") -> "ComposableValidator":
        """Combine validators with OR logic (at least one must pass)."""
        combined_name = f"({self.name} OR {other.name})"

        # Create a wrapper validator that implements OR logic
        left_validator = self
        right_validator = other

        class OrValidator(ComposableValidator):
            async def _perform_validation(
                self, text: str, result: SifakaResult
            ) -> tuple[bool, float, str]:
                # Run both validators
                result1 = await left_validator._perform_validation(text, result)
                result2 = await right_validator._perform_validation(text, result)

                # OR logic: pass if either passes
                passed = result1[0] or result2[0]
                score = max(result1[1], result2[1])
                details = f"Left: {result1[2]}\nRight: {result2[2]}"

                return passed, score, details

        validator = OrValidator(combined_name)
        # Don't store validators in rules since they're not ValidationRule objects
        return validator

    def __invert__(self) -> "ComposableValidator":
        """Create a NOT validator (inverts the result)."""
        inverted_name = f"NOT {self.name}"

        # Create inverted rules
        inverted_rules = []
        for rule in self.rules:
            orig_check = rule.check
            orig_score = rule.score_func
            orig_detail = rule.detail_func

            def make_inverted_check(
                orig: Callable[[str], bool]
            ) -> Callable[[str], bool]:
                return lambda text: not orig(text)

            def make_inverted_score(
                orig: Callable[[str], float]
            ) -> Callable[[str], float]:
                return lambda text: 1.0 - orig(text)

            def make_inverted_detail(
                orig: Callable[[str], str]
            ) -> Callable[[str], str]:
                return lambda text: f"NOT ({orig(text)})"

            inverted_rule = ValidationRule(
                name=f"NOT {rule.name}",
                check=make_inverted_check(orig_check),
                score_func=make_inverted_score(orig_score),
                detail_func=make_inverted_detail(orig_detail),
            )
            inverted_rules.append(inverted_rule)

        return ComposableValidator(inverted_name, inverted_rules)


class ValidatorBuilder:
    """Fluent interface for building validators."""

    def __init__(self, name: str = "custom"):
        """Initialize builder."""
        self.name = name
        self.rules: List[ValidationRule] = []

    def length(
        self, min_length: int = 0, max_length: int = 999999
    ) -> "ValidatorBuilder":
        """Add length validation."""
        rule = ValidationRule(
            name="length",
            check=lambda text: min_length <= len(text) <= max_length,
            score_func=lambda text: min(1.0, len(text) / max(min_length, 100)),
            detail_func=lambda text: f"Length {len(text)} (required: {min_length}-{max_length})",
        )
        self.rules.append(rule)
        return self

    def contains(self, keywords: List[str], mode: str = "all") -> "ValidatorBuilder":
        """Add keyword validation.

        Args:
            keywords: Keywords to check
            mode: "all" (must contain all) or "any" (must contain at least one)
        """

        def check(text: str) -> bool:
            text_lower = text.lower()
            if mode == "all":
                return all(kw.lower() in text_lower for kw in keywords)
            else:
                return any(kw.lower() in text_lower for kw in keywords)

        def score(text: str) -> float:
            text_lower = text.lower()
            found = sum(1 for kw in keywords if kw.lower() in text_lower)
            return found / len(keywords) if keywords else 0.0

        def detail(text: str) -> str:
            text_lower = text.lower()
            found = [kw for kw in keywords if kw.lower() in text_lower]
            return f"Contains {len(found)}/{len(keywords)} keywords: {found}"

        rule = ValidationRule(
            name=f"contains_{mode}", check=check, score_func=score, detail_func=detail
        )
        self.rules.append(rule)
        return self

    def matches(self, pattern: str, description: str = "pattern") -> "ValidatorBuilder":
        """Add regex pattern validation."""
        regex = re.compile(pattern)

        rule = ValidationRule(
            name=f"matches_{description}",
            check=lambda text: bool(regex.search(text)),
            score_func=lambda text: 1.0 if regex.search(text) else 0.0,
            detail_func=lambda text: f"Pattern '{description}' {'found' if regex.search(text) else 'not found'}",
        )
        self.rules.append(rule)
        return self

    def custom(
        self,
        name: str,
        check: Callable[[str], bool],
        score: Optional[Callable[[str], float]] = None,
        detail: Optional[Callable[[str], str]] = None,
    ) -> "ValidatorBuilder":
        """Add custom validation rule."""
        rule = ValidationRule(
            name=name,
            check=check,
            score_func=score or (lambda _: 1.0),
            detail_func=detail
            or (lambda text: f"{name}: {'passed' if check(text) else 'failed'}"),
        )
        self.rules.append(rule)
        return self

    def sentences(
        self, min_sentences: int = 1, max_sentences: int = 999999
    ) -> "ValidatorBuilder":
        """Add sentence count validation."""

        def count_sentences(text: str) -> int:
            # Simple sentence counting
            sentences = re.split(r"[.!?]+", text)
            return len([s for s in sentences if s.strip()])

        rule = ValidationRule(
            name="sentences",
            check=lambda text: min_sentences <= count_sentences(text) <= max_sentences,
            score_func=lambda text: min(
                1.0, count_sentences(text) / max(min_sentences, 5)
            ),
            detail_func=lambda text: f"{count_sentences(text)} sentences (required: {min_sentences}-{max_sentences})",
        )
        self.rules.append(rule)
        return self

    def words(self, min_words: int = 0, max_words: int = 999999) -> "ValidatorBuilder":
        """Add word count validation."""

        def count_words(text: str) -> int:
            return len(text.split())

        rule = ValidationRule(
            name="words",
            check=lambda text: min_words <= count_words(text) <= max_words,
            score_func=lambda text: min(1.0, count_words(text) / max(min_words, 50)),
            detail_func=lambda text: f"{count_words(text)} words (required: {min_words}-{max_words})",
        )
        self.rules.append(rule)
        return self

    def build(self) -> ComposableValidator:
        """Build the validator."""
        return ComposableValidator(self.name, self.rules)


# Convenience factory class
class Validator:
    """Factory class for creating validators with a fluent interface."""

    @staticmethod
    def create(name: str = "custom") -> ValidatorBuilder:
        """Create a new validator builder."""
        return ValidatorBuilder(name)

    @staticmethod
    def length(min_length: int = 0, max_length: int = 999999) -> ComposableValidator:
        """Create a length validator."""
        return ValidatorBuilder("length").length(min_length, max_length).build()

    @staticmethod
    def contains(keywords: List[str], mode: str = "all") -> ComposableValidator:
        """Create a keyword validator."""
        return ValidatorBuilder("contains").contains(keywords, mode).build()

    @staticmethod
    def matches(pattern: str, description: str = "pattern") -> ComposableValidator:
        """Create a pattern validator."""
        return ValidatorBuilder("matches").matches(pattern, description).build()

    @staticmethod
    def sentences(
        min_sentences: int = 1, max_sentences: int = 999999
    ) -> ComposableValidator:
        """Create a sentence count validator."""
        return (
            ValidatorBuilder("sentences")
            .sentences(min_sentences, max_sentences)
            .build()
        )

    @staticmethod
    def words(min_words: int = 0, max_words: int = 999999) -> ComposableValidator:
        """Create a word count validator."""
        return ValidatorBuilder("words").words(min_words, max_words).build()


# Example usage:
# validator = Validator.length(100, 500) & Validator.contains(["AI", "ML"])
# validator = Validator.create("essay").length(500, 1000).sentences(10, 50).contains(["thesis", "conclusion"]).build()
# validator = Validator.matches(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email') | Validator.matches(r'\d{3}-\d{3}-\d{4}', 'phone')
