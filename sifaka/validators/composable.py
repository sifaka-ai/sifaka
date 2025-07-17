"""Composable validator system for flexible, chainable validation rules.

This module provides a powerful, fluent interface for building complex validators
by composing simple validation rules. It supports logical operations (AND, OR, NOT)
and includes a builder pattern for creating custom validators with multiple criteria.

## Key Features:

- **Fluent Interface**: Chain validation rules with method calls
- **Logical Operations**: Combine validators with &, |, and ~ operators
- **Rule-Based**: Individual rules with check, score, and detail functions
- **Extensible**: Easy to add custom validation logic
- **Composable**: Build complex validators from simple building blocks

## Usage Examples:

    >>> # Simple fluent validation
    >>> validator = (Validator.length(100, 500) &
    ...              Validator.contains(["AI", "ML"]))
    >>>
    >>> # Complex essay validator
    >>> essay_validator = (Validator.create("essay")
    ...     .length(500, 1000)
    ...     .sentences(10, 50)
    ...     .contains(["thesis", "conclusion"])
    ...     .build())
    >>>
    >>> # Email or phone validation
    >>> contact_validator = (Validator.matches(r"\\b[\\w.-]+@[\\w.-]+\\.[a-z]{2,}\\b", "email") |
    ...                       Validator.matches(r"\\d{3}-\\d{3}-\\d{4}", "phone"))

## Design Philosophy:

This system prioritizes ease of use and readability while maintaining flexibility.
Validators are composed from individual rules that can be combined using familiar
logical operators, making complex validation logic intuitive to express and maintain.

## Logical Operations:

- **AND (&)**: All validators must pass
- **OR (|)**: At least one validator must pass
- **NOT (~)**: Inverts the validation result

## Built-in Validators:

- **Length**: Character count validation
- **Contains**: Keyword presence validation (all/any mode)
- **Matches**: Regex pattern validation
- **Sentences**: Sentence count validation
- **Words**: Word count validation
- **Custom**: User-defined validation logic

Each validator provides detailed scoring and feedback for comprehensive results.
"""

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from ..core.models import SifakaResult
from .base import BaseValidator


@dataclass
class ValidationRule:
    """Represents a single validation rule with check, scoring, and detail functions.

    A validation rule encapsulates a specific validation criterion with three
    key functions: checking if text passes the rule, calculating a score,
    and providing detailed feedback about the validation result.

    Attributes:
        name: Descriptive name for the rule (appears in validation results)
        check: Function that returns True if text passes the validation
        score_func: Function that returns a score (0.0-1.0) for the text
        detail_func: Function that returns detailed feedback about the validation

    Example:
        >>> rule = ValidationRule(
        ...     name="min_length",
        ...     check=lambda text: len(text) >= 100,
        ...     score_func=lambda text: min(1.0, len(text) / 100),
        ...     detail_func=lambda text: f"Length: {len(text)} chars (min: 100)"
        ... )
    """

    name: str
    check: Callable[[str], bool]
    score_func: Callable[[str], float]
    detail_func: Callable[[str], str]


class ComposableValidator(BaseValidator):
    """A validator that supports logical composition using operators.

    Provides a flexible validation system where individual validators can be
    combined using logical operators (&, |, ~) to create complex validation
    logic. Each validator is built from one or more ValidationRule objects.

    Key capabilities:
    - Logical AND (&): Both validators must pass
    - Logical OR (|): At least one validator must pass
    - Logical NOT (~): Inverts the validation result
    - Rule-based validation with detailed scoring and feedback
    - Exception handling for robust validation

    Example:
        >>> # Create individual validators
        >>> length_validator = Validator.length(100, 500)
        >>> keyword_validator = Validator.contains(["AI", "ML"])
        >>>
        >>> # Combine with logical operators
        >>> combined = length_validator & keyword_validator  # Both must pass
        >>> either = length_validator | keyword_validator    # Either can pass
        >>> inverted = ~length_validator                     # Opposite result
        >>>
        >>> # Use in validation
        >>> result = await combined.validate(text, sifaka_result)
        >>> print(f"Passed: {result.passed}, Score: {result.score}")

    The validator automatically handles rule execution, scoring calculation,
    and detailed feedback generation for comprehensive validation results.
    """

    def __init__(self, name: str, rules: Optional[List[ValidationRule]] = None):
        """Initialize composable validator with rules.

        Creates a validator that executes the provided validation rules
        and combines their results into a comprehensive validation outcome.

        Args:
            name: Descriptive name for the validator (used in results and logging)
            rules: List of ValidationRule objects to execute. If None or empty,
                validator will always pass with default score and message.

        Example:
            >>> # Create custom rules
            >>> rules = [
            ...     ValidationRule(
            ...         name="length",
            ...         check=lambda text: len(text) >= 100,
            ...         score_func=lambda text: len(text) / 100,
            ...         detail_func=lambda text: f"Length: {len(text)}"
            ...     )
            ... ]
            >>> validator = ComposableValidator("my_validator", rules)
        """
        super().__init__()
        self._name = name
        self.rules = rules or []

    @property
    def name(self) -> str:
        """Return the validator identifier.

        Returns:
            String identifier used in validation results, logging, and error messages
        """
        return self._name

    async def _perform_validation(
        self, text: str, result: SifakaResult
    ) -> tuple[bool, float, str]:
        """Execute all validation rules and combine their results.

        Runs each validation rule against the text, calculates overall scores,
        and generates comprehensive feedback. Handles rule execution errors
        gracefully to ensure validation robustness.

        Args:
            text: Text to validate against all rules
            result: SifakaResult for context (available but not currently used)

        Returns:
            Tuple containing:
            - bool: True if all rules pass, False if any rule fails
            - float: Average score across all rules (0.0-1.0)
            - str: Detailed feedback from all rules, including any errors

        Error handling:
            If a rule raises an exception during execution, it's caught and
            recorded as an error in the feedback rather than failing the
            entire validation process.

        Scoring:
            The overall score is the average of all successful rule scores.
            Failed rules contribute 0.0 to the average.
        """
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
                details.append(f"{rule.name}: Error - {e!s}")

        # Calculate overall results
        all_passed = passed_rules == len(self.rules)
        avg_score = total_score / len(self.rules) if self.rules else 0.0
        detail_text = "\n".join(details)

        return all_passed, avg_score, detail_text

    def __and__(self, other: "ComposableValidator") -> "ComposableValidator":
        """Combine validators with AND logic (both must pass).

        Creates a new validator that requires both the current validator
        and the other validator to pass for overall success.

        Args:
            other: Another ComposableValidator to combine with

        Returns:
            New ComposableValidator that implements AND logic by combining
            all rules from both validators

        Example:
            >>> length_val = Validator.length(100, 500)
            >>> keyword_val = Validator.contains(["AI", "ML"])
            >>> combined = length_val & keyword_val  # Both must pass
            >>>
            >>> # Text must be 100-500 chars AND contain "AI" and "ML"
            >>> result = await combined.validate(text, sifaka_result)
        """
        combined_name = f"({self.name} AND {other.name})"
        combined_rules = self.rules + other.rules
        return ComposableValidator(combined_name, combined_rules)

    def __or__(self, other: "ComposableValidator") -> "ComposableValidator":
        """Combine validators with OR logic (at least one must pass).

        Creates a new validator that passes if either the current validator
        or the other validator passes (or both pass).

        Args:
            other: Another ComposableValidator to combine with

        Returns:
            New ComposableValidator that implements OR logic by running both
            validators and accepting success from either one

        Example:
            >>> email_val = Validator.matches(r"\\b[\\w.-]+@[\\w.-]+\\.[a-z]{2,}\\b", "email")
            >>> phone_val = Validator.matches(r"\\d{3}-\\d{3}-\\d{4}", "phone")
            >>> contact = email_val | phone_val  # Either email OR phone
            >>>
            >>> # Text must contain either email or phone number
            >>> result = await contact.validate(text, sifaka_result)

        Note:
            The score is the maximum score from either validator, and detailed
            feedback includes results from both validators for transparency.
        """
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
        """Create a NOT validator that inverts all validation results.

        Creates a new validator that inverts the pass/fail status, scores,
        and detailed feedback of all rules in the current validator.

        Returns:
            New ComposableValidator with inverted logic for all rules

        Example:
            >>> spam_keywords = Validator.contains(["buy now", "urgent", "act fast"])
            >>> not_spam = ~spam_keywords  # Must NOT contain spam keywords
            >>>
            >>> # Text must not contain any of the spam keywords
            >>> result = await not_spam.validate(text, sifaka_result)

        Inversion behavior:
        - check function: Returns opposite boolean result
        - score function: Returns 1.0 - original_score
        - detail function: Prepends "NOT (...)" to original details

        Use cases:
        - Blacklist validation (must not contain certain patterns)
        - Exclusion rules (must not meet certain criteria)
        - Negative validation (opposite of positive criteria)
        """
        inverted_name = f"NOT {self.name}"

        # Create inverted rules
        inverted_rules = []
        for rule in self.rules:
            orig_check = rule.check
            orig_score = rule.score_func
            orig_detail = rule.detail_func

            def make_inverted_check(
                orig: Callable[[str], bool],
            ) -> Callable[[str], bool]:
                return lambda text: not orig(text)

            def make_inverted_score(
                orig: Callable[[str], float],
            ) -> Callable[[str], float]:
                return lambda text: 1.0 - orig(text)

            def make_inverted_detail(
                orig: Callable[[str], str],
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
    """Fluent interface for building complex validators through method chaining.

    Provides a convenient, readable way to construct validators with multiple
    validation criteria. Methods can be chained together to build sophisticated
    validation logic in a declarative style.

    Key features:
    - Method chaining for fluent construction
    - Built-in validation patterns (length, keywords, patterns, etc.)
    - Custom validation rule support
    - Automatic rule naming and feedback generation

    Example:
        >>> # Build a comprehensive essay validator
        >>> validator = (ValidatorBuilder("essay")
        ...     .length(500, 2000)           # 500-2000 characters
        ...     .words(100, 400)             # 100-400 words
        ...     .sentences(5, 20)            # 5-20 sentences
        ...     .contains(["thesis"], "all")  # Must contain "thesis"
        ...     .matches(r"\\bconclusion\\b", "conclusion")  # Must have conclusion
        ...     .custom("no_typos", lambda text: "teh" not in text.lower())
        ...     .build())
        >>>
        >>> # Use the built validator
        >>> result = await validator.validate(essay_text, sifaka_result)

    The builder automatically generates appropriate scoring functions and
    detailed feedback for each validation criterion.
    """

    def __init__(self, name: str = "custom"):
        """Initialize validator builder with a name.

        Args:
            name: Name for the final validator (used in results and logging)
        """
        self.name = name
        self.rules: List[ValidationRule] = []

    def length(
        self, min_length: int = 0, max_length: int = 999999
    ) -> "ValidatorBuilder":
        """Add character length validation to the builder.

        Validates that text falls within specified character count bounds.
        Automatically generates appropriate scoring based on length relative
        to minimum requirement.

        Args:
            min_length: Minimum allowed character count (inclusive)
            max_length: Maximum allowed character count (inclusive)

        Returns:
            Self for method chaining

        Example:
            >>> builder.length(100, 500)  # 100-500 characters required
            >>> builder.length(min_length=50)  # At least 50 characters
        """
        rule = ValidationRule(
            name="length",
            check=lambda text: min_length <= len(text) <= max_length,
            score_func=lambda text: min(1.0, len(text) / max(min_length, 100)),
            detail_func=lambda text: f"Length {len(text)} (required: {min_length}-{max_length})",
        )
        self.rules.append(rule)
        return self

    def contains(self, keywords: List[str], mode: str = "all") -> "ValidatorBuilder":
        """Add keyword presence validation to the builder.

        Validates that text contains specified keywords according to the mode.
        Case-insensitive matching is used for robust keyword detection.

        Args:
            keywords: List of keywords to search for in the text
            mode: Validation mode:
                - "all": Text must contain ALL keywords
                - "any": Text must contain at least ONE keyword

        Returns:
            Self for method chaining

        Example:
            >>> builder.contains(["AI", "machine learning"], "all")  # Both required
            >>> builder.contains(["email", "phone", "contact"], "any")  # Any one

        Scoring:
            Score is calculated as the proportion of keywords found:
            - mode="all": All keywords must be present for full score
            - mode="any": Score increases with each additional keyword found
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
        """Add regex pattern validation to the builder.

        Validates that text matches a specified regular expression pattern.
        Useful for format validation, specific content requirements, or
        structural patterns.

        Args:
            pattern: Regular expression pattern to match against
            description: Human-readable description of what the pattern validates
                (used in feedback messages)

        Returns:
            Self for method chaining

        Example:
            >>> builder.matches(r"\\b[\\w.-]+@[\\w.-]+\\.[a-z]{2,}\\b", "email")
            >>> builder.matches(r"^#+ .+$", "heading")
            >>> builder.matches(r"\\d{4}-\\d{2}-\\d{2}", "date")

        Note:
            Pattern is compiled once during rule creation for performance.
            Use raw strings (r"") to avoid escaping issues with regex patterns.
        """
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
        """Add a custom validation rule with user-defined logic.

        Allows adding arbitrary validation logic beyond the built-in patterns.
        Provides full control over checking, scoring, and feedback generation.

        Args:
            name: Descriptive name for the rule (appears in feedback)
            check: Function that takes text and returns True if valid
            score: Optional function that takes text and returns score (0.0-1.0).
                Defaults to returning 1.0 for any text.
            detail: Optional function that takes text and returns detailed feedback.
                Defaults to simple pass/fail message.

        Returns:
            Self for method chaining

        Example:
            >>> # Custom sentiment validation
            >>> def is_positive(text):
            ...     positive_words = ["good", "great", "excellent", "amazing"]
            ...     return any(word in text.lower() for word in positive_words)
            >>>
            >>> def sentiment_score(text):
            ...     # Simple scoring based on positive word count
            ...     positive_words = ["good", "great", "excellent", "amazing"]
            ...     count = sum(1 for word in positive_words if word in text.lower())
            ...     return min(1.0, count / 2)  # Max score with 2+ positive words
            >>>
            >>> builder.custom("positive_sentiment", is_positive, sentiment_score)
        """
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
        """Add sentence count validation to the builder.

        Validates that text contains an appropriate number of sentences based
        on simple sentence boundary detection (periods, exclamations, questions).

        Args:
            min_sentences: Minimum required number of sentences
            max_sentences: Maximum allowed number of sentences

        Returns:
            Self for method chaining

        Example:
            >>> builder.sentences(3, 10)  # 3-10 sentences required
            >>> builder.sentences(min_sentences=5)  # At least 5 sentences

        Note:
            Sentence counting is based on splitting text by [.!?] patterns
            and filtering out empty strings. This provides reasonable accuracy
            for most text types but may not handle complex punctuation perfectly.

        Scoring:
            Score is calculated relative to minimum requirement, with full
            score achieved at min_sentences and maintained within range.
        """

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
        """Add word count validation to the builder.

        Validates that text contains an appropriate number of words based
        on simple whitespace-separated word counting.

        Args:
            min_words: Minimum required number of words
            max_words: Maximum allowed number of words

        Returns:
            Self for method chaining

        Example:
            >>> builder.words(50, 200)  # 50-200 words required
            >>> builder.words(min_words=100)  # At least 100 words

        Note:
            Word counting is based on splitting text by whitespace, which
            provides reasonable accuracy for most text types. Hyphenated
            words are counted as single words.

        Scoring:
            Score is calculated relative to minimum requirement, with full
            score achieved at min_words and maintained within range.
        """

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
        """Build the final ComposableValidator with all configured rules.

        Creates a ComposableValidator instance containing all validation rules
        that were added through the builder's fluent interface.

        Returns:
            ComposableValidator ready for use in validation operations

        Example:
            >>> validator = (ValidatorBuilder("blog_post")
            ...     .length(200, 1000)
            ...     .contains(["technology"], "any")
            ...     .build())
            >>>
            >>> result = await validator.validate(text, sifaka_result)
        """
        return ComposableValidator(self.name, self.rules)


# Convenience factory class for common validator patterns
class Validator:
    """Factory class for creating validators with a fluent interface.

    Provides static methods for quickly creating common validators without
    needing to use the ValidatorBuilder directly. Combines the convenience
    of pre-built validators with the flexibility of the composable system.

    All methods return ComposableValidator instances that can be combined
    using logical operators (&, |, ~) for complex validation scenarios.

    Example:
        >>> # Quick validator creation
        >>> length_val = Validator.length(100, 500)
        >>> keyword_val = Validator.contains(["AI", "ML"])
        >>>
        >>> # Combine with operators
        >>> combined = length_val & keyword_val
        >>> either = length_val | keyword_val
        >>>
        >>> # Or use builder for complex cases
        >>> complex_val = Validator.create("essay").length(500).words(100).build()
    """

    @staticmethod
    def create(name: str = "custom") -> ValidatorBuilder:
        """Create a new validator builder for complex validation scenarios.

        Returns a ValidatorBuilder instance for fluent construction of
        validators with multiple criteria.

        Args:
            name: Name for the validator (used in results and logging)

        Returns:
            ValidatorBuilder instance ready for method chaining

        Example:
            >>> validator = (Validator.create("article")
            ...     .length(500, 2000)
            ...     .sentences(10, 50)
            ...     .contains(["introduction", "conclusion"])
            ...     .build())
        """
        return ValidatorBuilder(name)

    @staticmethod
    def length(min_length: int = 0, max_length: int = 999999) -> ComposableValidator:
        """Create a character length validator.

        Args:
            min_length: Minimum allowed character count
            max_length: Maximum allowed character count

        Returns:
            ComposableValidator that validates character length

        Example:
            >>> validator = Validator.length(100, 500)
            >>> result = await validator.validate(text, sifaka_result)
        """
        return ValidatorBuilder("length").length(min_length, max_length).build()

    @staticmethod
    def contains(keywords: List[str], mode: str = "all") -> ComposableValidator:
        """Create a keyword presence validator.

        Args:
            keywords: List of keywords to check for
            mode: "all" (all keywords required) or "any" (at least one required)

        Returns:
            ComposableValidator that validates keyword presence

        Example:
            >>> all_keywords = Validator.contains(["AI", "ML"], "all")
            >>> any_keyword = Validator.contains(["tech", "science"], "any")
        """
        return ValidatorBuilder("contains").contains(keywords, mode).build()

    @staticmethod
    def matches(pattern: str, description: str = "pattern") -> ComposableValidator:
        """Create a regex pattern validator.

        Args:
            pattern: Regular expression pattern to match
            description: Human-readable description for feedback

        Returns:
            ComposableValidator that validates regex patterns

        Example:
            >>> email_val = Validator.matches(r"\\b[\\w.-]+@[\\w.-]+\\.[a-z]{2,}\\b", "email")
            >>> phone_val = Validator.matches(r"\\d{3}-\\d{3}-\\d{4}", "phone")
        """
        return ValidatorBuilder("matches").matches(pattern, description).build()

    @staticmethod
    def sentences(
        min_sentences: int = 1, max_sentences: int = 999999
    ) -> ComposableValidator:
        """Create a sentence count validator.

        Args:
            min_sentences: Minimum required number of sentences
            max_sentences: Maximum allowed number of sentences

        Returns:
            ComposableValidator that validates sentence count

        Example:
            >>> validator = Validator.sentences(5, 20)
            >>> result = await validator.validate(text, sifaka_result)
        """
        return (
            ValidatorBuilder("sentences")
            .sentences(min_sentences, max_sentences)
            .build()
        )

    @staticmethod
    def words(min_words: int = 0, max_words: int = 999999) -> ComposableValidator:
        """Create a word count validator.

        Args:
            min_words: Minimum required number of words
            max_words: Maximum allowed number of words

        Returns:
            ComposableValidator that validates word count

        Example:
            >>> validator = Validator.words(100, 500)
            >>> result = await validator.validate(text, sifaka_result)
        """
        return ValidatorBuilder("words").words(min_words, max_words).build()


# Example usage patterns demonstrating the composable validator system:
#
# Simple combination:
# validator = Validator.length(100, 500) & Validator.contains(["AI", "ML"])
#
# Complex essay validator:
# validator = Validator.create("essay").length(500, 1000).sentences(10, 50).contains(["thesis", "conclusion"]).build()
#
# Contact information validator (email OR phone):
# validator = Validator.matches(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email') | Validator.matches(r'\d{3}-\d{3}-\d{4}', 'phone')
#
# Negative validation (must NOT contain spam keywords):
# spam_validator = Validator.contains(["buy now", "urgent", "limited time"])
# not_spam = ~spam_validator
