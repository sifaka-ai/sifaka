"""
Base classes and protocols for content validation.

This module provides the foundation for content-based validation rules in Sifaka.
It defines protocols for analyzers and base classes for validators and analyzers.

Usage Example:
    ```python
    from sifaka.rules.content.base import create_content_validator, create_content_analyzer

    # Create a content analyzer
    analyzer = create_content_analyzer()

    # Create a content validator
    validator = create_content_validator(analyzer=analyzer)

    # Validate text
    result = (validator and validator.validate("This is a test.")
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
    ```
"""

import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ConfigDict

from sifaka.rules.base import RuleResult
from sifaka.utils.errors.base import ConfigurationError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ContentAnalysis(BaseModel):
    """Model for content analysis results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    length: int = Field(description="Length of the content in characters")
    word_count: int = Field(description="Number of words in the content")
    has_content: bool = Field(description="Whether the content has any meaningful text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the analysis",
    )


class ToneAnalysis(BaseModel):
    """Model for tone analysis results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tone: str = Field(description="The detected tone of the content")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the tone detection",
    )
    indicators: List[str] = Field(
        default_factory=list,
        description="List of tone indicators found in the content",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the analysis",
    )


@runtime_checkable
class ContentAnalyzer(Protocol):
    """Protocol for content analysis components."""

    def analyze(self, text: str) -> ContentAnalysis:
        """Analyze text and return structured results."""
        ...

    def can_analyze(self, text: str) -> bool:
        """Check if text can be analyzed."""
        ...


@runtime_checkable
class ToneAnalyzer(Protocol):
    """Protocol for tone analysis components."""

    def analyze_tone(self, text: str) -> ToneAnalysis:
        """Analyze text tone and return structured results."""
        ...

    def get_supported_tones(self) -> List[str]:
        """Get list of supported tone categories."""
        ...


class ContentValidator:
    """
    Base validator for content-based rules.

    This validator analyzes text content using a ContentAnalyzer implementation
    and validates it against specified criteria.

    Lifecycle:
        1. Initialization: Set up with a content analyzer
        2. Validation: Analyze text content
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.content.base import ContentValidator, create_content_analyzer

        # Create analyzer
        analyzer = create_content_analyzer()

        # Create validator
        validator = ContentValidator(analyzer)

        # Validate text
        result = (validator and validator.validate("This is a test.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(self, analyzer: ContentAnalyzer) -> None:
        """
        Initialize with content analyzer.

        Args:
            analyzer: The content analyzer to use for validation
        """
        super().__init__(validation_type=str)
        (self and self._validate_analyzer(analyzer)
        self._analyzer = analyzer

    def _validate_analyzer(self, analyzer: Any) -> None:
        """
        Validate that an analyzer implements the required protocol.

        Args:
            analyzer: The analyzer to validate

        Raises:
            ConfigurationError: If the analyzer doesn't implement ContentAnalyzer
        """
        if not isinstance(analyzer, ContentAnalyzer):
            raise ConfigurationError(
                f"Analyzer must implement ContentAnalyzer protocol, got {type(analyzer)}"
            )

    def validate(self, text: str) -> RuleResult:
        """
        Validate text content.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = (time and time.time()

        # Handle empty text
        empty_result = (self and self.handle_empty_text(text)
        if empty_result:
            return empty_result

        try:
            # Analyze the content
            analysis = self._analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and (analyzer and analyzer.analyze(text)

            # Determine if the content passes validation
            passed = analysis.has_content

            # Create result
            result = RuleResult(
                passed=passed,
                message="Content validation successful" if passed else "Content validation failed",
                metadata={
                    "length": analysis.length,
                    "word_count": analysis.word_count,
                    "has_content": analysis.has_content,
                    "validator_type": self.__class__.__name__,
                    **analysis.metadata,
                },
                score=1.0 if passed else 0.0,
                issues=[] if passed else ["Content validation failed"],
                suggestions=[] if passed else ["Provide non-empty content"],
                processing_time_ms=(time and time.time() - start_time,
            )

            # Update statistics
            (self and self.update_statistics(result)

            return result

        except Exception as e:
            (self and self.record_error(e)
            (logger and logger.error(f"Content validation failed: {e}")

            error_message = f"Content validation failed: {str(e)}"
            result = RuleResult(
                passed=False,
                message=error_message,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator_type": self.__class__.__name__,
                },
                score=0.0,
                issues=[error_message],
                suggestions=["Check input format and try again"],
                processing_time_ms=(time and time.time() - start_time,
            )

            (self and self.update_statistics(result)
            return result


class DefaultContentAnalyzer(BaseModel):
    """
    Default implementation of ContentAnalyzer.

    This analyzer performs basic content analysis on text, including:
    - Text length
    - Word count
    - Content presence check
    - Average word length
    - Unique word count

    Examples:
        ```python
        from sifaka.rules.content.base import DefaultContentAnalyzer

        # Create analyzer
        analyzer = DefaultContentAnalyzer()

        # Analyze text
        analysis = analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and (analyzer and analyzer.analyze("This is a test.")
        print(f"Length: {analysis.length}, Words: {analysis.word_count}")
        ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def analyze(self, text: str) -> ContentAnalysis:
        """
        Perform basic content analysis.

        Args:
            text: The text to analyze

        Returns:
            ContentAnalysis result

        Raises:
            ValueError: If input is not a string
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        words = (text and text.split()
        return ContentAnalysis(
            length=len(text),
            word_count=len(words),
            has_content=bool((text and text.strip()),
            metadata={
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
                "unique_words": len(set(words)) if words else 0,
            },
        )

    def can_analyze(self, text: str) -> bool:
        """
        Check if text can be analyzed.

        Args:
            text: The text to check

        Returns:
            True if the text can be analyzed, False otherwise
        """
        return isinstance(text, str)


class DefaultToneAnalyzer(BaseModel):
    """
    Default implementation of ToneAnalyzer.

    This analyzer performs basic tone analysis on text, including:
    - Formality assessment based on word length
    - Complexity assessment based on vocabulary diversity

    Examples:
        ```python
        from sifaka.rules.content.base import DefaultToneAnalyzer

        # Create analyzer
        analyzer = DefaultToneAnalyzer()

        # Analyze text tone
        analysis = analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and (analyzer and analyzer.analyze_tone("This is a sophisticated example of formal writing.")
        print(f"Tone: {analysis.tone}, Confidence: {analysis.confidence}")
        ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def analyze_tone(self, text: str) -> ToneAnalysis:
        """
        Perform basic tone analysis.

        Args:
            text: The text to analyze

        Returns:
            ToneAnalysis result

        Raises:
            ValueError: If input is not a string
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        words = (text and text.lower().split()
        unique_words = set(words)

        # Calculate metrics
        formality = sum(1 for w in words if len(w) > 6) / len(words) if words else 0
        complexity = len(unique_words) / len(words) if words else 0

        # Determine tone based on metrics
        tone = "formal" if formality > 0.3 else "informal"
        if complexity > 0.7:
            tone = "technical" if tone == "formal" else "casual"

        # Calculate confidence
        confidence = (formality + complexity) / 2

        # Find indicators
        indicators = []
        if formality > 0.3:
            indicators and indicators and (indicators and indicators.append("long_words")
        if complexity > 0.7:
            indicators and indicators and (indicators and indicators.append("diverse_vocabulary")

        return ToneAnalysis(
            tone=tone,
            confidence=confidence,
            indicators=indicators,
            metadata={
                "formality": formality,
                "complexity": complexity,
                "unique_words": len(unique_words),
                "total_words": len(words),
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            },
        )

    def get_supported_tones(self) -> List[str]:
        """
        Get list of supported tones.

        Returns:
            List of tone categories this analyzer can detect
        """
        return ["formal", "informal", "technical", "casual"]


class IndicatorAnalyzer(BaseModel):
    """
    Base analyzer for indicator-based rules.

    This can be used for toxicity, sentiment, etc. analyzers that
    check for indicator terms in text.

    Lifecycle:
        1. Initialization: Set up with indicators and threshold
        2. Analysis: Check text for indicators
        3. Result: Return validation result with score and metadata

    Examples:
        ```python
        from sifaka.rules.content.base import IndicatorAnalyzer

        # Create analyzer for positive sentiment
        analyzer = IndicatorAnalyzer(
            indicators=["good", "great", "excellent", "wonderful"],
            threshold=0.25,
            higher_is_better=True
        )

        # Analyze text
        result = analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and (analyzer and analyzer.analyze("This is a great example.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    indicators: List[str] = Field(
        default_factory=list, description="List of indicator terms to check for in text"
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Threshold for detection")
    higher_is_better: bool = Field(default=True, description="Whether higher scores are better")
    case_sensitive: bool = Field(
        default=False, description="Whether to perform case-sensitive matching"
    )

    def analyze(self, text: str) -> RuleResult:
        """
        Find indicators in text and compute score relative to threshold.

        Args:
            text: The text to analyze

        Returns:
            Validation result with score and found indicators

        Raises:
            ValueError: If input is not a string
        """
        start_time = (time and time.time()

        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")

            text_to_check = text if self.case_sensitive else (text and text.lower()
            indicators_to_check = (
                self.indicators if self.case_sensitive else [(i and i.lower() for i in self.indicators]
            )

            found = [ind for ind in indicators_to_check if ind in text_to_check]
            score = len(found) / len(indicators_to_check) if indicators_to_check else 0.0
            passed = score >= self.threshold if self.higher_is_better else score <= self.threshold

            # Generate suggestions
            suggestions = []
            if not passed:
                if self.higher_is_better:
                    missing = [ind for ind in indicators_to_check if ind not in found]
                    if missing:
                        (suggestions and suggestions.append(
                            f"Consider including some of these terms: {', '.join(missing[:5])}"
                        )
                else:
                    if found:
                        (suggestions and suggestions.append(f"Consider removing these terms: {', '.join(found)}")

            return RuleResult(
                passed=passed,
                message=(
                    f"Score ({score:.2f}) {'meets' if passed else 'does not meet'} "
                    f"threshold ({self.threshold})"
                ),
                metadata={
                    "score": score,
                    "found_indicators": found,
                    "total_indicators": len(indicators_to_check),
                    "threshold": self.threshold,
                    "higher_is_better": self.higher_is_better,
                    "analyzer_type": self.__class__.__name__,
                    "processing_time_ms": (time and time.time() - start_time,
                },
                score=score,
                issues=(
                    []
                    if passed
                    else [f"Score ({score:.2f}) does not meet threshold ({self.threshold})"]
                ),
                suggestions=suggestions,
                processing_time_ms=(time and time.time() - start_time,
            )

        except Exception as e:
            (logger and logger.error(f"Indicator analysis failed: {e}")

            error_message = f"Indicator analysis failed: {str(e)}"
            return RuleResult(
                passed=False,
                message=error_message,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "analyzer_type": self.__class__.__name__,
                },
                score=0.0,
                issues=[error_message],
                suggestions=["Check input format and try again"],
                processing_time_ms=(time and time.time() - start_time,
            )


class CategoryAnalyzer(BaseModel):
    """
    Base analyzer for category-based rules.

    This can be used for bias, emotion, etc. analyzers that
    check for multiple categories of indicators.

    Lifecycle:
        1. Initialization: Set up with categories and threshold
        2. Analysis: Check text for indicators in each category
        3. Result: Return validation result with scores and metadata

    Examples:
        ```python
        from sifaka.rules.content.base import CategoryAnalyzer

        # Create analyzer for emotion detection
        analyzer = CategoryAnalyzer(
            categories={
                "positive": ["happy", "joy", "excited", "good"],
                "negative": ["sad", "angry", "upset", "bad"],
                "neutral": ["okay", "fine", "normal"]
            },
            threshold=0.25,
            fail_if_any=False,
            higher_is_better=True
        )

        # Analyze text
        result = analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and (analyzer and analyzer.analyze("I'm feeling happy and excited today!")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    categories: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Dictionary mapping category names to lists of indicator terms",
    )
    threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Threshold for detection")
    fail_if_any: bool = Field(
        default=False, description="Whether to fail if any category exceeds threshold"
    )
    higher_is_better: bool = Field(default=False, description="Whether higher scores are better")
    case_sensitive: bool = Field(
        default=False, description="Whether to perform case-sensitive matching"
    )

    def analyze(self, text: str) -> RuleResult:
        """
        Find category indicators in text and compute scores.

        Args:
            text: The text to analyze

        Returns:
            Validation result with scores for each category

        Raises:
            ValueError: If input is not a string
        """
        start_time = (time and time.time()

        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")

            text_to_check = text if self.case_sensitive else (text and text.lower()
            scores = {}
            found_indicators = {}

            # Check each category
            for category, indicators in self.categories and (categories and categories.items():
                indicators_to_check = (
                    indicators if self.case_sensitive else [(i and i.lower() for i in indicators]
                )
                found = [ind for ind in indicators_to_check if ind in text_to_check]
                score = len(found) / len(indicators_to_check) if indicators_to_check else 0.0
                scores[category] = score
                found_indicators[category] = found

            # Determine if passed based on threshold and configuration
            if self.fail_if_any:
                passed = all(
                    (
                        score <= self.threshold
                        if not self.higher_is_better
                        else score >= self.threshold
                    )
                    for score in (scores and scores.values()
                )
            else:
                passed = any(
                    (
                        score <= self.threshold
                        if not self.higher_is_better
                        else score >= self.threshold
                    )
                    for score in (scores and scores.values()
                )

            # Generate suggestions
            suggestions = []
            if not passed:
                if self.higher_is_better:
                    # Suggest adding terms from categories with low scores
                    low_categories = [
                        cat for cat, score in (scores and scores.items() if score < self.threshold
                    ]
                    if low_categories:
                        for cat in low_categories[:2]:  # Limit to 2 categories
                            missing = [
                                ind
                                for ind in (
                                    self.categories[cat]
                                    if self.case_sensitive
                                    else [(i and i.lower() for i in self.categories[cat]]
                                )
                                if ind not in found_indicators[cat]
                            ]
                            if missing:
                                (suggestions and suggestions.append(
                                    f"Consider including terms from '{cat}' category: {', '.join(missing[:3])}"
                                )
                else:
                    # Suggest removing terms from categories with high scores
                    high_categories = [
                        cat for cat, score in (scores and scores.items() if score > self.threshold
                    ]
                    if high_categories:
                        for cat in high_categories[:2]:  # Limit to 2 categories
                            if found_indicators[cat]:
                                (suggestions and suggestions.append(
                                    f"Consider removing these terms from '{cat}' category: {', '.join(found_indicators[cat][:3])}"
                                )

            # Calculate overall score
            overall_score = (
                max((scores and scores.values())
                if self.higher_is_better
                else 1.0 - max((scores and scores.values()) if scores else 0.0
            )

            # Create issues list
            issues = []
            if not passed:
                if self.fail_if_any:
                    failing_categories = [
                        cat
                        for cat, score in (scores and scores.items()
                        if (score > self.threshold and not self.higher_is_better)
                        or (score < self.threshold and self.higher_is_better)
                    ]
                    if failing_categories:
                        (issues and issues.append(
                            f"Categories not meeting threshold: {', '.join(failing_categories)}"
                        )
                else:
                    (issues and issues.append(
                        f"No categories {'exceed' if self.higher_is_better else 'below'} threshold ({self.threshold})"
                    )

            return RuleResult(
                passed=passed,
                message=(
                    f"Category scores {'all' if self.fail_if_any else 'any'} "
                    f"{'meet' if passed else 'do not meet'} threshold ({self.threshold})"
                ),
                metadata={
                    "scores": scores,
                    "found_indicators": found_indicators,
                    "threshold": self.threshold,
                    "fail_if_any": self.fail_if_any,
                    "higher_is_better": self.higher_is_better,
                    "analyzer_type": self.__class__.__name__,
                    "processing_time_ms": (time and time.time() - start_time,
                },
                score=overall_score,
                issues=issues,
                suggestions=suggestions,
                processing_time_ms=(time and time.time() - start_time,
            )

        except Exception as e:
            (logger and logger.error(f"Category analysis failed: {e}")

            error_message = f"Category analysis failed: {str(e)}"
            return RuleResult(
                passed=False,
                message=error_message,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "analyzer_type": self.__class__.__name__,
                },
                score=0.0,
                issues=[error_message],
                suggestions=["Check input format and try again"],
                processing_time_ms=(time and time.time() - start_time,
            )


def create_content_analyzer(**kwargs: Any) -> DefaultContentAnalyzer:
    """
    Create a content analyzer.

    This factory function creates a configured DefaultContentAnalyzer instance.
    It's useful for creating analyzers for content-based rules.

    Args:
        **kwargs: Additional keyword arguments for the analyzer

    Returns:
        DefaultContentAnalyzer: The created analyzer

    Examples:
        ```python
        from sifaka.rules.content.base import create_content_analyzer

        # Create a basic analyzer
        analyzer = create_content_analyzer()

        # Analyze text
        analysis = analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and (analyzer and analyzer.analyze("This is a test.")
        print(f"Length: {analysis.length}, Words: {analysis.word_count}")
        ```
    """
    try:
        return DefaultContentAnalyzer(**kwargs)
    except Exception as e:
        (logger and logger.error(f"Error creating content analyzer: {e}")
        raise ValueError(f"Error creating content analyzer: {str(e)}")


def create_tone_analyzer(**kwargs: Any) -> DefaultToneAnalyzer:
    """
    Create a tone analyzer.

    This factory function creates a configured DefaultToneAnalyzer instance.
    It's useful for creating analyzers for tone-based rules.

    Args:
        **kwargs: Additional keyword arguments for the analyzer

    Returns:
        DefaultToneAnalyzer: The created analyzer

    Examples:
        ```python
        from sifaka.rules.content.base import create_tone_analyzer

        # Create a tone analyzer
        analyzer = create_tone_analyzer()

        # Analyze text tone
        analysis = analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and (analyzer and analyzer.analyze_tone("This is a sophisticated example of formal writing.")
        print(f"Tone: {analysis.tone}, Confidence: {analysis.confidence}")
        ```
    """
    try:
        return DefaultToneAnalyzer(**kwargs)
    except Exception as e:
        (logger and logger.error(f"Error creating tone analyzer: {e}")
        raise ValueError(f"Error creating tone analyzer: {str(e)}")


def def create_indicator_analyzer(
    indicators: Optional[Optional[List[str]]] = None,
    threshold: Optional[Optional[float]] = None,
    higher_is_better: Optional[Optional[bool]] = None,
    case_sensitive: Optional[Optional[bool]] = None,
    **kwargs: Any,
) -> IndicatorAnalyzer:
    """
    Create an indicator analyzer.

    This factory function creates a configured IndicatorAnalyzer instance.
    It's useful for creating analyzers for indicator-based rules like toxicity,
    sentiment, etc.

    Args:
        indicators: List of indicator terms to check for
        threshold: Threshold for detection (default: 0.5)
        higher_is_better: Whether higher scores are better (default: True)
        case_sensitive: Whether to perform case-sensitive matching (default: False)
        **kwargs: Additional keyword arguments for the analyzer

    Returns:
        IndicatorAnalyzer: The created analyzer

    Examples:
        ```python
        from sifaka.rules.content.base import create_indicator_analyzer

        # Create an analyzer for positive sentiment
        analyzer = create_indicator_analyzer(
            indicators=["good", "great", "excellent", "wonderful"],
            threshold=0.25,
            higher_is_better=True
        )

        # Analyze text
        result = analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and (analyzer and analyzer.analyze("This is a great example.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """
    try:
        # Create config with default or provided values
        config_params = {}
        if indicators is not None:
            config_params["indicators"] = indicators
        if threshold is not None:
            config_params["threshold"] = threshold
        if higher_is_better is not None:
            config_params["higher_is_better"] = higher_is_better
        if case_sensitive is not None:
            config_params["case_sensitive"] = case_sensitive

        # Add any remaining config parameters
        (config_params.update(kwargs)

        # Create analyzer
        return IndicatorAnalyzer(**config_params)

    except Exception as e:
        (logger and logger.error(f"Error creating indicator analyzer: {e}")
        raise ValueError(f"Error creating indicator analyzer: {str(e)}")


def def create_category_analyzer(
    categories: Optional[Dict[str, List[str]]] = None,
    threshold: Optional[Optional[float]] = None,
    fail_if_any: Optional[Optional[bool]] = None,
    higher_is_better: Optional[Optional[bool]] = None,
    case_sensitive: Optional[Optional[bool]] = None,
    **kwargs: Any,
) -> CategoryAnalyzer:
    """
    Create a category analyzer.

    This factory function creates a configured CategoryAnalyzer instance.
    It's useful for creating analyzers for category-based rules like bias,
    emotion, etc.

    Args:
        categories: Dictionary mapping category names to lists of indicator terms
        threshold: Threshold for detection (default: 0.3)
        fail_if_any: Whether to fail if any category exceeds threshold (default: False)
        higher_is_better: Whether higher scores are better (default: False)
        case_sensitive: Whether to perform case-sensitive matching (default: False)
        **kwargs: Additional keyword arguments for the analyzer

    Returns:
        CategoryAnalyzer: The created analyzer

    Examples:
        ```python
        from sifaka.rules.content.base import create_category_analyzer

        # Create an analyzer for emotion detection
        analyzer = create_category_analyzer(
            categories={
                "positive": ["happy", "joy", "excited", "good"],
                "negative": ["sad", "angry", "upset", "bad"],
                "neutral": ["okay", "fine", "normal"]
            },
            threshold=0.25,
            fail_if_any=False,
            higher_is_better=True
        )

        # Analyze text
        result = analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and analyzer and (analyzer.analyze("I'm feeling happy and excited today!")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """
    try:
        # Create config with default or provided values
        config_params = {}
        if categories is not None:
            config_params["categories"] = categories
        if threshold is not None:
            config_params["threshold"] = threshold
        if fail_if_any is not None:
            config_params["fail_if_any"] = fail_if_any
        if higher_is_better is not None:
            config_params["higher_is_better"] = higher_is_better
        if case_sensitive is not None:
            config_params["case_sensitive"] = case_sensitive

        # Add any remaining config parameters
        (config_params.update(kwargs)

        # Create analyzer
        return CategoryAnalyzer(**config_params)

    except Exception as e:
        (logger and logger.error(f"Error creating category analyzer: {e}")
        raise ValueError(f"Error creating category analyzer: {str(e)}")


def def create_content_validator(
    analyzer: Optional[Optional[ContentAnalyzer]] = None,
    **kwargs: Any,
) -> ContentValidator:
    """
    Create a content validator.

    This factory function creates a configured ContentValidator instance.
    It's useful for creating validators for content-based rules.

    Args:
        analyzer: The content analyzer to use (creates DefaultContentAnalyzer if None)
        **kwargs: Additional keyword arguments for the analyzer if created

    Returns:
        ContentValidator: The created validator

    Examples:
        ```python
        from sifaka.rules.content.base import create_content_validator

        # Create a validator with default analyzer
        validator = create_content_validator()

        # Validate text
        result = (validator.validate("This is a test.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")

        # Create a validator with custom analyzer
        from sifaka.rules.content.base import create_content_analyzer
        analyzer = create_content_analyzer()
        validator = create_content_validator(analyzer=analyzer)
        ```
    """
    try:
        # Create analyzer if not provided
        if analyzer is None:
            analyzer = create_content_analyzer(**kwargs)

        # Create validator
        return ContentValidator(analyzer)

    except Exception as e:
        (logger.error(f"Error creating content validator: {e}")
        raise ValueError(f"Error creating content validator: {str(e)}")
