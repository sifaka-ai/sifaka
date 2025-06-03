"""Content validator for Sifaka.

This module provides validators for checking text content against prohibited
and required patterns, words, or phrases. Supports regex patterns, case sensitivity,
and whole word matching.
"""

import re
from typing import List, Optional, Pattern, Dict, Any
import asyncio

from sifaka.core.thought import SifakaThought
from sifaka.utils.errors import ValidationError
from sifaka.utils.logging import get_logger
from sifaka.validators.base import BaseValidator, ValidationResult, TimingMixin

logger = get_logger(__name__)


class ContentValidator(BaseValidator, TimingMixin):
    """Validator that checks text content against prohibited and required patterns.
    
    This validator can check for:
    - Prohibited content that must not appear in text
    - Required content that must appear in text
    - Support for regex patterns, case sensitivity, and whole word matching
    
    Attributes:
        prohibited: List of prohibited patterns/words
        required: List of required patterns/words
        case_sensitive: Whether matching is case-sensitive
        whole_word: Whether to match whole words only
        regex: Whether patterns are regular expressions
    """
    
    def __init__(
        self,
        prohibited: Optional[List[str]] = None,
        required: Optional[List[str]] = None,
        case_sensitive: bool = False,
        whole_word: bool = False,
        regex: bool = False,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the content validator.
        
        Args:
            prohibited: List of prohibited patterns/words
            required: List of required patterns/words
            case_sensitive: Whether matching is case-sensitive
            whole_word: Whether to match whole words only
            regex: Whether patterns are regular expressions
            name: Custom name for the validator
            description: Custom description for the validator
            
        Raises:
            ValidationError: If configuration is invalid
        """
        if not prohibited and not required:
            raise ValidationError(
                "Either prohibited or required patterns must be provided",
                error_code="invalid_config",
                suggestions=[
                    "Provide at least one prohibited pattern",
                    "Provide at least one required pattern",
                    "Provide both prohibited and required patterns",
                ]
            )
        
        # Set default name and description
        if name is None:
            parts = []
            if prohibited:
                parts.append(f"{len(prohibited)} prohibited")
            if required:
                parts.append(f"{len(required)} required")
            name = f"content_{'_'.join(parts)}"
        
        if description is None:
            parts = []
            if prohibited:
                parts.append(f"checks for {len(prohibited)} prohibited patterns")
            if required:
                parts.append(f"requires {len(required)} patterns")
            description = f"Content validator that {' and '.join(parts)}"
        
        super().__init__(name=name, description=description)
        
        self.prohibited = prohibited or []
        self.required = required or []
        self.case_sensitive = case_sensitive
        self.whole_word = whole_word
        self.regex = regex
        
        # Compile patterns for efficiency
        self._compiled_prohibited: List[Pattern[str]] = []
        self._compiled_required: List[Pattern[str]] = []
        self._compile_patterns()
        
        logger.debug(
            f"Created ContentValidator",
            extra={
                "validator_name": self.name,
                "prohibited_count": len(self.prohibited),
                "required_count": len(self.required),
                "case_sensitive": self.case_sensitive,
                "whole_word": self.whole_word,
                "regex": self.regex,
            }
        )
    
    def _compile_patterns(self) -> None:
        """Compile patterns into regex objects for efficient matching."""
        flags = 0 if self.case_sensitive else re.IGNORECASE
        
        # Compile prohibited patterns
        for i, pattern in enumerate(self.prohibited):
            try:
                regex_pattern = self._create_regex_pattern(pattern)
                compiled = re.compile(regex_pattern, flags)
                self._compiled_prohibited.append(compiled)
            except re.error as e:
                logger.warning(
                    f"Invalid prohibited pattern '{pattern}': {e}",
                    extra={"validator": self.name, "pattern_index": i}
                )
                # Skip invalid patterns rather than failing completely
        
        # Compile required patterns
        for i, pattern in enumerate(self.required):
            try:
                regex_pattern = self._create_regex_pattern(pattern)
                compiled = re.compile(regex_pattern, flags)
                self._compiled_required.append(compiled)
            except re.error as e:
                logger.warning(
                    f"Invalid required pattern '{pattern}': {e}",
                    extra={"validator": self.name, "pattern_index": i}
                )
                # Skip invalid patterns rather than failing completely
    
    def _create_regex_pattern(self, pattern: str) -> str:
        """Create a regex pattern from a string pattern.
        
        Args:
            pattern: Input pattern string
            
        Returns:
            Regex pattern string
        """
        if self.regex:
            # Pattern is already a regex
            return pattern
        else:
            # Escape special regex characters
            escaped = re.escape(pattern)
            if self.whole_word:
                # Add word boundaries
                return rf"\b{escaped}\b"
            else:
                return escaped
    
    async def validate_async(self, thought: SifakaThought) -> ValidationResult:
        """Validate text content asynchronously.
        
        Args:
            thought: The SifakaThought to validate
            
        Returns:
            ValidationResult with content validation information
        """
        # Check if we have text to validate
        text = thought.current_text
        if not text:
            logger.debug(
                f"Content validation failed: no text",
                extra={"validator": self.name, "thought_id": thought.id}
            )
            return self.create_empty_text_result()
        
        with self.time_operation("content_validation") as timer:
            # Check prohibited content
            prohibited_matches = []
            for i, pattern in enumerate(self._compiled_prohibited):
                matches = list(pattern.finditer(text))
                for match in matches:
                    prohibited_matches.append({
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": self.prohibited[i],
                        "pattern_index": i,
                    })
            
            # Check required content
            missing_required = []
            for i, pattern in enumerate(self._compiled_required):
                if not pattern.search(text):
                    missing_required.append(self.required[i])
            
            # Determine validation result
            violations = len(prohibited_matches) + len(missing_required)
            passed = violations == 0
            
            # Create issues and suggestions
            issues = []
            suggestions = []
            
            if prohibited_matches:
                unique_matches = set(match["match"] for match in prohibited_matches)
                unique_patterns = set(match["pattern"] for match in prohibited_matches)
                issues.append(f"Text contains prohibited content: {', '.join(unique_matches)}")
                suggestions.extend([
                    f"Remove or rephrase: {', '.join(unique_matches)}",
                    f"Avoid patterns like: {', '.join(unique_patterns)}",
                ])
            
            if missing_required:
                issues.append(f"Text missing required content: {', '.join(missing_required)}")
                suggestions.append(f"Include the following: {', '.join(missing_required)}")
            
            # Calculate score
            if passed:
                score = 1.0
            else:
                # Score based on proportion of violations
                total_checks = len(self.prohibited) + len(self.required)
                score = max(0.0, 1.0 - (violations / max(1, total_checks)))
            
            # Create result message
            if passed:
                message = "Content validation passed: all requirements met"
            else:
                message = f"Content validation failed: {violations} violation(s)"
            
            # Get processing time from timer context
            processing_time = getattr(timer, 'duration_ms', 0.0)
            
            result = self.create_validation_result(
                passed=passed,
                message=message,
                score=score,
                issues=issues,
                suggestions=suggestions,
                metadata={
                    "prohibited_matches": len(prohibited_matches),
                    "missing_required": len(missing_required),
                    "total_violations": violations,
                    "prohibited_patterns": len(self.prohibited),
                    "required_patterns": len(self.required),
                    "case_sensitive": self.case_sensitive,
                    "whole_word": self.whole_word,
                    "regex_mode": self.regex,
                    "text_length": len(text),
                },
                processing_time_ms=processing_time,
            )
            
            logger.debug(
                f"Content validation completed",
                extra={
                    "validator": self.name,
                    "thought_id": thought.id,
                    "passed": passed,
                    "violations": violations,
                    "prohibited_matches": len(prohibited_matches),
                    "missing_required": len(missing_required),
                    "score": score,
                }
            )
            
            return result


def create_content_validator(
    prohibited: Optional[List[str]] = None,
    required: Optional[List[str]] = None,
    case_sensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
    name: Optional[str] = None,
) -> ContentValidator:
    """Create a content validator with the specified parameters.
    
    Args:
        prohibited: List of prohibited patterns/words
        required: List of required patterns/words
        case_sensitive: Whether matching is case-sensitive
        whole_word: Whether to match whole words only
        regex: Whether patterns are regular expressions
        name: Custom name for the validator
        
    Returns:
        Configured ContentValidator instance
    """
    return ContentValidator(
        prohibited=prohibited,
        required=required,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        regex=regex,
        name=name,
    )


def prohibited_content_validator(
    prohibited: List[str],
    case_sensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
) -> ContentValidator:
    """Create a validator that checks for prohibited content only.
    
    Args:
        prohibited: List of prohibited patterns/words
        case_sensitive: Whether matching is case-sensitive
        whole_word: Whether to match whole words only
        regex: Whether patterns are regular expressions
        
    Returns:
        ContentValidator configured for prohibited content checking
    """
    return create_content_validator(
        prohibited=prohibited,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        regex=regex,
        name="prohibited_content",
    )


def required_content_validator(
    required: List[str],
    case_sensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
) -> ContentValidator:
    """Create a validator that checks for required content only.
    
    Args:
        required: List of required patterns/words
        case_sensitive: Whether matching is case-sensitive
        whole_word: Whether to match whole words only
        regex: Whether patterns are regular expressions
        
    Returns:
        ContentValidator configured for required content checking
    """
    return create_content_validator(
        required=required,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        regex=regex,
        name="required_content",
    )
