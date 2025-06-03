"""Format validator for Sifaka.

This module provides validators for checking text formatting requirements
such as structure, markdown formatting, JSON validity, etc.
"""

import json
import re
from typing import List, Optional, Dict, Any, Callable
import asyncio

from sifaka.core.thought import SifakaThought
from sifaka.utils.errors import ValidationError
from sifaka.utils.logging import get_logger
from sifaka.validators.base import BaseValidator, ValidationResult, TimingMixin

logger = get_logger(__name__)


class FormatValidator(BaseValidator, TimingMixin):
    """Validator that checks text formatting requirements.
    
    This validator can check for various formatting requirements:
    - JSON validity
    - Markdown structure
    - Custom format patterns
    - Line ending consistency
    - Indentation requirements
    
    Attributes:
        format_type: Type of format to validate ("json", "markdown", "custom")
        custom_checker: Custom validation function for "custom" format
        strict: Whether to fail on any formatting issue
    """
    
    def __init__(
        self,
        format_type: str = "text",
        custom_checker: Optional[Callable[[str], tuple[bool, List[str], List[str]]]] = None,
        strict: bool = True,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the format validator.
        
        Args:
            format_type: Type of format ("json", "markdown", "text", "custom")
            custom_checker: Custom validation function for "custom" format
                          Should return (is_valid, issues, suggestions)
            strict: Whether to fail on any formatting issue
            name: Custom name for the validator
            description: Custom description for the validator
            
        Raises:
            ValidationError: If configuration is invalid
        """
        supported_formats = ["json", "markdown", "text", "custom"]
        if format_type not in supported_formats:
            raise ValidationError(
                f"Unsupported format type: {format_type}",
                error_code="invalid_config",
                context={"format_type": format_type, "supported": supported_formats},
                suggestions=[f"Use one of: {', '.join(supported_formats)}"]
            )
        
        if format_type == "custom" and custom_checker is None:
            raise ValidationError(
                "Custom checker function required for 'custom' format type",
                error_code="invalid_config",
                suggestions=["Provide a custom_checker function", "Use a different format_type"]
            )
        
        # Set default name and description
        if name is None:
            name = f"format_{format_type}"
        
        if description is None:
            description = f"Validates {format_type} formatting requirements"
        
        super().__init__(name=name, description=description)
        
        self.format_type = format_type
        self.custom_checker = custom_checker
        self.strict = strict
        
        logger.debug(
            f"Created FormatValidator",
            extra={
                "validator_name": self.name,
                "format_type": self.format_type,
                "strict": self.strict,
                "has_custom_checker": custom_checker is not None,
            }
        )
    
    async def validate_async(self, thought: SifakaThought) -> ValidationResult:
        """Validate text formatting asynchronously.
        
        Args:
            thought: The SifakaThought to validate
            
        Returns:
            ValidationResult with format validation information
        """
        # Check if we have text to validate
        text = thought.current_text
        if not text:
            logger.debug(
                f"Format validation failed: no text",
                extra={"validator": self.name, "thought_id": thought.id}
            )
            return self.create_empty_text_result()
        
        with self.time_operation("format_validation") as timer:
            # Dispatch to appropriate format checker
            if self.format_type == "json":
                is_valid, issues, suggestions, metadata = self._validate_json(text)
            elif self.format_type == "markdown":
                is_valid, issues, suggestions, metadata = self._validate_markdown(text)
            elif self.format_type == "text":
                is_valid, issues, suggestions, metadata = self._validate_text(text)
            elif self.format_type == "custom":
                is_valid, issues, suggestions = self.custom_checker(text)
                metadata = {"format_type": "custom"}
            else:
                # Should not reach here due to constructor validation
                raise ValidationError(
                    f"Unknown format type: {self.format_type}",
                    error_code="internal_error"
                )
            
            # Calculate score
            if is_valid:
                score = 1.0
            elif self.strict:
                score = 0.0
            else:
                # Proportional score based on number of issues
                score = max(0.1, 1.0 - (len(issues) * 0.2))
            
            # Create result message
            if is_valid:
                message = f"Format validation passed: valid {self.format_type}"
            else:
                message = f"Format validation failed: {len(issues)} issue(s) in {self.format_type}"
            
            # Get processing time from timer context
            processing_time = getattr(timer, 'duration_ms', 0.0)
            
            result = self.create_validation_result(
                passed=is_valid,
                message=message,
                score=score,
                issues=issues,
                suggestions=suggestions,
                metadata={
                    **metadata,
                    "format_type": self.format_type,
                    "strict_mode": self.strict,
                    "text_length": len(text),
                },
                processing_time_ms=processing_time,
            )
            
            logger.debug(
                f"Format validation completed",
                extra={
                    "validator": self.name,
                    "thought_id": thought.id,
                    "passed": is_valid,
                    "format_type": self.format_type,
                    "issues_count": len(issues),
                    "score": score,
                }
            )
            
            return result
    
    def _validate_json(self, text: str) -> tuple[bool, List[str], List[str], Dict[str, Any]]:
        """Validate JSON formatting.
        
        Returns:
            Tuple of (is_valid, issues, suggestions, metadata)
        """
        issues = []
        suggestions = []
        metadata = {"format_type": "json"}
        
        try:
            parsed = json.loads(text)
            metadata["json_type"] = type(parsed).__name__
            if isinstance(parsed, dict):
                metadata["json_keys"] = len(parsed)
            elif isinstance(parsed, list):
                metadata["json_length"] = len(parsed)
            
            return True, issues, suggestions, metadata
            
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON: {str(e)}")
            suggestions.extend([
                "Check for missing quotes around strings",
                "Verify all brackets and braces are properly closed",
                "Ensure no trailing commas",
                "Use double quotes for strings, not single quotes",
            ])
            metadata["json_error"] = str(e)
            metadata["json_error_line"] = getattr(e, 'lineno', None)
            metadata["json_error_col"] = getattr(e, 'colno', None)
            
            return False, issues, suggestions, metadata
    
    def _validate_markdown(self, text: str) -> tuple[bool, List[str], List[str], Dict[str, Any]]:
        """Validate Markdown formatting.
        
        Returns:
            Tuple of (is_valid, issues, suggestions, metadata)
        """
        issues = []
        suggestions = []
        metadata = {"format_type": "markdown"}
        
        # Check for common markdown issues
        lines = text.split('\n')
        
        # Count markdown elements
        headers = len(re.findall(r'^#+\s', text, re.MULTILINE))
        links = len(re.findall(r'\[.*?\]\(.*?\)', text))
        code_blocks = len(re.findall(r'```', text)) // 2
        
        metadata.update({
            "headers": headers,
            "links": links,
            "code_blocks": code_blocks,
            "lines": len(lines),
        })
        
        # Check for unclosed code blocks
        code_block_markers = text.count('```')
        if code_block_markers % 2 != 0:
            issues.append("Unclosed code block (odd number of ``` markers)")
            suggestions.append("Ensure all code blocks are properly closed with ```")
        
        # Check for malformed links
        malformed_links = re.findall(r'\[.*?\]\([^)]*$', text, re.MULTILINE)
        if malformed_links:
            issues.append(f"Malformed links found: {len(malformed_links)}")
            suggestions.append("Check that all links have closing parentheses")
        
        # Check for inconsistent header levels (jumping from # to ###)
        header_matches = re.findall(r'^(#+)', text, re.MULTILINE)
        if header_matches:
            header_levels = [len(match) for match in header_matches]
            for i in range(1, len(header_levels)):
                if header_levels[i] > header_levels[i-1] + 1:
                    issues.append("Inconsistent header levels (skipping levels)")
                    suggestions.append("Use consecutive header levels (# then ##, not # then ###)")
                    break
        
        is_valid = len(issues) == 0
        return is_valid, issues, suggestions, metadata
    
    def _validate_text(self, text: str) -> tuple[bool, List[str], List[str], Dict[str, Any]]:
        """Validate general text formatting.
        
        Returns:
            Tuple of (is_valid, issues, suggestions, metadata)
        """
        issues = []
        suggestions = []
        metadata = {"format_type": "text"}
        
        lines = text.split('\n')
        
        # Check for basic text issues
        trailing_whitespace_lines = sum(1 for line in lines if line.endswith(' ') or line.endswith('\t'))
        empty_lines = sum(1 for line in lines if not line.strip())
        
        metadata.update({
            "lines": len(lines),
            "trailing_whitespace_lines": trailing_whitespace_lines,
            "empty_lines": empty_lines,
            "characters": len(text),
        })
        
        # Check for excessive trailing whitespace
        if trailing_whitespace_lines > len(lines) * 0.1:  # More than 10% of lines
            issues.append(f"Excessive trailing whitespace on {trailing_whitespace_lines} lines")
            suggestions.append("Remove trailing spaces and tabs from lines")
        
        # Check for excessive empty lines
        if empty_lines > len(lines) * 0.3:  # More than 30% empty lines
            issues.append(f"Excessive empty lines: {empty_lines} out of {len(lines)}")
            suggestions.append("Reduce the number of empty lines")
        
        is_valid = len(issues) == 0
        return is_valid, issues, suggestions, metadata


def create_format_validator(
    format_type: str = "text",
    custom_checker: Optional[Callable[[str], tuple[bool, List[str], List[str]]]] = None,
    strict: bool = True,
    name: Optional[str] = None,
) -> FormatValidator:
    """Create a format validator with the specified parameters.
    
    Args:
        format_type: Type of format to validate
        custom_checker: Custom validation function for "custom" format
        strict: Whether to fail on any formatting issue
        name: Custom name for the validator
        
    Returns:
        Configured FormatValidator instance
    """
    return FormatValidator(
        format_type=format_type,
        custom_checker=custom_checker,
        strict=strict,
        name=name,
    )


def json_validator(strict: bool = True) -> FormatValidator:
    """Create a validator for JSON formatting.
    
    Args:
        strict: Whether to fail on any JSON issue
        
    Returns:
        FormatValidator configured for JSON validation
    """
    return create_format_validator(
        format_type="json",
        strict=strict,
        name="json_format",
    )


def markdown_validator(strict: bool = True) -> FormatValidator:
    """Create a validator for Markdown formatting.
    
    Args:
        strict: Whether to fail on any Markdown issue
        
    Returns:
        FormatValidator configured for Markdown validation
    """
    return create_format_validator(
        format_type="markdown",
        strict=strict,
        name="markdown_format",
    )
