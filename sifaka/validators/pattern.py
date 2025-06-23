"""Pattern-based validator for Sifaka."""

import re
from typing import Optional, Dict, Pattern as PatternType

from ..core.interfaces import Validator
from ..core.models import ValidationResult, SifakaResult


class PatternValidator(Validator):
    """Validates text against regex patterns."""
    
    def __init__(
        self,
        required_patterns: Optional[Dict[str, str]] = None,
        forbidden_patterns: Optional[Dict[str, str]] = None,
        pattern_counts: Optional[Dict[str, tuple[int, Optional[int]]]] = None,
    ):
        """Initialize pattern validator.
        
        Args:
            required_patterns: Dict of {name: regex} patterns that must match
            forbidden_patterns: Dict of {name: regex} patterns that must not match
            pattern_counts: Dict of {pattern_name: (min, max)} for pattern occurrence counts
        """
        self.required_patterns: Dict[str, PatternType[str]] = {}
        self.forbidden_patterns: Dict[str, PatternType[str]] = {}
        self.pattern_counts = pattern_counts or {}
        
        # Compile required patterns
        if required_patterns:
            for name, pattern in required_patterns.items():
                self.required_patterns[name] = re.compile(pattern, re.MULTILINE)
        
        # Compile forbidden patterns
        if forbidden_patterns:
            for name, pattern in forbidden_patterns.items():
                self.forbidden_patterns[name] = re.compile(pattern, re.MULTILINE)
    
    @property
    def name(self) -> str:
        return "pattern_validator"
    
    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate text against patterns."""
        issues = []
        
        # Check required patterns
        for name, pattern in self.required_patterns.items():
            matches = pattern.findall(text)
            
            if name in self.pattern_counts:
                min_count, max_count = self.pattern_counts[name]
                match_count = len(matches)
                
                if match_count < min_count:
                    issues.append(f"Pattern '{name}' must occur at least {min_count} times, found {match_count}")
                elif max_count is not None and match_count > max_count:
                    issues.append(f"Pattern '{name}' must occur at most {max_count} times, found {match_count}")
            else:
                # Just check if pattern exists
                if not matches:
                    issues.append(f"Required pattern '{name}' not found")
        
        # Check forbidden patterns
        for name, pattern in self.forbidden_patterns.items():
            matches = pattern.findall(text)
            if matches:
                sample = matches[0] if len(matches[0]) < 50 else matches[0][:50] + "..."
                issues.append(f"Forbidden pattern '{name}' found: '{sample}'")
        
        # Build result
        if issues:
            return ValidationResult(
                validator=self.name,
                passed=False,
                score=0.0,
                details="; ".join(issues[:3])  # Limit to first 3 issues
            )
        
        # Calculate score based on pattern matching quality
        total_patterns = len(self.required_patterns) + len(self.forbidden_patterns)
        if total_patterns == 0:
            score = 1.0
            details = "No patterns configured"
        else:
            score = 1.0
            details = f"All {total_patterns} pattern(s) validated successfully"
        
        return ValidationResult(
            validator=self.name,
            passed=True,
            score=score,
            details=details
        )


# Convenience factory functions

def create_code_validator() -> PatternValidator:
    """Create a validator for code blocks."""
    return PatternValidator(
        required_patterns={
            "code_block": r"```[\w]*\n[\s\S]+?\n```",
        },
        pattern_counts={
            "code_block": (1, None),  # At least one code block
        }
    )


def create_citation_validator() -> PatternValidator:
    """Create a validator for academic citations."""
    return PatternValidator(
        required_patterns={
            "citation": r"\[\d+\]|\(\w+,?\s*\d{4}\)",  # [1] or (Author, 2023)
        },
        pattern_counts={
            "citation": (1, None),  # At least one citation
        }
    )


def create_structured_validator() -> PatternValidator:
    """Create a validator for structured documents."""
    return PatternValidator(
        required_patterns={
            "heading": r"^#+\s+.+$|^.+\n[=-]+$",  # Markdown or underline headings
            "list_item": r"^[\s]*[-*+•]\s+.+$|^[\s]*\d+\.\s+.+$",  # Bullet or numbered lists
        },
        pattern_counts={
            "heading": (1, None),  # At least one heading
            "list_item": (2, None),  # At least two list items
        }
    )