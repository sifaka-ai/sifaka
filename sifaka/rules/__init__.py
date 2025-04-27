"""
Sifaka Rules Package

This package provides a collection of validation rules for content validation in the
Sifaka framework. Each rule implements specific validation logic and follows a
consistent interface for easy integration and extension.

Architecture Overview:
- Each rule inherits from the base Rule class
- Rules follow the single responsibility principle
- All rules return RuleResult objects with consistent structure
- Rules can be combined in a Reflector for comprehensive validation

Available Rules:
1. LengthRule: Validates text length against constraints
   - Supports minimum, maximum, and exact length requirements
   - Useful for ensuring content meets size requirements

2. ProhibitedContentRule: Checks for prohibited terms
   - Case-insensitive matching
   - Returns detailed information about found terms
   - Ideal for content filtering

3. SentimentRule: Analyzes text sentiment
   - Threshold-based validation
   - Placeholder for sentiment analysis integration
   - Useful for maintaining content tone

4. ToxicityRule: Checks for toxic content
   - Threshold-based validation
   - Placeholder for toxicity analysis integration
   - Important for content safety

5. FormatRule: Validates text format
   - Supports markdown, JSON, and plain text
   - Basic syntax validation
   - Ensures content meets formatting requirements

6. Pattern Rules: Analyze text patterns and structure
   - SymmetryRule: Checks for text symmetry
   - RepetitionRule: Detects repeated patterns
   - Useful for structural validation

Usage Example:
    from sifaka.rules import LengthRule, ProhibitedContentRule, FormatRule
    from sifaka import Reflector

    # Create rules
    length_rule = LengthRule(min_length=10, max_length=1000)
    content_rule = ProhibitedContentRule(["bad", "inappropriate"])
    format_rule = FormatRule(required_format="markdown")

    # Create reflector with rules
    reflector = Reflector(
        rules=[length_rule, content_rule, format_rule],
        critique=True
    )

    # Validate content
    result = reflector.validate("Your content here")
"""

from .base import Rule, RuleResult
from .safety import ToxicityRule, BiasRule, HarmfulContentRule
from .classifier_rule import ClassifierRule
from .prohibited_content import ProhibitedContentRule
from .format import FormatRule
from .sentiment import SentimentRule
from .length import LengthRule
from .pattern_rules import SymmetryRule, RepetitionRule

__all__ = [
    "Rule",
    "RuleResult",
    "ToxicityRule",
    "BiasRule",
    "HarmfulContentRule",
    "ClassifierRule",
    "ProhibitedContentRule",
    "FormatRule",
    "SentimentRule",
    "LengthRule",
    "SymmetryRule",
    "RepetitionRule",
]
