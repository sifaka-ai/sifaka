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
1. Base Rules:
   - Rule (base class)
   - RuleResult
   - FunctionRule

2. Content Rules:
   - LengthRule: Validates text length
   - ProhibitedContentRule: Checks for prohibited terms
   - FormatRule: Validates text format
   - ParagraphRule: Validates paragraph structure
   - StyleRule: Validates text style
   - FormattingRule: Validates text formatting

3. Pattern Rules:
   - SymmetryRule: Checks for text symmetry
   - RepetitionRule: Detects repeated patterns

4. Safety Rules:
   - ToxicityRule: Checks for toxic content
   - BiasRule: Checks for bias
   - HarmfulContentRule: Detects harmful content

5. Domain-Specific Rules:
   - MedicalRule: Medical content validation
   - LegalRule: Legal content validation
   - LegalCitationRule: Validates legal citations
   - LegalTermsRule: Validates legal terminology
   - PythonRule: Python code validation
   - ConsistencyRule: Content consistency validation

6. Factual Rules:
   - FactualConsistencyRule: Checks fact consistency
   - ConfidenceRule: Validates confidence levels
   - CitationRule: Validates citations
   - FactualAccuracyRule: Checks factual accuracy

7. Sentiment Rules:
   - SentimentRule: Analyzes text sentiment
   - EmotionalContentRule: Validates emotional content

8. Wrapper Rules:
   - WrapperRule: Base wrapper for rules
   - CodeBlockRule: Validates code blocks

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

from .base import FunctionRule, Rule, RuleResult
from .classifier_rule import ClassifierRule
from .domain import (
    ConsistencyRule,
    LegalCitationRule,
    LegalRule,
    LegalTermsRule,
    MedicalRule,
    PythonRule,
    create_consistency_rule,
    create_legal_citation_rule,
    create_legal_rule,
    create_legal_terms_rule,
    create_medical_rule,
    create_python_rule,
)
from .factual import (
    CitationRule,
    ConfidenceRule,
    FactualAccuracyRule,
    FactualConsistencyRule,
)
from .format import FormatRule
from .formatting import FormattingRule, ParagraphRule, StyleRule
from .length import LengthRule
from .pattern_rules import RepetitionRule, SymmetryRule
from .prohibited_content import ProhibitedContentRule
from .safety import BiasRule, HarmfulContentRule, ToxicityRule
from .sentiment import EmotionalContentRule, SentimentRule
from .wrapper import CodeBlockRule, WrapperRule

__all__ = [
    # Base
    "Rule",
    "RuleResult",
    "FunctionRule",
    # Safety
    "ToxicityRule",
    "BiasRule",
    "HarmfulContentRule",
    # Classifier
    "ClassifierRule",
    # Content
    "ProhibitedContentRule",
    "FormatRule",
    "LengthRule",
    # Pattern
    "SymmetryRule",
    "RepetitionRule",
    # Domain
    "LegalRule",
    "MedicalRule",
    "PythonRule",
    "ConsistencyRule",
    # Legal-specific
    "LegalCitationRule",
    "LegalTermsRule",
    "create_legal_citation_rule",
    "create_legal_terms_rule",
    "create_legal_rule",
    "create_medical_rule",
    "create_python_rule",
    "create_consistency_rule",
    # Factual
    "FactualConsistencyRule",
    "ConfidenceRule",
    "CitationRule",
    "FactualAccuracyRule",
    # Sentiment
    "SentimentRule",
    "EmotionalContentRule",
    # Formatting
    "ParagraphRule",
    "StyleRule",
    "FormattingRule",
    # Wrapper
    "WrapperRule",
    "CodeBlockRule",
]
