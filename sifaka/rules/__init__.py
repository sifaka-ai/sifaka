"""
Sifaka Rules Package

This package provides a collection of validation rules for content validation in the
Sifaka framework. Each rule implements specific validation logic and follows a
consistent interface for easy integration and extension.

Architecture Overview:
- Each rule inherits from the base Rule class
- Rules follow the single responsibility principle
- All rules return RuleResult objects with consistent structure
- Rules can be combined in a Chain for comprehensive validation

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

8. Adapter Rules:
   - ClassifierRule: Converts classifiers to rules

9. Wrapper Rules:
   - WrapperRule: Base wrapper for rules
   - CodeBlockRule: Validates code blocks

Usage Example:
    from sifaka.chain import Chain
    from sifaka.models import OpenAIProvider
    from sifaka.rules import LengthRule, ProhibitedContentRule, FormatRule
    from sifaka.critics.prompt import PromptCritic, PromptCriticConfig

    # Configure provider
    provider = OpenAIProvider(
        model_name="gpt-4-turbo-preview",
        config={"api_key": "your-api-key"}
    )

    # Create rules with type hints for better IDE support
    length_rule = LengthRule(
        name="length",
        config=RuleConfig(
            metadata={
                "min_length": 10,
                "max_length": 1000,
                "unit": "characters"
            },
            priority=RulePriority.MEDIUM,
            cost=1.0
        )
    )
    content_rule = ProhibitedContentRule(
        name="content_filter",
        config=RuleConfig(
            metadata={
                "prohibited_terms": ["bad", "inappropriate"],
                "case_sensitive": False
            },
            priority=RulePriority.HIGH,
            cost=1.0
        )
    )
    format_rule = FormatRule(
        name="format",
        config=RuleConfig(
            metadata={
                "required_format": "markdown",
                "strict": False
            },
            priority=RulePriority.LOW,
            cost=1.0
        )
    )

    # Create critic for content improvement
    critic = PromptCritic(
        model=provider,
        config=PromptCriticConfig(
            name="content_improver",
            system_prompt="You are an expert editor that improves text while maintaining its original meaning."
        )
    )

    # Create chain with rules and critic
    chain = Chain(
        model=provider,
        rules=[length_rule, content_rule, format_rule],
        critic=critic,
        max_attempts=3
    )

    # Validate and improve content
    result = chain.run("Your content here")

    # Access validation results
    print(f"Content: {result.content}")
    print(f"Validation passed: {result.passed_validation}")
    if not result.passed_validation:
        print("Failed rules:")
        for rule_name, rule_result in result.rule_results.items():
            if not rule_result.passed:
                print(f"- {rule_name}: {rule_result.message}")
"""

import warnings

from .base import FunctionRule, Rule, RuleResult

# Deprecated - use adapters instead
from .classifier_rule import ClassifierRule as _DeprecatedClassifierRule
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

# Import from the new adapters module
from .adapters import (
    ClassifierAdapter,
    ClassifierRule,
    create_classifier_rule,
)

# Emit deprecation warning for old ClassifierRule
ClassifierRule = _DeprecatedClassifierRule  # Keep backward compatibility
warnings.warn(
    "Direct import from classifier_rule is deprecated. "
    "Use rules.adapters.ClassifierRule instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Base
    "Rule",
    "RuleResult",
    "FunctionRule",
    # Safety
    "ToxicityRule",
    "BiasRule",
    "HarmfulContentRule",
    # Classifier (both deprecated and new)
    "ClassifierRule",
    "ClassifierAdapter",
    "create_classifier_rule",
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
