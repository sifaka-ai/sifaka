"""
Factual validation rules for Sifaka.

This module provides rules for validating factual content in text, including:
- Citation validation
- Confidence validation
- Consistency validation
- Accuracy validation

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - Factory functions handle configuration
    - Validator factory functions create standalone validators

Usage Example:
    from sifaka.rules.factual import (
        create_citation_rule,
        create_confidence_rule,
        create_consistency_rule,
        create_accuracy_rule
    )

    # Create a citation rule
    citation_rule = create_citation_rule(
        citation_patterns=[
            r"\(\d{4}\)",  # (2024)
            r"\[.*?\]",    # [Author, 2024]
            r"\(.*?\)"     # (Author, 2024)
        ],
        min_citations=1,
        max_citations=5
    )

    # Create a confidence rule
    confidence_rule = create_confidence_rule(
        confidence_indicators=[
            "highly confident",
            "very certain",
            "definitely",
            "without a doubt",
            "absolutely"
        ],
        threshold=0.8
    )

    # Create a consistency rule
    consistency_rule = create_consistency_rule(
        contradiction_indicators=[
            "however",
            "but",
            "although",
            "despite",
            "in contrast"
        ],
        threshold=0.7
    )

    # Create an accuracy rule
    accuracy_rule = create_accuracy_rule(
        knowledge_base=[
            "The Earth is round",
            "Water boils at 100°C at sea level",
            "The capital of France is Paris"
        ],
        threshold=0.8
    )
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict

from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.factual.base import BaseFactualValidator
from sifaka.rules.factual.citation import (
    CitationConfig,
    DefaultCitationValidator,
    CitationRule,
    create_citation_validator,
    create_citation_rule,
)
from sifaka.rules.factual.confidence import (
    ConfidenceConfig,
    DefaultConfidenceValidator,
    ConfidenceRule,
    create_confidence_validator,
    create_confidence_rule,
)
from sifaka.rules.factual.consistency import (
    ConsistencyConfig,
    DefaultConsistencyValidator,
    ConsistencyRule,
    create_consistency_validator,
    create_consistency_rule,
)
from sifaka.rules.factual.accuracy import (
    AccuracyConfig,
    DefaultAccuracyValidator,
    AccuracyRule,
    create_accuracy_validator,
    create_accuracy_rule,
)


__all__ = [
    # Config classes
    "CitationConfig",
    "ConfidenceConfig",
    "ConsistencyConfig",
    "AccuracyConfig",
    # Validator classes
    "DefaultCitationValidator",
    "DefaultConfidenceValidator",
    "DefaultConsistencyValidator",
    "DefaultAccuracyValidator",
    # Rule classes
    "CitationRule",
    "ConfidenceRule",
    "ConsistencyRule",
    "AccuracyRule",
    # Factory functions
    "create_citation_validator",
    "create_citation_rule",
    "create_confidence_validator",
    "create_confidence_rule",
    "create_consistency_validator",
    "create_consistency_rule",
    "create_accuracy_validator",
    "create_accuracy_rule",
]
