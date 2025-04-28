"""
Fact-checking rules for Sifaka.

.. deprecated:: 1.0.0
   This module is deprecated and will be removed in version 2.0.0.
   Use the following modules instead:

   - :mod:`sifaka.rules.factual.consistency` for factual consistency validation
   - :mod:`sifaka.rules.factual.confidence` for confidence level validation
   - :mod:`sifaka.rules.factual.citation` for citation validation
   - :mod:`sifaka.rules.factual.accuracy` for factual accuracy validation

Migration guide:
1. Replace imports:
   - Old: from sifaka.rules.factual import FactualConsistencyRule, ConfidenceRule, CitationRule, FactualAccuracyRule
   - New: from sifaka.rules.factual.consistency import FactualConsistencyRule
         from sifaka.rules.factual.confidence import ConfidenceRule
         from sifaka.rules.factual.citation import CitationRule
         from sifaka.rules.factual.accuracy import FactualAccuracyRule

2. Update configuration:
   - Each factual rule now has its own configuration class
   - Each has its own set of parameters and validation logic
   - See the respective module documentation for details

Example:
    Old code:
    >>> from sifaka.rules.factual import FactualConsistencyRule
    >>> rule = FactualConsistencyRule()

    New code:
    >>> from sifaka.rules.factual.consistency import FactualConsistencyRule
    >>> rule = FactualConsistencyRule()
"""

import warnings

# Re-export classes for backward compatibility
from sifaka.rules.factual.accuracy import (
    DefaultFactualAccuracyValidator,
    FactualAccuracyConfig,
    FactualAccuracyRule,
    create_factual_accuracy_rule,
)
from sifaka.rules.factual.base import BaseFactualValidator, FactualValidator
from sifaka.rules.factual.citation import (
    CitationConfig,
    CitationRule,
    DefaultCitationValidator,
    create_citation_rule,
)
from sifaka.rules.factual.confidence import (
    ConfidenceConfig,
    ConfidenceRule,
    DefaultConfidenceValidator,
    create_confidence_rule,
)
from sifaka.rules.factual.consistency import (
    DefaultFactualConsistencyValidator,
    FactualConsistencyConfig,
    FactualConsistencyRule,
    create_factual_consistency_rule,
)

warnings.warn(
    "The factual module is deprecated and will be removed in version 2.0.0. "
    "Use sifaka.rules.factual.consistency, sifaka.rules.factual.confidence, "
    "sifaka.rules.factual.citation, and sifaka.rules.factual.accuracy instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Export public classes and functions
__all__ = [
    "FactualConsistencyRule",
    "FactualConsistencyConfig",
    "DefaultFactualConsistencyValidator",
    "ConfidenceRule",
    "ConfidenceConfig",
    "DefaultConfidenceValidator",
    "CitationRule",
    "CitationConfig",
    "DefaultCitationValidator",
    "FactualAccuracyRule",
    "FactualAccuracyConfig",
    "DefaultFactualAccuracyValidator",
    "BaseFactualValidator",
    "FactualValidator",
    "create_factual_consistency_rule",
    "create_confidence_rule",
    "create_citation_rule",
    "create_factual_accuracy_rule",
]
