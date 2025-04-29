"""
Domain-specific rules for Sifaka.

.. deprecated:: 1.0.0
   This module is deprecated and will be removed in version 2.0.0.
   Use the following modules instead:

   - :mod:`sifaka.rules.domain.legal` for legal content validation
   - :mod:`sifaka.rules.domain.medical` for medical content validation
   - :mod:`sifaka.rules.domain.python` for Python code validation
   - :mod:`sifaka.rules.domain.consistency` for content consistency validation

Migration guide:
1. Replace imports:
   - Old: from sifaka.rules.domain import LegalRule, MedicalRule, PythonRule, ConsistencyRule
   - New: from sifaka.rules.domain.legal import LegalRule
         from sifaka.rules.domain.medical import MedicalRule
         from sifaka.rules.domain.python import PythonRule
         from sifaka.rules.domain.consistency import ConsistencyRule

2. Update configuration:
   - Each domain now has its own configuration class
   - Each has its own set of parameters and validation logic
   - See the respective module documentation for details

Example:
    Old code:
    >>> from sifaka.rules.domain import LegalRule
    >>> rule = LegalRule()

    New code:
    >>> from sifaka.rules.domain.legal import LegalRule
    >>> rule = LegalRule()
"""

import warnings

# Re-export classes for backward compatibility
from sifaka.rules.domain.base import BaseDomainValidator, DomainValidator
from sifaka.rules.domain.consistency import (
    ConsistencyConfig,
    ConsistencyRule,
    ConsistencyValidator,
    DefaultConsistencyValidator,
    create_consistency_rule,
)
from sifaka.rules.domain.legal import (
    DefaultLegalCitationValidator,
    DefaultLegalTermsValidator,
    DefaultLegalValidator,
    LegalCitationConfig,
    LegalCitationRule,
    LegalCitationValidator,
    LegalConfig,
    LegalRule,
    LegalTermsConfig,
    LegalTermsRule,
    LegalTermsValidator,
    LegalValidator,
    create_legal_citation_rule,
    create_legal_rule,
    create_legal_terms_rule,
)
from sifaka.rules.domain.medical import (
    DefaultMedicalValidator,
    MedicalConfig,
    MedicalRule,
    MedicalValidator,
    create_medical_rule,
)
from sifaka.rules.domain.python import (
    DefaultPythonValidator,
    PythonConfig,
    PythonRule,
    PythonValidator,
    create_python_rule,
)

warnings.warn(
    "The domain module is deprecated and will be removed in version 2.0.0. "
    "Use sifaka.rules.domain.legal, sifaka.rules.domain.medical, "
    "sifaka.rules.domain.python, and sifaka.rules.domain.consistency instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Export public classes and functions
__all__ = [
    "MedicalRule",
    "MedicalConfig",
    "DefaultMedicalValidator",
    "LegalRule",
    "LegalConfig",
    "DefaultLegalValidator",
    "LegalCitationRule",
    "LegalCitationConfig",
    "DefaultLegalCitationValidator",
    "LegalTermsRule",
    "LegalTermsConfig",
    "DefaultLegalTermsValidator",
    "PythonRule",
    "PythonConfig",
    "DefaultPythonValidator",
    "ConsistencyRule",
    "ConsistencyConfig",
    "DefaultConsistencyValidator",
    "create_medical_rule",
    "create_legal_rule",
    "create_legal_citation_rule",
    "create_legal_terms_rule",
    "create_python_rule",
    "create_consistency_rule",
    "DomainValidator",
    "BaseDomainValidator",
    "MedicalValidator",
    "LegalValidator",
    "LegalCitationValidator",
    "LegalTermsValidator",
    "PythonValidator",
    "ConsistencyValidator",
]
