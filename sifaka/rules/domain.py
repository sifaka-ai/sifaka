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

This module previously provided rules for validating content in specific domains,
including:
- Legal (general legal content, citations, and terminology)
- Medical
- Python code
- Content consistency

Each domain has specialized validators and configurations.
"""

import warnings

# Re-export everything from the new package
from sifaka.rules.domain import *

warnings.warn(
    "The domain.py module is deprecated and will be removed in version 2.0.0. "
    "Use sifaka.rules.domain.legal, sifaka.rules.domain.medical, "
    "sifaka.rules.domain.python, and sifaka.rules.domain.consistency instead.",
    DeprecationWarning,
    stacklevel=2,
)
