"""
Legal-specific rules for Sifaka.

DEPRECATION WARNING: This module is deprecated and will be removed in a future version.
Please import from sifaka.rules.domain instead.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "The legal.py module is deprecated and will be removed in a future version. "
    "Please import from sifaka.rules.domain instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from domain
from sifaka.rules.domain import (
    DefaultLegalCitationValidator,
    DefaultLegalTermsValidator,
    LegalCitationConfig,
    LegalCitationRule,
    LegalCitationValidator,
    LegalTermsConfig,
    LegalTermsRule,
    LegalTermsValidator,
    create_legal_citation_rule,
    create_legal_terms_rule,
)

# For backward compatibility
__all__ = [
    "LegalCitationConfig",
    "LegalTermsConfig",
    "LegalCitationValidator",
    "LegalTermsValidator",
    "DefaultLegalCitationValidator",
    "DefaultLegalTermsValidator",
    "LegalCitationRule",
    "LegalTermsRule",
    "create_legal_citation_rule",
    "create_legal_terms_rule",
]
