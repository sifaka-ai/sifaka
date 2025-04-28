"""
Content validation rules for Sifaka.

.. deprecated:: 1.0.0
   This module is deprecated and will be removed in version 2.0.0.
   Use the following modules instead:

   - :mod:`sifaka.rules.content.prohibited` for prohibited content validation
   - :mod:`sifaka.rules.content.tone` for tone consistency validation

Migration guide:
1. Replace imports:
   - Old: from sifaka.rules.content import ProhibitedContentRule, ToneConsistencyRule
   - New: from sifaka.rules.content.prohibited import ProhibitedContentRule
         from sifaka.rules.content.tone import ToneConsistencyRule

2. Update configuration:
   - Prohibited content and tone configurations are now separate classes
   - Each has its own set of parameters and validation logic
   - See the respective module documentation for details

Example:
    Old code:
    >>> from sifaka.rules.content import ProhibitedContentRule
    >>> rule = ProhibitedContentRule()

    New code:
    >>> from sifaka.rules.content.prohibited import ProhibitedContentRule
    >>> rule = ProhibitedContentRule()
"""

import warnings

# Re-export everything from the new package
from sifaka.rules.content import *

warnings.warn(
    "The content.py module is deprecated and will be removed in version 2.0.0. "
    "Use sifaka.rules.content.prohibited and sifaka.rules.content.tone instead.",
    DeprecationWarning,
    stacklevel=2,
)
