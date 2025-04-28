"""
DEPRECATED: This module has been replaced by pattern_rules.py

The functionality previously provided by the Reflector class has been moved to
specialized pattern detection rules in sifaka.rules.pattern_rules.

Instead of using this module, use the new pattern rules:

```python
from sifaka.rules.pattern_rules import SymmetryRule, RepetitionRule

# For symmetry detection (previously horizontal/vertical reflection)
symmetry_rule = SymmetryRule(
    name="symmetry_check",
    config=SymmetryConfig(
        mirror_mode="both",
        symmetry_threshold=0.8
    )
)

# For pattern detection (previously pattern matching)
repetition_rule = RepetitionRule(
    name="repetition_check",
    config=RepetitionConfig(
        pattern_type="repeat",
        pattern_length=3
    )
)
```

This module will be removed in a future version.
"""

import warnings
from typing import Dict, Optional

from sifaka.rules.pattern_rules import (
    RepetitionConfig,
    RepetitionRule,
    RuleResult,
    SymmetryConfig,
    SymmetryRule,
)


class Reflector:
    """
    DEPRECATED: Use SymmetryRule and RepetitionRule instead.

    This class has been replaced by specialized pattern rules in
    sifaka.rules.pattern_rules.
    """

    def __init__(
        self,
        reflection_config: Optional[SymmetryConfig] = None,
        pattern_config: Optional[RepetitionConfig] = None,
    ) -> None:
        """Initialize with warning about deprecation."""
        warnings.warn(
            "The Reflector class is deprecated and will be removed in a future version. "
            "Use SymmetryRule and RepetitionRule from sifaka.rules.pattern_rules instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self._symmetry_rule = SymmetryRule(
            name="symmetry_validator",
            config=reflection_config or SymmetryConfig(),
        )

        self._pattern_rule = RepetitionRule(
            name="pattern_detector",
            config=pattern_config or RepetitionConfig(),
        )

    @property
    def reflection_config(self) -> SymmetryConfig:
        """Get symmetry configuration."""
        return self._symmetry_rule.config

    @property
    def pattern_config(self) -> RepetitionConfig:
        """Get pattern configuration."""
        return self._pattern_rule.config

    def validate_reflection(self, text: str) -> RuleResult:
        """Validate text symmetry (deprecated)."""
        return self._symmetry_rule.validate(text)

    def validate_pattern(self, text: str) -> RuleResult:
        """Validate text patterns (deprecated)."""
        return self._pattern_rule.validate(text)

    def validate(self, text: str) -> Dict[str, RuleResult]:
        """Run all validations (deprecated)."""
        return {
            "reflection": self.validate_reflection(text),
            "pattern": self.validate_pattern(text),
        }
