"""
Test script to check if the rules import without deprecation warnings.
"""

import warnings

# Enable all warnings
warnings.filterwarnings("always")

# Import from rules module
from sifaka.rules import (
    Rule,
    RuleResult,
    RuleConfig,
    LengthRule,
    ProhibitedContentRule,
    FormatRule,
    ToxicityRule,
    BiasRule,
    create_length_rule,
    create_plain_text_rule,
    create_json_rule,
)

print("Successfully imported rules!")

# Create some rule instances to verify
length_rule = create_length_rule(min_chars=10, max_chars=1000)
prohibited_rule = ProhibitedContentRule(
    config=RuleConfig(params={"terms": ["bad", "inappropriate"]})
)
format_rule = FormatRule(format_type="plain_text")

print("Successfully created rule instances!")
