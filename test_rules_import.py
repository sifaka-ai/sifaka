"""
Test script to check if the rules import without deprecation warnings.
"""

import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
from sifaka.rules.content.prohibited import (
    DefaultProhibitedContentValidator,
    create_prohibited_content_rule,
)

print("Successfully imported rules!")

# Create some rule instances to verify
length_rule = create_length_rule(min_chars=10, max_chars=1000)
prohibited_rule = create_prohibited_content_rule(terms=["bad", "inappropriate"])
format_rule = FormatRule(format_type="plain_text")

print("Successfully created rule instances!")
