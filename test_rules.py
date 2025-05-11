#!/usr/bin/env python3
"""
Test script for the rules module.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import rules components directly
from sifaka.rules.base import Rule
from sifaka.rules.config import RuleConfig, RulePriority
from sifaka.rules.result import RuleResult
from sifaka.rules.validators import BaseValidator, FunctionValidator, RuleValidator

# Print success message
print('Successfully imported rules components')
