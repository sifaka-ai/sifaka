#!/usr/bin/env python3
"""
OpenAI Integration Example for Sifaka.

This example demonstrates:
1. Setting up an OpenAI model provider
2. Configuring various validation rules
3. Analyzing text against these rules
4. Using a critic to improve text based on violations

Usage:
    python openai_example.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
    - OpenAI API key in OPENAI_API_KEY environment variable
"""

import os
import sys

# Add parent directory to system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing dotenv package. Install with: pip install python-dotenv")
    sys.exit(1)

from sifaka.rules import LengthRule, ProhibitedContentRule, RepetitionRule
from sifaka.rules.adapters import ClassifierRuleAdapter
from sifaka.classifiers.readability import ReadabilityClassifier
from sifaka.classifiers.language import LanguageClassifier
from sifaka.rules.base import RuleConfig, RulePriority
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.utils.logging import get_logger

# Initialize logger from Sifaka
logger = get_logger(__name__)


def main():
    """Run the OpenAI example with pattern analysis."""
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    logger.info("Starting OpenAI integration example...")

    # Initialize OpenAI provider
    openai_provider = OpenAIProvider(
        model_name="gpt-4-turbo-preview",
        config=ModelConfig(
            api_key=api_key,
            temperature=0.7,
            max_tokens=1000,
        ),
    )

    # Create rules with proper configurations
    logger.info("Creating validation rules...")

    length_rule = LengthRule(
        name="length_validator",
        description="Validates text length between 100 and 500 characters",
        config=RuleConfig(
            metadata={
                "min_length": 100,
                "max_length": 500,
                "unit": "characters",
            },
            priority=RulePriority.HIGH,
            cost=0.1,
        ),
    )

    content_rule = ProhibitedContentRule(
        name="content_filter",
        description="Checks for prohibited or inappropriate content",
        config=RuleConfig(
            metadata={
                "prohibited_terms": ["hate", "violence", "profanity"],
                "case_sensitive": False,
            },
            priority=RulePriority.HIGH,
            cost=0.1,
        ),
    )

    # Create a readability classifier rule adapter
    readability_rule = ClassifierRuleAdapter(
        classifier_cls=ReadabilityClassifier,
        rule_config=RuleConfig(
            priority=RulePriority.MEDIUM,
            cost=1.0,
        ),
    )

    # Create a language classifier rule adapter
    language_rule = ClassifierRuleAdapter(
        classifier_cls=LanguageClassifier,
        rule_config=RuleConfig(
            priority=RulePriority.MEDIUM,
            cost=1.0,
        ),
    )

    repetition_rule = RepetitionRule(
        name="repetition_detector",
        description="Detects repetitive patterns in text",
        config=RuleConfig(
            metadata={
                "pattern_type": "repeat",
                "pattern_length": 2,
                "case_sensitive": True,
                "allow_overlap": False,
            },
            priority=RulePriority.MEDIUM,
            cost=1.0,
        ),
    )

    # Create a critic with the OpenAI provider
    critic = PromptCritic(
        model=openai_provider,
        config=PromptCriticConfig(
            name="openai_critic",
            description="A critic that uses OpenAI to improve text",
            system_prompt="You are an expert editor that improves text.",
            temperature=0.7,
            max_tokens=1000,
        ),
    )

    # Example text to validate
    text = "This is a short test text. It needs to be longer to pass the length rule."
    logger.info(f"Analyzing text: '{text}'")

    # Validate text with each rule and collect violations
    all_rules = [length_rule, content_rule, readability_rule, language_rule, repetition_rule]
    violations = []

    for rule in all_rules:
        logger.info(f"Validating with {rule.name}...")
        result = rule._validate_impl(text)
        logger.info(f"Passed: {result.passed}, Message: {result.message}")

        if not result.passed:
            violations.append(
                {"rule": rule.name, "message": result.message, "metadata": result.metadata}
            )

    # If there are violations, use the critic to improve the text
    if violations:
        logger.info(f"Found {len(violations)} violations. Using critic to improve text...")
        improved_text = critic.improve(text, violations)
        logger.info(f"Original text: '{text}'")
        logger.info(f"Improved text: '{improved_text}'")
    else:
        logger.info("No violations found.")

    logger.info("OpenAI integration example completed.")


if __name__ == "__main__":
    main()
