"""
OpenAI Integration Example for Sifaka Pattern Analysis.

This example demonstrates how to use Sifaka with OpenAI's models:
1. Setting up the OpenAI model provider
2. Configuring validation rules with proper parameters
3. Analyzing text for patterns and content constraints
4. Handling validation results and logging

Usage:
    python openai_example.py

Requirements:
    - Sifaka library
    - OpenAI API key in environment variables
    - Python dotenv for environment management
"""

import logging
import os
from typing import Dict, Any
from dotenv import load_dotenv

from sifaka.rules import LengthRule, ProhibitedContentRule, SymmetryRule, RepetitionRule
from sifaka.rules.base import RuleConfig, RulePriority
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models import OpenAIProvider
from sifaka.models.base import ModelConfig

# Configure logging with a consistent format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze text using multiple validation rules and pattern detection.

    Args:
        text (str): The text to analyze

    Returns:
        Dict[str, Any]: Analysis results for each rule type
    """
    results = {}

    # Load environment variables
    load_dotenv()

    # Initialize OpenAI provider
    openai_provider = OpenAIProvider(
        model_name="gpt-4-turbo-preview",
        config=ModelConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=1000,
        ),
    )

    # Create rules with proper configurations
    length_rule = LengthRule(
        name="length_validator",
        description="Validates text length between 100 and 500 characters",
        config={
            "min_length": 100,
            "max_length": 500,
            "unit": "characters",
            "priority": RulePriority.HIGH,
            "cache_size": 1000,
            "cost": 0.1,
        },
    )

    content_rule = ProhibitedContentRule(
        name="content_filter",
        description="Checks for prohibited or inappropriate content",
        config={
            "prohibited_terms": ["hate", "violence", "profanity"],
            "case_sensitive": False,
            "priority": RulePriority.HIGH,
            "cache_size": 1000,
            "cost": 0.1,
        },
    )

    symmetry_rule = SymmetryRule(
        name="symmetry_validator",
        description="Validates text symmetry patterns",
        config=RuleConfig(
            metadata={
                "mirror_mode": "horizontal",
                "preserve_whitespace": True,
                "preserve_case": True,
                "ignore_punctuation": False,
                "symmetry_threshold": 1.0,
            },
            priority=RulePriority.MEDIUM,
            cache_size=100,
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
            cache_size=100,
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
    text = "This is a test text that we want to validate and potentially improve. It may have some issues with length, prohibited content, symmetry, or repetition."

    # Validate text with each rule and collect violations
    violations = []
    for rule in [length_rule, content_rule, symmetry_rule, repetition_rule]:
        result = rule._validate_impl(text)
        print(f"{rule.__class__.__name__} validation result:", result)
        if not result.passed:
            violations.append(
                {"rule": rule.name, "message": result.message, "metadata": result.metadata}
            )

    # If there are violations, use the critic to improve the text
    if violations:
        improved_text = critic.improve(text, violations)
        print("\nImproved text:", improved_text)

    return results


def main():
    """Run the OpenAI example with pattern analysis."""
    # Load environment variables
    load_dotenv()

    # Initialize OpenAI provider
    openai_provider = OpenAIProvider(
        model_name="gpt-4-turbo-preview",
        config=ModelConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=1000,
        ),
    )

    # Create rules with proper configurations
    length_rule = LengthRule(
        name="length_validator",
        description="Validates text length between 100 and 500 characters",
        config={
            "min_length": 100,
            "max_length": 500,
            "unit": "characters",
            "priority": RulePriority.HIGH,
            "cache_size": 1000,
            "cost": 0.1,
        },
    )

    content_rule = ProhibitedContentRule(
        name="content_filter",
        description="Checks for prohibited or inappropriate content",
        config={
            "prohibited_terms": ["hate", "violence", "profanity"],
            "case_sensitive": False,
            "priority": RulePriority.HIGH,
            "cache_size": 1000,
            "cost": 0.1,
        },
    )

    symmetry_rule = SymmetryRule(
        name="symmetry_validator",
        description="Validates text symmetry patterns",
        config=RuleConfig(
            metadata={
                "mirror_mode": "horizontal",
                "preserve_whitespace": True,
                "preserve_case": True,
                "ignore_punctuation": False,
                "symmetry_threshold": 1.0,
            },
            priority=RulePriority.MEDIUM,
            cache_size=100,
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
            cache_size=100,
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
    text = "This is a test text that we want to validate and potentially improve. It may have some issues with length, prohibited content, symmetry, or repetition."

    # Validate text with each rule and collect violations
    violations = []
    for rule in [length_rule, content_rule, symmetry_rule, repetition_rule]:
        result = rule._validate_impl(text)
        print(f"{rule.__class__.__name__} validation result:", result)
        if not result.passed:
            violations.append(
                {"rule": rule.name, "message": result.message, "metadata": result.metadata}
            )

    # If there are violations, use the critic to improve the text
    if violations:
        improved_text = critic.improve(text, violations)
        print("\nImproved text:", improved_text)


if __name__ == "__main__":
    main()
