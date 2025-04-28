#!/usr/bin/env python3
"""
OpenAI Integration Example for Sifaka.

This example demonstrates:
1. Setting up an OpenAI model provider
2. Configuring validation rules
3. Using pattern rules to analyze and improve text

Usage:
    python openai_example.py

Requirements:
    - Python environment with Sifaka installed
    - OpenAI API key in OPENAI_API_KEY environment variable
"""

from sifaka.models import OpenAIProvider
from sifaka.rules import ProhibitedContentRule
from sifaka.rules.pattern_rules import SymmetryRule, RepetitionRule
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.rules.base import RuleConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI provider
openai_model = OpenAIProvider(model_name="gpt-4-turbo-preview")

# Create rules
prohibited_terms = ProhibitedContentRule(
    name="content_filter",
    description="Checks for prohibited or inappropriate content",
    config=RuleConfig(
        params={
            "prohibited_terms": ["hate", "violence", "profanity"],
            "case_sensitive": False,
        }
    ),
)

symmetry_rule = SymmetryRule(
    name="symmetry_checker",
    description="Checks for symmetrical patterns in text",
    config=RuleConfig(),
)

repetition_rule = RepetitionRule(
    name="repetition_checker",
    description="Checks for repetitive patterns in text",
    config=RuleConfig(),
)

# Create a critic for improving outputs that fail validation
critic_config = PromptCriticConfig(
    name="openai_critic",
    description="A critic that uses OpenAI to improve text",
    system_prompt="You are an expert editor that improves text.",
    temperature=0.7,
    max_tokens=1000,
)

critic = PromptCritic(config=critic_config, model=openai_model)

# Example text to validate and improve
text = "Write a professional email about a project update that avoids any inappropriate content."

# Apply rules and get results
rules = [prohibited_terms, symmetry_rule, repetition_rule]
results = []
violations = []

for rule in rules:
    result = rule.validate(text)
    results.append(result)
    print(f"\nRule: {rule.name}")
    print(f"Passed: {result.passed}")
    print(f"Message: {result.message}")

    if not result.passed:
        violations.append({"rule": rule.name, "message": result.message})

# If any rule failed, use the critic to improve
if violations:
    print("\nImproving text with critic...")
    improved_text = critic.improve(text, violations)
    print(f"\nOriginal text: {text}")
    print(f"Improved text: {improved_text}")
else:
    print("\nAll rules passed!")
