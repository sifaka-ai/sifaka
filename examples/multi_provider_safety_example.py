#!/usr/bin/env python3
"""
Multi-Provider Safety Example for Sifaka.

This example demonstrates:
1. Using multiple LLM providers (OpenAI and Anthropic)
2. Creating safety rules for content moderation
3. Comparing results between different providers
4. Using a validation chain for content checking

Usage:
    python multi_provider_safety_example.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
    - Sifaka toxicity extras: pip install sifaka[toxicity]
    - OpenAI API key in OPENAI_API_KEY environment variable
    - Anthropic API key in ANTHROPIC_API_KEY environment variable
"""

import os
import sys
from typing import Dict, List

# Add parent directory to system path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing dotenv package. Install with: pip install python-dotenv")
    sys.exit(1)

from sifaka.chain import Chain
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models import AnthropicProvider, OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.base import Rule, RulePriority
from sifaka.rules.content import create_prohibited_content_rule
from sifaka.rules.safety import create_harmful_content_rule, create_toxicity_rule
from sifaka.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

def create_safety_rules() -> List[Rule]:
    """Create a set of safety rules for content moderation."""
    return [
        # Toxicity rule
        create_toxicity_rule(
            name="toxicity_detector",
            description="Detects toxic or offensive content",
            config={
                "threshold": 0.1,  # Even lower threshold to catch subtle toxicity
                "indicators": [
                    # Personal attacks and insults
                    "hate",
                    "offensive",
                    "vulgar",
                    "profanity",
                    "racist",
                    "sexist",
                    "threatening",
                    "violent",
                    "stupid",
                    "trash",
                    "anger",
                    "insult",
                    "idiot",
                    "dumb",
                    "fool",
                    "disgust",
                    "contempt",
                    "hostile",
                    # Additional emotional indicators
                    "furious",
                    "outraged",
                    "despise",
                    "loathe",
                    "detest",
                    # Derogatory terms
                    "worthless",
                    "useless",
                    "pathetic",
                    "incompetent",
                    # Aggressive language
                    "shut up",
                    "get lost",
                    "go away",
                ],
                "priority": RulePriority.HIGH.value,
            },
        ),
        # Harmful content rule
        create_harmful_content_rule(
            name="harmful_content_detector",
            description="Detects potentially harmful content",
            config={
                "categories": {
                    "violence": [
                        "violent",
                        "threatening",
                        "dangerous",
                        "weapon",
                        "hurt",
                        "harm",
                        "kill",
                        "attack",
                        "fight",
                        "aggressive",
                        "hate",
                        "destroy",
                        "crush",
                        "eliminate",
                        "assault",
                        "beat",
                    ],
                    "self_harm": [
                        "suicide",
                        "self-harm",
                        "self-injury",
                        "hurt myself",
                        "end it all",
                        "give up",
                        "worthless",
                        "hopeless",
                        "can't go on",
                        "no point",
                        "better off dead",
                    ],
                    "misinformation": [
                        "false",
                        "misleading",
                        "conspiracy",
                        "fake",
                        "hoax",
                        "propaganda",
                        "lie",
                        "deceptive",
                        "made up",
                        "not true",
                        "fake news",
                        "disinformation",
                        "untrue",
                    ],
                    "harassment": [
                        "harass",
                        "bully",
                        "mock",
                        "ridicule",
                        "shame",
                        "insult",
                        "taunt",
                        "humiliate",
                        "belittle",
                        "degrade",
                        "demean",
                        "make fun",
                        "laugh at",
                        "pick on",
                        "target",
                        "abuse",
                    ],
                    "discrimination": [
                        "those people",
                        "them",
                        "they all",
                        "their kind",
                        "these ones",
                        "that group",
                        "stereotypes",
                    ],
                },
                "priority": RulePriority.CRITICAL.value,
            },
        ),
        # Prohibited content rule
        create_prohibited_content_rule(
            name="prohibited_terms_detector",
            description="Checks for explicitly prohibited terms",
            config={
                "terms": [
                    # Explicit content terms
                    "profanity",
                    "obscenity",
                    "hate speech",
                    "explicit content",
                    "nsfw",
                    "terrorist",
                    # Common toxic terms
                    "trash",
                    "stupid",
                    "idiot",
                    "hate",
                    "offensive",
                    "racist",
                    "sexist",
                    # Derogatory expressions
                    "those people",
                    "their kind",
                    "that group",
                    # Aggressive phrases
                    "shut up",
                    "get lost",
                    "go away",
                ],
                "case_sensitive": False,
                "priority": RulePriority.HIGH.value,
            },
        ),
    ]

def setup_providers() -> Dict:
    """Set up LLM providers and their critics."""
    providers = {}
    critics = {}

    # Get API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_api_key:
        logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Create providers with consistent configuration
    model_args = {"temperature": 0.7, "max_tokens": 500}

    # OpenAI provider
    providers["openai"] = OpenAIProvider(
        model_name="gpt-4-turbo", config=ModelConfig(api_key=openai_api_key, **model_args)
    )

    # Anthropic provider (if available)
    if anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(
            model_name="claude-3-haiku-20240307",
            config=ModelConfig(api_key=anthropic_api_key, **model_args),
        )
    else:
        logger.warning("Anthropic API key not found. Only OpenAI will be used.")

    # Create critics for each provider
    for name, provider in providers.items():
        critics[name] = PromptCritic(
            model=provider,
            config=PromptCriticConfig(
                name=f"{name}_safety_critic",
                description=f"Improves content safety using {name}",
                system_prompt=(
                    "You are a safety moderator. Improve flagged text by making minimal "
                    "changes to address safety concerns while preserving original meaning."
                ),
                temperature=0.2,
                max_tokens=1000,
            ),
        )

    return {"providers": providers, "critics": critics}

def validate_content(text, provider_name, provider, critic, rules):
    """Validate and potentially improve content using a provider."""
    logger.info(f"\nValidating with {provider_name.upper()}:")

    # Create validation chain
    chain = Chain(model=provider, rules=rules, critic=critic, max_attempts=3)

    try:
        # Validate content
        result = chain.run(text)
        logger.info(f"Valid: {all(r.passed for r in result.rule_results)}")

        # Handle violations
        violations = [r for r in result.rule_results if not r.passed]
        if violations:
            logger.info(f"Found {len(violations)} violations:")
            for i, v in enumerate(violations, 1):
                logger.info(f"{i}. Rule: {v.message}")

            if result.output != text:
                logger.info(f"Improved text: '{result.output}'")
                if result.critique_details:
                    logger.info(f"Critique details: {result.critique_details}")
        else:
            logger.info("No violations found.")

    except ValueError as e:
        logger.error(f"Validation failed: {e}")
    except Exception as e:
        logger.error(f"Error during validation: {e}")

def main():
    """Run the multi-provider safety example."""
    # Load environment variables
    load_dotenv()
    logger.info("Starting multi-provider safety example...")

    # Set up rules and providers
    safety_rules = create_safety_rules()
    setup = setup_providers()
    providers = setup["providers"]
    critics = setup["critics"]

    # Test content samples
    content_samples = [
        {"name": "Neutral", "text": "The weather today is sunny with a high of 75 degrees."},
        {"name": "Borderline", "text": "That decision was stupid and makes me angry."},
        {"name": "Problematic", "text": "I hate those people. They're absolute trash."},
    ]

    # Process each sample with each provider
    for sample in content_samples:
        logger.info(f"\n===== Sample: {sample['name']} =====")
        logger.info(f"Text: '{sample['text']}'")

        for name, provider in providers.items():
            validate_content(
                text=sample["text"],
                provider_name=name,
                provider=provider,
                critic=critics[name],
                rules=safety_rules,
            )

    logger.info("\nMulti-provider safety example completed.")

if __name__ == "__main__":
    main()
