#!/usr/bin/env python3
"""
Comprehensive Content Validation Example for Sifaka.

This example demonstrates:
1. Generating text with Anthropic Claude
2. Validating content using multiple classifiers:
   - Genre classification (news vs fiction)
   - Topic classification
   - Toxicity detection
   - Bias detection
3. Using a critic to fix content issues

Usage:
    python comprehensive_content_validation.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
    - anthropic package: pip install anthropic
    - Anthropic API key in ANTHROPIC_API_KEY environment variable
    - OpenAI API key in OPENAI_API_KEY environment variable (for critic)
"""

import os
import sys
from typing import Any, Dict

# Add parent directory to system path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import anthropic
except ImportError:
    print("The anthropic package is required. Install with: pip install anthropic")
    sys.exit(1)

from sifaka.classifiers.base import ClassifierConfig
from sifaka.classifiers.bias import BiasConfig, BiasDetector
from sifaka.classifiers.toxicity import (
    ToxicityClassifier,
    ToxicityConfig,
    ToxicityThresholds,
)
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.safety import (
    create_bias_rule,
    create_harmful_content_rule,
)
from sifaka.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Sample data for classifier training
SAMPLES = {
    "genre": {
        "news": [
            "Breaking news: A major earthquake struck the coast yesterday, causing significant damage.",
            "The president announced a new economic plan aimed at reducing inflation and creating jobs.",
            "Scientists have discovered a new species of deep-sea marine life according to a recent report.",
            "Tech giant unveils revolutionary quantum computer with unprecedented processing power.",
            "Global climate summit concludes with new agreements on emission reductions.",
            "Stock markets reach record highs following positive economic indicators.",
        ],
        "fiction": [
            "The moon cast an eerie glow on the abandoned mansion as Sarah approached the door.",
            "The dragon unfurled its massive wings and roared, sending a tremor through the forest.",
            "Time seemed to slow as he leaned in for the kiss, the years of longing fading away.",
            "In the depths of space, the ancient starship drifted silently between the stars.",
            "The wizard's crystal ball pulsed with an otherworldly blue light.",
            "As the clock struck midnight, the old toys slowly came to life.",
        ],
    },
    "bias": {
        "gender": [
            "Women are naturally better at nurturing roles while men excel at leadership positions.",
            "All men are aggressive and competitive by nature.",
            "Women can't handle technical jobs as well as men can.",
        ],
        "neutral": [
            "The study included participants from diverse backgrounds and controlled for demographic factors.",
            "The company's hiring policy states that candidates are evaluated based on qualifications.",
            "Both approaches have advantages and disadvantages that should be considered.",
        ],
    },
}


def setup_classifiers():
    """Set up classifiers for content validation."""
    classifiers = {}

    # Configure toxicity detection
    toxicity_config = ToxicityConfig(
        model_name="original",
        thresholds=ToxicityThresholds(severe_toxic=0.7, threat=0.7, general=0.3),
    )

    # Configure bias detection
    bias_config = BiasConfig(
        min_confidence=0.25,
        max_features=3000,
        bias_types=[
            "gender",
            "racial",
            "age",
            "cultural",
            "socioeconomic",
            "religious",
            "political",
            "educational",
            "geographical",
            "neutral",
        ],
    )

    # Configure harmful content detection
    harmful_content_config = {
        "categories": {
            "violence": [
                "violence",
                "assault",
                "kill",
                "attack",
                "fight",
                "hurt",
                "hate",
                "trash",
                "destroy",
                "harm",
                "abuse",
                "cruel",
            ],
            "self_harm": [
                "suicide",
                "self-harm",
                "cutting",
                "depression",
                "kill yourself",
                "harm yourself",
                "end your life",
                "self-destructive",
                "self-injury",
            ],
            "hate_speech": [
                "hate",
                "bigot",
                "racist",
                "discrimination",
                "sexist",
                "misogynist",
                "xenophobic",
                "prejudice",
                "stereotype",
                "slur",
                "derogatory",
            ],
            "misinformation": [
                "fake news",
                "conspiracy",
                "hoax",
                "propaganda",
                "flat earth",
                "anti-vax",
                "autism",
                "false",
                "misleading",
                "disinformation",
                "pseudoscience",
            ],
        }
    }

    # Initialize classifiers
    classifiers["toxicity"] = ToxicityClassifier(
        name="toxicity_classifier",
        description="Detects toxic content using Detoxify",
        toxicity_config=toxicity_config,
        min_confidence=0.3,
    )

    # Initialize bias detector with minimal configuration
    classifiers["bias"] = BiasDetector(
        name="bias_detector",
        description="Detects various forms of bias in text",
        config=ClassifierConfig(
            labels=bias_config.bias_types, cost=2.5, min_confidence=bias_config.min_confidence
        ),
    )

    # Set up safety rules
    safety_rules = []

    # Add harmful content rule
    harmful_content_rule = create_harmful_content_rule(config=harmful_content_config)
    safety_rules.append(harmful_content_rule)

    # Add bias rule
    bias_rule_config = {
        "threshold": 0.3,
        "categories": {
            "gender": [
                "sexist",
                "misogynistic",
                "patriarchal",
                "gender bias",
                "gender stereotype",
            ],
            "racial": [
                "racist",
                "racial bias",
                "ethnic stereotype",
                "racial prejudice",
                "discriminatory",
            ],
            "age": [
                "ageist",
                "age discrimination",
                "age bias",
                "age stereotype",
                "generational bias",
            ],
            "cultural": [
                "cultural bias",
                "xenophobic",
                "ethnocentric",
                "cultural stereotype",
                "cultural prejudice",
            ],
        },
        "cache_size": 100,
        "priority": 1,
        "cost": 1.0,
    }
    bias_rule = create_bias_rule(config=bias_rule_config)
    safety_rules.append(bias_rule)

    # Add rules to classifiers
    classifiers["safety_rules"] = safety_rules

    return classifiers


def setup_critic(api_key):
    """Set up content critic using OpenAI."""
    if not api_key:
        return None

    openai_provider = OpenAIProvider(
        model_name="gpt-4-turbo",
        config=ModelConfig(
            api_key=api_key,
            temperature=0.7,
            max_tokens=1000,
        ),
    )

    return PromptCritic(
        model=openai_provider,
        config=PromptCriticConfig(
            name="content_critic",
            description="Improves content based on validation results",
            system_prompt="You are an editor that improves text based on specific validation criteria.",
            temperature=0.7,
            max_tokens=1000,
        ),
    )


def validate_content(text: str, classifiers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates the given text content using multiple classifiers and safety rules.

    Args:
        text: The text content to validate
        classifiers: Dictionary containing various classifiers and rules

    Returns:
        Dictionary containing validation results and explanations
    """
    validation_results = {"is_valid": True, "issues": [], "explanations": []}

    # Apply safety rules first
    for rule in classifiers.get("safety_rules", []):
        try:
            rule_result = rule.validate(text)
            if not rule_result.passed:
                validation_results["is_valid"] = False
                validation_results["issues"].append(
                    {"type": rule.name, "severity": "high", "details": rule_result.message}
                )
                validation_results["explanations"].append(
                    f"Safety rule '{rule.name}' violation: {rule_result.message}"
                )
        except Exception as e:
            validation_results["issues"].append(
                {
                    "type": "error",
                    "severity": "high",
                    "details": f"Error applying safety rule {rule.name}: {str(e)}",
                }
            )

    # Apply toxicity classifier
    toxicity_classifier = classifiers.get("toxicity")
    if toxicity_classifier:
        try:
            toxicity_result = toxicity_classifier.classify(text)
            if toxicity_result.label in ["toxic", "severe_toxic", "threat"]:
                validation_results["is_valid"] = False
                validation_results["issues"].append(
                    {
                        "type": "toxicity",
                        "severity": "high",
                        "score": toxicity_result.confidence,
                        "details": f"Detected {toxicity_result.label} content",
                    }
                )
                validation_results["explanations"].append(
                    f"Toxicity detected: {toxicity_result.label} content with {toxicity_result.confidence:.2f} confidence"
                )
        except Exception as e:
            validation_results["issues"].append(
                {
                    "type": "error",
                    "severity": "high",
                    "details": f"Error in toxicity classification: {str(e)}",
                }
            )

    # Apply bias classifier
    bias_classifier = classifiers.get("bias")
    if bias_classifier:
        try:
            bias_result = bias_classifier.classify(text)
            if bias_result.label != "neutral":
                validation_results["is_valid"] = False
                validation_results["issues"].append(
                    {
                        "type": "bias",
                        "severity": "high",
                        "bias_type": bias_result.label,
                        "confidence": bias_result.confidence,
                    }
                )
                validation_results["explanations"].append(
                    f"Bias detected: {bias_result.label} bias with {bias_result.confidence:.2f} confidence"
                )
        except Exception as e:
            validation_results["issues"].append(
                {
                    "type": "error",
                    "severity": "high",
                    "details": f"Error in bias classification: {str(e)}",
                }
            )

    return validation_results


def validate_improvements(improved_content, classifiers, expected_genre):
    """Validate if the improved content fixed the issues."""
    improved_genre = classifiers["genre"].classify(improved_content).label
    improved_toxicity = classifiers["toxicity"].classify(improved_content).label
    improved_bias = classifiers["bias"].classify(improved_content).label

    logger.info(
        f"After improvement - Genre: {improved_genre}, Toxicity: {improved_toxicity}, Bias: {improved_bias}"
    )

    # Check if genre was fixed
    if expected_genre and improved_genre == expected_genre:
        logger.info(f"✓ Genre issue fixed: now {improved_genre}")

    # Check if toxicity was fixed
    if improved_toxicity == "non_toxic":
        logger.info("✓ Toxicity issue fixed")

    # Check if bias was fixed
    if improved_bias == "neutral":
        logger.info("✓ Bias issue fixed")


def test_validation(classifiers):
    """Test the enhanced content validation with various examples."""
    test_cases = [
        "I hate those people. They're absolute trash.",
        "This is a normal, friendly message about cooperation.",
        "You should harm yourself, it's the only way.",
        "Women are not good at technical jobs.",
        "The earth is flat and vaccines cause autism.",
    ]

    logger.info("\nTesting enhanced content validation:")
    for text in test_cases:
        logger.info(f"\nTesting content: {text}")
        result = validate_content(text, classifiers)

        if not result["is_valid"]:
            logger.info("Content flagged as invalid:")
            for explanation in result["explanations"]:
                logger.info(f"- {explanation}")
        else:
            logger.info("Content passed all validation checks")


def main():
    """Run the comprehensive content validation example."""
    # Check for API keys
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not anthropic_api_key:
        logger.error("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    if not openai_api_key:
        logger.warning("OpenAI API key not found. Content improvement will be disabled.")

    logger.info("Starting comprehensive content validation example...")

    # Initialize classifiers and critic
    classifiers = setup_classifiers()
    setup_critic(openai_api_key)

    # Generate content with Claude
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    prompts = [
        {
            "text": "Write a news article about a recent technological advancement",
            "expected_genre": "news",
        },
        {"text": "Write a short story about a magical forest", "expected_genre": "fiction"},
    ]

    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\n===== Content {i} =====")
        logger.info(f"Prompt: '{prompt['text']}'")

        # Generate content
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            system="You are a helpful assistant that writes content on various topics.",
            messages=[{"role": "user", "content": prompt["text"]}],
        )

        content = response.content[0].text
        logger.info(f"Generated content (preview): '{content[:100]}...'")

        # Validate content
        validate_content(content, classifiers)

    # Run the test validation
    test_validation(classifiers)

    logger.info("\nComprehensive content validation example completed.")


if __name__ == "__main__":
    main()
