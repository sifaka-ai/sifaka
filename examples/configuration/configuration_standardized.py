"""
Example demonstrating standardized configuration approaches in Sifaka.

This example shows how to use the standardized configuration pattern
across different components in Sifaka.
"""

from typing import Dict, Any, List

from sifaka.rules.base import RuleConfig, Rule, RulePriority
from sifaka.classifiers.base import ClassifierConfig
from sifaka.critics.models import CriticConfig, PromptCriticConfig
from sifaka.utils.config import (
    standardize_rule_config,
    standardize_critic_config,
)
from sifaka.utils.config import standardize_classifier_config
from sifaka.adapters.rules import create_classifier_rule
from sifaka.classifiers.toxicity import create_toxicity_classifier
from sifaka.critics.factories import create_prompt_critic
from sifaka.models.mock import MockProvider


def demonstrate_rule_config() -> None:
    """Demonstrate standardized rule configuration."""
    print("\n=== Rule Configuration Examples ===")

    # 1. Create a rule config with direct parameters
    config1 = RuleConfig(
        priority=RulePriority.HIGH,
        cost=5,
        params={
            "min_length": 10,
            "max_length": 100,
        },
    )
    print(f"Direct config: {config1}")

    # 2. Create a rule config using standardize_rule_config
    config2 = standardize_rule_config(
        priority=RulePriority.MEDIUM,
        params={
            "min_length": 20,
            "max_length": 200,
        },
    )
    print(f"Standardized config: {config2}")

    # 3. Update an existing config
    config3 = standardize_rule_config(config=config1, params={"min_length": 30}, cost=10)
    print(f"Updated config: {config3}")

    # 4. Create from dictionary
    config_dict = {
        "priority": RulePriority.LOW,
        "params": {
            "min_length": 5,
            "max_length": 50,
        },
    }
    config4 = standardize_rule_config(config=config_dict)
    print(f"Config from dict: {config4}")


def demonstrate_classifier_config() -> None:
    """Demonstrate standardized classifier configuration."""
    print("\n=== Classifier Configuration Examples ===")

    # 1. Create a classifier config with direct parameters
    config1 = ClassifierConfig(
        labels=["positive", "negative", "neutral"],
        min_confidence=0.7,
        params={
            "model_name": "sentiment-large",
            "threshold": 0.7,
        },
    )
    print(f"Direct config: {config1}")

    # 2. Create a classifier config using standardize_classifier_config
    config2 = standardize_classifier_config(
        labels=["spam", "ham"],
        min_confidence=0.8,
        params={
            "max_features": 1000,
            "use_bigrams": True,
        },
    )
    print(f"Standardized config: {config2}")

    # 3. Update an existing config
    config3 = standardize_classifier_config(
        config=config1, params={"threshold": 0.8}, min_confidence=0.9
    )
    print(f"Updated config: {config3}")

    # 4. Create from dictionary
    config_dict = {
        "labels": ["toxic", "non-toxic"],
        "min_confidence": 0.6,
        "params": {
            "model_name": "toxicity-v2",
            "threshold": 0.6,
        },
    }
    config4 = standardize_classifier_config(config=config_dict)
    print(f"Config from dict: {config4}")


def demonstrate_critic_config() -> None:
    """Demonstrate standardized critic configuration."""
    print("\n=== Critic Configuration Examples ===")

    # 1. Create a critic config with direct parameters
    config1 = CriticConfig(
        name="basic_critic",
        description="A basic critic",
        min_confidence=0.7,
        max_attempts=3,
        params={
            "system_prompt": "You are an expert editor.",
        },
    )
    print(f"Direct config: {config1}")

    # 2. Create a critic config using standardize_critic_config
    config2 = standardize_critic_config(
        name="advanced_critic",
        description="An advanced critic",
        min_confidence=0.8,
        max_attempts=5,
        params={
            "system_prompt": "You are an expert technical editor.",
            "temperature": 0.7,
        },
    )
    print(f"Standardized config: {config2}")

    # 3. Create a specialized critic config
    prompt_config = standardize_critic_config(
        config_class=PromptCriticConfig,
        name="prompt_critic",
        description="A prompt-based critic",
        system_prompt="You are an expert editor.",
        temperature=0.7,
        max_tokens=1000,
    )
    print(f"Specialized config: {prompt_config}")

    # 4. Update an existing config
    config3 = standardize_critic_config(
        config=config1,
        params={"system_prompt": "You are an expert technical writer."},
        min_confidence=0.9,
    )
    print(f"Updated config: {config3}")


def demonstrate_factory_functions() -> None:
    """Demonstrate factory functions with standardized configuration."""
    print("\n=== Factory Function Examples ===")

    # 1. Create a classifier with standardized configuration
    toxicity_classifier = create_toxicity_classifier(
        name="toxicity_classifier",
        description="Detects toxic content",
        general_threshold=0.5,
        severe_toxic_threshold=0.7,
        threat_threshold=0.7,
        cache_size=100,
        min_confidence=0.6,
        cost=5,
    )
    print(f"Toxicity classifier: {toxicity_classifier.name}")
    print(f"Classifier config: {toxicity_classifier.config}")

    # 2. Create a rule from the classifier with standardized configuration
    rule_config = RuleConfig(
        priority="HIGH",
        cost=5,
        params={
            "threshold": 0.8,
            "valid_labels": ["non-toxic"],
        },
    )
    toxicity_rule = create_classifier_rule(
        classifier=toxicity_classifier,
        config=rule_config,
        name="toxicity_rule",
        description="Ensures text is not toxic",
    )
    print(f"Toxicity rule: {toxicity_rule.name}")
    print(f"Rule config: {toxicity_rule.config}")

    # 3. Create a critic with standardized configuration
    model = MockProvider()
    critic_config = PromptCriticConfig(
        name="content_critic",
        description="Improves content quality",
        min_confidence=0.7,
        max_attempts=3,
        system_prompt="You are an expert editor.",
        temperature=0.7,
        max_tokens=1000,
    )
    critic = create_prompt_critic(llm_provider=model, config=critic_config)
    print(f"Critic: {critic.config.name}")
    print(f"Critic config: {critic.config}")


def main() -> None:
    """Run the example."""
    print("Standardized Configuration Examples")
    print("==================================")

    demonstrate_rule_config()
    demonstrate_classifier_config()
    demonstrate_critic_config()
    demonstrate_factory_functions()


if __name__ == "__main__":
    main()
