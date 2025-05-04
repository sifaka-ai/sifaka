"""
Example demonstrating standardized configuration approaches for models and chains in Sifaka.

This example shows how to use the standardized configuration pattern
for models and chains in Sifaka.
"""

import os
from typing import Dict, Any, List

from sifaka.models.config import ModelConfig, OpenAIConfig, AnthropicConfig
from sifaka.chain.config import ChainConfig, RetryConfig, BackoffRetryConfig, ValidationConfig
from sifaka.utils.config import (
    standardize_model_config,
    standardize_chain_config,
    standardize_retry_config,
    standardize_validation_config,
)
from sifaka.models.openai import OpenAIProvider
from sifaka.models.anthropic import AnthropicProvider
from sifaka.chain import ChainOrchestrator
from sifaka.chain.strategies import SimpleRetryStrategy, BackoffRetryStrategy
from sifaka.chain.managers import ValidationManager
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.content.prohibited import create_prohibited_content_rule


def demonstrate_model_config() -> None:
    """Demonstrate standardized model configuration."""
    print("\n=== Model Configuration Examples ===")

    # 1. Create a model config with direct parameters
    config1 = ModelConfig(
        temperature=0.7,
        max_tokens=1000,
        params={
            "system_prompt": "You are a helpful assistant.",
            "top_p": 0.9,
        },
    )
    print(f"Direct config: {config1}")

    # 2. Create a model config using standardize_model_config
    config2 = standardize_model_config(
        temperature=0.8,
        max_tokens=2000,
        params={
            "system_prompt": "You are an expert coder.",
            "top_p": 0.95,
        },
    )
    print(f"Standardized config: {config2}")

    # 3. Update an existing config
    config3 = standardize_model_config(
        config=config1, params={"system_prompt": "You are an expert writer."}, temperature=0.9
    )
    print(f"Updated config: {config3}")

    # 4. Create from dictionary
    config_dict = {
        "temperature": 0.6,
        "max_tokens": 500,
        "params": {
            "system_prompt": "You are a helpful assistant.",
            "top_p": 0.8,
        },
    }
    config4 = standardize_model_config(config=config_dict)
    print(f"Config from dict: {config4}")

    # 5. Create specialized configs
    openai_config = standardize_model_config(
        config_class=OpenAIConfig,
        temperature=0.7,
        max_tokens=1000,
        params={
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
        },
    )
    print(f"OpenAI config: {openai_config}")

    anthropic_config = standardize_model_config(
        config_class=AnthropicConfig,
        temperature=0.7,
        max_tokens=1000,
        params={
            "top_k": 50,
            "top_p": 0.9,
        },
    )
    print(f"Anthropic config: {anthropic_config}")


def demonstrate_chain_config() -> None:
    """Demonstrate standardized chain configuration."""
    print("\n=== Chain Configuration Examples ===")

    # 1. Create a chain config with direct parameters
    config1 = ChainConfig(
        max_attempts=3,
        params={
            "system_prompt": "You are a helpful assistant.",
            "use_critic": True,
        },
    )
    print(f"Direct config: {config1}")

    # 2. Create a chain config using standardize_chain_config
    config2 = standardize_chain_config(
        max_attempts=5,
        params={
            "system_prompt": "You are an expert coder.",
            "use_critic": False,
        },
    )
    print(f"Standardized config: {config2}")

    # 3. Update an existing config
    config3 = standardize_chain_config(
        config=config1, params={"system_prompt": "You are an expert writer."}, max_attempts=4
    )
    print(f"Updated config: {config3}")

    # 4. Create from dictionary
    config_dict = {
        "max_attempts": 2,
        "params": {
            "system_prompt": "You are a helpful assistant.",
            "use_critic": True,
        },
    }
    config4 = standardize_chain_config(config=config_dict)
    print(f"Config from dict: {config4}")


def demonstrate_retry_config() -> None:
    """Demonstrate standardized retry configuration."""
    print("\n=== Retry Configuration Examples ===")

    # 1. Create a retry config with direct parameters
    config1 = RetryConfig(
        max_attempts=3,
        params={
            "use_backoff": False,
        },
    )
    print(f"Direct config: {config1}")

    # 2. Create a retry config using standardize_retry_config
    config2 = standardize_retry_config(
        max_attempts=5,
        params={
            "use_backoff": True,
        },
    )
    print(f"Standardized config: {config2}")

    # 3. Create a backoff retry config
    backoff_config = standardize_retry_config(
        config_class=BackoffRetryConfig,
        max_attempts=5,
        initial_backoff=1.0,
        backoff_factor=2.0,
        max_backoff=60.0,
        params={
            "jitter": True,
        },
    )
    print(f"Backoff config: {backoff_config}")


def demonstrate_validation_config() -> None:
    """Demonstrate standardized validation configuration."""
    print("\n=== Validation Configuration Examples ===")

    # 1. Create a validation config with direct parameters
    config1 = ValidationConfig(
        prioritize_by_cost=True,
        params={
            "fail_fast": True,
        },
    )
    print(f"Direct config: {config1}")

    # 2. Create a validation config using standardize_validation_config
    config2 = standardize_validation_config(
        prioritize_by_cost=False,
        params={
            "fail_fast": False,
        },
    )
    print(f"Standardized config: {config2}")


def demonstrate_model_with_config() -> None:
    """Demonstrate using a model with standardized configuration."""
    print("\n=== Model with Configuration Examples ===")

    # Create an OpenAI configuration
    openai_config = standardize_model_config(
        config_class=OpenAIConfig,
        temperature=0.7,
        max_tokens=1000,
        api_key=os.environ.get("OPENAI_API_KEY"),
        params={
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
        },
    )

    # Create an OpenAI provider with the configuration
    try:
        openai_provider = OpenAIProvider(model_name="gpt-3.5-turbo", config=openai_config)
        print(f"Created OpenAI provider with config: {openai_provider.config}")
    except Exception as e:
        print(f"Could not create OpenAI provider: {e}")

    # Create an Anthropic configuration
    anthropic_config = standardize_model_config(
        config_class=AnthropicConfig,
        temperature=0.7,
        max_tokens=1000,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        params={
            "top_k": 50,
            "top_p": 0.9,
        },
    )

    # Create an Anthropic provider with the configuration
    try:
        anthropic_provider = AnthropicProvider(
            model_name="claude-3-sonnet-20240229", config=anthropic_config
        )
        print(f"Created Anthropic provider with config: {anthropic_provider.config}")
    except Exception as e:
        print(f"Could not create Anthropic provider: {e}")


def demonstrate_chain_with_config() -> None:
    """Demonstrate using a chain with standardized configuration."""
    print("\n=== Chain with Configuration Examples ===")

    # Create rules
    rules = [
        create_length_rule(min_words=10, max_words=100),
        create_prohibited_content_rule(prohibited_terms=["bad", "inappropriate"]),
    ]

    # Create a validation configuration
    validation_config = standardize_validation_config(
        prioritize_by_cost=True,
        params={
            "fail_fast": True,
        },
    )

    # Create a validation manager with the configuration
    validation_manager = ValidationManager(
        rules=rules, prioritize_by_cost=validation_config.prioritize_by_cost
    )
    print(f"Created validation manager with {len(validation_manager.rules)} rules")

    # Create a retry configuration
    retry_config = standardize_retry_config(
        config_class=BackoffRetryConfig,
        max_attempts=5,
        initial_backoff=1.0,
        backoff_factor=2.0,
        max_backoff=60.0,
        params={
            "jitter": True,
        },
    )

    # Create a retry strategy with the configuration
    retry_strategy = BackoffRetryStrategy(
        max_attempts=retry_config.max_attempts,
        initial_backoff=retry_config.initial_backoff,
        backoff_factor=retry_config.backoff_factor,
        max_backoff=retry_config.max_backoff,
    )
    print(f"Created retry strategy with max_attempts: {retry_strategy._max_attempts}")

    # Create a chain configuration
    chain_config = standardize_chain_config(
        max_attempts=3,
        params={
            "system_prompt": "You are a helpful assistant.",
            "use_critic": True,
        },
    )
    print(f"Created chain config: {chain_config}")


def main() -> None:
    """Run the example."""
    print("Standardized Model and Chain Configuration Examples")
    print("==================================================")

    demonstrate_model_config()
    demonstrate_chain_config()
    demonstrate_retry_config()
    demonstrate_validation_config()
    demonstrate_model_with_config()
    demonstrate_chain_with_config()


if __name__ == "__main__":
    main()
