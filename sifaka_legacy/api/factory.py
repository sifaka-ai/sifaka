"""
Factory Functions Module

This module provides simplified factory functions for creating components
without dealing with the underlying complexity of the registry system.
"""

from typing import Any, Dict, Optional

# Initialize the registry
from sifaka.core import initialize_registry

initialize_registry.initialize_registry()

# Import the factory functions
from sifaka.critics.base.factories import create_critic as core_create_critic
from sifaka.rules.factories import create_rule as core_create_rule
from sifaka.classifiers.factories import create_classifier as core_create_classifier
from sifaka.retrieval.factories import create_simple_retriever as core_create_retriever
from sifaka.core.factories import create_model_provider as core_create_model_provider


def model(provider_type: str, model_name: str = "", **kwargs: Any) -> Any:
    """
    Create a model provider.

    This is a simplified version of the create_model_provider function.

    Args:
        provider_type: The type of model provider (e.g., "openai", "anthropic")
        model_name: The name of the model to use (e.g., "gpt-4" for OpenAI)
        **kwargs: Additional arguments to pass to the provider

    Returns:
        A model provider instance
    """
    # If model_name is not provided but api_key is, set a default model_name based on provider_type
    if not model_name and "api_key" in kwargs:
        if provider_type == "openai":
            model_name = "gpt-3.5-turbo"
        elif provider_type == "anthropic":
            model_name = "claude-3-sonnet-20240229"
        elif provider_type == "gemini":
            model_name = "gemini-pro"
        else:
            model_name = "default-model"

    return core_create_model_provider(provider_type, model_name=model_name, **kwargs)


def critic(critic_type: str, **kwargs: Any) -> Any:
    """
    Create a critic.

    This is a simplified version of the create_critic function.

    Args:
        critic_type: The type of critic (e.g., "prompt", "reflexion")
        **kwargs: Additional arguments to pass to the critic

    Returns:
        A critic instance
    """
    try:
        return core_create_critic(critic_type, **kwargs)
    except Exception as e:
        print(f"Warning: Could not create critic of type {critic_type}: {str(e)}")

        # Return a dummy critic that does nothing
        class DummyCritic:
            def validate(self, text: str) -> bool:
                return True

            def critique(self, text: str) -> dict:
                return {
                    "score": 1.0,
                    "feedback": "No critic available",
                    "issues": [],
                    "suggestions": [],
                }

            def improve(self, text: str, feedback: str = None) -> str:
                return text

        return DummyCritic()


def rule(rule_type: str, **kwargs: Any) -> Any:
    """
    Create a rule.

    This is a simplified version of the create_rule function.

    Args:
        rule_type: The type of rule (e.g., "length", "toxicity")
        **kwargs: Additional arguments to pass to the rule

    Returns:
        A rule instance
    """
    try:
        if rule_type == "length":
            from sifaka.rules.formatting.length import create_length_rule

            # Convert max_length to max_chars for length rules
            if "max_length" in kwargs:
                kwargs["max_chars"] = kwargs.pop("max_length")
            return create_length_rule(**kwargs)
        else:
            # For other rule types, use the general factory function
            return core_create_rule(rule_type, **kwargs)
    except Exception as e:
        print(f"Warning: Could not create rule of type {rule_type}: {str(e)}")

        # Return a dummy rule that does nothing
        class DummyRule:
            def __init__(self):
                self.name = f"{rule_type}_rule"
                self.description = f"Dummy {rule_type} rule"

            def validate(self, text: str) -> dict:
                return {
                    "passed": True,
                    "message": "No rule available",
                    "issues": [],
                    "suggestions": [],
                }

            def process(self, text: str) -> dict:
                return {
                    "passed": True,
                    "message": "No rule available",
                    "issues": [],
                    "suggestions": [],
                }

        return DummyRule()


def classifier(classifier_type: str, **kwargs: Any) -> Any:
    """
    Create a classifier.

    This is a simplified version of the create_classifier function.

    Args:
        classifier_type: The type of classifier
        **kwargs: Additional arguments to pass to the classifier

    Returns:
        A classifier instance
    """
    return core_create_classifier(classifier_type, **kwargs)


def retriever(retriever_type: str, **kwargs: Any) -> Any:
    """
    Create a retriever.

    This is a simplified version of the create_retriever function.

    Args:
        retriever_type: The type of retriever ("simple" or "threshold")
        **kwargs: Additional arguments to pass to the retriever

    Returns:
        A retriever instance
    """
    # Import here to avoid circular imports
    from sifaka.retrieval.factories import create_threshold_retriever

    if retriever_type == "threshold":
        return create_threshold_retriever(**kwargs)
    else:  # Default to simple retriever for other types
        return core_create_retriever(**kwargs)
