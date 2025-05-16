"""
Component Registry Module

This module provides a centralized registry for component factory functions.
It allows for dynamic registration and retrieval of factory functions without
direct imports, helping to break circular dependencies.

## Overview
The registry is a solution to circular dependency issues in the factory system.
Instead of directly importing factory functions from implementation modules,
this registry stores references to factory functions that can be retrieved at runtime.

## Usage
- Implementation modules register their factory functions with the registry
- Factory modules retrieve factory functions from the registry
- No direct imports between factory modules and implementation modules

## Example
```python
# In implementation module
from sifaka.core.registry import register_factory

def create_my_component(**kwargs):
    # Implementation
    pass

register_factory("my_component", create_my_component)

# In factory module
from sifaka.core.registry import get_factory

def create_component(component_type, **kwargs):
    factory = get_factory(component_type)
    if factory:
        return factory(**kwargs)
    else:
        raise ValueError(f"Unknown component type: {component_type}")
```
"""

from typing import Any, Callable, Dict, Optional, TypeVar, cast

# Type for factory functions
T = TypeVar("T")
FactoryFunction = Callable[..., Any]

# Registry storage
_factory_registry: Dict[str, Dict[str, FactoryFunction]] = {
    "critic": {},
    "rule": {},
    "classifier": {},
    "retriever": {},
    "adapter": {},
    "model_provider": {},
    "chain": {},
}


def register_factory(
    component_type: str, factory_name: str, factory_function: FactoryFunction
) -> None:
    """
    Register a factory function in the registry.

    Args:
        component_type: The type of component ("critic", "rule", etc.)
        factory_name: The name of the factory function (e.g., "prompt", "length")
        factory_function: The factory function to register

    Raises:
        ValueError: If the component type is not supported
    """
    if component_type not in _factory_registry:
        raise ValueError(
            f"Unsupported component type: {component_type}. "
            f"Supported types: {list(_factory_registry.keys())}"
        )

    _factory_registry[component_type][factory_name] = factory_function


def get_factory(component_type: str, factory_name: str) -> Optional[FactoryFunction]:
    """
    Get a factory function from the registry.

    Args:
        component_type: The type of component ("critic", "rule", etc.)
        factory_name: The name of the factory ("prompt", "length", etc.)

    Returns:
        The factory function or None if not found
    """
    if component_type not in _factory_registry:
        return None

    return _factory_registry[component_type].get(factory_name)


def get_available_factories(component_type: str) -> Dict[str, FactoryFunction]:
    """
    Get all available factory functions for a component type.

    Args:
        component_type: The type of component ("critic", "rule", etc.)

    Returns:
        A dictionary of factory functions by name
    """
    if component_type not in _factory_registry:
        return {}

    return dict(_factory_registry[component_type])


def register_critic_factory(name: str, factory_function: FactoryFunction) -> None:
    """
    Register a critic factory function.

    Args:
        name: The name of the critic type (e.g., "prompt", "reflexion")
        factory_function: The factory function to register
    """
    register_factory("critic", name, factory_function)


def register_rule_factory(name: str, factory_function: FactoryFunction) -> None:
    """
    Register a rule factory function.

    Args:
        name: The name of the rule type (e.g., "length", "toxicity")
        factory_function: The factory function to register
    """
    register_factory("rule", name, factory_function)


def register_classifier_factory(name: str, factory_function: FactoryFunction) -> None:
    """
    Register a classifier factory function.

    Args:
        name: The name of the classifier type (e.g., "toxicity", "sentiment")
        factory_function: The factory function to register
    """
    register_factory("classifier", name, factory_function)


def register_model_provider_factory(name: str, factory_function: FactoryFunction) -> None:
    """
    Register a model provider factory function.

    Args:
        name: The name of the provider type (e.g., "openai", "anthropic")
        factory_function: The factory function to register
    """
    register_factory("model_provider", name, factory_function)


def get_critic_factory(name: str) -> Optional[FactoryFunction]:
    """
    Get a critic factory function.

    Args:
        name: The name of the critic type (e.g., "prompt", "reflexion")

    Returns:
        The factory function or None if not found
    """
    return get_factory("critic", name)


def get_rule_factory(name: str) -> Optional[FactoryFunction]:
    """
    Get a rule factory function.

    Args:
        name: The name of the rule type (e.g., "length", "toxicity")

    Returns:
        The factory function or None if not found
    """
    return get_factory("rule", name)


def get_classifier_factory(name: str) -> Optional[FactoryFunction]:
    """
    Get a classifier factory function.

    Args:
        name: The name of the classifier type (e.g., "toxicity", "sentiment")

    Returns:
        The factory function or None if not found
    """
    return get_factory("classifier", name)


def get_model_provider_factory(name: str) -> Optional[FactoryFunction]:
    """
    Get a model provider factory function.

    Args:
        name: The name of the provider type (e.g., "openai", "anthropic")

    Returns:
        The factory function or None if not found
    """
    return get_factory("model_provider", name)
