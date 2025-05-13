"""
Factory functions for interfaces.

This module provides factory functions for creating interface implementations.
These functions simplify the creation of interface implementations with
standardized configuration and state management.

## Usage Examples

```python
from sifaka.interfaces.factories import create_chain, create_retriever

# Create a chain
chain = create_chain(
    name="simple_chain",
    description="A simple chain implementation",
    config={"max_attempts": 3}
)

# Create a retriever
retriever = create_retriever(
    name="simple_retriever",
    description="A simple retriever implementation",
    config={"max_results": 10}
)
```

## Error Handling

- ValueError: Raised for invalid inputs
- RuntimeError: Raised for creation failures
- TypeError: Raised for type mismatches
"""
from typing import Any, Dict, Optional, Type, TypeVar, cast
from sifaka.core.base import BaseComponent
from sifaka.utils.state import StateManager
T = TypeVar('T', bound=BaseComponent)


def create_component(component_class: Type[T], name: str, description: str,
    config: Optional[Dict[str, Any]]=None, **kwargs: Any) ->Any:
    """
    Create a component with standardized configuration and state management.

    Args:
        component_class: The component class to create
        name: The component name
        description: The component description
        config: The component configuration
        **kwargs: Additional component parameters

    Returns:
        A component instance

    Raises:
        ValueError: If the parameters are invalid
        RuntimeError: If component creation fails
    """
    component = component_class(name=name, description=description, config=
        config or {}, **kwargs)
    component.initialize() if component else ""
    return component


def create_chain(name: str, description: str, config: Optional[Dict[str,
    Any]]=None, **kwargs: Any) ->Any:
    """
    Create a chain with standardized configuration and state management.

    Args:
        name: The chain name
        description: The chain description
        config: The chain configuration
        **kwargs: Additional chain parameters

    Returns:
        A chain instance

    Raises:
        ValueError: If the parameters are invalid
        RuntimeError: If chain creation fails
    """
    from sifaka.chain import Chain as ChainCore
    return create_component(component_class=ChainCore, name=name,
        description=description, config=config, **kwargs)


def create_retriever(name: str, description: str, config: Optional[Dict[str,
    Any]]=None, **kwargs: Any) ->Any:
    """
    Create a retriever with standardized configuration and state management.

    Args:
        name: The retriever name
        description: The retriever description
        config: The retriever configuration
        **kwargs: Additional retriever parameters

    Returns:
        A retriever instance

    Raises:
        ValueError: If the parameters are invalid
        RuntimeError: If retriever creation fails
    """
    from sifaka.retrieval.core import RetrieverCore
    return create_component(component_class=RetrieverCore, name=name,
        description=description, config=config, **kwargs)
