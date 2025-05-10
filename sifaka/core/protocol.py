"""
Protocol Implementation Module

This module provides utilities for working with protocols in Sifaka.
It defines functions for checking protocol compliance, generating protocol
implementation templates, and documenting protocol requirements.

## Usage Examples

```python
from sifaka.core.protocol import check_protocol_compliance, generate_implementation_template
from sifaka.interfaces import Chain

# Check if a class implements a protocol
class MyChain:
    def run(self, prompt, **kwargs):
        return "Result"

compliance_result = check_protocol_compliance(MyChain, Chain)
print(f"Compliant: {compliance_result.compliant}")
print(f"Missing methods: {compliance_result.missing_methods}")

# Generate an implementation template
template = generate_implementation_template(Chain)
print(template)
```
"""

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    get_type_hints,
    runtime_checkable,
)

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=Protocol)
C = TypeVar("C")


@dataclass
class ProtocolComplianceResult:
    """Result of protocol compliance check."""

    protocol: Type[Protocol]
    target: Type[Any]
    compliant: bool
    missing_methods: List[str]
    missing_properties: List[str]
    type_mismatches: Dict[str, Dict[str, Any]]


def check_protocol_compliance(
    target_class: Type[C], protocol_class: Type[T]
) -> ProtocolComplianceResult:
    """
    Check if a class implements a protocol.

    This function checks if a class implements a protocol by examining
    its methods and properties.

    Args:
        target_class: The class to check
        protocol_class: The protocol to check against

    Returns:
        A ProtocolComplianceResult with compliance information
    """
    # Get protocol methods and properties
    protocol_methods = _get_protocol_methods(protocol_class)
    protocol_properties = _get_protocol_properties(protocol_class)

    # Get target methods and properties
    target_methods = _get_class_methods(target_class)
    target_properties = _get_class_properties(target_class)

    # Check for missing methods
    missing_methods = []
    for method_name, method_info in protocol_methods.items():
        if method_name not in target_methods:
            missing_methods.append(method_name)

    # Check for missing properties
    missing_properties = []
    for prop_name in protocol_properties:
        if prop_name not in target_properties:
            missing_properties.append(prop_name)

    # Check for type mismatches
    type_mismatches = {}
    for method_name, method_info in protocol_methods.items():
        if method_name in target_methods:
            protocol_sig = method_info["signature"]
            target_sig = target_methods[method_name]["signature"]

            # Check return type
            if protocol_sig.return_annotation != inspect.Signature.empty:
                if target_sig.return_annotation == inspect.Signature.empty:
                    if method_name not in type_mismatches:
                        type_mismatches[method_name] = {}
                    type_mismatches[method_name]["return"] = {
                        "expected": protocol_sig.return_annotation,
                        "actual": "unspecified",
                    }
                elif not _is_compatible_type(
                    target_sig.return_annotation, protocol_sig.return_annotation
                ):
                    if method_name not in type_mismatches:
                        type_mismatches[method_name] = {}
                    type_mismatches[method_name]["return"] = {
                        "expected": protocol_sig.return_annotation,
                        "actual": target_sig.return_annotation,
                    }

            # Check parameter types
            for param_name, param in protocol_sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    if param_name in target_sig.parameters:
                        target_param = target_sig.parameters[param_name]
                        if target_param.annotation == inspect.Parameter.empty:
                            if method_name not in type_mismatches:
                                type_mismatches[method_name] = {}
                            if "params" not in type_mismatches[method_name]:
                                type_mismatches[method_name]["params"] = {}
                            type_mismatches[method_name]["params"][param_name] = {
                                "expected": param.annotation,
                                "actual": "unspecified",
                            }
                        elif not _is_compatible_type(target_param.annotation, param.annotation):
                            if method_name not in type_mismatches:
                                type_mismatches[method_name] = {}
                            if "params" not in type_mismatches[method_name]:
                                type_mismatches[method_name]["params"] = {}
                            type_mismatches[method_name]["params"][param_name] = {
                                "expected": param.annotation,
                                "actual": target_param.annotation,
                            }

    # Determine overall compliance
    compliant = (
        len(missing_methods) == 0 and len(missing_properties) == 0 and len(type_mismatches) == 0
    )

    return ProtocolComplianceResult(
        protocol=protocol_class,
        target=target_class,
        compliant=compliant,
        missing_methods=missing_methods,
        missing_properties=missing_properties,
        type_mismatches=type_mismatches,
    )


def generate_implementation_template(protocol_class: Type[T]) -> str:
    """
    Generate a template for implementing a protocol.

    This function generates a Python code template for implementing
    a protocol, including all required methods and properties.

    Args:
        protocol_class: The protocol to generate a template for

    Returns:
        A string containing the implementation template
    """
    # Get protocol methods and properties
    protocol_methods = _get_protocol_methods(protocol_class)
    protocol_properties = _get_protocol_properties(protocol_class)

    # Generate class definition
    template = f"class {protocol_class.__name__}Implementation:\n"
    template += f'    """Implementation of {protocol_class.__name__} protocol."""\n\n'

    # Generate property implementations
    for prop_name in protocol_properties:
        template += f"    @property\n"
        template += f"    def {prop_name}(self):\n"
        template += f'        """Get the {prop_name}."""\n'
        template += f"        # TODO: Implement {prop_name} property\n"
        template += f"        raise NotImplementedError()\n\n"

    # Generate method implementations
    for method_name, method_info in protocol_methods.items():
        signature = method_info["signature"]
        docstring = method_info["docstring"] or f"{method_name} method."

        # Generate method signature
        params = []
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                params.append("self")
            else:
                if param.annotation != inspect.Parameter.empty:
                    params.append(f"{param_name}: {_format_annotation(param.annotation)}")
                else:
                    params.append(param_name)

        # Add return type annotation if available
        return_annotation = ""
        if signature.return_annotation != inspect.Signature.empty:
            return_annotation = f" -> {_format_annotation(signature.return_annotation)}"

        # Generate method definition
        template += f"    def {method_name}({', '.join(params)}){return_annotation}:\n"

        # Generate docstring
        template += f'        """{docstring}"""\n'

        # Generate method body
        template += f"        # TODO: Implement {method_name} method\n"
        template += f"        raise NotImplementedError()\n\n"

    return template


def get_protocol_requirements(protocol_class: Type[T]) -> Dict[str, Any]:
    """
    Get the requirements for implementing a protocol.

    This function returns a dictionary containing the methods and properties
    required to implement a protocol.

    Args:
        protocol_class: The protocol to get requirements for

    Returns:
        A dictionary containing protocol requirements
    """
    # Get protocol methods and properties
    protocol_methods = _get_protocol_methods(protocol_class)
    protocol_properties = _get_protocol_properties(protocol_class)

    # Format method requirements
    method_requirements = {}
    for method_name, method_info in protocol_methods.items():
        signature = method_info["docstring"]
        docstring = method_info["docstring"] or f"{method_name} method."

        method_requirements[method_name] = {
            "signature": str(signature),
            "docstring": docstring,
            "parameters": {
                param_name: {
                    "annotation": (
                        _format_annotation(param.annotation)
                        if param.annotation != inspect.Parameter.empty
                        else None
                    ),
                    "default": param.default if param.default != inspect.Parameter.empty else None,
                }
                for param_name, param in signature.parameters.items()
                if param_name != "self"
            },
            "return_type": (
                _format_annotation(signature.return_annotation)
                if signature.return_annotation != inspect.Parameter.empty
                else None
            ),
        }

    # Format property requirements
    property_requirements = {prop_name: {} for prop_name in protocol_properties}

    return {
        "name": protocol_class.__name__,
        "module": protocol_class.__module__,
        "methods": method_requirements,
        "properties": property_requirements,
    }


def _get_protocol_methods(protocol_class: Type[Protocol]) -> Dict[str, Dict[str, Any]]:
    """
    Get the methods defined in a protocol.

    Args:
        protocol_class: The protocol to get methods from

    Returns:
        A dictionary mapping method names to method information
    """
    methods = {}

    # Get all attributes defined in the protocol
    for attr_name in dir(protocol_class):
        # Skip special methods and private attributes
        if attr_name.startswith("_") and attr_name != "__call__":
            continue

        # Get the attribute
        attr = getattr(protocol_class, attr_name)

        # Check if it's a method
        if inspect.isfunction(attr) or inspect.ismethod(attr) or inspect.ismethoddescriptor(attr):
            # Get method signature and docstring
            try:
                signature = inspect.signature(attr)
                docstring = inspect.getdoc(attr)

                methods[attr_name] = {
                    "signature": signature,
                    "docstring": docstring,
                }
            except (ValueError, TypeError):
                # Skip methods that can't be inspected
                pass

    return methods


def _get_protocol_properties(protocol_class: Type[Protocol]) -> Set[str]:
    """
    Get the properties defined in a protocol.

    Args:
        protocol_class: The protocol to get properties from

    Returns:
        A set of property names
    """
    properties = set()

    # Get all attributes defined in the protocol
    for attr_name in dir(protocol_class):
        # Skip special methods and private attributes
        if attr_name.startswith("_"):
            continue

        # Get the attribute
        attr = getattr(protocol_class, attr_name)

        # Check if it's a property
        if isinstance(attr, property):
            properties.add(attr_name)

    return properties


def _get_class_methods(cls: Type[Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get the methods defined in a class.

    Args:
        cls: The class to get methods from

    Returns:
        A dictionary mapping method names to method information
    """
    methods = {}

    # Get all attributes defined in the class
    for attr_name in dir(cls):
        # Skip private attributes (except __call__)
        if attr_name.startswith("_") and attr_name != "__call__":
            continue

        # Get the attribute
        attr = getattr(cls, attr_name)

        # Check if it's a method
        if inspect.isfunction(attr) or inspect.ismethod(attr) or inspect.ismethoddescriptor(attr):
            # Get method signature and docstring
            try:
                signature = inspect.signature(attr)
                docstring = inspect.getdoc(attr)

                methods[attr_name] = {
                    "signature": signature,
                    "docstring": docstring,
                }
            except (ValueError, TypeError):
                # Skip methods that can't be inspected
                pass

    return methods


def _get_class_properties(cls: Type[Any]) -> Set[str]:
    """
    Get the properties defined in a class.

    Args:
        cls: The class to get properties from

    Returns:
        A set of property names
    """
    properties = set()

    # Get all attributes defined in the class
    for attr_name in dir(cls):
        # Skip special methods and private attributes
        if attr_name.startswith("_"):
            continue

        # Get the attribute
        attr = getattr(cls, attr_name)

        # Check if it's a property
        if isinstance(attr, property):
            properties.add(attr_name)

    return properties


def _is_compatible_type(actual_type: Any, expected_type: Any) -> bool:
    """
    Check if an actual type is compatible with an expected type.

    Args:
        actual_type: The actual type
        expected_type: The expected type

    Returns:
        True if the types are compatible, False otherwise
    """
    # TODO: Implement more sophisticated type compatibility checking
    return actual_type == expected_type


def _format_annotation(annotation: Any) -> str:
    """
    Format a type annotation for display.

    Args:
        annotation: The type annotation

    Returns:
        A string representation of the annotation
    """
    if annotation is inspect.Signature.empty:
        return "Any"

    # Handle special cases
    if annotation is None:
        return "None"
    elif annotation is type(None):
        return "None"
    elif annotation is Any:
        return "Any"

    # Try to get the name
    try:
        return annotation.__name__
    except AttributeError:
        # Fall back to string representation
        return str(annotation)
