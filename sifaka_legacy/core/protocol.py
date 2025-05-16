from typing import Any
"""
Protocol Implementation Module

This module provides utilities for working with protocols in Sifaka, enabling runtime protocol compliance checking and implementation assistance.

## Overview
The protocol module provides tools for working with Python's Protocol type hints:
- Checking if a class implements a protocol
- Generating implementation templates for protocols
- Documenting protocol requirements
- Analyzing protocol methods and properties

## Components
- **ProtocolComplianceResult**: Result of protocol compliance checks
- **check_protocol_compliance**: Function to check if a class implements a protocol
- **generate_implementation_template**: Function to generate protocol implementation templates
- **get_protocol_requirements**: Function to get protocol implementation requirements

## Usage Examples
```python
from sifaka.core.protocol import check_protocol_compliance, generate_implementation_template
from sifaka.interfaces import Chain

# Check if a class implements a protocol
class MyChain:
    def run(self, prompt: Any, **kwargs: Any) -> None:
        return "Result"

compliance_result = check_protocol_compliance(MyChain, Chain)
print(f"Compliant: {compliance_result.compliant}")
print(f"Missing methods: {compliance_result.missing_methods}")

# Generate an implementation template
template = generate_implementation_template(Chain)
print(template)

# Get protocol requirements
requirements = get_protocol_requirements(Chain)
print(f"Required methods: {list(requirements['methods'].keys()))")
```

## Error Handling
The module handles various error conditions:
- Gracefully handles methods that can't be inspected
- Provides detailed information about type mismatches
- Logs warnings for potential issues
"""

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Set,
    Type,
    TypeVar,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T")
C = TypeVar("C")


@dataclass
class ProtocolComplianceResult:
    """
    Result of protocol compliance check.

    This class contains the results of checking if a class implements a protocol,
    including whether the class is compliant, which methods and properties are
    missing, and any type mismatches.

    ## Architecture
    The class uses Python's dataclass for simple data storage with type hints.
    It provides a structured way to report protocol compliance issues.

    Attributes:
        protocol (Type[Protocol]): The protocol that was checked
        target (Type[Any]): The target class that was checked
        compliant (bool): Whether the target class implements the protocol
        missing_methods (List[str]): Methods required by the protocol but missing from the target
        missing_properties (List[str]): Properties required by the protocol but missing from the target
        type_mismatches (Dict[str, Dict[str, Any]]): Type mismatches between protocol and target
    """

    protocol: Type[Any]
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

    This function performs a comprehensive check to determine if a class
    implements a protocol by examining its methods and properties. It checks
    for missing methods, missing properties, and type mismatches between
    the protocol and the target class.

    The function performs the following checks:
    1. Identifies methods required by the protocol but missing from the target class
    2. Identifies properties required by the protocol but missing from the target class
    3. Checks for type mismatches in method parameters and return types
    4. Determines overall compliance based on all checks

    Args:
        target_class (Type[C]): The class to check for protocol compliance
        protocol_class (Type[T]): The protocol to check against

    Returns:
        ProtocolComplianceResult: A result object containing compliance information,
            including whether the class is compliant, missing methods and properties,
            and any type mismatches

    Example:
        ```python
        from sifaka.interfaces import Chain

        class MyChain:
            def run(self, prompt: Any, **kwargs: Any) -> None:
                return "Result"

        result = check_protocol_compliance(MyChain, Chain)
        if result.compliant:
            print("MyChain implements the Chain protocol")
        else:
            print(f"Missing methods: {result.missing_methods}")
            print(f"Type mismatches: {result.type_mismatches}")
        ```
    """
    protocol_methods = _get_protocol_methods(protocol_class)
    protocol_properties = _get_protocol_properties(protocol_class)
    target_methods = _get_class_methods(target_class)
    target_properties = _get_class_properties(target_class)
    missing_methods = []
    if protocol_methods:
        for method_name, method_info in protocol_methods.items():
            if method_name not in target_methods:
                missing_methods.append(method_name)
    missing_properties = []
    for prop_name in protocol_properties:
        if prop_name not in target_properties:
            missing_properties.append(prop_name)
    type_mismatches: Dict[str, Dict[str, Any]] = {}
    if protocol_methods:
        for method_name, method_info in protocol_methods.items():
            if method_name in target_methods:
                protocol_sig = method_info["signature"]
                target_sig = target_methods[method_name]["signature"]
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
                if hasattr(protocol_sig, "parameters"):
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
                                elif not _is_compatible_type(
                                    target_param.annotation, param.annotation
                                ):
                                    if method_name not in type_mismatches:
                                        type_mismatches[method_name] = {}
                                    if "params" not in type_mismatches[method_name]:
                                        type_mismatches[method_name]["params"] = {}
                                    type_mismatches[method_name]["params"][param_name] = {
                                        "expected": param.annotation,
                                        "actual": target_param.annotation,
                                    }
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

    This function generates a Python code template for implementing a protocol,
    including all required methods and properties. The template includes:
    - A class definition with the protocol name
    - Property implementations with appropriate getters
    - Method implementations with correct signatures and type annotations
    - Docstrings for all methods and properties
    - TODO comments indicating what needs to be implemented

    The generated template serves as a starting point for implementing a protocol,
    ensuring that all required methods and properties are included with the
    correct signatures.

    Args:
        protocol_class (Type[T]): The protocol to generate a template for

    Returns:
        str: A string containing the Python code for the implementation template

    Example:
        ```python
        from sifaka.interfaces import Chain

        # Generate an implementation template for the Chain protocol
        template = generate_implementation_template(Chain)
        print(template)

        # Output:
        # class ChainImplementation:
        #     '''Implementation of Chain protocol.'''
        #
        #     def run(self, prompt: str, **kwargs) -> str:
        #         '''Run the chain with the given prompt.'''
        #         # TODO: Implement run method
        #         raise NotImplementedError()
        ```
    """
    protocol_methods = _get_protocol_methods(protocol_class)
    protocol_properties = _get_protocol_properties(protocol_class)
    template = f"class {protocol_class.__name__}Implementation:\n"
    template += f'    """Implementation of {protocol_class.__name__} protocol."""\n\n'
    for prop_name in protocol_properties:
        template += f"    @property\n"
        template += f"    def {prop_name}(self):\n"
        template += f'        """Get the {prop_name}."""\n'
        template += f"        # TODO: Implement {prop_name} property\n"
        template += f"        raise NotImplementedError()\n\n"
    if protocol_methods:
        for method_name, method_info in protocol_methods.items():
            signature = method_info["signature"]
            docstring = method_info["docstring"] or f"{method_name} method."
            params = []
            if hasattr(signature, "parameters"):
                for param_name, param in signature.parameters.items():
                    if param_name == "self":
                        params.append("self")
                    elif param.annotation != inspect.Parameter.empty:
                        params.append(f"{param_name}: {_format_annotation(param.annotation)}")
                    else:
                        params.append(param_name)
            return_annotation = ""
            if signature.return_annotation != inspect.Signature.empty:
                return_annotation = f" -> {_format_annotation(signature.return_annotation)}"
            template += f"    def {method_name}({', '.join(params)}){return_annotation}:\n"
            template += f'        """{docstring}"""\n'
            template += f"        # TODO: Implement {method_name} method\n"
            template += f"        raise NotImplementedError()\n\n"
    return template


def get_protocol_requirements(protocol_class: Type[T]) -> Dict[str, Any]:
    """
    Get the requirements for implementing a protocol.

    This function analyzes a protocol and returns a structured dictionary
    containing detailed information about the methods and properties required
    to implement the protocol. The returned dictionary includes method signatures,
    parameter types, return types, and docstrings.

    The requirements dictionary can be used to:
    - Generate documentation for a protocol
    - Create code generators for protocol implementations
    - Validate existing implementations against protocol requirements
    - Provide developer assistance for implementing protocols

    Args:
        protocol_class (Type[T]): The protocol to analyze for requirements

    Returns:
        Dict[str, Any]: A dictionary containing protocol requirements with the following structure:
            {
                "name": str,              # Protocol name
                "module": str,            # Protocol module
                "methods": {              # Dictionary of methods
                    "method_name": {      # Method information
                        "signature": str,     # Method signature
                        "docstring": str,     # Method docstring
                        "parameters": dict,   # Parameter information
                        "return_type": str    # Return type
                    }
                },
                "properties": {           # Dictionary of properties
                    "property_name": {}   # Property information
                }
            }

    Example:
        ```python
        from sifaka.interfaces import Chain

        # Get requirements for the Chain protocol
        requirements = get_protocol_requirements(Chain)

        # Print method requirements
        for method_name, method_info in requirements["methods"].items():
            print(f"Method: {method_name}")
            print(f"  Signature: {method_info['signature']}")
            print(f"  Return type: {method_info['return_type']}")
            print(f"  Parameters:")
            for param_name, param_info in method_info['parameters'].items():
                print(f"    {param_name}: {param_info['annotation']}")
        ```
    """
    protocol_methods = _get_protocol_methods(protocol_class)
    protocol_properties = _get_protocol_properties(protocol_class)
    method_requirements = {}
    for method_name, method_info in protocol_methods.items() if protocol_methods else {}:
        signature = method_info["signature"]
        docstring = method_info["docstring"] or f"{method_name} method."
        parameters = {}
        if signature and hasattr(signature, "parameters"):
            for param_name, param in signature.parameters.items():
                if param_name != "self":
                    parameters[param_name] = {
                        "annotation": (
                            _format_annotation(param.annotation)
                            if param.annotation != inspect.Parameter.empty
                            else None
                        ),
                        "default": (
                            param.default if param.default != inspect.Parameter.empty else None
                        ),
                    }

        method_requirements[method_name] = {
            "signature": str(signature),
            "docstring": docstring,
            "parameters": parameters,
            "return_type": (
                _format_annotation(signature.return_annotation)
                if signature.return_annotation != inspect.Parameter.empty
                else None
            ),
        }
    property_requirements: Dict[str, Dict[str, Any]] = {
        prop_name: {} for prop_name in protocol_properties
    }
    return {
        "name": protocol_class.__name__,
        "module": protocol_class.__module__,
        "methods": method_requirements,
        "properties": property_requirements,
    }


def _get_protocol_methods(protocol_class: Type[Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get the methods defined in a protocol.

    This function inspects a protocol class and extracts information about
    all its methods, including signatures and docstrings. It skips special
    methods (those starting with '_') except for '__call__'.

    Args:
        protocol_class (Type[Any]): The protocol to inspect for methods

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping method names to method information,
            where each method's information includes:
            - signature: The method's signature
            - docstring: The method's docstring

    Raises:
        None: Errors during inspection are caught and the method is skipped
    """
    methods = {}
    for attr_name in dir(protocol_class):
        if attr_name and attr_name.startswith("_") and attr_name != "__call__":
            continue
        attr = getattr(protocol_class, attr_name)
        if (
            (inspect and inspect.isfunction(attr))
            or (inspect and inspect.ismethod(attr))
            or (inspect and inspect.ismethoddescriptor(attr))
        ):
            try:
                signature = inspect.signature(attr) if inspect else None
                docstring = inspect.getdoc(attr) if inspect else None
                methods[attr_name] = {"signature": signature, "docstring": docstring}
            except (ValueError, TypeError):
                pass
    return methods


def _get_protocol_properties(protocol_class: Type[Any]) -> Set[str]:
    """
    Get the properties defined in a protocol.

    This function inspects a protocol class and extracts information about
    all its properties. It skips special properties (those starting with '_').

    Args:
        protocol_class (Type[Any]): The protocol to inspect for properties

    Returns:
        Set[str]: A set of property names defined in the protocol
    """
    properties = set()
    for attr_name in dir(protocol_class):
        if attr_name and attr_name.startswith("_"):
            continue
        attr = getattr(protocol_class, attr_name)
        if isinstance(attr, property):
            properties.add(attr_name)
    return properties


def _get_class_methods(cls: Type[Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get the methods defined in a class.

    This function inspects a class and extracts information about all its methods,
    including signatures and docstrings. It skips special methods (those starting
    with '_') except for '__call__'.

    Args:
        cls (Type[Any]): The class to inspect for methods

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping method names to method information,
            where each method's information includes:
            - signature: The method's signature
            - docstring: The method's docstring

    Raises:
        None: Errors during inspection are caught and the method is skipped
    """
    methods = {}
    for attr_name in dir(cls):
        if attr_name and attr_name.startswith("_") and attr_name != "__call__":
            continue
        attr = getattr(cls, attr_name)
        if (
            (inspect and inspect.isfunction(attr))
            or (inspect and inspect.ismethod(attr))
            or (inspect and inspect.ismethoddescriptor(attr))
        ):
            try:
                signature = inspect.signature(attr) if inspect else None
                docstring = inspect.getdoc(attr) if inspect else None
                methods[attr_name] = {"signature": signature, "docstring": docstring}
            except (ValueError, TypeError):
                pass
    return methods


def _get_class_properties(cls: Type[Any]) -> Set[str]:
    """
    Get the properties defined in a class.

    This function inspects a class and extracts information about all its properties.
    It skips special properties (those starting with '_').

    Args:
        cls (Type[Any]): The class to inspect for properties

    Returns:
        Set[str]: A set of property names defined in the class
    """
    properties = set()
    for attr_name in dir(cls):
        if attr_name and attr_name.startswith("_"):
            continue
        attr = getattr(cls, attr_name)
        if isinstance(attr, property):
            properties.add(attr_name)
    return properties


def _is_compatible_type(actual_type: Any, expected_type: Any) -> bool:
    """
    Check if an actual type is compatible with an expected type.

    This function determines if an actual type is compatible with an expected type.
    Currently, it performs a simple equality check, but could be extended to handle
    more sophisticated type compatibility rules such as subtyping, type variables,
    and union types.

    Args:
        actual_type (Any): The actual type to check
        expected_type (Any): The expected type to check against

    Returns:
        bool: True if the actual type is compatible with the expected type, False otherwise

    Note:
        This is a simple implementation that only checks for exact type matches.
        Future implementations could handle more complex type compatibility rules.
    """
    return bool(actual_type == expected_type)


def _format_annotation(annotation: Any) -> str:
    """
    Format a type annotation for display.

    This function converts a type annotation object into a string representation
    suitable for display in documentation, error messages, or code generation.
    It handles special cases like None, Any, and missing annotations.

    Args:
        annotation (Any): The type annotation to format

    Returns:
        str: A string representation of the type annotation

    Examples:
        - None -> "None"
        - Any -> "Any"
        - str -> "str"
        - List[int] -> "List[int]"
        - inspect.Signature.empty -> "Any"
    """
    if annotation is inspect.Signature.empty:
        return "Any"
    if annotation is None:
        return "None"
    elif annotation is type(None):
        return "None"
    elif hasattr(annotation, "__name__") and annotation.__name__ == "Any":
        return "Any"
    try:
        return str(annotation.__name__)
    except AttributeError:
        return str(annotation)
