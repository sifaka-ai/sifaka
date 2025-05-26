# Docstring Standards for Sifaka

This document outlines the standards for writing docstrings in the Sifaka project. Consistent and comprehensive docstrings are essential for maintaining code quality, enabling effective collaboration, and providing clear documentation for users.

## Table of Contents

- [General Guidelines](#general-guidelines)
- [Style Guide](#style-guide)
- [Module Docstrings](#module-docstrings)
- [Class Docstrings](#class-docstrings)
- [Method and Function Docstrings](#method-and-function-docstrings)
- [Property Docstrings](#property-docstrings)
- [Examples](#examples)
- [References](#references)

## General Guidelines

- Every module, class, method, function, and property should have a docstring
- Docstrings should be clear, concise, and informative
- Use complete sentences with proper punctuation
- Include type hints in the function/method signature in addition to documenting types in the docstring
- For complex functionality, include examples in the docstring
- For implementations of academic papers or algorithms, include references to the original sources

## Style Guide

We follow the Google style for docstrings. This style is well-supported by documentation generators like Sphinx and provides a good balance between readability and comprehensiveness.

### Basic Structure

```python
"""Summary line.

Extended description of function.

Args:
    param1 (type): Description of param1.
    param2 (type): Description of param2.

Returns:
    type: Description of return value.

Raises:
    ExceptionType: When and why this exception is raised.

Examples:
    >>> example_function(1, 2)
    3
"""
```

## Module Docstrings

Module docstrings should appear at the top of the file, before any imports. They should provide an overview of the module's purpose and functionality.

```python
"""
Module for text validation in Sifaka.

This module provides validators for checking if text meets specific criteria,
such as length, content, and format requirements. It includes both built-in
validators and utilities for creating custom validators.

Example:
    ```python
    from sifaka.validators import length, prohibited_content

    # Create validators
    length_validator = length(min_words=100, max_words=500)
    content_validator = prohibited_content(prohibited=["harmful", "offensive"])

    # Use validators
    result = length_validator.validate("Some text to validate")
    ```
"""
```

## Class Docstrings

Class docstrings should describe the purpose and behavior of the class. Include information about important attributes and usage patterns.

```python
"""A validator for checking text length.

This class validates if text meets specified length requirements,
such as minimum and maximum word or character counts.

Attributes:
    min_words (Optional[int]): Minimum number of words required.
    max_words (Optional[int]): Maximum number of words allowed.
    min_chars (Optional[int]): Minimum number of characters required.
    max_chars (Optional[int]): Maximum number of characters allowed.
    name (str): Name of the validator.
"""
```

## Method and Function Docstrings

Method and function docstrings should describe what the function does, its parameters, return values, and any exceptions it might raise.

```python
"""Validate text against length requirements.

Args:
    text (str): The text to validate.

Returns:
    ValidationResult: A result object indicating whether the text meets
        the length requirements, with details about any issues found.

Raises:
    ValidationError: If the validation process encounters an error.
"""
```

## Property Docstrings

Property docstrings should describe what the property represents.

```python
"""The minimum number of words required for validation.

Returns:
    Optional[int]: The minimum word count, or None if not set.
"""
```

## Examples

### Module Example

```python
"""
Module for text improvement using critics in Sifaka.

This module provides critics that analyze and improve text quality.
It includes various critic implementations such as self-refine,
constitutional, and reflexion critics.

Example:
    ```python
    from sifaka.critics.self_refine import create_self_refine_critic
    from sifaka.models.openai import OpenAIModel

    # Create a model
    model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

    # Create a critic
    critic = create_self_refine_critic(
        model=model,
        max_refinement_iterations=3
    )

    # Improve text
    improved_text, result = critic.improve("Text to improve")
    ```
"""
```

### Class Example

```python
class Chain:
    """Main orchestrator for text generation, validation, and improvement.

    The Chain class coordinates the process of generating text using a model,
    validating it against specified criteria, and improving it using critics.
    It follows a fluent interface pattern for easy configuration.

    Attributes:
        model (Optional[Model]): The model used for text generation.
        prompt (Optional[str]): The prompt used for text generation.
        validators (List[Validator]): Validators used to check text quality.
        critics (List[Critic]): Critics used to improve text quality.
        max_attempts (int): Maximum number of improvement attempts.
        config (SifakaConfig): Configuration for the chain.

    Example:
        ```python
        from sifaka import Chain
        from sifaka.models.openai import OpenAIModel
        from sifaka.validators import length
        from sifaka.critics.self_refine import create_self_refine_critic

        # Create a chain
        chain = (Chain()
            .with_model(OpenAIModel(model_name="gpt-4", api_key="your-api-key"))
            .with_prompt("Write a short story about a robot.")
            .validate_with(length(min_words=100, max_words=500))
            .improve_with(create_self_refine_critic(model=model))
        )

        # Run the chain
        result = chain.run()
        ```
    """
```

### Method Example

```python
def improve(self, text: str) -> Tuple[str, ImprovementResult]:
    """Improve text using this critic.

    This method analyzes the provided text and generates an improved version
    based on the critic's improvement strategy.

    Args:
        text (str): The text to improve.

    Returns:
        Tuple[str, ImprovementResult]: A tuple containing the improved text
            and a result object with details about the improvement process.

    Raises:
        ImproverError: If the improvement process encounters an error.

    Example:
        ```python
        improved_text, result = critic.improve("Text to improve")
        print(f"Original: {result.original_text}")
        print(f"Improved: {result.improved_text}")
        print(f"Changes made: {result.changes_made}")
        ```
    """
```

## References

For implementations of academic papers or algorithms, include references to the original sources:

```python
"""Implementation of the ReflexionCritic based on the Reflexion paper.

This critic implements the Reflexion approach for improving text quality
through self-reflection and refinement.

References:
    Shinn, N., Cassano, F., Labash, B., Gopinath, A., Narasimhan, K., & Yao, S. (2023).
    Reflexion: Language Agents with Verbal Reinforcement Learning.
    arXiv preprint arXiv:2303.11366.
    https://arxiv.org/abs/2303.11366
"""
```

## Best Practices

1. **Be Specific**: Avoid vague descriptions. Clearly state what a function does, what parameters it expects, and what it returns.

2. **Document Edge Cases**: Mention any edge cases or special behaviors that might not be immediately obvious.

3. **Keep Updated**: Update docstrings when you change code functionality.

4. **Use Examples**: For complex functions or classes, include usage examples.

5. **Document Exceptions**: List all exceptions that might be raised and under what conditions.

6. **Include References**: For implementations of academic papers or algorithms, include proper citations.

7. **Avoid Redundancy**: Don't repeat information that's already clear from the function signature or context.

By following these standards, we ensure that Sifaka's codebase remains well-documented, maintainable, and accessible to new contributors and users.
