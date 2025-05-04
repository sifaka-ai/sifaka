"""
Generation module for Sifaka.

This module provides components for generating outputs using model providers. It includes:
- Generator: A generic class that handles output generation
- ModelProvider: Interface for model providers (imported from models.base)

The generation process follows these steps:
1. Initialize a Generator with a model provider
2. Pass a prompt to the generate() method
3. Receive the generated output

Example:
    ```python
    from sifaka.generation import Generator
    from sifaka.models.base import ModelProvider

    # Create a model provider
    model = ModelProvider()

    # Create a generator
    generator = Generator(model)

    # Generate output
    output = generator.generate("Write a short story about a robot.")
    print(f"Generated: {output}")
    ```
"""

from typing import TypeVar, Generic

from .models.base import ModelProvider

OutputType = TypeVar("OutputType")


class Generator(Generic[OutputType]):
    """
    Generator class that handles output generation using model providers.

    This class is responsible for:
    1. Managing model providers for generation
    2. Handling prompt processing
    3. Generating outputs from prompts
    4. Type-safe output handling

    The generator follows a simple workflow:
    1. Initialize with a model provider
    2. Pass prompts to generate()
    3. Receive typed outputs

    Example:
        ```python
        generator = Generator(ModelProvider())
        output = generator.generate("Write a poem about nature.")
        ```
    """

    def __init__(self, model: ModelProvider):
        """
        Initialize a Generator instance.

        Args:
            model: The model provider to use for generation. Must implement
                  the ModelProvider protocol and provide a generate() method.

        Raises:
            ValueError: If model is None
            TypeError: If model does not implement ModelProvider
        """
        if model is None:
            raise ValueError("Model provider cannot be None")
        self.model = model

    def generate(self, prompt: str) -> OutputType:
        """
        Generate an output from a prompt.

        This method:
        1. Validates the input prompt
        2. Passes the prompt to the model provider
        3. Returns the generated output

        Args:
            prompt: The prompt to generate from. Must be a non-empty string.

        Returns:
            Generated output of the specified type (OutputType)

        Raises:
            ValueError: If prompt is empty or None
            RuntimeError: If model fails during generation
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        try:
            return self.model.generate(prompt)
        except Exception as e:
            raise RuntimeError(f"Model failed during generation: {str(e)}")