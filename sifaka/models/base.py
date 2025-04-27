"""
Base classes for Sifaka model providers.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict


class ModelProvider(BaseModel):
    """
    Base class for all Sifaka model providers.

    A model provider is responsible for generating text using an LLM.

    Attributes:
        name: The name of the model provider
        config: Configuration for the model provider
    """

    name: str
    config: Dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a model provider.

        Args:
            name: The name of the model provider
            config: Configuration for the model provider
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            config=config or {},
            **kwargs,
        )

    @property
    def provider_name(self) -> str:
        """Return the name of the model provider."""
        return self.name

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The generated text

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.

        This allows the model provider to be called like a function.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The generated text
        """
        return self.generate(prompt, **kwargs)
