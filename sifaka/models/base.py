"""
Base classes for Sifaka model providers.
"""

from typing import Optional
from pydantic import BaseModel


class ModelProvider(BaseModel):
    """
    Base class for all Sifaka model providers.

    A model provider is responsible for generating text using an LLM.

    Attributes:
        name (str): The name of the model provider
    """

    name: str

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name: Optional[str] = None, **data):
        """
        Initialize a model provider.

        Args:
            name (Optional[str]): The name of the model provider
            **data: Additional data for the model provider
        """
        if name is not None:
            data["name"] = name
        elif "name" not in data:
            data["name"] = self.__class__.__name__

        super().__init__(**data)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt (str): The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            str: The generated text
        """
        raise NotImplementedError("Subclasses must implement generate()")
