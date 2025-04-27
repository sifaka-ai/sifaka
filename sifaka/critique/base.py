"""
Base critique class.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class Critique(BaseModel):
    """
    Base class for critique systems.

    Attributes:
        name: The name of the critique system
        description: Description of the critique system
        config: Configuration for the critique system
    """

    name: str
    description: str
    config: Dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a critique system.

        Args:
            name: The name of the critique system
            description: Description of the critique system
            config: Configuration for the critique system
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            **kwargs,
        )

    def critique(self, prompt: str) -> str:
        """
        Critique a prompt.

        Args:
            prompt: The prompt to critique

        Returns:
            The critiqued prompt
        """
        raise NotImplementedError
