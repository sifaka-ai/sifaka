"""Mock model provider for testing."""

from typing import Dict, Any, Optional
from sifaka.models.base import ModelProvider


class MockProvider(ModelProvider):
    """Mock model provider for testing."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the mock provider."""
        super().__init__(config)

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate a mock response."""
        return {
            "text": f"Mock response to: {prompt}",
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 10,
                "total_tokens": len(prompt.split()) + 10
            }
        }

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the configuration."""
        if not config.get("name"):
            raise ValueError("Name is required")
        if not config.get("description"):
            raise ValueError("Description is required")