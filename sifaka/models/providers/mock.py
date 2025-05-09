"""Mock model provider for testing."""

from typing import Dict, Any, Optional, ClassVar
from pydantic import PrivateAttr

from sifaka.models.base import ModelProvider, ModelConfig, APIClient, TokenCounter
from sifaka.utils.logging import get_logger
from sifaka.utils.state import ModelState

logger = get_logger(__name__)


class MockAPIClient(APIClient):
    """Mock API client for testing."""

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Send a mock prompt and return a response."""
        return f"Mock response to: {prompt}"


class MockTokenCounter(TokenCounter):
    """Mock token counter for testing."""

    def count_tokens(self, text: str) -> int:
        """Count tokens in text by splitting on whitespace."""
        return len(text.split())


class MockProvider(ModelProvider):
    """Mock model provider for testing."""

    # Class constants
    DEFAULT_MODEL: ClassVar[str] = "mock-model"

    def __init__(
        self, model_name: str = DEFAULT_MODEL, config: Optional[ModelConfig] = None, **kwargs: Any
    ):
        """
        Initialize the mock provider.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            **kwargs: Additional configuration parameters
        """
        # Create default config if not provided
        if config is None:
            try:
                from sifaka.utils.config import standardize_model_config

                config = standardize_model_config(
                    temperature=0.7,
                    max_tokens=100,
                    api_key="mock-api-key",
                    trace_enabled=True,
                    **kwargs,
                )
            except (ImportError, AttributeError):
                config = ModelConfig(
                    temperature=0.7,
                    max_tokens=100,
                    api_key="mock-api-key",
                    trace_enabled=True,
                    **kwargs,
                )

        # Initialize base class
        super().__init__(model_name, config)

    def _create_default_client(self) -> APIClient:
        """Create a default mock API client."""
        return MockAPIClient()

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default mock token counter."""
        return MockTokenCounter()

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Generate a mock response.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional keyword arguments

        Returns:
            A dictionary containing the generated text and usage statistics
        """
        # Calculate token counts
        prompt_tokens = len(prompt.split())
        completion_tokens = 10
        total_tokens = prompt_tokens + completion_tokens

        return {
            "text": f"Mock response to: {prompt}",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        Invoke the model with a prompt and return structured output.

        This is a convenience method that extracts the text from generate.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional keyword arguments

        Returns:
            The generated text
        """
        # Generate response
        response = self.generate(prompt, **kwargs)

        # Return just the text
        return response["text"]

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration.

        Args:
            config: The configuration to validate

        Raises:
            ValueError: If the configuration is invalid
        """
        if not config.get("name"):
            raise ValueError("Name is required")
        if not config.get("description"):
            raise ValueError("Description is required")
