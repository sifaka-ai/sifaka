"""
OpenAI client manager for model providers.

This module provides the OpenAIClientManager class which is responsible for
managing OpenAI API clients for model providers.
"""

import os
from typing import Any, Dict, Optional
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.models.managers.client import ClientManager
from sifaka.utils.config.models import ModelConfig
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient(APIClient):
    """OpenAI API client implementation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the OpenAI client.

        Args:
            api_key: Optional API key for OpenAI
        """
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                logger.debug("Retrieved API key from environment")
        if not api_key:
            logger.warning(
                "No OpenAI API key provided and OPENAI_API_KEY environment variable not set"
            )
        elif not api_key.startswith("sk-"):
            logger.warning(
                f"API key format appears incorrect. Expected to start with 'sk-', got: {api_key[:5]}..."
            )
        self.api_key = api_key
        logger.debug("Initialized OpenAI client")

    def send_prompt(self, prompt: str, config: Any) -> str:
        """
        Send a prompt to OpenAI and return the response.

        Args:
            prompt: The prompt to send
            config: Configuration for the request

        Returns:
            The generated text response

        Raises:
            ValueError: If API key is missing
            RuntimeError: If the API call fails
        """
        api_key = config.api_key or self.api_key
        if not api_key:
            raise ValueError(
                "No API key provided. Please provide an API key either by setting the OPENAI_API_KEY environment variable or by passing it explicitly."
            )

        def generate_operation() -> Any:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

            # Get model name from config
            model_name = getattr(config, "model_name", "gpt-3.5-turbo")
            if hasattr(config, "params") and config.params and "model_name" in config.params:
                model_name = config.params.get("model_name")

            messages = [{"role": "user", "content": prompt}]

            # Build parameters
            params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }

            # Add optional parameters if available
            if hasattr(config, "params"):
                if "top_p" in config.params:
                    params["top_p"] = config.params.get("top_p", 1.0)
                if "frequency_penalty" in config.params:
                    params["frequency_penalty"] = config.params.get("frequency_penalty", 0.0)
                if "presence_penalty" in config.params:
                    params["presence_penalty"] = config.params.get("presence_penalty", 0.0)
                if "stop" in config.params:
                    params["stop"] = config.params.get("stop")

            # Make API call
            response = client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content.strip() if content is not None else ""

        from sifaka.utils.errors.safe_execution import safely_execute_component
        from sifaka.utils.errors.component import ModelError

        result = safely_execute_component(
            operation=generate_operation,
            component_name="OpenAIClient",
            component_type="APIClient",
            error_class=ModelError,
            additional_metadata={"model_name": getattr(config, "model_name", "gpt-3.5-turbo")},
        )
        return str(result)


class OpenAIClientManager(ClientManager[OpenAIClient]):
    """
    Manages OpenAI API clients for model providers.

    This class extends the ClientManager to provide OpenAI-specific
    client management functionality.
    """

    def _create_default_client(self) -> OpenAIClient:
        """
        Create a default OpenAI client if none was provided.

        Returns:
            A default OpenAI client for the model

        Raises:
            RuntimeError: If a default client cannot be created
        """
        logger.debug(f"Creating default OpenAI client for {self._model_name}")
        return OpenAIClient(api_key=self._config.api_key)
