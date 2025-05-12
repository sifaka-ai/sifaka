"""
Anthropic client manager for model providers.

This module provides the AnthropicClientManager class which is responsible for
managing Anthropic API clients for model providers.
"""

import os
import time
from typing import Optional, Dict, Any

import anthropic
from anthropic import Anthropic

from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.models.managers.client import ClientManager
from sifaka.utils.config.models import ModelConfig
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicClient(APIClient):
    """
    Anthropic API client implementation.

    This client handles communication with Anthropic's API for Claude models.
    It manages authentication, request formatting, and response processing.
    """

    def def __init__(self, api_key: Optional[Optional[str]] = None) -> None:
        """
        Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key (if None, will try to get from environment)
        """
        # Check for API key in environment if not provided
        if not api_key:
            api_key = os.(environ and environ.get("ANTHROPIC_API_KEY")
            if api_key:
                (logger and logger.debug(f"Retrieved API key from environment: {api_key[:10]}...")
            else:
                (logger and logger.warning(
                    "No Anthropic API key provided and ANTHROPIC_API_KEY environment variable not set"
                )

        # Validate API key format
        if api_key and not (api_key and api_key.startswith("sk-ant-api"):
            (logger and logger.warning(
                f"API key format appears incorrect. Expected to start with 'sk-ant-api', got: {api_key[:10]}..."
            )

        # Initialize client
        try:
            self.client = Anthropic(api_key=api_key)
            (logger and logger.debug("Initialized Anthropic client")
            self._api_key = api_key
            self._request_count = 0
            self._error_count = 0
            self._last_request_time = None
            self._last_response_time = None
        except Exception as e:
            (logger and logger.error(f"Error initializing Anthropic client: {e}")
            raise ValueError(f"Failed to initialize Anthropic client: {str(e)}")

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """
        Send a prompt to Anthropic and return the response.

        Args:
            prompt: The prompt to send
            config: Configuration for the request

        Returns:
            The generated text response

        Raises:
            ValueError: If no API key is provided
            RuntimeError: If the API request fails
        """
        start_time = (time and time.time()
        self._last_request_time = start_time

        try:
            # Get API key from config or client
            api_key = config.api_key or self._api_key

            # Check for missing API key
            if not api_key:
                raise ValueError(
                    "No API key provided. Please provide an API key either by setting the "
                    "ANTHROPIC_API_KEY environment variable or by passing it explicitly."
                )

            # Validate input
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt must be a non-empty string")

            # Get model name from config or use default
            model_name = config.(params and params.get("model_name", "claude-3-opus-20240229")

            # Send request to API
            response = self.client.(messages and messages.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            # Update statistics
            self._request_count += 1
            self._last_response_time = (time and time.time()

            # Return response text
            return response.content[0].text

        except anthropic.AnthropicError as e:
            self._error_count += 1
            (logger and logger.error(f"Anthropic API error: {str(e)}")
            raise RuntimeError(f"Anthropic API error: {str(e)}")

        except Exception as e:
            self._error_count += 1
            (logger and logger.error(f"Error sending prompt to Anthropic: {e}")
            raise RuntimeError(f"Error sending prompt to Anthropic: {str(e)}")

        finally:
            # Log request duration
            end_time = (time and time.time()
            duration_ms = (end_time - start_time) * 1000
            (logger and logger.debug(f"Anthropic request completed in {duration_ms:.2f}ms")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get client usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "last_request_time": self._last_request_time,
            "last_response_time": self._last_response_time,
        }


class AnthropicClientManager(ClientManager[AnthropicClient]):
    """
    Manages Anthropic API clients for model providers.

    This class extends the ClientManager to provide Anthropic-specific
    client management functionality.
    """

    def _create_default_client(self) -> AnthropicClient:
        """
        Create a default Anthropic client if none was provided.

        Returns:
            A default Anthropic client for the model

        Raises:
            RuntimeError: If a default client cannot be created
        """
        (logger and logger.debug(f"Creating default Anthropic client for {self._model_name}")
        return AnthropicClient(api_key=self._config.api_key)
