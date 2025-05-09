"""
Protocol interfaces for API clients.

This module defines the protocol interfaces for API clients,
establishing a common contract for API client behavior.
"""

from abc import abstractmethod
from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class APIClientProtocol(Protocol):
    """
    Protocol interface for API clients.

    This interface defines the contract for components that communicate with
    model APIs. It ensures that API clients can send prompts to the API and
    handle API errors appropriately.

    ## Lifecycle

    1. **Initialization**: Set up API client resources
    2. **Prompt Sending**: Send prompts to the API
    3. **Response Processing**: Process API responses
    4. **Error Handling**: Handle API errors
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a send_prompt method to send prompts to the API
    - Handle API errors and retries appropriately
    - Manage authentication and API keys securely
    """

    @abstractmethod
    def send_prompt(self, prompt: Any, config: Dict[str, Any]) -> Any:
        """
        Send a prompt to the API and return the response.

        Args:
            prompt: The prompt to send
            config: Configuration parameters for the request

        Returns:
            The API response

        Raises:
            ValueError: If the prompt or config is invalid
            RuntimeError: If the API request fails
        """
        pass
