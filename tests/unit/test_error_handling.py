"""Tests for error handling when no API key is available."""

import os
from unittest.mock import patch

import pytest

from sifaka import improve_sync
from sifaka.core.exceptions import ModelProviderError


class TestErrorHandling:
    """Test error handling for missing API keys and providers."""

    def test_no_api_key_error_message(self):
        """Test that a clear error message is shown when no API key is found."""
        # Mock dotenv at import time to prevent loading actual .env file
        with patch("dotenv.load_dotenv"):
            # Clear all API keys from environment
            with patch.dict(os.environ, {}, clear=True):
                try:
                    result = improve_sync("Test text")
                    # If we get here, check what we actually got
                    print(f"Unexpected result: {result}")
                    print(f"Final text: {result.final_text}")
                    pytest.fail("Expected ModelProviderError but got result instead")
                except ModelProviderError as e:
                    # This is what we expect
                    assert "Cannot improve text" in str(e)
                    assert "No API key found" in str(e)
                    assert e.error_code == "authentication"
                except Exception as e:
                    # Unexpected exception type
                    print(f"Unexpected exception type: {type(e)}")
                    print(f"Exception message: {str(e)}")
                    raise

    def test_invalid_provider_error(self):
        """Test error when using invalid provider."""
        from sifaka import Config
        from sifaka.core.config import LLMConfig

        with pytest.raises(ValueError) as exc_info:
            config = Config(llm=LLMConfig(provider="invalid-provider"))
            improve_sync("Test text", config=config)

        assert "invalid-provider" in str(exc_info.value)

    def test_api_key_error_lists_all_providers(self):
        """Test that error message lists all available providers."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ModelProviderError) as exc_info:
                improve_sync("Test text")

            error_msg = str(exc_info.value)
            assert "OPENAI_API_KEY" in error_msg
            assert "ANTHROPIC_API_KEY" in error_msg
            assert "GROQ_API_KEY" in error_msg
            assert "GEMINI_API_KEY" in error_msg
            assert "OLLAMA_API_KEY" in error_msg

    @patch("sifaka.core.llm_client.LLMManager.get_client")
    def test_client_creation_error(self, mock_get_client):
        """Test error handling when client creation fails."""
        # Mock client creation to raise an exception
        mock_get_client.side_effect = Exception("Connection failed")

        with pytest.raises(ModelProviderError) as exc_info:
            improve_sync("Test text")

        assert "Failed to generate improved text" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value)

    def test_ollama_no_api_key_works(self):
        """Test that Ollama works without an API key."""
        from sifaka import Config
        from sifaka.core.config import LLMConfig
        from sifaka.core.llm_client import LLMClient

        # Clear all API keys
        with patch.dict(os.environ, {}, clear=True):
            # Set OLLAMA_BASE_URL to indicate Ollama is available
            with patch.dict(
                os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434/v1"}
            ):
                # Mock the LLM client to avoid actual API calls
                with patch.object(LLMClient, "complete") as mock_complete:
                    from sifaka.core.llm_client import LLMResponse

                    mock_complete.return_value = LLMResponse(
                        content="Improved text",
                        model="llama3.2",
                        usage={"total_tokens": 15},
                    )

                    # This should work without error
                    config = Config(llm=LLMConfig(provider="ollama", model="llama3.2"))
                    result = improve_sync("Test text", config=config)

                    # Should return improved text, not original
                    assert result.final_text == "Improved text"
                    assert not result.final_text == "Test text"
