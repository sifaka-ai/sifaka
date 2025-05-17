"""
Tests for the models module.

This module contains tests for the models in the Sifaka framework.
"""

import pytest
import os  # noqa
from typing import Any, Dict, List, Optional  # noqa
from unittest.mock import patch, MagicMock  # noqa

from sifaka.models.base import create_model
from sifaka.models.openai import OpenAIModel, create_openai_model


class TestCreateModel:
    """Tests for the create_model function."""

    def test_create_model_with_provider_and_model_name(self, monkeypatch) -> None:
        """Test creating a model with separate provider and model name."""

        # Mock the factory_create_model function
        def mock_factory_create_model(provider: str, model_name: str, **options: Any) -> str:
            return f"{provider}:{model_name}"

        monkeypatch.setattr("sifaka.factories.create_model", mock_factory_create_model)

        model = create_model("openai", "gpt-4")
        assert model == "openai:gpt-4"

    def test_create_model_with_combined_string(self, monkeypatch) -> None:
        """Test creating a model with a combined provider:model string."""

        # Mock the parse_model_string function
        def mock_parse_model_string(model_string: str) -> tuple[str, str]:
            return "openai", "gpt-4"

        # Mock the factory_create_model function
        def mock_factory_create_model(provider: str, model_name: str, **options: Any) -> str:
            return f"{provider}:{model_name}"

        monkeypatch.setattr("sifaka.factories.parse_model_string", mock_parse_model_string)
        monkeypatch.setattr("sifaka.factories.create_model", mock_factory_create_model)

        # Test with a combined string that includes a colon
        model = create_model("openai:gpt-4")
        assert model == "openai:gpt-4:"

    def test_create_model_with_options(self, monkeypatch) -> None:
        """Test creating a model with options."""

        # Mock the factory_create_model function
        def mock_factory_create_model(
            provider: str, model_name: str, **options: Any
        ) -> Dict[str, Any]:
            return {"provider": provider, "model_name": model_name, "options": options}

        monkeypatch.setattr("sifaka.factories.create_model", mock_factory_create_model)

        model = create_model("openai", "gpt-4", api_key="test-key", temperature=0.7)
        assert model["provider"] == "openai"
        assert model["model_name"] == "gpt-4"
        assert model["options"]["api_key"] == "test-key"
        assert model["options"]["temperature"] == 0.7


class TestOpenAIModel:
    """Tests for the OpenAIModel class."""

    @pytest.fixture
    def mock_openai(self, monkeypatch) -> None:
        """Fixture to mock the OpenAI package."""
        # Create mock classes
        mock_message = MagicMock()
        mock_message.content = "Mock response"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion_choice = MagicMock()
        mock_completion_choice.text = "Mock response"

        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [mock_choice]

        mock_completion = MagicMock()
        mock_completion.choices = [mock_completion_choice]

        # Create mock client
        mock_chat = MagicMock()
        mock_chat.completions.create.return_value = mock_chat_completion

        mock_completions = MagicMock()
        mock_completions.create.return_value = mock_completion

        mock_client = MagicMock()
        mock_client.chat = mock_chat
        mock_client.completions = mock_completions

        # Create mock OpenAI class
        mock_openai_class = MagicMock()
        mock_openai_class.return_value = mock_client

        # Create mock error classes
        mock_api_error = MagicMock()
        mock_rate_limit_error = MagicMock()
        mock_api_connection_error = MagicMock()

        # Patch the imports
        monkeypatch.setattr("sifaka.models.openai.OpenAI", mock_openai_class)
        monkeypatch.setattr("sifaka.models.openai.APIError", mock_api_error)
        monkeypatch.setattr("sifaka.models.openai.RateLimitError", mock_rate_limit_error)
        monkeypatch.setattr("sifaka.models.openai.APIConnectionError", mock_api_connection_error)
        monkeypatch.setattr("sifaka.models.openai.OPENAI_AVAILABLE", True)

        return mock_client

    @pytest.fixture
    def mock_tiktoken(self, monkeypatch) -> None:
        """Fixture to mock the tiktoken package."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

        mock_tiktoken = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_encoding
        mock_tiktoken.get_encoding.return_value = mock_encoding

        monkeypatch.setattr("sifaka.models.openai.tiktoken", mock_tiktoken)
        return mock_tiktoken

    def test_init_with_defaults(self, mock_openai, set_env_vars, env_vars) -> None:  # noqa
        """Test initializing an OpenAIModel with default parameters."""
        model = OpenAIModel(model_name="gpt-4")
        assert model.model_name == "gpt-4"
        assert model.api_key == env_vars["OPENAI_API_KEY"]
        assert model.organization is None
        assert model.options == {}

    def test_init_with_api_key(self, mock_openai) -> None:  # noqa
        """Test initializing an OpenAIModel with an API key."""
        model = OpenAIModel(model_name="gpt-4", api_key="test-api-key")
        assert model.api_key == "test-api-key"

    def test_init_with_organization(self, mock_openai) -> None:  # noqa
        """Test initializing an OpenAIModel with an organization ID."""
        model = OpenAIModel(model_name="gpt-4", api_key="test-api-key", organization="test-org")
        assert model.organization == "test-org"

    def test_init_with_options(self, mock_openai) -> None:  # noqa
        """Test initializing an OpenAIModel with options."""
        model = OpenAIModel(
            model_name="gpt-4", api_key="test-api-key", temperature=0.7, max_tokens=500
        )
        assert model.options["temperature"] == 0.7
        assert model.options["max_tokens"] == 500

    def test_configure(self, mock_openai) -> None:  # noqa
        """Test configuring an OpenAIModel with new options."""
        model = OpenAIModel(model_name="gpt-4", api_key="test-api-key")
        model.configure(temperature=0.7, max_tokens=500)
        assert model.options["temperature"] == 0.7
        assert model.options["max_tokens"] == 500

    def test_configure_with_api_key(self, mock_openai) -> None:  # noqa
        """Test configuring an OpenAIModel with a new API key."""
        model = OpenAIModel(model_name="gpt-4", api_key="test-api-key")
        model.configure(api_key="new-api-key")
        assert model.api_key == "new-api-key"

    def test_generate_chat_model(self, mock_openai) -> None:
        """Test generating text with a chat model."""
        model = OpenAIModel(model_name="gpt-4", api_key="test-api-key")
        response = model.generate("Test prompt")

        assert response == "Mock response"
        mock_openai.chat.completions.create.assert_called_once()
        kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4"
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["messages"][0]["content"] == "Test prompt"

    def test_generate_chat_model_with_system_message(self, mock_openai) -> None:
        """Test generating text with a chat model and a system message."""
        model = OpenAIModel(model_name="gpt-4", api_key="test-api-key")
        response = model.generate("Test prompt", system_message="You are a helpful assistant.")

        assert response == "Mock response"
        mock_openai.chat.completions.create.assert_called_once()
        kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4"
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][0]["content"] == "You are a helpful assistant."
        assert kwargs["messages"][1]["role"] == "user"
        assert kwargs["messages"][1]["content"] == "Test prompt"

    def test_generate_completion_model(self, mock_openai) -> None:
        """Test generating text with a completion model."""
        model = OpenAIModel(model_name="text-davinci-003", api_key="test-api-key")
        response = model.generate("Test prompt")

        assert response == "Mock response"
        mock_openai.completions.create.assert_called_once()
        kwargs = mock_openai.completions.create.call_args.kwargs
        assert kwargs["model"] == "text-davinci-003"
        assert kwargs["prompt"] == "Test prompt"

    def test_generate_with_options(self, mock_openai) -> None:
        """Test generating text with options."""
        model = OpenAIModel(model_name="gpt-4", api_key="test-api-key")
        response = model.generate("Test prompt", temperature=0.7, max_tokens=500, top_p=0.9)

        assert response == "Mock response"
        mock_openai.chat.completions.create.assert_called_once()
        kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 500
        assert kwargs["top_p"] == 0.9

    def test_count_tokens(self, mock_openai, mock_tiktoken) -> None:  # noqa
        """Test counting tokens in text."""
        model = OpenAIModel(model_name="gpt-4", api_key="test-api-key")
        token_count = model.count_tokens("Test text")

        assert token_count == 5
        mock_tiktoken.encoding_for_model.assert_called_once_with("gpt-4")
        mock_tiktoken.encoding_for_model.return_value.encode.assert_called_once_with("Test text")

    def test_create_openai_model(self, mock_openai) -> None:  # noqa
        """Test creating an OpenAIModel using the factory function."""
        model = create_openai_model(model_name="gpt-4", api_key="test-api-key")
        assert isinstance(model, OpenAIModel)
        assert model.model_name == "gpt-4"
        assert model.api_key == "test-api-key"

    def test_create_openai_model_with_options(self, mock_openai) -> None:  # noqa
        """Test creating an OpenAIModel with options using the factory function."""
        model = create_openai_model(
            model_name="gpt-4", api_key="test-api-key", temperature=0.7, max_tokens=500
        )
        assert isinstance(model, OpenAIModel)
        assert model.model_name == "gpt-4"
        assert model.api_key == "test-api-key"
        assert model.options["temperature"] == 0.7
        assert model.options["max_tokens"] == 500
