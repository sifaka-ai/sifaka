"""
Tests for the Chain class.

This module contains tests for the Chain class, which is the central component
of the Sifaka framework.
"""

import pytest
from typing import Any, Dict, List, Optional, Tuple

from sifaka import Chain
from sifaka.config import SifakaConfig, ModelConfig
from sifaka.results import Result, ValidationResult, ImprovementResult


class TestChainInitialization:
    """Tests for Chain initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initializing a Chain with default parameters."""
        chain = Chain()
        assert chain._model is None
        assert chain._prompt is None
        assert chain._validators == []
        assert chain._improvers == []
        assert chain._config is not None

    def test_init_with_config(self) -> None:
        """Test initializing a Chain with a custom configuration."""
        config = SifakaConfig(model=ModelConfig(temperature=0.8))
        chain = Chain(config)
        assert chain._config is config
        assert chain._config.model.temperature == 0.8


class TestChainConfiguration:
    """Tests for Chain configuration methods."""

    def test_with_model_instance(self, mock_model) -> None:
        """Test configuring a Chain with a model instance."""
        chain = Chain().with_model(mock_model)
        assert chain._model is mock_model

    def test_with_model_string(self, monkeypatch) -> None:
        """Test configuring a Chain with a model string."""

        # Mock the create_model_from_string function
        def mock_create_model_from_string(model_string: str, **options: Any) -> Any:
            return "mock_model"

        monkeypatch.setattr("sifaka.chain.create_model_from_string", mock_create_model_from_string)

        chain = Chain().with_model("openai:gpt-4")
        assert chain._model == "mock_model"

    def test_with_prompt(self) -> None:
        """Test configuring a Chain with a prompt."""
        prompt = "Write a short story about a robot."
        chain = Chain().with_prompt(prompt)
        assert chain._prompt == prompt

    def test_validate_with(self, mock_validator) -> None:
        """Test adding a validator to a Chain."""
        chain = Chain().validate_with(mock_validator)
        assert len(chain._validators) == 1
        assert chain._validators[0] is mock_validator

    def test_improve_with(self, mock_critic) -> None:
        """Test adding a critic to a Chain."""
        chain = Chain().improve_with(mock_critic)
        assert len(chain._improvers) == 1
        assert chain._improvers[0] is mock_critic

    def test_with_options(self) -> None:
        """Test configuring a Chain with options."""
        options = {"temperature": 0.7, "max_tokens": 500}
        chain = Chain().with_options(**options)
        assert chain._config.model.temperature == 0.7
        assert chain._config.model.max_tokens == 500

    def test_with_config(self) -> None:
        """Test configuring a Chain with a new configuration."""
        config = SifakaConfig(model=ModelConfig(temperature=0.8))
        chain = Chain().with_config(config)
        assert chain._config is config


class TestChainExecution:
    """Tests for Chain execution."""

    def test_run_without_model_raises_error(self) -> None:
        """Test that running a Chain without a model raises an error."""
        from sifaka.errors import ChainError

        chain = Chain().with_prompt("Write a short story about a robot.")
        with pytest.raises(ChainError, match="Model not specified"):
            chain.run()

    def test_run_without_prompt_raises_error(self, mock_model) -> None:
        """Test that running a Chain without a prompt raises an error."""
        from sifaka.errors import ChainError

        chain = Chain().with_model(mock_model)
        with pytest.raises(ChainError, match="Prompt not specified"):
            chain.run()

    def test_run_basic(self, mock_model) -> None:
        """Test running a Chain with a model and prompt."""
        mock_model.set_response("This is a generated response.")
        chain = Chain().with_model(mock_model).with_prompt("Write a short story about a robot.")
        result = chain.run()

        assert result.text == "This is a generated response."
        assert result.passed is True
        assert len(result.validation_results) == 0
        assert len(result.improvement_results) == 0
        assert len(mock_model.generate_calls) == 1
        assert mock_model.generate_calls[0][0] == "Write a short story about a robot."

    def test_run_with_validator_pass(self, mock_model, mock_validator) -> None:
        """Test running a Chain with a validator that passes."""
        mock_model.set_response("This is a generated response.")
        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .validate_with(mock_validator)
        )
        result = chain.run()

        assert result.text == "This is a generated response."
        assert result.passed is True
        assert len(result.validation_results) == 1
        assert result.validation_results[0].passed is True
        assert len(mock_validator.validate_calls) == 1
        assert mock_validator.validate_calls[0] == "This is a generated response."

    @pytest.mark.parametrize("mock_validator", [False], indirect=True)
    def test_run_with_validator_fail(self, mock_model, mock_validator) -> None:
        """Test running a Chain with a validator that fails."""
        mock_model.set_response("This is a generated response.")
        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .validate_with(mock_validator)
        )
        result = chain.run()

        assert result.text == "This is a generated response."
        assert result.passed is False
        assert len(result.validation_results) == 1
        assert result.validation_results[0].passed is False
        assert len(mock_validator.validate_calls) == 1
        assert mock_validator.validate_calls[0] == "This is a generated response."

    def test_run_with_critic(self, mock_model, mock_critic) -> None:
        """Test running a Chain with a critic."""
        mock_model.set_response("This is a generated response.")
        mock_critic.set_improved_text("This is an improved response.")
        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .improve_with(mock_critic)
        )
        result = chain.run()

        assert result.text == "This is an improved response."
        assert result.passed is True
        assert len(result.improvement_results) == 1
        assert result.improvement_results[0].changes_made is True
        assert len(mock_critic.improve_calls) == 1
        assert mock_critic.improve_calls[0] == "This is a generated response."

    @pytest.mark.parametrize("mock_critic", [False], indirect=True)
    def test_run_with_critic_no_improvement(self, mock_model, mock_critic) -> None:
        """Test running a Chain with a critic that doesn't improve the text."""
        mock_model.set_response("This is a generated response.")
        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .improve_with(mock_critic)
        )
        result = chain.run()

        assert result.text == "This is a generated response."
        assert result.passed is True
        assert len(result.improvement_results) == 1
        assert result.improvement_results[0].changes_made is False
        assert len(mock_critic.improve_calls) == 1
        assert mock_critic.improve_calls[0] == "This is a generated response."

    def test_run_with_validator_and_critic(self, mock_model, mock_validator, mock_critic) -> None:
        """Test running a Chain with a validator and a critic."""
        mock_model.set_response("This is a generated response.")
        mock_critic.set_improved_text("This is an improved response.")
        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .validate_with(mock_validator)
            .improve_with(mock_critic)
        )
        result = chain.run()

        assert result.text == "This is an improved response."
        assert result.passed is True
        assert len(result.validation_results) == 1
        assert result.validation_results[0].passed is True
        assert len(result.improvement_results) == 1
        assert result.improvement_results[0].changes_made is True
        assert len(mock_validator.validate_calls) == 1
        assert mock_validator.validate_calls[0] == "This is a generated response."
        assert len(mock_critic.improve_calls) == 1
        assert mock_critic.improve_calls[0] == "This is a generated response."

    @pytest.mark.parametrize("mock_validator", [False], indirect=True)
    def test_run_with_validator_fail_and_critic(
        self, mock_model, mock_validator, mock_critic
    ) -> None:
        """Test running a Chain with a validator that fails and a critic."""
        mock_model.set_response("This is a generated response.")
        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .validate_with(mock_validator)
            .improve_with(mock_critic)
        )
        result = chain.run()

        assert result.text == "This is a generated response."
        assert result.passed is False
        assert len(result.validation_results) == 1
        assert result.validation_results[0].passed is False
        assert (
            len(result.improvement_results) == 0
        )  # Critic should not be called if validation fails
        assert len(mock_validator.validate_calls) == 1
        assert mock_validator.validate_calls[0] == "This is a generated response."
        assert len(mock_critic.improve_calls) == 0
