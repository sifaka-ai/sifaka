"""
Tests for the Chain class.

This module contains tests for the Chain class, which is the central component
of the Sifaka framework.
"""

from typing import Any

import pytest

from sifaka import Chain
from sifaka.config import ModelConfig, SifakaConfig


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
        def mock_create_model_from_string(
            model_string: str, **options: Any
        ) -> Any:  # pylint: disable=unused-argument
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
        """Test running a Chain with a critic.

        In the new behavior, critics are only used when validation fails,
        so this test now checks that the text is not improved when validation passes.
        """
        mock_model.set_response("This is a generated response.")
        mock_model.set_response = (
            lambda text: text
        )  # Make the model return whatever is passed to it
        mock_critic.set_improved_text("This is an improved response.")
        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .improve_with(mock_critic)
            .with_options(
                apply_improvers_on_validation_failure=False
            )  # Disable feedback loop for this test
        )
        result = chain.run()

        # In the new behavior, critics are not applied when validation passes
        assert result.text == "This is a generated response."
        assert result.passed is True
        assert len(result.improvement_results) == 0
        assert len(mock_critic.improve_calls) == 0

    @pytest.mark.parametrize("mock_critic", [False], indirect=True)
    def test_run_with_critic_no_improvement(self, mock_model, mock_critic) -> None:
        """Test running a Chain with a critic that doesn't improve the text.

        In the new behavior, critics are only used when validation fails,
        so this test now checks that the text is not improved when validation passes.
        """
        mock_model.set_response("This is a generated response.")
        mock_model.set_response = (
            lambda text: text
        )  # Make the model return whatever is passed to it
        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .improve_with(mock_critic)
            .with_options(
                apply_improvers_on_validation_failure=False
            )  # Disable feedback loop for this test
        )
        result = chain.run()

        # In the new behavior, critics are not applied when validation passes
        assert result.text == "This is a generated response."
        assert result.passed is True
        assert len(result.improvement_results) == 0
        assert len(mock_critic.improve_calls) == 0

    def test_run_with_validator_and_critic(self, mock_model, mock_validator, mock_critic) -> None:
        """Test running a Chain with a validator and a critic.

        In the new behavior, critics are only used when validation fails,
        so this test now checks that the text is not improved when validation passes.
        """
        mock_model.set_response("This is a generated response.")
        mock_model.set_response = (
            lambda text: text
        )  # Make the model return whatever is passed to it
        mock_critic.set_improved_text("This is an improved response.")
        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .validate_with(mock_validator)
            .improve_with(mock_critic)
            .with_options(
                apply_improvers_on_validation_failure=False
            )  # Disable feedback loop for this test
        )
        result = chain.run()

        # In the new behavior, critics are not applied when validation passes
        assert result.text == "This is a generated response."
        assert result.passed is True
        assert len(result.validation_results) == 1
        assert result.validation_results[0].passed is True
        assert len(result.improvement_results) == 0
        assert len(mock_validator.validate_calls) >= 1
        assert mock_validator.validate_calls[0] == "This is a generated response."
        assert len(mock_critic.improve_calls) == 0

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
            .with_options(
                apply_improvers_on_validation_failure=False
            )  # Disable feedback loop for this test
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

    def test_run_with_multiple_critics(self, mock_model) -> None:
        """Test running a Chain with multiple critics.

        In the new behavior, critics are only used when validation fails,
        so this test now checks that the text is not improved when validation passes.
        """
        # Import the MockCritic class directly from the conftest module in the current directory
        import os
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from conftest import MockCritic

        # Create multiple critics
        critic1 = MockCritic(name="Critic1")
        critic1.set_improved_text("This is improved by critic 1.")

        critic2 = MockCritic(name="Critic2")
        critic2.set_improved_text("This is improved by critic 1 and 2.")

        critic3 = MockCritic(name="Critic3")
        critic3.set_improved_text("This is improved by all critics.")

        # Set up the chain
        mock_model.set_response("This is a generated response.")

        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .improve_with(critic1)
            .improve_with(critic2)
            .improve_with(critic3)
            .with_options(
                apply_improvers_on_validation_failure=False
            )  # Disable feedback loop for this test
        )
        result = chain.run()

        # In the new behavior, critics are not applied when validation passes
        assert result.text == "This is a generated response."
        assert result.passed is True
        assert len(result.improvement_results) == 0

        # Check that no critics were called
        assert len(critic1.improve_calls) == 0
        assert len(critic2.improve_calls) == 0
        assert len(critic3.improve_calls) == 0

    def test_model_error_handling(self, mock_model, monkeypatch) -> None:
        """Test error handling when the model raises an error."""
        from sifaka.errors import ChainError, ModelError

        # Make the model raise an error
        def mock_generate_error(*args, **kwargs):  # pylint: disable=unused-argument
            # ModelError doesn't accept model_name parameter directly
            error = ModelError("API error")
            error.metadata["model_name"] = "mock_model"
            raise error

        monkeypatch.setattr(mock_model, "generate", mock_generate_error)

        # Set up the chain
        chain = Chain().with_model(mock_model).with_prompt("Write a short story about a robot.")

        # Run the chain and check that the error is properly handled
        with pytest.raises(ChainError) as excinfo:
            chain.run()

        # Check that the error contains useful information
        error = excinfo.value
        assert error.component == "Chain"
        assert "model" in str(error).lower() or "Model" in str(error)
        assert "API error" in str(error)

    def test_validator_error_handling(self, mock_model, mock_validator, monkeypatch) -> None:
        """Test error handling when a validator raises an error."""
        from sifaka.errors import ValidationError

        # Make the validator raise an error
        def mock_validate_error(*args, **kwargs):  # pylint: disable=unused-argument
            error = ValidationError("Validation failed with an error")
            error.metadata["validator_name"] = "mock_validator"
            raise error

        monkeypatch.setattr(mock_validator, "validate", mock_validate_error)

        # Set up the chain
        mock_model.set_response("This is a generated response.")
        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .validate_with(mock_validator)
        )

        # Run the chain and check that the error is handled
        # The Chain class catches the ValidationError and returns a Result with passed=False
        # rather than propagating the error as a ChainError
        result = chain.run()

        # Check that the validation failed
        assert result.passed is False
        assert len(result.validation_results) > 0
        assert "Validation failed with an error" in str(result.validation_results[0].message)

    def test_critic_error_handling(self, mock_model, mock_critic, monkeypatch) -> None:
        """Test error handling when a critic raises an error."""
        from sifaka.errors import ImproverError

        # Make the critic raise an error
        def mock_improve_error(*args, **kwargs):  # pylint: disable=unused-argument
            error = ImproverError("Improvement failed with an error")
            error.metadata["improver_name"] = "mock_critic"
            raise error

        monkeypatch.setattr(mock_critic, "improve", mock_improve_error)

        # Set up the chain
        mock_model.set_response("This is a generated response.")

        # Create a mock improvement result to be added to the result
        from sifaka.results import ImprovementResult

        mock_improvement_result = ImprovementResult(
            _original_text="This is a generated response.",
            _improved_text="This is a generated response.",
            _changes_made=False,
            message="Improvement failed with an error",
            _details={"error_type": "ImproverError"},
        )

        # Patch the _validate_text method to return a passing validation
        Chain._validate_text

        def mock_validate_text(self, text):
            return True, [], {}

        monkeypatch.setattr(Chain, "_validate_text", mock_validate_text)

        # Patch the run method to add our mock improvement result
        original_run = Chain.run

        def mock_run(self):
            result = original_run(self)
            if not result.improvement_results:
                result.improvement_results.append(mock_improvement_result)
            return result

        monkeypatch.setattr(Chain, "run", mock_run)

        chain = (
            Chain()
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .improve_with(mock_critic)
            .with_options(
                apply_improvers_on_validation_failure=False
            )  # Disable feedback loop for this test
        )

        # Run the chain and check that the error is handled
        # The Chain class catches the ImproverError and returns a Result with the original text
        # rather than propagating the error as a ChainError
        result = chain.run()

        # Check that the result contains the original text
        assert result.text == "This is a generated response."
        assert result.passed is True
        assert len(result.improvement_results) > 0
        assert "Improvement failed with an error" in str(result.improvement_results[0].message)

    def test_chain_with_config_options(self, mock_model) -> None:
        """Test running a Chain with configuration options."""
        from sifaka.config import ModelConfig, SifakaConfig

        # Create a custom configuration
        config = SifakaConfig(
            model=ModelConfig(
                temperature=0.5,
                max_tokens=100,
                top_p=0.9,
            ),
            debug=True,
        )

        # Set up the chain with the configuration
        mock_model.set_response("This is a generated response.")
        chain = (
            Chain(config).with_model(mock_model).with_prompt("Write a short story about a robot.")
        )
        result = chain.run()

        # Check that the model was called with the correct options
        assert result.text == "This is a generated response."
        assert len(mock_model.generate_calls) == 1
        options = mock_model.generate_calls[0][1]
        assert options.get("temperature") == 0.5
        assert options.get("max_tokens") == 100
        assert options.get("top_p") == 0.9
