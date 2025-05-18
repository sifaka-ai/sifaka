"""
Tests for the GuardrailsValidator.

This module contains tests for the GuardrailsValidator in Sifaka.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from sifaka.errors import ValidationError


# Mock the guardrails module for testing
class MockGuardrailsModule:
    """Mock for the guardrails module."""

    class Guard:
        """Mock for the Guard class."""

        def __init__(self):
            self.validators = []

        def use(self, validator_class, **kwargs):
            """Mock for the use method."""
            self.validators.append((validator_class, kwargs))
            return self

        def parse(self, text, metadata=None):
            """Mock for the parse method."""
            result = MagicMock()
            result.validation_passed = True
            result.validation_errors = []
            result.fixed_output = None
            return result

    @staticmethod
    def install(validator_path):
        """Mock for the install function."""
        # Extract validator name from path
        validator_name = validator_path.split("/")[-1]

        # Create a mock module with a validator class
        mock_module = MagicMock()

        # Convert snake_case to CamelCase for the class name
        class_name = "".join(word.title() for word in validator_name.split("_"))
        setattr(mock_module, class_name, MagicMock())

        return mock_module


class TestGuardrailsValidator:
    """Tests for the GuardrailsValidator class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Patch the _load_guardrails method to avoid actual imports
        with patch.object(GuardrailsValidator, "_load_guardrails", return_value=MagicMock()):
            validator = GuardrailsValidator()

            assert validator.name == "guardrails_validator"
            assert validator.description == "Validates text using GuardrailsAI"
            assert validator._initialized is False
            assert validator._guard is None
            assert validator._validators_config is None
            assert validator._validator_args_config == {}

    def test_init_with_both_guard_and_validators(self):
        """Test initialization with both guard and validators."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Create a mock guard
        mock_guard = MagicMock()

        # Try to initialize with both guard and validators
        with pytest.raises(ValidationError) as excinfo:
            GuardrailsValidator(guard=mock_guard, validators=["regex_match"])

        error = excinfo.value
        assert "Cannot provide both a guard and validators" in str(error)
        assert error.component == "GuardrailsValidator"
        assert error.operation == "initialization"
        # The error message itself contains the suggestion
        assert "Choose one approach" in str(error)

    def test_init_with_guard(self):
        """Test initialization with a pre-configured guard."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Create a mock guard
        mock_guard = MagicMock()

        # Initialize with the guard
        validator = GuardrailsValidator(guard=mock_guard)

        assert validator._guard_config == mock_guard
        assert validator._validators_config is None

    def test_init_with_validators(self):
        """Test initialization with a list of validators."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Initialize with validators
        validator = GuardrailsValidator(
            validators=["regex_match", "profanity_free"],
            validator_args={
                "regex_match": {"pattern": r"\d+"},
                "profanity_free": {"threshold": 0.8},
            },
        )

        assert validator._validators_config == ["regex_match", "profanity_free"]
        assert validator._validator_args_config == {
            "regex_match": {"pattern": r"\d+"},
            "profanity_free": {"threshold": 0.8},
        }

    def test_init_with_api_key(self):
        """Test initialization with an API key."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Initialize with an API key
        validator = GuardrailsValidator(api_key="test-api-key")

        assert validator._api_key == "test-api-key"

    def test_init_with_api_key_from_env(self):
        """Test initialization with an API key from the environment."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Set the API key in the environment
        with patch.dict(os.environ, {"GUARDRAILS_API_KEY": "env-test-key"}):
            validator = GuardrailsValidator()

            assert validator._api_key == "env-test-key"

    def test_load_guardrails_success(self):
        """Test successful loading of the guardrails module."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Create a mock guardrails module
        mock_guardrails = MagicMock()

        # Patch importlib.import_module to return the mock
        with patch("importlib.import_module", return_value=mock_guardrails):
            validator = GuardrailsValidator()
            result = validator._load_guardrails()

            assert result == mock_guardrails

    def test_load_guardrails_import_error(self):
        """Test handling of ImportError when loading guardrails."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Patch importlib.import_module to raise an ImportError
        with patch(
            "importlib.import_module",
            side_effect=ImportError("No module named 'guardrails'"),
        ):
            validator = GuardrailsValidator()

            with pytest.raises(ImportError) as excinfo:
                validator._load_guardrails()

            assert "guardrails-ai package is required" in str(excinfo.value)
            assert "pip install guardrails-ai" in str(excinfo.value)

    def test_initialize_with_guard_config(self):
        """Test initialization with a pre-configured guard."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Create a mock guard
        mock_guard = MagicMock()

        # Patch _load_guardrails to return a mock
        with patch.object(GuardrailsValidator, "_load_guardrails", return_value=MagicMock()):
            # Initialize with the guard
            validator = GuardrailsValidator(guard=mock_guard)

            # Initialize the validator
            validator._initialize()

            # Check that the guard was set
            assert validator._guard == mock_guard
            assert validator._initialized is True

    @patch("importlib.import_module", return_value=MockGuardrailsModule())
    def test_initialize_with_validators(self, mock_import):
        """Test initialization with validators."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Create a mock Guard class that properly tracks validators
        mock_guard = MagicMock()
        mock_guard.validators = []

        # Create a use method that adds validators to the list
        def mock_use(validator_class, **kwargs):
            mock_guard.validators.append((validator_class, kwargs))
            return mock_guard

        mock_guard.use = mock_use

        # Patch the Guard class to return our mock
        with patch.object(MockGuardrailsModule, "Guard", return_value=mock_guard):
            # Initialize with validators
            validator = GuardrailsValidator(
                validators=["regex_match", "profanity_free"],
                validator_args={
                    "regex_match": {"pattern": r"\d+"},
                    "profanity_free": {"threshold": 0.8},
                },
            )

            # Initialize the validator
            validator._initialize()

            # Check that the guard was created and validators were added
            assert validator._initialized is True
            assert validator._guard is not None
            assert len(validator._guard.validators) == 2

    def test_to_camel_case(self):
        """Test the _to_camel_case method."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        validator = GuardrailsValidator()

        assert validator._to_camel_case("regex_match") == "RegexMatch"
        assert validator._to_camel_case("profanity_free") == "ProfanityFree"
        assert validator._to_camel_case("llm_guard") == "LlmGuard"

    @patch("importlib.import_module", return_value=MockGuardrailsModule())
    def test_validate_empty_text(self, mock_import):
        """Test validation of empty text."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Initialize with validators
        validator = GuardrailsValidator(validators=["regex_match"])

        # Validate empty text
        result = validator._validate("")

        # Check the result
        assert result.passed is False
        assert "Input text is empty" in result.message
        assert result._details["input_length"] == 0

    @patch("importlib.import_module", return_value=MockGuardrailsModule())
    def test_validate_success(self, mock_import):
        """Test successful validation."""
        # Import here to avoid early import errors
        from sifaka.validators.guardrails import GuardrailsValidator

        # Initialize with validators
        validator = GuardrailsValidator(validators=["regex_match"])

        # Validate text
        result = validator._validate("Test text")

        # Check the result
        assert result.passed is True
        assert "passed GuardrailsAI validation" in result.message
        assert result._details["input_length"] == len("Test text")
        assert result._details["validation_passed"] is True
