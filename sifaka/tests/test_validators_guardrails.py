"""Tests for the GuardrailsAI validator module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import asyncio
import sys

from sifaka.validators.base import ValidatorConfig
from sifaka.core.models import SifakaResult


class TestGuardrailsValidatorImportError:
    """Test GuardrailsValidator when GuardrailsAI is not available."""

    def test_init_without_guardrails(self):
        """Test that ImportError is raised when GuardrailsAI is not installed."""
        # Mock the import to fail
        with patch.dict("sys.modules", {"guardrails": None, "guardrails.hub": None}):
            # Force reload to get the mocked version
            if "sifaka.validators.guardrails" in sys.modules:
                del sys.modules["sifaka.validators.guardrails"]

            from sifaka.validators.guardrails import GuardrailsValidator, HAS_GUARDRAILS

            if not HAS_GUARDRAILS:
                with pytest.raises(ImportError) as exc_info:
                    GuardrailsValidator(["toxic-language"])

                assert "GuardrailsAI is not installed" in str(exc_info.value)
                assert "pip install sifaka[guardrails]" in str(exc_info.value)


class TestGuardrailsValidatorMocked:
    """Test GuardrailsValidator with mocked GuardrailsAI."""

    @pytest.fixture
    def mock_guardrails(self):
        """Mock the guardrails module."""
        mock_gr = MagicMock()
        mock_install = MagicMock()

        with patch.dict(
            "sys.modules",
            {"guardrails": mock_gr, "guardrails.hub": MagicMock(install=mock_install)},
        ):
            # Force reload with mocks
            if "sifaka.validators.guardrails" in sys.modules:
                del sys.modules["sifaka.validators.guardrails"]

            with patch("sifaka.validators.guardrails.HAS_GUARDRAILS", True):
                from sifaka.validators.guardrails import GuardrailsValidator

                yield GuardrailsValidator, mock_gr, mock_install

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Test", final_text="Test")

    def test_init_default(self, mock_guardrails):
        """Test initialization with default values."""
        GuardrailsValidator, _, _ = mock_guardrails
        validators = ["toxic-language", "detect-pii"]
        validator = GuardrailsValidator(validators)

        assert validator.validators == validators
        assert validator.on_fail == "fix"
        assert validator._installed_validators == set()
        assert validator._guard is None
        assert isinstance(validator._lock, asyncio.Lock)

    def test_init_with_config(self, mock_guardrails):
        """Test initialization with custom config."""
        GuardrailsValidator, _, _ = mock_guardrails
        config = ValidatorConfig(enabled=False)
        validator = GuardrailsValidator(["toxic-language"], config=config)

        assert validator.config == config

    def test_name_property(self, mock_guardrails):
        """Test name property."""
        GuardrailsValidator, _, _ = mock_guardrails
        validator = GuardrailsValidator(["toxic-language", "detect-pii"])
        assert validator.name == "guardrails[toxic-language, detect-pii]"

    def test_ensure_validators_installed(self, mock_guardrails):
        """Test validator installation."""
        GuardrailsValidator, _, mock_install = mock_guardrails
        validator = GuardrailsValidator(["toxic-language", "detect-pii"])
        validator._ensure_validators_installed()

        assert mock_install.call_count == 2
        assert validator._installed_validators == {"toxic-language", "detect-pii"}

    def test_ensure_validators_installed_failure(self, mock_guardrails):
        """Test validator installation failure."""
        GuardrailsValidator, _, mock_install = mock_guardrails
        mock_install.side_effect = Exception("Installation failed")

        validator = GuardrailsValidator(["bad-validator"])

        with pytest.raises(ValueError) as exc_info:
            validator._ensure_validators_installed()

        assert "Failed to install GuardrailsAI validator 'bad-validator'" in str(
            exc_info.value
        )

    def test_build_guard(self, mock_guardrails):
        """Test building guard."""
        GuardrailsValidator, mock_gr, mock_install = mock_guardrails

        mock_guard = MagicMock()
        mock_gr.Guard.from_rail_string.return_value = mock_guard

        validator = GuardrailsValidator(["toxic-language"])
        validator._build_guard()

        assert validator._guard == mock_guard
        assert mock_gr.Guard.from_rail_string.called

        # Check rail spec
        rail_spec = mock_gr.Guard.from_rail_string.call_args[0][0]
        assert '<validator name="toxic-language" on-fail="fix"/>' in rail_spec

    @pytest.mark.asyncio
    async def test_perform_validation_success(self, mock_guardrails, sample_result):
        """Test successful validation."""
        GuardrailsValidator, mock_gr, mock_install = mock_guardrails

        # Setup mock guard
        mock_guard = MagicMock()
        mock_output = MagicMock()
        mock_output.validation_passed = True
        mock_guard.validate.return_value = mock_output
        mock_gr.Guard.from_rail_string.return_value = mock_guard

        validator = GuardrailsValidator(["toxic-language"])
        passed, score, details = await validator._perform_validation(
            "Safe text", sample_result
        )

        assert passed is True
        assert score == 1.0
        assert details == "All GuardrailsAI validators passed"

    @pytest.mark.asyncio
    async def test_perform_validation_failure(self, mock_guardrails, sample_result):
        """Test validation failure."""
        GuardrailsValidator, mock_gr, mock_install = mock_guardrails

        # Setup mock guard with failures
        mock_guard = MagicMock()
        mock_output = MagicMock()
        mock_output.validation_passed = False

        # Mock validator logs
        fail1 = MagicMock()
        fail1.passed = False
        fail1.validator_name = "toxic-language"
        fail1.failure_reason = "Toxic content detected"

        mock_output.validator_logs = [fail1]
        mock_guard.validate.return_value = mock_output
        mock_gr.Guard.from_rail_string.return_value = mock_guard

        validator = GuardrailsValidator(["toxic-language"])
        passed, score, details = await validator._perform_validation(
            "Bad text", sample_result
        )

        assert passed is False
        assert score == 0.0
        assert "toxic-language: Toxic content detected" in details

    @pytest.mark.asyncio
    async def test_perform_validation_partial_failure(
        self, mock_guardrails, sample_result
    ):
        """Test validation with partial failures."""
        GuardrailsValidator, mock_gr, mock_install = mock_guardrails

        # Setup mock guard with mixed results
        mock_guard = MagicMock()
        mock_output = MagicMock()
        mock_output.validation_passed = False

        # Mock validator logs
        fail1 = MagicMock()
        fail1.passed = False
        fail1.validator_name = "toxic-language"
        fail1.failure_reason = "Toxic"

        pass1 = MagicMock()
        pass1.passed = True
        pass1.validator_name = "detect-pii"

        mock_output.validator_logs = [fail1, pass1]
        mock_guard.validate.return_value = mock_output
        mock_gr.Guard.from_rail_string.return_value = mock_guard

        validator = GuardrailsValidator(["toxic-language", "detect-pii"])
        passed, score, details = await validator._perform_validation(
            "Mixed text", sample_result
        )

        assert passed is False
        assert score == 0.5  # 1 out of 2 passed
        assert "toxic-language: Toxic" in details

    @pytest.mark.asyncio
    async def test_validate_full_flow(self, mock_guardrails, sample_result):
        """Test full validation flow."""
        GuardrailsValidator, mock_gr, mock_install = mock_guardrails

        # Setup mock guard
        mock_guard = MagicMock()
        mock_output = MagicMock()
        mock_output.validation_passed = True
        mock_guard.validate.return_value = mock_output
        mock_gr.Guard.from_rail_string.return_value = mock_guard

        validator = GuardrailsValidator(["toxic-language"])
        result = await validator.validate("Safe text", sample_result)

        assert result.passed is True
        assert result.score == 1.0
        assert result.validator == "guardrails[toxic-language]"

    def test_build_guard_idempotent(self, mock_guardrails):
        """Test that guard is only built once."""
        GuardrailsValidator, mock_gr, mock_install = mock_guardrails

        mock_guard = MagicMock()
        mock_gr.Guard.from_rail_string.return_value = mock_guard

        validator = GuardrailsValidator(["toxic-language"])

        # Build guard twice
        validator._build_guard()
        validator._build_guard()

        # Should only be called once
        assert mock_gr.Guard.from_rail_string.call_count == 1

    @pytest.mark.asyncio
    async def test_perform_validation_no_guard(self, mock_guardrails, sample_result):
        """Test validation when guard is not initialized properly."""
        GuardrailsValidator, mock_gr, mock_install = mock_guardrails

        validator = GuardrailsValidator(["toxic-language"])
        validator._guard = None

        # Mock _build_guard to not set guard
        validator._build_guard = Mock()

        with pytest.raises(RuntimeError) as exc_info:
            await validator._perform_validation("Text", sample_result)

        assert "Guard not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_perform_validation_empty_logs(self, mock_guardrails, sample_result):
        """Test validation with no validator logs."""
        GuardrailsValidator, mock_gr, mock_install = mock_guardrails

        # Setup mock guard
        mock_guard = MagicMock()
        mock_output = MagicMock()
        mock_output.validation_passed = False
        mock_output.validator_logs = []
        mock_guard.validate.return_value = mock_output
        mock_gr.Guard.from_rail_string.return_value = mock_guard

        validator = GuardrailsValidator([])
        passed, score, details = await validator._perform_validation(
            "Text", sample_result
        )

        assert passed is False
        assert score == 0.0
        assert details == "Validation failed"
