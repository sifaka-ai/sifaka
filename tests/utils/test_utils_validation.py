"""
Tests for validation utilities.
"""

import pytest
from sifaka.utils.validation import validate_config


class TestValidationUtils:
    """Tests for validation utilities."""

    def test_validate_config_valid(self):
        """Test validate_config with valid configuration."""
        config = {
            "name": "test_config",
            "description": "A test configuration"
        }

        # Should not raise an exception
        validate_config(config)

    def test_validate_config_with_params(self):
        """Test validate_config with valid params."""
        config = {
            "name": "test_config",
            "description": "A test configuration",
            "params": {
                "param1": "value1",
                "param2": 42
            }
        }

        # Should not raise an exception
        validate_config(config)

    def test_validate_config_not_dict(self):
        """Test validate_config with non-dictionary input."""
        config = "not a dictionary"

        with pytest.raises(ValueError) as excinfo:
            validate_config(config)

        assert "Configuration must be a dictionary" in str(excinfo.value)

    def test_validate_config_no_name(self):
        """Test validate_config with missing name."""
        config = {
            "description": "A test configuration"
        }

        with pytest.raises(ValueError) as excinfo:
            validate_config(config)

        assert "Configuration must have a name" in str(excinfo.value)

    def test_validate_config_no_description(self):
        """Test validate_config with missing description."""
        config = {
            "name": "test_config"
        }

        with pytest.raises(ValueError) as excinfo:
            validate_config(config)

        assert "Configuration must have a description" in str(excinfo.value)

    def test_validate_config_invalid_params(self):
        """Test validate_config with invalid params type."""
        config = {
            "name": "test_config",
            "description": "A test configuration",
            "params": "not a dictionary"
        }

        with pytest.raises(ValueError) as excinfo:
            validate_config(config)

        assert "Parameters must be a dictionary" in str(excinfo.value)

    def test_validate_config_empty_name(self):
        """Test validate_config with empty name."""
        config = {
            "name": "",
            "description": "A test configuration"
        }

        with pytest.raises(ValueError) as excinfo:
            validate_config(config)

        assert "Configuration must have a name" in str(excinfo.value)

    def test_validate_config_empty_description(self):
        """Test validate_config with empty description."""
        config = {
            "name": "test_config",
            "description": ""
        }

        with pytest.raises(ValueError) as excinfo:
            validate_config(config)

        assert "Configuration must have a description" in str(excinfo.value)