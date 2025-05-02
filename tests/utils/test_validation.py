"""Test module for sifaka.utils.validation."""

import pytest
from sifaka.utils.validation import validate_config


class TestValidateConfig:
    """Tests for the validate_config function."""

    def test_valid_config(self):
        """Test validate_config with a valid configuration."""
        config = {
            "name": "test_config",
            "description": "A test configuration",
            "params": {"param1": "value1", "param2": "value2"},
        }
        # Should not raise an exception
        validate_config(config)

    def test_valid_config_without_params(self):
        """Test validate_config with a valid configuration without params."""
        config = {
            "name": "test_config",
            "description": "A test configuration",
        }
        # Should not raise an exception
        validate_config(config)

    def test_non_dict_config(self):
        """Test validate_config with a non-dictionary input."""
        config = "not a dictionary"
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            validate_config(config)

    def test_missing_name(self):
        """Test validate_config with a configuration missing a name."""
        config = {
            "description": "A test configuration",
        }
        with pytest.raises(ValueError, match="Configuration must have a name"):
            validate_config(config)

    def test_empty_name(self):
        """Test validate_config with a configuration with an empty name."""
        config = {
            "name": "",
            "description": "A test configuration",
        }
        with pytest.raises(ValueError, match="Configuration must have a name"):
            validate_config(config)

    def test_missing_description(self):
        """Test validate_config with a configuration missing a description."""
        config = {
            "name": "test_config",
        }
        with pytest.raises(ValueError, match="Configuration must have a description"):
            validate_config(config)

    def test_empty_description(self):
        """Test validate_config with a configuration with an empty description."""
        config = {
            "name": "test_config",
            "description": "",
        }
        with pytest.raises(ValueError, match="Configuration must have a description"):
            validate_config(config)

    def test_invalid_params(self):
        """Test validate_config with invalid params (not a dictionary)."""
        config = {
            "name": "test_config",
            "description": "A test configuration",
            "params": "not a dictionary",
        }
        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            validate_config(config)