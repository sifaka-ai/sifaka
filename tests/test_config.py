"""
Tests for the configuration utilities.

This module contains tests for the configuration utilities in the Sifaka framework.
"""

import json
import os
import tempfile

import pytest

from sifaka.errors import ConfigurationError
from sifaka.utils.config import load_config, save_config


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_from_path(self) -> None:
        """Test loading configuration from a specified path."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {"key": "value", "nested": {"key": "value"}}
            json.dump(config, f)
            config_path = f.name

        try:
            # Load the config
            loaded_config = load_config(config_path)
            assert loaded_config == config
        finally:
            # Clean up
            os.unlink(config_path)

    def test_load_nonexistent_path(self) -> None:
        """Test loading configuration from a nonexistent path."""
        # Use a path that doesn't exist
        config_path = "/path/that/does/not/exist.json"

        # Load the config (should return empty dict)
        loaded_config = load_config(config_path)
        assert loaded_config == {}

    def test_load_invalid_json(self) -> None:
        """Test loading configuration from a file with invalid JSON."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("This is not valid JSON")
            config_path = f.name

        try:
            # Load the config (should raise ConfigurationError)
            with pytest.raises(ConfigurationError):
                load_config(config_path)
        finally:
            # Clean up
            os.unlink(config_path)

    def test_load_default_locations(self, monkeypatch) -> None:
        """Test loading configuration from default locations."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {"key": "value"}
            json.dump(config, f)
            config_path = f.name

        try:
            # Mock os.path.exists to return True only for our config path
            def mock_exists(path: str) -> bool:
                return path == config_path

            monkeypatch.setattr(os.path, "exists", mock_exists)

            # Mock os.getcwd to return a directory containing our config file
            def mock_getcwd() -> str:
                return os.path.dirname(config_path)

            monkeypatch.setattr(os, "getcwd", mock_getcwd)

            # Mock os.path.join to return our config path
            def mock_join(directory: str, filename: str) -> str:
                if filename == "sifaka_config.json":
                    return config_path
                return os.path.join(directory, filename)

            monkeypatch.setattr(os.path, "join", mock_join)

            # Load the config
            loaded_config = load_config()
            assert loaded_config == config
        finally:
            # Clean up
            os.unlink(config_path)


class TestSaveConfig:
    """Tests for the save_config function."""

    def test_save_config(self) -> None:
        """Test saving configuration to a file."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            config = {"key": "value", "nested": {"key": "value"}}

            # Save the config
            save_config(config, config_path)

            # Verify the config was saved correctly
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                assert loaded_config == config

    def test_save_config_creates_directory(self) -> None:
        """Test saving configuration to a file in a nonexistent directory."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "subdir", "config.json")
            config = {"key": "value"}

            # Save the config (should create the directory)
            save_config(config, config_path)

            # Verify the directory was created and the config was saved
            assert os.path.exists(os.path.dirname(config_path))
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                assert loaded_config == config

    def test_save_config_permission_error(self, monkeypatch) -> None:
        """Test saving configuration to a file with permission error."""

        # Mock open to raise a permission error
        def mock_open(*args, **kwargs):
            raise PermissionError("Permission denied")

        monkeypatch.setattr("builtins.open", mock_open)

        # Try to save the config (should raise ConfigurationError)
        with pytest.raises(ConfigurationError):
            save_config({"key": "value"}, "/path/to/config.json")
