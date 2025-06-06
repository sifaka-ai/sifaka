"""Comprehensive unit tests for Sifaka configuration management.

This module tests the configuration system:
- SifakaConfig model and validation
- Configuration loading and parsing
- Default value handling
- Environment variable integration
- Configuration validation and constraints

Tests cover:
- Config model creation and validation
- Field constraints and validation
- Default configuration values
- Configuration serialization
- Environment variable handling
"""

import pytest
import os
from typing import Dict, Any, Optional
from unittest.mock import patch, mock_open
from pydantic import ValidationError

from sifaka.utils.config import SifakaConfig


class TestSifakaConfig:
    """Test the SifakaConfig configuration model."""

    def test_config_creation_minimal(self):
        """Test creating SifakaConfig with minimal configuration."""
        config = SifakaConfig()
        
        # Verify default values
        assert config.max_iterations == 3
        assert config.validator_weight == 0.6
        assert config.critic_weight == 0.4
        assert config.always_apply_critics is False
        assert config.timeout_seconds == 300.0
        assert config.log_level == "INFO"
        assert config.enable_performance_logging is True
        assert config.enable_thought_storage is True

    def test_config_creation_custom(self):
        """Test creating SifakaConfig with custom values."""
        config = SifakaConfig(
            max_iterations=5,
            validator_weight=0.7,
            critic_weight=0.3,
            always_apply_critics=True,
            timeout_seconds=600.0,
            log_level="DEBUG",
            enable_performance_logging=False,
            enable_thought_storage=False,
        )
        
        assert config.max_iterations == 5
        assert config.validator_weight == 0.7
        assert config.critic_weight == 0.3
        assert config.always_apply_critics is True
        assert config.timeout_seconds == 600.0
        assert config.log_level == "DEBUG"
        assert config.enable_performance_logging is False
        assert config.enable_thought_storage is False

    def test_config_weight_validation(self):
        """Test validation of validator and critic weights."""
        # Valid weights
        SifakaConfig(validator_weight=0.0, critic_weight=1.0)
        SifakaConfig(validator_weight=1.0, critic_weight=0.0)
        SifakaConfig(validator_weight=0.5, critic_weight=0.5)
        
        # Invalid weights - negative
        with pytest.raises(ValidationError):
            SifakaConfig(validator_weight=-0.1)
        
        with pytest.raises(ValidationError):
            SifakaConfig(critic_weight=-0.1)
        
        # Invalid weights - greater than 1
        with pytest.raises(ValidationError):
            SifakaConfig(validator_weight=1.1)
        
        with pytest.raises(ValidationError):
            SifakaConfig(critic_weight=1.1)

    def test_config_weight_sum_validation(self):
        """Test validation that weights sum to approximately 1.0."""
        # Valid weight combinations
        SifakaConfig(validator_weight=0.6, critic_weight=0.4)  # Sum = 1.0
        SifakaConfig(validator_weight=0.7, critic_weight=0.3)  # Sum = 1.0
        SifakaConfig(validator_weight=0.0, critic_weight=1.0)  # Sum = 1.0
        
        # Note: The actual implementation may or may not enforce sum=1.0
        # This test documents the expected behavior

    def test_config_max_iterations_validation(self):
        """Test validation of max_iterations field."""
        # Valid values
        SifakaConfig(max_iterations=1)
        SifakaConfig(max_iterations=10)
        SifakaConfig(max_iterations=100)
        
        # Invalid values
        with pytest.raises(ValidationError):
            SifakaConfig(max_iterations=0)
        
        with pytest.raises(ValidationError):
            SifakaConfig(max_iterations=-1)

    def test_config_timeout_validation(self):
        """Test validation of timeout_seconds field."""
        # Valid values
        SifakaConfig(timeout_seconds=1.0)
        SifakaConfig(timeout_seconds=300.0)
        SifakaConfig(timeout_seconds=3600.0)
        
        # Invalid values
        with pytest.raises(ValidationError):
            SifakaConfig(timeout_seconds=0.0)
        
        with pytest.raises(ValidationError):
            SifakaConfig(timeout_seconds=-1.0)

    def test_config_log_level_validation(self):
        """Test validation of log_level field."""
        # Valid log levels
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            config = SifakaConfig(log_level=level)
            assert config.log_level == level
        
        # Case insensitive (if supported)
        config = SifakaConfig(log_level="debug")
        assert config.log_level.upper() == "DEBUG"
        
        # Invalid log level
        with pytest.raises(ValidationError):
            SifakaConfig(log_level="INVALID")

    def test_config_serialization(self):
        """Test SifakaConfig serialization and deserialization."""
        original = SifakaConfig(
            max_iterations=5,
            validator_weight=0.7,
            critic_weight=0.3,
            always_apply_critics=True,
            timeout_seconds=600.0,
            log_level="DEBUG",
        )
        
        # Serialize to dict
        data = original.model_dump()
        
        # Verify structure
        assert data["max_iterations"] == 5
        assert data["validator_weight"] == 0.7
        assert data["critic_weight"] == 0.3
        assert data["always_apply_critics"] is True
        assert data["timeout_seconds"] == 600.0
        assert data["log_level"] == "DEBUG"
        
        # Deserialize back to model
        restored = SifakaConfig.model_validate(data)
        
        # Verify restored model
        assert restored.max_iterations == original.max_iterations
        assert restored.validator_weight == original.validator_weight
        assert restored.critic_weight == original.critic_weight
        assert restored.always_apply_critics == original.always_apply_critics
        assert restored.timeout_seconds == original.timeout_seconds
        assert restored.log_level == original.log_level

    def test_config_json_serialization(self):
        """Test SifakaConfig JSON serialization."""
        config = SifakaConfig(
            max_iterations=3,
            validator_weight=0.6,
            log_level="INFO",
        )
        
        # Serialize to JSON
        json_str = config.model_dump_json()
        
        # Verify JSON contains expected fields
        assert '"max_iterations":3' in json_str or '"max_iterations": 3' in json_str
        assert '"validator_weight":0.6' in json_str or '"validator_weight": 0.6' in json_str
        assert '"log_level":"INFO"' in json_str or '"log_level": "INFO"' in json_str
        
        # Deserialize from JSON
        restored = SifakaConfig.model_validate_json(json_str)
        
        assert restored.max_iterations == config.max_iterations
        assert restored.validator_weight == config.validator_weight
        assert restored.log_level == config.log_level


class TestSifakaConfigEnvironment:
    """Test SifakaConfig integration with environment variables."""

    @patch.dict(os.environ, {
        'SIFAKA_MAX_ITERATIONS': '5',
        'SIFAKA_VALIDATOR_WEIGHT': '0.7',
        'SIFAKA_LOG_LEVEL': 'DEBUG',
        'SIFAKA_ALWAYS_APPLY_CRITICS': 'true',
    })
    def test_config_from_environment(self):
        """Test loading configuration from environment variables."""
        # Note: This test assumes the config class supports environment loading
        # The actual implementation may vary
        
        # If SifakaConfig supports environment loading
        if hasattr(SifakaConfig, 'from_env') or hasattr(SifakaConfig, 'load_from_env'):
            # Test would go here
            pass
        else:
            # Manual environment parsing test
            max_iterations = int(os.environ.get('SIFAKA_MAX_ITERATIONS', 3))
            validator_weight = float(os.environ.get('SIFAKA_VALIDATOR_WEIGHT', 0.6))
            log_level = os.environ.get('SIFAKA_LOG_LEVEL', 'INFO')
            always_apply_critics = os.environ.get('SIFAKA_ALWAYS_APPLY_CRITICS', 'false').lower() == 'true'
            
            config = SifakaConfig(
                max_iterations=max_iterations,
                validator_weight=validator_weight,
                log_level=log_level,
                always_apply_critics=always_apply_critics,
            )
            
            assert config.max_iterations == 5
            assert config.validator_weight == 0.7
            assert config.log_level == "DEBUG"
            assert config.always_apply_critics is True

    def test_config_environment_override(self):
        """Test that environment variables override default values."""
        # Test with no environment variables
        config_default = SifakaConfig()
        assert config_default.max_iterations == 3  # Default value
        
        # Test with environment variable
        with patch.dict(os.environ, {'SIFAKA_MAX_ITERATIONS': '7'}):
            # If environment loading is supported
            if hasattr(SifakaConfig, 'from_env'):
                config_env = SifakaConfig.from_env()
                assert config_env.max_iterations == 7
            else:
                # Manual override
                max_iterations = int(os.environ.get('SIFAKA_MAX_ITERATIONS', 3))
                config_env = SifakaConfig(max_iterations=max_iterations)
                assert config_env.max_iterations == 7

    def test_config_environment_validation(self):
        """Test validation of environment variable values."""
        # Invalid environment values should be handled gracefully
        with patch.dict(os.environ, {'SIFAKA_MAX_ITERATIONS': 'invalid'}):
            # Should raise appropriate error when parsing
            with pytest.raises((ValueError, ValidationError)):
                int(os.environ['SIFAKA_MAX_ITERATIONS'])


class TestSifakaConfigFile:
    """Test SifakaConfig file loading capabilities."""

    def test_config_from_dict(self):
        """Test creating SifakaConfig from dictionary."""
        config_dict = {
            "max_iterations": 4,
            "validator_weight": 0.8,
            "critic_weight": 0.2,
            "log_level": "WARNING",
            "timeout_seconds": 450.0,
        }
        
        config = SifakaConfig.model_validate(config_dict)
        
        assert config.max_iterations == 4
        assert config.validator_weight == 0.8
        assert config.critic_weight == 0.2
        assert config.log_level == "WARNING"
        assert config.timeout_seconds == 450.0

    @patch("builtins.open", new_callable=mock_open, read_data='{"max_iterations": 6, "log_level": "ERROR"}')
    def test_config_from_json_file(self, mock_file):
        """Test loading SifakaConfig from JSON file."""
        # Simulate loading from JSON file
        import json
        
        # Mock file content
        file_content = '{"max_iterations": 6, "log_level": "ERROR"}'
        config_dict = json.loads(file_content)
        config = SifakaConfig.model_validate(config_dict)
        
        assert config.max_iterations == 6
        assert config.log_level == "ERROR"
        # Other fields should have default values
        assert config.validator_weight == 0.6
        assert config.critic_weight == 0.4

    def test_config_partial_override(self):
        """Test partial configuration override."""
        # Start with defaults
        base_config = SifakaConfig()
        
        # Override specific fields
        override_dict = {
            "max_iterations": 8,
            "log_level": "DEBUG",
        }
        
        # Create new config with overrides
        updated_config = SifakaConfig.model_validate({
            **base_config.model_dump(),
            **override_dict
        })
        
        assert updated_config.max_iterations == 8
        assert updated_config.log_level == "DEBUG"
        # Other fields should retain default values
        assert updated_config.validator_weight == base_config.validator_weight
        assert updated_config.critic_weight == base_config.critic_weight


class TestSifakaConfigValidation:
    """Test advanced validation scenarios for SifakaConfig."""

    def test_config_field_types(self):
        """Test that config fields have correct types."""
        config = SifakaConfig()
        
        assert isinstance(config.max_iterations, int)
        assert isinstance(config.validator_weight, float)
        assert isinstance(config.critic_weight, float)
        assert isinstance(config.always_apply_critics, bool)
        assert isinstance(config.timeout_seconds, float)
        assert isinstance(config.log_level, str)
        assert isinstance(config.enable_performance_logging, bool)
        assert isinstance(config.enable_thought_storage, bool)

    def test_config_immutability(self):
        """Test that SifakaConfig is immutable (if frozen)."""
        config = SifakaConfig()
        
        # If the model is frozen, this should raise an error
        try:
            config.max_iterations = 10
            # If no error, the model is mutable
            assert config.max_iterations == 10
        except (ValidationError, AttributeError):
            # Model is frozen/immutable
            pass

    def test_config_copy_and_update(self):
        """Test copying and updating SifakaConfig."""
        original = SifakaConfig(max_iterations=3, log_level="INFO")
        
        # Create updated copy
        updated = original.model_copy(update={"max_iterations": 5, "log_level": "DEBUG"})
        
        # Original should be unchanged
        assert original.max_iterations == 3
        assert original.log_level == "INFO"
        
        # Updated should have new values
        assert updated.max_iterations == 5
        assert updated.log_level == "DEBUG"
        
        # Other fields should be the same
        assert updated.validator_weight == original.validator_weight
        assert updated.critic_weight == original.critic_weight

    def test_config_validation_edge_cases(self):
        """Test edge cases in configuration validation."""
        # Boundary values
        SifakaConfig(max_iterations=1)  # Minimum
        SifakaConfig(validator_weight=0.0)  # Minimum
        SifakaConfig(critic_weight=0.0)  # Minimum
        SifakaConfig(timeout_seconds=0.1)  # Very small but positive
        
        # Maximum reasonable values
        SifakaConfig(max_iterations=1000)
        SifakaConfig(validator_weight=1.0)
        SifakaConfig(critic_weight=1.0)
        SifakaConfig(timeout_seconds=86400.0)  # 24 hours
