"""Configuration management for Sifaka.

This module provides comprehensive configuration management with support for:
- Environment variables
- File-based configuration (YAML/JSON)
- Default values and validation
- Type safety and documentation

Example Usage:
    ```python
    # Load from environment variables
    config = SifakaConfig.from_env()
    
    # Load from file
    config = SifakaConfig.from_file("config.yaml")
    
    # Use defaults
    config = SifakaConfig()
    
    # Custom configuration
    config = SifakaConfig(
        default_model="anthropic:claude-3-sonnet",
        max_iterations=5,
        enable_critics=True
    )
    ```
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from sifaka.utils.errors import ConfigurationError


@dataclass
class SifakaConfig:
    """Configuration for Sifaka engine.
    
    This class provides comprehensive configuration management with support
    for environment variables, file-based configuration, and validation.
    
    Attributes:
        default_model: Default model for text generation
        max_iterations: Maximum number of iterations per thought
        enable_critics: Whether to enable critic feedback
        enable_validators: Whether to enable validation
        timeout_seconds: Timeout for model API calls
        retry_attempts: Number of retry attempts for failed operations
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        storage_backend: Storage backend type (memory, file, redis)
        storage_config: Storage-specific configuration
        critic_models: Model configuration for specific critics
        validator_config: Configuration for validators
    """
    
    # Core configuration
    default_model: str = "openai:gpt-4o-mini"
    max_iterations: int = 3
    enable_critics: bool = True
    enable_validators: bool = True
    
    # Performance and reliability
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    log_level: str = "INFO"
    
    # Storage configuration
    storage_backend: str = "memory"
    storage_config: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced configuration
    critic_models: Dict[str, str] = field(default_factory=dict)
    validator_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values."""
        # Validate max_iterations
        if self.max_iterations < 1:
            raise ConfigurationError(
                "max_iterations must be at least 1",
                config_key="max_iterations",
                config_value=self.max_iterations,
                suggestions=[
                    "Set max_iterations to a positive integer",
                    "Use default value of 3 for most use cases",
                ]
            )
        
        if self.max_iterations > 20:
            raise ConfigurationError(
                "max_iterations cannot exceed 20 for safety",
                config_key="max_iterations", 
                config_value=self.max_iterations,
                suggestions=[
                    "Reduce max_iterations to 10 or less",
                    "Consider if your use case really needs many iterations",
                ]
            )
        
        # Validate timeout
        if self.timeout_seconds <= 0:
            raise ConfigurationError(
                "timeout_seconds must be positive",
                config_key="timeout_seconds",
                config_value=self.timeout_seconds,
                suggestions=[
                    "Set timeout_seconds to a positive number",
                    "Use 30.0 seconds as a reasonable default",
                ]
            )
        
        # Validate retry attempts
        if self.retry_attempts < 0:
            raise ConfigurationError(
                "retry_attempts cannot be negative",
                config_key="retry_attempts",
                config_value=self.retry_attempts,
                suggestions=[
                    "Set retry_attempts to 0 or positive integer",
                    "Use 3 retries as a reasonable default",
                ]
            )
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigurationError(
                f"Invalid log_level: {self.log_level}",
                config_key="log_level",
                config_value=self.log_level,
                suggestions=[
                    f"Use one of: {', '.join(valid_log_levels)}",
                    "Use 'INFO' for normal operation",
                ]
            )
        
        # Validate storage backend
        valid_backends = ["memory", "file", "redis"]
        if self.storage_backend not in valid_backends:
            raise ConfigurationError(
                f"Invalid storage_backend: {self.storage_backend}",
                config_key="storage_backend",
                config_value=self.storage_backend,
                suggestions=[
                    f"Use one of: {', '.join(valid_backends)}",
                    "Use 'memory' for development, 'redis' for production",
                ]
            )
    
    @classmethod
    def from_env(cls, prefix: str = "SIFAKA_") -> "SifakaConfig":
        """Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix (default: "SIFAKA_")
            
        Returns:
            SifakaConfig instance with values from environment
            
        Example:
            ```bash
            export SIFAKA_DEFAULT_MODEL="anthropic:claude-3-sonnet"
            export SIFAKA_MAX_ITERATIONS="5"
            export SIFAKA_ENABLE_CRITICS="true"
            ```
        """
        def get_env_bool(key: str, default: bool) -> bool:
            """Get boolean from environment variable."""
            value = os.getenv(key)
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes", "on")
        
        def get_env_int(key: str, default: int) -> int:
            """Get integer from environment variable."""
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                raise ConfigurationError(
                    f"Invalid integer value for {key}: {value}",
                    config_key=key,
                    config_value=value,
                )
        
        def get_env_float(key: str, default: float) -> float:
            """Get float from environment variable."""
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return float(value)
            except ValueError:
                raise ConfigurationError(
                    f"Invalid float value for {key}: {value}",
                    config_key=key,
                    config_value=value,
                )
        
        def get_env_dict(key: str, default: Dict[str, Any]) -> Dict[str, Any]:
            """Get dictionary from environment variable (JSON format)."""
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise ConfigurationError(
                    f"Invalid JSON value for {key}: {value}",
                    config_key=key,
                    config_value=value,
                    context={"json_error": str(e)},
                )
        
        return cls(
            default_model=os.getenv(f"{prefix}DEFAULT_MODEL", cls.default_model),
            max_iterations=get_env_int(f"{prefix}MAX_ITERATIONS", cls.max_iterations),
            enable_critics=get_env_bool(f"{prefix}ENABLE_CRITICS", cls.enable_critics),
            enable_validators=get_env_bool(f"{prefix}ENABLE_VALIDATORS", cls.enable_validators),
            timeout_seconds=get_env_float(f"{prefix}TIMEOUT_SECONDS", cls.timeout_seconds),
            retry_attempts=get_env_int(f"{prefix}RETRY_ATTEMPTS", cls.retry_attempts),
            log_level=os.getenv(f"{prefix}LOG_LEVEL", cls.log_level),
            storage_backend=os.getenv(f"{prefix}STORAGE_BACKEND", cls.storage_backend),
            storage_config=get_env_dict(f"{prefix}STORAGE_CONFIG", {}),
            critic_models=get_env_dict(f"{prefix}CRITIC_MODELS", {}),
            validator_config=get_env_dict(f"{prefix}VALIDATOR_CONFIG", {}),
        )
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SifakaConfig":
        """Load configuration from YAML or JSON file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            SifakaConfig instance with values from file
            
        Raises:
            ConfigurationError: If file cannot be read or parsed
        """
        path = Path(path)
        
        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {path}",
                config_key="file_path",
                config_value=str(path),
                suggestions=[
                    "Check that the file path is correct",
                    "Create the configuration file",
                    "Use SifakaConfig.from_env() or defaults instead",
                ]
            )
        
        try:
            content = path.read_text()
        except Exception as e:
            raise ConfigurationError(
                f"Cannot read configuration file: {path}",
                config_key="file_path",
                config_value=str(path),
                context={"error": str(e)},
            )
        
        # Determine file format and parse
        if path.suffix.lower() in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ConfigurationError(
                    "YAML support not available. Install PyYAML: pip install pyyaml",
                    suggestions=["Install PyYAML: pip install pyyaml"]
                )
            try:
                data = yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise ConfigurationError(
                    f"Invalid YAML in configuration file: {path}",
                    config_key="file_content",
                    context={"yaml_error": str(e)},
                )
        elif path.suffix.lower() == ".json":
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise ConfigurationError(
                    f"Invalid JSON in configuration file: {path}",
                    config_key="file_content",
                    context={"json_error": str(e)},
                )
        else:
            raise ConfigurationError(
                f"Unsupported configuration file format: {path.suffix}",
                config_key="file_format",
                config_value=path.suffix,
                suggestions=[
                    "Use .yaml, .yml, or .json file extension",
                    "Convert file to supported format",
                ]
            )
        
        if not isinstance(data, dict):
            raise ConfigurationError(
                "Configuration file must contain a dictionary/object at root level",
                config_key="file_content",
                config_value=type(data).__name__,
            )
        
        # Create config with file data
        try:
            return cls(**data)
        except TypeError as e:
            raise ConfigurationError(
                f"Invalid configuration keys in file: {path}",
                config_key="file_content",
                context={"error": str(e)},
                suggestions=[
                    "Check configuration keys match SifakaConfig attributes",
                    "Review documentation for valid configuration options",
                ]
            )
