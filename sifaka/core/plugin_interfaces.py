"""Plugin interface base classes for Sifaka extensibility.

This module provides the foundation for Sifaka's plugin system, enabling
third-party extensions for critics, validators, storage backends, and other
components.

## Plugin System Architecture:

The plugin system is designed around several key principles:

1. **Type Safety**: All plugins must implement proper interfaces
2. **Lifecycle Management**: Plugins have clear initialization and cleanup phases
3. **Metadata Requirements**: All plugins must provide descriptive metadata
4. **Dependency Management**: Plugins can declare dependencies on other plugins
5. **Version Compatibility**: Plugins declare compatible Sifaka versions

## Plugin Types:

- **CriticPlugin**: Extends text analysis capabilities
- **ValidatorPlugin**: Adds new text quality validators
- **StoragePlugin**: Provides new storage backends
- **MiddlewarePlugin**: Adds request/response processing

## Example Plugin:

    from sifaka.core.plugin_interfaces import CriticPlugin
    from sifaka.core.models import CritiqueResult

    class MyCriticPlugin(CriticPlugin):
        @property
        def metadata(self) -> PluginMetadata:
            return PluginMetadata(
                name="my_critic",
                version="1.0.0",
                author="Your Name",
                description="A custom critic plugin",
                dependencies=[],
                sifaka_version=">=0.1.0"
            )

        async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
            # Your critique logic here
            pass
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .interfaces import Critic, Validator
from .models import CritiqueResult, SifakaResult, ValidationResult


class PluginType(str, Enum):
    """Types of plugins supported by Sifaka."""

    CRITIC = "critic"
    VALIDATOR = "validator"
    STORAGE = "storage"
    MIDDLEWARE = "middleware"
    TOOL = "tool"


class PluginStatus(str, Enum):
    """Plugin lifecycle status."""

    UNKNOWN = "unknown"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class PluginMetadata(BaseModel):
    """Metadata describing a plugin."""

    name: str = Field(..., description="Unique plugin name")
    version: str = Field(..., description="Plugin version (semver)")
    author: str = Field(..., description="Plugin author")
    description: str = Field(..., description="Plugin description")
    plugin_type: PluginType = Field(..., description="Type of plugin")

    # Dependencies and compatibility
    dependencies: List[str] = Field(
        default_factory=list, description="List of required plugin names"
    )
    sifaka_version: str = Field(
        default=">=0.1.0", description="Compatible Sifaka version range"
    )
    python_version: str = Field(default=">=3.10", description="Required Python version")

    # Optional metadata
    homepage: Optional[str] = Field(default=None, description="Plugin homepage URL")
    documentation: Optional[str] = Field(
        default=None, description="Plugin documentation URL"
    )
    license: Optional[str] = Field(default=None, description="Plugin license")
    keywords: List[str] = Field(
        default_factory=list, description="Plugin keywords for discovery"
    )

    # Plugin configuration
    config_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON schema for plugin configuration"
    )
    default_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Default configuration values"
    )


class PluginInterface(ABC):
    """Base interface for all Sifaka plugins.

    This interface defines the common contract that all plugins must implement,
    regardless of their specific type (critic, validator, storage, etc.).
    """

    def __init__(self) -> None:
        """Initialize the plugin.

        Plugins should perform minimal initialization here. Heavy initialization
        should be deferred to the initialize() method.
        """
        self._status = PluginStatus.LOADED
        self._config: Dict[str, Any] = {}
        self._error: Optional[Exception] = None

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata.

        Returns:
            PluginMetadata containing plugin information
        """
        pass

    @property
    def status(self) -> PluginStatus:
        """Current plugin status.

        Returns:
            Current plugin lifecycle status
        """
        return self._status

    @property
    def config(self) -> Dict[str, Any]:
        """Plugin configuration.

        Returns:
            Current plugin configuration
        """
        return self._config.copy()

    @property
    def error(self) -> Optional[Exception]:
        """Last error encountered by the plugin.

        Returns:
            Last exception, or None if no error
        """
        return self._error

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        This method is called after plugin discovery and should perform
        any heavy initialization tasks.

        Args:
            config: Plugin configuration dictionary

        Raises:
            PluginInitializationError: If initialization fails
        """
        try:
            self._status = PluginStatus.LOADING
            self._config = config or {}
            self._on_initialize()
            self._status = PluginStatus.INITIALIZED
        except Exception as e:
            self._error = e
            self._status = PluginStatus.ERROR
            raise PluginInitializationError(
                f"Failed to initialize plugin {self.metadata.name}: {e}"
            ) from e

    def activate(self) -> None:
        """Activate the plugin.

        This method is called when the plugin should become active and
        start processing requests.

        Raises:
            PluginActivationError: If activation fails
        """
        try:
            if self._status != PluginStatus.INITIALIZED:
                raise PluginActivationError(
                    f"Plugin {self.metadata.name} must be initialized before activation"
                )

            self._on_activate()
            self._status = PluginStatus.ACTIVE
        except Exception as e:
            self._error = e
            self._status = PluginStatus.ERROR
            raise PluginActivationError(
                f"Failed to activate plugin {self.metadata.name}: {e}"
            ) from e

    def deactivate(self) -> None:
        """Deactivate the plugin.

        This method is called when the plugin should stop processing
        requests but remain available for reactivation.

        Raises:
            PluginDeactivationError: If deactivation fails
        """
        try:
            if self._status != PluginStatus.ACTIVE:
                return  # Already deactivated

            self._on_deactivate()
            self._status = PluginStatus.INITIALIZED
        except Exception as e:
            self._error = e
            self._status = PluginStatus.ERROR
            raise PluginDeactivationError(
                f"Failed to deactivate plugin {self.metadata.name}: {e}"
            ) from e

    def cleanup(self) -> None:
        """Clean up plugin resources.

        This method is called when the plugin is being unloaded and
        should release any resources it holds.

        Raises:
            PluginCleanupError: If cleanup fails
        """
        try:
            if self._status == PluginStatus.ACTIVE:
                self.deactivate()

            self._on_cleanup()
            self._status = PluginStatus.DISABLED
        except Exception as e:
            self._error = e
            self._status = PluginStatus.ERROR
            raise PluginCleanupError(
                f"Failed to cleanup plugin {self.metadata.name}: {e}"
            ) from e

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid

        Raises:
            PluginConfigurationError: If configuration is invalid
        """
        try:
            return self._validate_config(config)
        except Exception as e:
            raise PluginConfigurationError(
                f"Invalid configuration for plugin {self.metadata.name}: {e}"
            ) from e

    def get_health_status(self) -> Dict[str, Any]:
        """Get plugin health status.

        Returns:
            Dictionary containing health information
        """
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "status": self.status.value,
            "error": str(self.error) if self.error else None,
            "config": self.config,
            "metadata": self.metadata.model_dump(),
        }

    # Protected methods that subclasses can override

    def _on_initialize(self) -> None:
        """Plugin-specific initialization logic.

        Override this method to add custom initialization logic.
        """
        pass

    def _on_activate(self) -> None:
        """Plugin-specific activation logic.

        Override this method to add custom activation logic.
        """
        pass

    def _on_deactivate(self) -> None:
        """Plugin-specific deactivation logic.

        Override this method to add custom deactivation logic.
        """
        pass

    def _on_cleanup(self) -> None:
        """Plugin-specific cleanup logic.

        Override this method to add custom cleanup logic.
        """
        pass

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Plugin-specific configuration validation.

        Override this method to add custom configuration validation.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        return True


# Plugin exceptions


class PluginError(Exception):
    """Base exception for plugin errors."""

    pass


class PluginInitializationError(PluginError):
    """Raised when plugin initialization fails."""

    pass


class PluginActivationError(PluginError):
    """Raised when plugin activation fails."""

    pass


class PluginDeactivationError(PluginError):
    """Raised when plugin deactivation fails."""

    pass


class PluginCleanupError(PluginError):
    """Raised when plugin cleanup fails."""

    pass


class PluginConfigurationError(PluginError):
    """Raised when plugin configuration is invalid."""

    pass


class PluginDiscoveryError(PluginError):
    """Raised when plugin discovery fails."""

    pass


class PluginRegistrationError(PluginError):
    """Raised when plugin registration fails."""

    pass


class PluginDependencyError(PluginError):
    """Raised when plugin dependencies cannot be resolved."""

    pass


class PluginVersionError(PluginError):
    """Raised when plugin version is incompatible."""

    pass


# Plugin registry interface


class PluginRegistry:
    """Registry for managing plugin instances and metadata."""

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        self._plugins: Dict[str, PluginInterface] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
        self._status: Dict[str, PluginStatus] = {}

    def register(self, plugin: PluginInterface) -> None:
        """Register a plugin.

        Args:
            plugin: Plugin instance to register

        Raises:
            PluginRegistrationError: If registration fails
        """
        try:
            metadata = plugin.metadata
            name = metadata.name

            if name in self._plugins:
                raise PluginRegistrationError(f"Plugin {name} is already registered")

            self._plugins[name] = plugin
            self._metadata[name] = metadata
            self._status[name] = plugin.status

        except Exception as e:
            raise PluginRegistrationError(f"Failed to register plugin: {e}") from e

    def unregister(self, name: str) -> None:
        """Unregister a plugin.

        Args:
            name: Name of plugin to unregister

        Raises:
            PluginRegistrationError: If unregistration fails
        """
        try:
            if name not in self._plugins:
                raise PluginRegistrationError(f"Plugin {name} is not registered")

            plugin = self._plugins[name]
            plugin.cleanup()

            del self._plugins[name]
            del self._metadata[name]
            del self._status[name]

        except Exception as e:
            raise PluginRegistrationError(
                f"Failed to unregister plugin {name}: {e}"
            ) from e

    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)

    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by name.

        Args:
            name: Plugin name

        Returns:
            Plugin metadata or None if not found
        """
        return self._metadata.get(name)

    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[str]:
        """List all registered plugins.

        Args:
            plugin_type: Optional filter by plugin type

        Returns:
            List of plugin names
        """
        if plugin_type is None:
            return list(self._plugins.keys())

        return [
            name
            for name, metadata in self._metadata.items()
            if metadata.plugin_type == plugin_type
        ]

    def get_status(self, name: str) -> Optional[PluginStatus]:
        """Get plugin status by name.

        Args:
            name: Plugin name

        Returns:
            Plugin status or None if not found
        """
        plugin = self._plugins.get(name)
        return plugin.status if plugin else None

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered plugins.

        Returns:
            Dictionary mapping plugin names to their health status
        """
        return {
            name: plugin.get_health_status() for name, plugin in self._plugins.items()
        }


# Global plugin registry instance
_global_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance.

    Returns:
        Global plugin registry
    """
    return _global_registry


def register_plugin(plugin: PluginInterface) -> None:
    """Register a plugin with the global registry.

    Args:
        plugin: Plugin to register
    """
    _global_registry.register(plugin)


def get_plugin(name: str) -> Optional[PluginInterface]:
    """Get a plugin from the global registry.

    Args:
        name: Plugin name

    Returns:
        Plugin instance or None if not found
    """
    return _global_registry.get_plugin(name)


def list_plugins(plugin_type: Optional[PluginType] = None) -> List[str]:
    """List all registered plugins.

    Args:
        plugin_type: Optional filter by plugin type

    Returns:
        List of plugin names
    """
    return _global_registry.list_plugins(plugin_type)


# Specific plugin interfaces for Sifaka components


class CriticPlugin(PluginInterface, Critic):
    """Plugin interface for critics that analyze text and provide feedback.

    This interface combines the generic plugin capabilities with Sifaka's
    Critic interface, enabling third-party critics to be loaded as plugins.

    Example:
        class MyCriticPlugin(CriticPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="my_critic",
                    version="1.0.0",
                    author="Your Name",
                    description="A custom critic plugin",
                    plugin_type=PluginType.CRITIC,
                    dependencies=[],
                    sifaka_version=">=0.1.0"
                )

            async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
                # Your critique logic here
                return CritiqueResult(
                    critic=self.name,
                    feedback="Your feedback here",
                    suggestions=["Suggestion 1", "Suggestion 2"],
                    needs_improvement=True,
                    confidence=0.8
                )
    """

    def __init__(self) -> None:
        """Initialize the critic plugin.

        Sets up both plugin lifecycle and critic-specific state.
        """
        super().__init__()
        self._model_config: Optional[Dict[str, Any]] = None
        self._temperature: float = 0.7
        self._max_tokens: Optional[int] = None

    @property
    def name(self) -> str:
        """Get the critic name from plugin metadata.

        Returns:
            The plugin name, used as the critic identifier
        """
        return self.metadata.name

    @abstractmethod
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Analyze text and provide structured improvement feedback.

        This method must be implemented by critic plugins to provide
        their unique perspective on text quality and improvement opportunities.

        Args:
            text: The current text to critique
            result: The complete SifakaResult containing all history

        Returns:
            CritiqueResult with feedback, suggestions, and metadata
        """
        pass

    def set_model_config(
        self, model: str, temperature: float = 0.7, max_tokens: Optional[int] = None
    ) -> None:
        """Configure the LLM model settings for this critic.

        Args:
            model: Model name (e.g., 'gpt-4o-mini')
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
        """
        self._model_config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_config(self) -> Optional[Dict[str, Any]]:
        """Get current model configuration.

        Returns:
            Dictionary with model, temperature, and max_tokens
        """
        return self._model_config

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate critic-specific configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate model configuration if provided
        if "model" in config:
            model = config["model"]
            if not isinstance(model, str) or not model.strip():
                raise ValueError("Model must be a non-empty string")

        if "temperature" in config:
            temp = config["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 2.0:
                raise ValueError("Temperature must be between 0.0 and 2.0")

        if "max_tokens" in config:
            tokens = config["max_tokens"]
            if tokens is not None and (not isinstance(tokens, int) or tokens <= 0):
                raise ValueError("max_tokens must be a positive integer or None")

        return True

    def _on_initialize(self) -> None:
        """Initialize critic-specific components.

        Sets up model configuration from the plugin config.
        """
        if "model" in self._config:
            self.set_model_config(
                model=self._config["model"],
                temperature=self._config.get("temperature", 0.7),
                max_tokens=self._config.get("max_tokens"),
            )

    def get_health_status(self) -> Dict[str, Any]:
        """Get critic plugin health status.

        Returns:
            Extended health status including model configuration
        """
        status = super().get_health_status()
        status.update(
            {
                "model_config": self._model_config,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
        )
        return status


class ValidatorPlugin(PluginInterface, Validator):
    """Plugin interface for validators that check text quality.

    This interface combines the generic plugin capabilities with Sifaka's
    Validator interface, enabling third-party validators to be loaded as plugins.

    Example:
        class MyValidatorPlugin(ValidatorPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="my_validator",
                    version="1.0.0",
                    author="Your Name",
                    description="A custom validator plugin",
                    plugin_type=PluginType.VALIDATOR,
                    dependencies=[],
                    sifaka_version=">=0.1.0"
                )

            async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
                # Your validation logic here
                passed = len(text) > 10  # Example validation
                return ValidationResult(
                    validator=self.name,
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    details=f"Text length: {len(text)}"
                )
    """

    def __init__(self) -> None:
        """Initialize the validator plugin.

        Sets up both plugin lifecycle and validator-specific state.
        """
        super().__init__()
        self._validation_config: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Get the validator name from plugin metadata.

        Returns:
            The plugin name, used as the validator identifier
        """
        return self.metadata.name

    @abstractmethod
    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate the given text against this validator's criteria.

        This method must be implemented by validator plugins to provide
        their specific validation logic.

        Args:
            text: The current text to validate
            result: The complete SifakaResult containing all history

        Returns:
            ValidationResult with passed status, score, and details
        """
        pass

    def set_validation_config(self, **kwargs: Any) -> None:
        """Configure validator-specific settings.

        Args:
            **kwargs: Validator-specific configuration options
        """
        self._validation_config.update(kwargs)

    @property
    def validation_config(self) -> Dict[str, Any]:
        """Get current validation configuration.

        Returns:
            Dictionary with validator-specific settings
        """
        return self._validation_config.copy()

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate validator-specific configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid

        Note:
            Base implementation accepts any configuration.
            Subclasses should override for specific validation.
        """
        return True

    def _on_initialize(self) -> None:
        """Initialize validator-specific components.

        Sets up validation configuration from the plugin config.
        """
        self._validation_config.update(self._config)

    def get_health_status(self) -> Dict[str, Any]:
        """Get validator plugin health status.

        Returns:
            Extended health status including validation configuration
        """
        status = super().get_health_status()
        status.update(
            {
                "validation_config": self._validation_config,
            }
        )
        return status
