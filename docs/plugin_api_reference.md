# Plugin API Reference

This document provides detailed API reference for Sifaka's plugin system.

## Table of Contents

1. [Core Interfaces](#core-interfaces)
2. [Plugin Metadata](#plugin-metadata)
3. [Plugin Lifecycle](#plugin-lifecycle)
4. [Data Models](#data-models)
5. [Exceptions](#exceptions)
6. [Utilities](#utilities)

## Core Interfaces

### `PluginInterface`

Base interface for all Sifaka plugins.

```python
from sifaka.core.plugin_interfaces import PluginInterface

class PluginInterface(ABC):
    def __init__(self) -> None

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata

    @property
    def status(self) -> PluginStatus

    @property
    def config(self) -> Dict[str, Any]

    @property
    def error(self) -> Optional[Exception]

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None
    def activate(self) -> None
    def deactivate(self) -> None
    def cleanup(self) -> None
    def validate_config(self, config: Dict[str, Any]) -> bool
    def get_health_status(self) -> Dict[str, Any]

    # Protected methods for subclasses
    def _on_initialize(self) -> None
    def _on_activate(self) -> None
    def _on_deactivate(self) -> None
    def _on_cleanup(self) -> None
    def _validate_config(self, config: Dict[str, Any]) -> bool
```

#### Properties

- **`metadata`**: Plugin metadata (name, version, author, etc.)
- **`status`**: Current plugin status (loaded, initialized, active, etc.)
- **`config`**: Current plugin configuration
- **`error`**: Last error encountered (if any)

#### Methods

- **`initialize(config)`**: Initialize plugin with configuration
- **`activate()`**: Activate the plugin for use
- **`deactivate()`**: Deactivate the plugin
- **`cleanup()`**: Clean up plugin resources
- **`validate_config(config)`**: Validate plugin configuration
- **`get_health_status()`**: Get plugin health information

### `CriticPlugin`

Interface for critic plugins that analyze text and provide feedback.

```python
from sifaka.core.plugin_interfaces import CriticPlugin

class CriticPlugin(PluginInterface, Critic):
    def __init__(self) -> None

    @property
    def name(self) -> str

    @abstractmethod
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult

    def set_model_config(self, model: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> None

    @property
    def model_config(self) -> Optional[Dict[str, Any]]
```

#### Methods

- **`critique(text, result)`**: Analyze text and return critique
- **`set_model_config(model, temperature, max_tokens)`**: Configure LLM settings
- **`model_config`**: Get current model configuration

### `ValidatorPlugin`

Interface for validator plugins that check text quality.

```python
from sifaka.core.plugin_interfaces import ValidatorPlugin

class ValidatorPlugin(PluginInterface, Validator):
    def __init__(self) -> None

    @property
    def name(self) -> str

    @abstractmethod
    async def validate(self, text: str, result: SifakaResult) -> ValidationResult

    def set_validation_config(self, **kwargs: Any) -> None

    @property
    def validation_config(self) -> Dict[str, Any]
```

#### Methods

- **`validate(text, result)`**: Validate text and return result
- **`set_validation_config(**kwargs)`**: Configure validation settings
- **`validation_config`**: Get current validation configuration

## Plugin Metadata

### `PluginMetadata`

Metadata describing a plugin.

```python
from sifaka.core.plugin_interfaces import PluginMetadata

class PluginMetadata(BaseModel):
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = []
    sifaka_version: str = ">=0.1.0"
    python_version: str = ">=3.10"
    homepage: Optional[str] = None
    documentation: Optional[str] = None
    license: Optional[str] = None
    keywords: List[str] = []
    config_schema: Optional[Dict[str, Any]] = None
    default_config: Optional[Dict[str, Any]] = None
```

#### Fields

- **`name`**: Unique plugin name
- **`version`**: Plugin version (semantic versioning)
- **`author`**: Plugin author name
- **`description`**: Plugin description
- **`plugin_type`**: Type of plugin (critic, validator, etc.)
- **`dependencies`**: List of required plugin names
- **`sifaka_version`**: Compatible Sifaka version range
- **`python_version`**: Required Python version
- **`homepage`**: Plugin homepage URL
- **`documentation`**: Plugin documentation URL
- **`license`**: Plugin license
- **`keywords`**: Plugin keywords for discovery
- **`config_schema`**: JSON schema for plugin configuration
- **`default_config`**: Default configuration values

### `PluginType`

Enumeration of plugin types.

```python
from sifaka.core.plugin_interfaces import PluginType

class PluginType(str, Enum):
    CRITIC = "critic"
    VALIDATOR = "validator"
    STORAGE = "storage"
    MIDDLEWARE = "middleware"
    TOOL = "tool"
```

### `PluginStatus`

Enumeration of plugin lifecycle states.

```python
from sifaka.core.plugin_interfaces import PluginStatus

class PluginStatus(str, Enum):
    UNKNOWN = "unknown"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"
```

## Plugin Lifecycle

### Lifecycle States

1. **`UNKNOWN`**: Initial state
2. **`LOADING`**: Plugin is being loaded
3. **`LOADED`**: Plugin class instantiated
4. **`INITIALIZED`**: Plugin initialized with configuration
5. **`ACTIVE`**: Plugin is active and ready for use
6. **`ERROR`**: Plugin encountered an error
7. **`DISABLED`**: Plugin has been disabled/cleaned up

### Lifecycle Methods

#### `initialize(config: Optional[Dict[str, Any]] = None)`

Initialize plugin with configuration.

**Parameters:**
- `config`: Optional configuration dictionary

**Raises:**
- `PluginInitializationError`: If initialization fails

**Example:**
```python
plugin = MyPlugin()
plugin.initialize({"threshold": 0.8, "model": "gpt-4o-mini"})
```

#### `activate()`

Activate the plugin for use.

**Raises:**
- `PluginActivationError`: If activation fails

#### `deactivate()`

Deactivate the plugin.

**Raises:**
- `PluginDeactivationError`: If deactivation fails

#### `cleanup()`

Clean up plugin resources.

**Raises:**
- `PluginCleanupError`: If cleanup fails

## Data Models

### `CritiqueResult`

Result of a critic's analysis.

```python
from sifaka.core.models import CritiqueResult

class CritiqueResult(BaseModel):
    critic: str
    feedback: str
    suggestions: List[str]
    needs_improvement: bool
    confidence: float
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)
```

#### Fields

- **`critic`**: Name of the critic that produced this result
- **`feedback`**: Qualitative feedback about the text
- **`suggestions`**: List of specific improvement suggestions
- **`needs_improvement`**: Whether the text needs improvement
- **`confidence`**: Confidence score (0.0 to 1.0)
- **`metadata`**: Additional metadata
- **`timestamp`**: When the critique was generated

### `ValidationResult`

Result of a validator's check.

```python
from sifaka.core.models import ValidationResult

class ValidationResult(BaseModel):
    validator: str
    passed: bool
    score: Optional[float] = None
    details: str = ""
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)
```

#### Fields

- **`validator`**: Name of the validator that produced this result
- **`passed`**: Whether validation passed
- **`score`**: Optional quality score (0.0 to 1.0)
- **`details`**: Human-readable explanation
- **`metadata`**: Additional metadata
- **`timestamp`**: When the validation was performed

### `SifakaResult`

Complete result object passed to plugins.

```python
from sifaka.core.models import SifakaResult

class SifakaResult(BaseModel):
    id: str
    original_text: str
    final_text: str
    iteration: int
    processing_time: float
    created_at: datetime
    updated_at: datetime
    generations: List[Generation]
    critiques: List[CritiqueResult]
    validations: List[ValidationResult]
    metadata: Dict[str, Any] = {}
```

This object contains the complete history of text improvement, including all previous generations, critiques, and validations.

## Exceptions

### Plugin Exception Hierarchy

```python
from sifaka.core.plugin_interfaces import (
    PluginError,
    PluginInitializationError,
    PluginActivationError,
    PluginDeactivationError,
    PluginCleanupError,
    PluginConfigurationError,
    PluginDiscoveryError,
    PluginRegistrationError,
    PluginDependencyError,
    PluginVersionError,
)

class PluginError(Exception):
    """Base exception for plugin errors."""

class PluginInitializationError(PluginError):
    """Raised when plugin initialization fails."""

class PluginActivationError(PluginError):
    """Raised when plugin activation fails."""

class PluginDeactivationError(PluginError):
    """Raised when plugin deactivation fails."""

class PluginCleanupError(PluginError):
    """Raised when plugin cleanup fails."""

class PluginConfigurationError(PluginError):
    """Raised when plugin configuration is invalid."""

class PluginDiscoveryError(PluginError):
    """Raised when plugin discovery fails."""

class PluginRegistrationError(PluginError):
    """Raised when plugin registration fails."""

class PluginDependencyError(PluginError):
    """Raised when plugin dependencies cannot be resolved."""

class PluginVersionError(PluginError):
    """Raised when plugin version is incompatible."""
```

## Utilities

### `PluginRegistry`

Registry for managing plugin instances.

```python
from sifaka.core.plugin_interfaces import PluginRegistry

class PluginRegistry:
    def __init__(self) -> None

    def register(self, plugin: PluginInterface) -> None
    def unregister(self, name: str) -> None
    def get_plugin(self, name: str) -> Optional[PluginInterface]
    def get_metadata(self, name: str) -> Optional[PluginMetadata]
    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[str]
    def get_status(self, name: str) -> Optional[PluginStatus]
    def get_all_status(self) -> Dict[str, Dict[str, Any]]
```

### `PluginLoader`

Utility for discovering and loading plugins.

```python
from sifaka.core.plugin_loader import PluginLoader

class PluginLoader:
    def __init__(self) -> None

    def load_from_directory(self, directory: Union[str, Path]) -> List[PluginInterface]
    def load_from_entry_points(self, group: str) -> List[PluginInterface]
    def load_all_plugins(self) -> List[PluginInterface]
    def resolve_dependencies(self, plugins: List[PluginInterface]) -> List[PluginInterface]
    def validate_plugin(self, plugin: PluginInterface) -> bool
    def get_loaded_plugins(self) -> Dict[str, PluginInterface]
    def get_failed_plugins(self) -> Dict[str, Exception]
```

### Global Functions

```python
from sifaka.core.plugin_interfaces import (
    get_plugin_registry,
    register_plugin,
    get_plugin,
    list_plugins,
)

def get_plugin_registry() -> PluginRegistry
def register_plugin(plugin: PluginInterface) -> None
def get_plugin(name: str) -> Optional[PluginInterface]
def list_plugins(plugin_type: Optional[PluginType] = None) -> List[str]
```

```python
from sifaka.core.plugin_loader import (
    get_plugin_loader,
    load_plugins_from_directory,
    load_plugins_from_entry_points,
    load_all_plugins,
    discover_and_load_plugins,
)

def get_plugin_loader() -> PluginLoader
def load_plugins_from_directory(directory: Union[str, Path]) -> List[PluginInterface]
def load_plugins_from_entry_points(group: str) -> List[PluginInterface]
def load_all_plugins() -> List[PluginInterface]
def discover_and_load_plugins() -> Dict[str, List[PluginInterface]]
```

## Usage Examples

### Basic Plugin Implementation

```python
from sifaka.core.plugin_interfaces import CriticPlugin, PluginMetadata, PluginType
from sifaka.core.models import CritiqueResult, SifakaResult

class SimpleCritic(CriticPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="simple_critic",
            version="1.0.0",
            author="Your Name",
            description="Simple example critic",
            plugin_type=PluginType.CRITIC
        )

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        word_count = len(text.split())
        needs_improvement = word_count < 50

        return CritiqueResult(
            critic=self.name,
            feedback=f"Text has {word_count} words",
            suggestions=["Add more content"] if needs_improvement else [],
            needs_improvement=needs_improvement,
            confidence=0.9
        )
```

### Plugin with Configuration

```python
class ConfigurableCritic(CriticPlugin):
    def __init__(self):
        super().__init__()
        self.min_words = 50

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="configurable_critic",
            version="1.0.0",
            author="Your Name",
            description="Configurable critic",
            plugin_type=PluginType.CRITIC,
            default_config={"min_words": 50}
        )

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        if "min_words" in config:
            if not isinstance(config["min_words"], int) or config["min_words"] < 1:
                raise ValueError("min_words must be a positive integer")
        return True

    def _on_initialize(self):
        self.min_words = self.config.get("min_words", 50)

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        word_count = len(text.split())
        needs_improvement = word_count < self.min_words

        return CritiqueResult(
            critic=self.name,
            feedback=f"Text has {word_count} words (minimum: {self.min_words})",
            suggestions=["Add more content"] if needs_improvement else [],
            needs_improvement=needs_improvement,
            confidence=0.9
        )
```

### Plugin Discovery and Loading

```python
from sifaka.core.plugin_loader import discover_and_load_plugins
from sifaka.core.plugin_interfaces import get_plugin_registry

# Discover all plugins
plugins_by_type = discover_and_load_plugins()

# Get registry
registry = get_plugin_registry()

# List all critics
critic_names = registry.list_plugins(PluginType.CRITIC)
print(f"Available critics: {critic_names}")

# Get specific plugin
plugin = registry.get_plugin("simple_critic")
if plugin:
    print(f"Plugin status: {plugin.status}")
    print(f"Plugin metadata: {plugin.metadata}")
```

This completes the comprehensive API reference for Sifaka's plugin system.
