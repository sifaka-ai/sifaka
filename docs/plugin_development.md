# Plugin Development Guide

This guide covers everything you need to know to develop plugins for Sifaka.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Plugin Types](#plugin-types)
4. [Plugin Architecture](#plugin-architecture)
5. [Creating Your First Plugin](#creating-your-first-plugin)
6. [Testing Your Plugin](#testing-your-plugin)
7. [Publishing Your Plugin](#publishing-your-plugin)
8. [Advanced Topics](#advanced-topics)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

Sifaka's plugin system allows you to extend the framework with custom critics and validators. Plugins are Python packages that implement specific interfaces and can be distributed independently.

### Key Benefits

- **Extensibility**: Add new analysis capabilities without modifying core Sifaka code
- **Distribution**: Share plugins as Python packages via PyPI
- **Auto-discovery**: Plugins are automatically discovered via entry points
- **Type Safety**: Full type checking and validation
- **Lifecycle Management**: Comprehensive plugin lifecycle with health monitoring

## Quick Start

### 1. Generate Plugin Template

Use the built-in plugin generator:

```bash
# Create a critic plugin
python sifaka/templates/create_plugin.py critic my_critic "My custom critic" "Your Name"

# Create a validator plugin
python sifaka/templates/create_plugin.py validator my_validator "My custom validator" "Your Name"
```

### 2. Implement Your Logic

Edit the generated `plugin.py` file and implement your analysis logic in the `critique()` or `validate()` methods.

### 3. Test Your Plugin

```bash
cd my_critic_critic  # or my_validator_validator
pip install -e ".[dev]"
pytest
```

### 4. Use Your Plugin

```python
from sifaka.api import improve_text
from sifaka.core.config import Config

config = Config(
    critics=["my_critic"],  # or validators=["my_validator"]
    model="gpt-4o-mini"
)

result = improve_text("Your text here", config=config)
```

## Plugin Types

### Critic Plugins

Critics analyze text and provide feedback for improvement. They implement the `CriticPlugin` interface.

**Use cases:**
- Content analysis (clarity, coherence, style)
- Domain-specific checks (technical writing, academic papers)
- Quality assessment (readability, engagement)

### Validator Plugins

Validators check text against specific criteria and determine if it meets quality standards. They implement the `ValidatorPlugin` interface.

**Use cases:**
- Quality gates (minimum word count, grammar)
- Content filtering (profanity, sensitive information)
- Compliance checks (style guides, formatting rules)

## Plugin Architecture

### Core Components

```python
from sifaka.core.plugin_interfaces import CriticPlugin, ValidatorPlugin
from sifaka.core.models import CritiqueResult, ValidationResult
```

### Plugin Interface Hierarchy

```
PluginInterface (abstract base)
├── CriticPlugin (extends Critic interface)
└── ValidatorPlugin (extends Validator interface)
```

### Plugin Lifecycle

1. **Discovery**: Plugins are discovered via entry points
2. **Loading**: Plugin classes are loaded and instantiated
3. **Initialization**: `initialize()` is called with configuration
4. **Activation**: `activate()` is called when plugin becomes active
5. **Execution**: `critique()` or `validate()` methods are called
6. **Deactivation**: `deactivate()` is called when plugin is no longer needed
7. **Cleanup**: `cleanup()` is called to release resources

## Creating Your First Plugin

### Example: Readability Critic

```python
from sifaka.core.plugin_interfaces import CriticPlugin, PluginMetadata, PluginType
from sifaka.core.models import CritiqueResult, SifakaResult

class ReadabilityCritic(CriticPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="readability_critic",
            version="1.0.0",
            author="Your Name",
            description="Analyzes text readability",
            plugin_type=PluginType.CRITIC,
            dependencies=[],
            sifaka_version=">=0.1.0"
        )

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # Calculate readability score
        score = self._calculate_readability(text)

        needs_improvement = score < 0.7
        suggestions = []

        if needs_improvement:
            suggestions.append("Consider shorter sentences")
            suggestions.append("Use simpler vocabulary")

        return CritiqueResult(
            critic=self.name,
            feedback=f"Readability score: {score:.2f}",
            suggestions=suggestions,
            needs_improvement=needs_improvement,
            confidence=0.9
        )

    def _calculate_readability(self, text: str) -> float:
        # Implement your readability algorithm
        words = len(text.split())
        sentences = text.count('.') + text.count('!') + text.count('?')

        if sentences == 0:
            return 0.0

        avg_sentence_length = words / sentences
        return max(0.0, 1.0 - (avg_sentence_length / 30))
```

### Example: Word Count Validator

```python
from sifaka.core.plugin_interfaces import ValidatorPlugin, PluginMetadata, PluginType
from sifaka.core.models import ValidationResult, SifakaResult

class WordCountValidator(ValidatorPlugin):
    def __init__(self):
        super().__init__()
        self.min_words = 50
        self.max_words = 1000

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="word_count_validator",
            version="1.0.0",
            author="Your Name",
            description="Validates text word count",
            plugin_type=PluginType.VALIDATOR,
            default_config={
                "min_words": 50,
                "max_words": 1000
            }
        )

    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        word_count = len(text.split())

        if word_count < self.min_words:
            return ValidationResult(
                validator=self.name,
                passed=False,
                score=0.0,
                details=f"Text has {word_count} words, minimum is {self.min_words}"
            )

        if word_count > self.max_words:
            return ValidationResult(
                validator=self.name,
                passed=False,
                score=0.0,
                details=f"Text has {word_count} words, maximum is {self.max_words}"
            )

        return ValidationResult(
            validator=self.name,
            passed=True,
            score=1.0,
            details=f"Text has {word_count} words (within range)"
        )

    def _on_initialize(self):
        self.min_words = self.validation_config.get("min_words", 50)
        self.max_words = self.validation_config.get("max_words", 1000)
```

## Testing Your Plugin

### Test Structure

```python
import pytest
from datetime import datetime
from your_plugin import YourPlugin
from sifaka.core.models import SifakaResult

class TestYourPlugin:
    def test_plugin_metadata(self):
        plugin = YourPlugin()
        assert plugin.metadata.name == "your_plugin"
        assert plugin.metadata.version == "1.0.0"

    def test_plugin_lifecycle(self):
        plugin = YourPlugin()
        plugin.initialize()
        plugin.activate()
        plugin.deactivate()
        plugin.cleanup()

    @pytest.mark.asyncio
    async def test_functionality(self):
        plugin = YourPlugin()
        plugin.initialize()

        result = SifakaResult(
            id="test",
            original_text="test",
            final_text="test",
            iteration=1,
            processing_time=0.1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            generations=[],
            critiques=[],
            validations=[]
        )

        # Test your plugin's main functionality
        critique = await plugin.critique("test text", result)
        assert critique.critic == "your_plugin"
```

### Running Tests

```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest --cov=your_plugin  # With coverage
pytest -k "test_name"     # Run specific test
```

## Publishing Your Plugin

### 1. Package Structure

```
your_plugin_critic/
├── your_plugin_critic/
│   ├── __init__.py
│   ├── plugin.py
│   └── py.typed
├── tests/
│   ├── __init__.py
│   └── test_your_plugin.py
├── pyproject.toml
├── README.md
└── LICENSE
```

### 2. Entry Points

In `pyproject.toml`:

```toml
[project.entry-points."sifaka.critics"]
your_plugin = "your_plugin_critic:YourPlugin"

# Or for validators:
[project.entry-points."sifaka.validators"]
your_plugin = "your_plugin_validator:YourPlugin"
```

### 3. Publishing to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

### 4. Installation

Once published:

```bash
pip install your-plugin-critic
```

The plugin will be automatically discovered by Sifaka.

## Advanced Topics

### Configuration Management

```python
def _validate_config(self, config: Dict[str, Any]) -> bool:
    """Validate plugin configuration."""
    if "threshold" in config:
        if not isinstance(config["threshold"], float):
            raise ValueError("threshold must be a float")
        if not 0.0 <= config["threshold"] <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
    return True

def _on_initialize(self):
    """Initialize with configuration."""
    self.threshold = self.config.get("threshold", 0.8)
```

### Error Handling

```python
async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
    try:
        # Your analysis logic
        analysis = self._analyze(text)
        return self._create_critique(analysis)
    except Exception as e:
        logger.error(f"Error in {self.name}: {e}")
        return CritiqueResult(
            critic=self.name,
            feedback=f"Analysis failed: {str(e)}",
            suggestions=["Please check the input text"],
            needs_improvement=False,
            confidence=0.0,
            metadata={"error": str(e)}
        )
```

### Async Operations

```python
import aiohttp

async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
    async with aiohttp.ClientSession() as session:
        async with session.post('https://api.example.com/analyze',
                               json={'text': text}) as response:
            data = await response.json()
            return self._process_response(data)
```

### Plugin Dependencies

```python
@property
def metadata(self) -> PluginMetadata:
    return PluginMetadata(
        name="advanced_critic",
        dependencies=["basic_critic", "sentiment_analyzer"],
        sifaka_version=">=0.1.0"
    )
```

## Best Practices

### Code Quality

1. **Use type hints**: Full type annotation for all methods
2. **Follow PEP 8**: Use black, ruff, and mypy
3. **Document everything**: Docstrings for all public methods
4. **Handle errors gracefully**: Never let exceptions crash the pipeline
5. **Log appropriately**: Use structured logging for debugging

### Performance

1. **Minimize blocking operations**: Use async/await for I/O
2. **Cache expensive computations**: Store results between calls
3. **Validate inputs early**: Fail fast on invalid data
4. **Use appropriate data structures**: Choose efficient algorithms

### Testing

1. **Test all code paths**: Include success and failure cases
2. **Use meaningful test data**: Test with realistic inputs
3. **Mock external dependencies**: Don't rely on external services
4. **Test configuration validation**: Ensure bad configs are rejected

### Documentation

1. **Clear README**: Installation, usage, and examples
2. **API documentation**: Document all configuration options
3. **Version changelog**: Track changes between versions
4. **Usage examples**: Show real-world use cases

## Troubleshooting

### Common Issues

#### Plugin Not Discovered

```bash
# Check if plugin is installed
pip list | grep your-plugin

# Check entry points
python -c "import pkg_resources; print(list(pkg_resources.iter_entry_points('sifaka.critics')))"

# Check if plugin is loaded in Sifaka
# (Plugins are auto-discovered from entry points)
```

#### Import Errors

```python
# Check dependencies
pip install -e ".[dev]"

# Test import
python -c "from your_plugin import YourPlugin; print('OK')"
```

#### Configuration Issues

```python
# Test configuration validation
plugin = YourPlugin()
plugin.validate_config({"key": "value"})
```

#### Runtime Errors

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check plugin health
plugin = YourPlugin()
plugin.initialize()
status = plugin.get_health_status()
print(status)
```

### Debug Mode

```python
# Enable detailed error reporting
import os
os.environ['SIFAKA_DEBUG'] = '1'

# Check plugin status programmatically
from sifaka.core.plugin_loader import get_plugin_loader
loader = get_plugin_loader()
status = loader.get_plugin_status('your_plugin')
```

### Getting Help

1. **Check existing plugins**: Look at examples in `sifaka/examples/plugins/`
2. **Read the source**: Study the plugin interfaces in `sifaka/core/plugin_interfaces.py`
3. **API Documentation**: See the plugin API reference for detailed information
4. **Community**: Join the Sifaka community for support

---

This completes the comprehensive plugin development guide. For API reference and additional examples, see the other documentation files.
