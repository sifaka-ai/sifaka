# Sifaka Plugin Examples

This directory contains example plugins demonstrating how to extend Sifaka with custom functionality.

## Overview

Sifaka supports plugin extensions for:

1. **Critics** - Domain-specific text improvement strategies
2. **Validators** - Custom validation rules for generated content
3. **Storage Backends** - Custom storage solutions for persisting results

## Example Plugins in This Directory

### 1. Critic Plugin (`example_critic_plugin.py`)

Demonstrates how to create a custom critic plugin using the new `CriticPlugin` interface:

- Shows the complete plugin lifecycle (initialize, activate, deactivate, cleanup)
- Implements a readability-focused critic
- Demonstrates proper error handling and logging
- Uses the modern plugin interface

**Key features:**
- Readability analysis (Flesch score, complex words)
- Specific improvement suggestions
- Confidence scoring based on text analysis
- Proper async/await patterns

### 2. Validator Plugin (`example_validator_plugin.py`)

Shows how to create a custom validator plugin using the `ValidatorPlugin` interface:

- Implements comprehensive text quality validation
- Shows plugin metadata and versioning
- Demonstrates validation result formatting
- Includes both warnings and errors

**Key features:**
- Multiple quality checks (length, structure, terminology)
- Configurable validation rules
- Detailed validation messages
- Async validation support

## Installing Plugins

There are two ways to use these plugins:

### Method 1: Direct Registration

```python
from sifaka import register_storage_backend, register_critic, register_validator
from custom_storage_backend import RedisStorageBackend
from custom_critic import AcademicWritingCritic
from custom_validator import SEOValidator

# Register plugins
register_storage_backend("redis", RedisStorageBackend)
register_critic("academic", AcademicWritingCritic)
register_validator("seo", SEOValidator)

# Use in improve calls
result = improve_sync(
    text="Your content here",
    critics=["academic"],
    validators=["seo"],
    storage_backend="redis"
)
```

### Method 2: Package Entry Points

Create a `setup.py` for your plugin package:

```python
setup(
    name="sifaka-custom-plugins",
    entry_points={
        "sifaka.storage": [
            "redis = my_plugins:RedisStorageBackend",
        ],
        "sifaka.critics": [
            "academic = my_plugins:AcademicWritingCritic",
            "tech_docs = my_plugins:TechnicalDocumentationCritic",
        ],
        "sifaka.validators": [
            "seo = my_plugins:SEOValidator",
            "code_quality = my_plugins:CodeQualityValidator",
            "accessibility = my_plugins:AccessibilityValidator",
        ],
    }
)
```

Then install your package:
```bash
pip install -e .
```

Plugins will be automatically discovered and available for use.

## Creating Your Own Plugins

### Critic Plugin Template

```python
from sifaka.core.plugin_interfaces import CriticPlugin
from sifaka.core.models import CritiqueResult, SifakaResult

class MyCriticPlugin(CriticPlugin):
    """Custom critic plugin for domain-specific improvements."""

    def __init__(self):
        super().__init__()
        self.name = "my_critic"
        self.version = "1.0.0"
        self.author = "Your Name"
        self.description = "Description of what your critic does"

    async def initialize(self) -> None:
        """Initialize plugin resources."""
        # Setup any required resources
        pass

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Perform critique on the text."""
        # Analyze text and return critique
        return CritiqueResult(
            feedback="Your detailed feedback here",
            suggestions=["Suggestion 1", "Suggestion 2"],
            confidence=0.85
        )

    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        # Release any resources
        pass
```

### Validator Plugin Template

```python
from sifaka.core.plugin_interfaces import ValidatorPlugin
from sifaka.core.models import ValidationResult, SifakaResult

class MyValidatorPlugin(ValidatorPlugin):
    """Custom validator plugin for specific requirements."""

    def __init__(self):
        super().__init__()
        self.name = "my_validator"
        self.version = "1.0.0"
        self.author = "Your Name"
        self.description = "Description of validation rules"

    async def initialize(self) -> None:
        """Initialize plugin resources."""
        # Setup validation rules
        pass

    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate the text against custom rules."""
        errors = []
        warnings = []

        # Perform validation checks
        if len(text) < 10:
            errors.append("Text too short")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass
```

### Storage Backend Template

```python
from sifaka.storage.base import StorageBackend
from sifaka.core.models import SifakaResult

class MyStorageBackend(StorageBackend):
    """Custom storage backend for persisting results."""

    async def save(self, result: SifakaResult) -> str:
        """Save result and return ID."""
        # Implementation
        pass

    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load result by ID."""
        # Implementation
        pass

    async def list(self, limit: int = 100, offset: int = 0) -> List[SifakaResult]:
        """List stored results."""
        # Implementation
        pass

    async def delete(self, result_id: str) -> bool:
        """Delete result by ID."""
        # Implementation
        pass

    async def search(self, query: str, limit: int = 10) -> List[SifakaResult]:
        """Search results."""
        # Implementation
        pass
```

## Best Practices

1. **Error Handling**: Always handle potential errors gracefully
2. **Async Support**: Use async/await for I/O operations
3. **Configuration**: Accept configuration parameters in `__init__`
4. **Documentation**: Document your plugin's purpose and parameters
5. **Testing**: Include tests for your plugins
6. **Dependencies**: Declare any additional dependencies clearly

## Running the Examples

To see the plugin examples in action:

```bash
# Run the critic plugin example
uv run python examples/plugins/example_critic_plugin.py

# Run the validator plugin example
uv run python examples/plugins/example_validator_plugin.py
```

For full integration testing, you'll need to provide API keys for the LLM providers in your environment or `.env` file:
- `OPENAI_API_KEY` for OpenAI models
- `ANTHROPIC_API_KEY` for Claude models
- `GEMINI_API_KEY` for Gemini models
