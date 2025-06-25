# Sifaka Plugin Examples

This directory contains example plugins demonstrating how to extend Sifaka with custom functionality.

## Overview

Sifaka supports three types of plugins:

1. **Storage Backends** - Custom storage solutions for persisting results
2. **Critics** - Domain-specific text improvement strategies
3. **Validators** - Custom validation rules for generated content

## Plugin Examples

### 1. Custom Storage Backend (`custom_storage_backend.py`)

Demonstrates a Redis-style storage backend that persists Sifaka results. In this example:

- Shows the complete `StorageBackend` interface implementation
- Demonstrates async methods for CRUD operations
- Includes search functionality
- Shows both file-based mock and production Redis patterns

**Use cases:**
- Persistent storage across sessions
- Distributed caching for team collaboration
- Integration with existing data infrastructure

### 2. Custom Critics (`custom_critic.py`)

Contains two domain-specific critics:

#### Academic Writing Critic
- Improves text for academic publications
- Enforces citation standards (APA, MLA, etc.)
- Ensures formal tone and clear argumentation
- Validates logical structure

#### Technical Documentation Critic
- Optimizes API documentation
- Validates code examples
- Ensures completeness of parameter descriptions
- Maintains consistent documentation style

**Use cases:**
- Research paper refinement
- Technical writing improvement
- Documentation standardization
- Domain-specific content enhancement

### 3. Custom Validators (`custom_validator.py`)

Includes three specialized validators:

#### SEO Validator
- Checks keyword density
- Validates content length
- Ensures proper header structure
- Monitors meta descriptions

#### Code Quality Validator
- Validates syntax in code blocks
- Checks line length limits
- Enforces docstring requirements
- Language-specific validation

#### Accessibility Validator
- Ensures WCAG compliance
- Validates alt text presence
- Checks heading hierarchy
- Reviews link text clarity

**Use cases:**
- Content marketing optimization
- Documentation quality assurance
- Web accessibility compliance
- Code example validation

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

### Storage Backend Template

```python
from sifaka.storage.base import StorageBackend

class MyStorageBackend(StorageBackend):
    async def save(self, result):
        # Implementation
        pass

    async def load(self, result_id):
        # Implementation
        pass

    async def list(self, limit=100, offset=0):
        # Implementation
        pass

    async def delete(self, result_id):
        # Implementation
        pass

    async def search(self, query, limit=10):
        # Implementation
        pass
```

### Critic Template

```python
from sifaka.critics.core.base import BaseCritic

class MyCritic(BaseCritic):
    name = "my_critic"

    def _build_critique_prompt(self, original_text, current_text, attempt, metadata=None):
        # Return critique prompt
        pass

    def _build_improvement_prompt(self, original_text, current_text, critique, attempt, metadata=None):
        # Return improvement prompt
        pass
```

### Validator Template

```python
from sifaka.validators.base import BaseValidator

class MyValidator(BaseValidator):
    async def validate(self, text, metadata=None):
        # Return ValidationResult
        pass
```

## Best Practices

1. **Error Handling**: Always handle potential errors gracefully
2. **Async Support**: Use async/await for I/O operations
3. **Configuration**: Accept configuration parameters in `__init__`
4. **Documentation**: Document your plugin's purpose and parameters
5. **Testing**: Include tests for your plugins
6. **Dependencies**: Declare any additional dependencies clearly

## Testing Plugins

Run the example files directly to see basic functionality:

```bash
python custom_storage_backend.py
python custom_critic.py
python custom_validator.py
```

For full integration testing, you'll need to provide API keys for the LLM providers.
