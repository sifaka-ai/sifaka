# Domain Configuration

The domain module in Sifaka provides a flexible way to define and manage different domains of operation, each with its own set of rules, models, and critics.

## Overview

A domain in Sifaka represents a specific area of operation (e.g., "code", "text", "image") with its own:
- Configuration settings
- Rule sets
- Model providers
- Critics
- Validation logic

## Configuration Structure

```python
from sifaka.domain import Domain

# Example domain configuration
domain_config = {
    "name": "code",
    "description": "Code generation and analysis domain",
    "rules": {
        "syntax": {
            "enabled": True,
            "config": {
                "language": "python",
                "strict": True
            }
        },
        "security": {
            "enabled": True,
            "config": {
                "check_sql_injection": True,
                "check_xss": True
            }
        }
    },
    "models": {
        "default": {
            "provider": "openai",
            "config": {
                "model": "gpt-4",
                "temperature": 0.7
            }
        }
    },
    "critics": {
        "code_quality": {
            "enabled": True,
            "config": {
                "check_complexity": True,
                "check_style": True
            }
        }
    }
}

# Create domain instance
domain = Domain(config=domain_config)
```

## Configuration Options

### Domain Level
- `name`: Unique identifier for the domain
- `description`: Human-readable description
- `enabled`: Whether the domain is active (default: True)

### Rules Configuration
Each rule can be configured with:
- `enabled`: Whether the rule is active
- `config`: Rule-specific configuration
- `priority`: Rule execution priority (higher numbers execute first)

### Model Configuration
Model providers can be configured with:
- `provider`: Name of the model provider
- `config`: Provider-specific configuration
- `fallback`: Alternative provider to use if primary fails

### Critic Configuration
Critics can be configured with:
- `enabled`: Whether the critic is active
- `config`: Critic-specific configuration
- `threshold`: Minimum score for acceptance

## Usage Examples

### Basic Domain Creation
```python
from sifaka.domain import Domain

# Create a simple domain
domain = Domain({
    "name": "text",
    "description": "Text generation domain",
    "rules": {
        "length": {"enabled": True}
    }
})
```

### Domain with Multiple Rules
```python
domain = Domain({
    "name": "code",
    "rules": {
        "syntax": {
            "enabled": True,
            "config": {"language": "python"}
        },
        "security": {
            "enabled": True,
            "priority": 1
        }
    }
})
```

### Domain with Custom Models
```python
domain = Domain({
    "name": "image",
    "models": {
        "primary": {
            "provider": "stability",
            "config": {"model": "stable-diffusion-v2"}
        },
        "fallback": {
            "provider": "openai",
            "config": {"model": "dall-e-2"}
        }
    }
})
```

## Validation

The domain configuration is validated using Pydantic models to ensure:
- Required fields are present
- Field types are correct
- Configuration values are within valid ranges
- Dependencies between components are satisfied

Example validation:
```python
from sifaka.domain import DomainConfig

# Validate configuration
config = DomainConfig(**domain_config)
```

## Error Handling

The domain module provides comprehensive error handling:
- Configuration validation errors
- Rule execution errors
- Model provider errors
- Critic evaluation errors

Example error handling:
```python
try:
    domain = Domain(config=domain_config)
except ValidationError as e:
    print(f"Configuration error: {e}")
except DomainError as e:
    print(f"Domain error: {e}")
```

## Best Practices

1. **Configuration Management**
   - Use environment variables for sensitive data
   - Keep configurations in version control
   - Document all configuration options

2. **Rule Organization**
   - Group related rules together
   - Use meaningful rule names
   - Set appropriate priorities

3. **Model Selection**
   - Choose models appropriate for the domain
   - Configure fallback options
   - Monitor model performance

4. **Critic Configuration**
   - Set appropriate thresholds
   - Balance critic coverage
   - Monitor critic effectiveness

## Testing

See [Testing Guide](testing/README.md) for information on testing domain configurations.

## API Reference

For detailed API information, see [API Reference](api/README.md).