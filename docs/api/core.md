# Core API Documentation

This document provides detailed API documentation for Sifaka's core functionality.

## Rules and Validators

### Base Classes

#### Rule
```python
class Rule(Generic[T, R, V, H]):
    """Base class for all rules in Sifaka."""

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[RuleConfig] = None,
        validator: Optional[V] = None,
        **kwargs
    ):
        """
        Initialize a new rule.

        Args:
            name: Unique identifier for the rule
            description: Human-readable description
            config: Rule configuration
            validator: Validator instance
            **kwargs: Additional configuration
        """
        pass

    def validate(self, text: T, **kwargs) -> R:
        """
        Validate the given text.

        Args:
            text: Text to validate
            **kwargs: Additional validation parameters

        Returns:
            Validation result
        """
        pass
```

#### BaseValidator
```python
class BaseValidator(Generic[T]):
    """Base class for all validators in Sifaka."""

    def validate(self, text: T, **kwargs) -> RuleResult:
        """
        Validate the given text.

        Args:
            text: Text to validate
            **kwargs: Additional validation parameters

        Returns:
            Validation result
        """
        pass

    def handle_empty_text(self, text: T) -> Optional[RuleResult]:
        """
        Handle empty text cases.

        Args:
            text: Text to check

        Returns:
            RuleResult if text is empty, None otherwise
        """
        pass
```

### Example Usage

#### Creating a Custom Rule
```python
from sifaka.rules.base import Rule, RuleResult, RuleConfig, BaseValidator
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class CustomConfig(RuleConfig):
    """Configuration for custom rule."""
    min_length: int = 10
    max_length: int = 100

class CustomValidator(BaseValidator[str]):
    """Custom validator implementation."""

    def __init__(self, config: CustomConfig):
        self.config = config

    def validate(self, text: str, **kwargs) -> RuleResult:
        if len(text) < self.config.min_length:
            return RuleResult(
                passed=False,
                message=f"Text too short: {len(text)} chars (min {self.config.min_length})"
            )
        return RuleResult(passed=True, message="Validation passed")

class CustomRule(Rule[str, RuleResult, CustomValidator, None]):
    """Custom rule implementation."""

    def _create_default_validator(self) -> CustomValidator:
        config = CustomConfig(**self._rule_params)
        return CustomValidator(config)
```

#### Using the Rule
```python
# Create a rule instance
rule = CustomRule(
    name="custom_rule",
    description="Custom validation rule",
    config=CustomConfig(min_length=20, max_length=200)
)

# Validate text
result = rule.validate("This is a test")
print(f"Validation passed: {result.passed}")
print(f"Message: {result.message}")
```

## Model Providers

### Base Model Provider
```python
class ModelProvider(Generic[T]):
    """Base class for model providers."""

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generation result
        """
        pass

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        pass
```

### Example Usage
```python
from sifaka.models.base import ModelProvider

class CustomModelProvider(ModelProvider):
    """Custom model provider implementation."""

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return {
            "text": f"Generated response to: {prompt}",
            "tokens": len(prompt.split())
        }

    def count_tokens(self, text: str) -> int:
        return len(text.split())

# Create and use provider
provider = CustomModelProvider()
result = provider.generate("Test prompt")
print(f"Generated text: {result['text']}")
print(f"Token count: {result['tokens']}")
```

## Critics

### Base Critic
```python
class Critic(Generic[T]):
    """Base class for critics."""

    def critique(self, text: T, **kwargs) -> Dict[str, Any]:
        """
        Critique the given text.

        Args:
            text: Text to critique
            **kwargs: Additional critique parameters

        Returns:
            Critique result
        """
        pass

    def improve(self, text: T, violations: List[str], **kwargs) -> T:
        """
        Improve the given text.

        Args:
            text: Text to improve
            violations: List of violations to address
            **kwargs: Additional improvement parameters

        Returns:
            Improved text
        """
        pass
```

### Example Usage
```python
from sifaka.critics.base import Critic

class CustomCritic(Critic[str]):
    """Custom critic implementation."""

    def critique(self, text: str, **kwargs) -> Dict[str, Any]:
        return {
            "score": 0.8,
            "feedback": "Good text",
            "issues": ["Minor issue"],
            "suggestions": ["Minor improvement"]
        }

    def improve(self, text: str, violations: List[str], **kwargs) -> str:
        return f"Improved: {text}"

# Create and use critic
critic = CustomCritic()
result = critic.critique("Test text")
print(f"Score: {result['score']}")
print(f"Feedback: {result['feedback']}")
```

## Domains

### Base Domain
```python
class Domain(Generic[T]):
    """Base class for domains."""

    def __init__(self, config: DomainConfig):
        """
        Initialize a new domain.

        Args:
            config: Domain configuration
        """
        pass

    def validate(self, text: T, **kwargs) -> ValidationResult:
        """
        Validate text against domain rules.

        Args:
            text: Text to validate
            **kwargs: Additional validation parameters

        Returns:
            Validation result
        """
        pass
```

### Example Usage
```python
from sifaka.domain.base import Domain, DomainConfig
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.formatting.style import create_style_rule

# Create domain configuration
config = DomainConfig(
    name="custom_domain",
    description="Custom domain for testing",
    rules=[
        create_length_rule(min_chars=10, max_chars=100),
        create_style_rule(capitalization="sentence")
    ]
)

# Create and use domain
domain = Domain(config)
result = domain.validate("Test text")
print(f"Validation passed: {result.all_passed}")
print(f"Rule results: {result.rule_results}")
```