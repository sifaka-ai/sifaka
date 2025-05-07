# Critics API Reference

This document provides detailed API reference for all Critics in Sifaka.

## Overview

Critics are components in Sifaka that provide feedback and suggestions for improving text. They analyze text, identify issues, and generate improvements, serving as a key component in the Sifaka feedback loop. Critics work alongside rules and classifiers to create a complete validation and improvement system.

Critics implement three main interfaces:
- **TextValidator**: Validates text against specific criteria
- **TextImprover**: Improves text based on identified issues
- **TextCritic**: Provides detailed feedback and suggestions

## Core Interfaces

### BaseCritic

`BaseCritic` is the abstract base class for all critics in Sifaka.

```python
from sifaka.critics.base import BaseCritic, CriticConfig, CriticMetadata
from typing import List, Dict, Any

class MyCritic(BaseCritic[str, str]):
    """Custom critic implementation."""

    def validate(self, text: str) -> bool:
        """Validate the text."""
        return len(text) > 10

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Improve the text based on violations."""
        if len(text) <= 10:
            return text + " Additional content to make it longer."
        return text

    def critique(self, text: str) -> CriticMetadata[str]:
        """Critique the text."""
        if len(text) <= 10:
            return CriticMetadata(
                score=0.5,
                feedback="Text is too short",
                issues=["Text length is below minimum"],
                suggestions=["Add more content"]
            )
        return CriticMetadata(
            score=0.9,
            feedback="Text length is good",
            issues=[],
            suggestions=[]
        )
```

### TextValidator

`TextValidator` is a protocol for text validation components.

```python
from sifaka.critics.base import TextValidator

class MyValidator(TextValidator[str]):
    """Custom text validator implementation."""

    def validate(self, text: str) -> bool:
        """Validate the text."""
        return len(text) > 10
```

### TextImprover

`TextImprover` is a protocol for text improvement components.

```python
from sifaka.critics.base import TextImprover
from typing import List, Dict, Any

class MyImprover(TextImprover[str, str]):
    """Custom text improver implementation."""

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Improve the text based on violations."""
        if any(v.get("issue") == "too_short" for v in violations):
            return text + " Additional content to make it longer."
        return text
```

### TextCritic

`TextCritic` is a protocol for text critiquing components.

```python
from sifaka.critics.base import TextCritic, CriticMetadata

class MyCritic(TextCritic[str, str]):
    """Custom text critic implementation."""

    def critique(self, text: str) -> CriticMetadata[str]:
        """Critique the text."""
        if len(text) <= 10:
            return CriticMetadata(
                score=0.5,
                feedback="Text is too short",
                issues=["Text length is below minimum"],
                suggestions=["Add more content"]
            )
        return CriticMetadata(
            score=0.9,
            feedback="Text length is good",
            issues=[],
            suggestions=[]
        )
```

## Configuration

### CriticConfig

`CriticConfig` is the configuration class for critics.

```python
from sifaka.critics.base import CriticConfig

# Create a critic configuration
config = CriticConfig(
    name="my_critic",
    description="A custom critic",
    min_confidence=0.7,
    max_attempts=3,
    params={
        "min_length": 10,
    }
)

# Access configuration values
print(f"Name: {config.name}")
print(f"Min confidence: {config.min_confidence}")
print(f"Max attempts: {config.max_attempts}")
print(f"Min length: {config.params['min_length']}")

# Create a new configuration with updated options
updated_config = config.with_options(
    min_confidence=0.8,
    params={"min_length": 20}
)
```

## Results

### CriticMetadata

`CriticMetadata` represents the result of a critique.

```python
from sifaka.critics.base import CriticMetadata

# Create critic metadata
metadata = CriticMetadata(
    score=0.7,
    feedback="Text needs improvement",
    issues=["Text is too short", "Text lacks detail"],
    suggestions=["Add more content", "Include specific examples"],
    processing_time_ms=150
)

# Access metadata values
print(f"Score: {metadata.score}")
print(f"Feedback: {metadata.feedback}")
print(f"Issues: {metadata.issues}")
print(f"Suggestions: {metadata.suggestions}")
print(f"Processing time: {metadata.processing_time_ms} ms")
```

### CriticOutput

`CriticOutput` represents the output of a critic operation.

```python
from sifaka.critics.base import CriticOutput, CriticMetadata

# Create critic metadata
metadata = CriticMetadata(
    score=0.7,
    feedback="Text needs improvement",
    issues=["Text is too short"],
    suggestions=["Add more content"]
)

# Create critic output
output = CriticOutput(
    original_text="Short text",
    improved_text="Short text with additional content",
    metadata=metadata,
    attempt=1,
    max_attempts=3
)

# Access output values
print(f"Original text: {output.original_text}")
print(f"Improved text: {output.improved_text}")
print(f"Score: {output.metadata.score}")
print(f"Attempt: {output.attempt} of {output.max_attempts}")
```

## Factory Functions

Sifaka provides factory functions for creating critics. Always use these factory functions instead of instantiating critic classes directly.

### create_prompt_critic

```python
def create_prompt_critic(
    system_prompt: str,
    name: str = "prompt_critic",
    description: str = "A prompt-based critic for text improvement",
    model: Optional[ModelProvider] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    **kwargs
) -> BaseCritic[str, str]:
    """
    Create a prompt-based critic.

    Args:
        system_prompt: System prompt to guide the model
        name: Name of the critic
        description: Description of the critic
        model: Model provider to use (if None, uses default OpenAI)
        temperature: Temperature for text generation
        max_tokens: Maximum tokens for text generation
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        **kwargs: Additional keyword arguments

    Returns:
        A prompt critic instance

    Raises:
        ValueError: If parameters are invalid
    """
```

### create_reflexion_critic

```python
def create_reflexion_critic(
    model: ModelProvider,
    name: str = "reflexion_critic",
    description: str = "A reflexion-based critic for text improvement",
    memory_size: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    **kwargs
) -> BaseCritic[str, str]:
    """
    Create a reflexion-based critic.

    Args:
        model: Model provider to use
        name: Name of the critic
        description: Description of the critic
        memory_size: Number of past interactions to remember
        temperature: Temperature for text generation
        max_tokens: Maximum tokens for text generation
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        **kwargs: Additional keyword arguments

    Returns:
        A reflexion critic instance

    Raises:
        ValueError: If parameters are invalid
    """
```

### create_self_refine_critic

```python
def create_self_refine_critic(
    model: ModelProvider,
    name: str = "self_refine_critic",
    description: str = "A self-refine critic for text improvement",
    max_iterations: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    **kwargs
) -> BaseCritic[str, str]:
    """
    Create a self-refine critic.

    Args:
        model: Model provider to use
        name: Name of the critic
        description: Description of the critic
        max_iterations: Maximum number of refinement iterations
        temperature: Temperature for text generation
        max_tokens: Maximum tokens for text generation
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        **kwargs: Additional keyword arguments

    Returns:
        A self-refine critic instance

    Raises:
        ValueError: If parameters are invalid
    """
```

### create_constitutional_critic

```python
def create_constitutional_critic(
    model: ModelProvider,
    constitution: List[str],
    name: str = "constitutional_critic",
    description: str = "A constitutional critic for text improvement",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    **kwargs
) -> BaseCritic[str, str]:
    """
    Create a constitutional critic.

    Args:
        model: Model provider to use
        constitution: List of constitutional principles
        name: Name of the critic
        description: Description of the critic
        temperature: Temperature for text generation
        max_tokens: Maximum tokens for text generation
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        **kwargs: Additional keyword arguments

    Returns:
        A constitutional critic instance

    Raises:
        ValueError: If parameters are invalid
    """
```

## Critic Types

Sifaka provides several types of critics:

### PromptCritic

`PromptCritic` uses prompt-based guidance to improve text.

```python
from sifaka.critics.prompt import create_prompt_critic

# Create a prompt critic
critic = create_prompt_critic(
    system_prompt="You are an expert editor. Improve the text to make it more concise and clear.",
    name="prompt_critic",
    description="Improves text clarity and conciseness"
)
```

### ReflexionCritic

`ReflexionCritic` uses reflection and memory to improve text over time.

```python
from sifaka.critics.reflexion import create_reflexion_critic
from sifaka.models.openai import create_openai_chat_provider

# Create a model provider
model = create_openai_chat_provider(model_name="gpt-4")

# Create a reflexion critic
critic = create_reflexion_critic(
    model=model,
    name="reflexion_critic",
    description="Improves text through reflection and memory"
)
```

### SelfRefineCritic

`SelfRefineCritic` uses iterative self-improvement to refine text.

```python
from sifaka.critics.self_refine import create_self_refine_critic
from sifaka.models.openai import create_openai_chat_provider

# Create a model provider
model = create_openai_chat_provider(model_name="gpt-4")

# Create a self-refine critic
critic = create_self_refine_critic(
    model=model,
    name="self_refine_critic",
    description="Improves text through iterative self-refinement",
    max_iterations=3
)
```

### ConstitutionalCritic

`ConstitutionalCritic` uses constitutional principles to guide text improvement.

```python
from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.models.openai import create_openai_chat_provider

# Create a model provider
model = create_openai_chat_provider(model_name="gpt-4")

# Create a constitutional critic
critic = create_constitutional_critic(
    model=model,
    constitution=[
        "Text should be respectful and avoid harmful content",
        "Text should be factually accurate",
        "Text should be clear and concise"
    ],
    name="constitutional_critic",
    description="Improves text based on constitutional principles"
)
```

## Usage Examples

### Basic Critic Usage

```python
from sifaka.critics.prompt import create_prompt_critic

# Create a critic
critic = create_prompt_critic(
    system_prompt="You are an expert editor. Improve the text to make it more concise and clear."
)

# Critique text
metadata = critic.critique("This is a very long and verbose text that could be made more concise.")
print(f"Score: {metadata.score}")
print(f"Feedback: {metadata.feedback}")
print(f"Issues: {metadata.issues}")
print(f"Suggestions: {metadata.suggestions}")

# Improve text
improved_text = critic.improve(
    "This is a very long and verbose text that could be made more concise.",
    violations=[{"issue": "verbose"}]
)
print(f"Improved text: {improved_text}")
```

### Custom Critic Implementation

```python
from sifaka.critics.base import BaseCritic, CriticConfig, CriticMetadata
from typing import List, Dict, Any

class LengthCritic(BaseCritic[str, str]):
    """Critic for text length."""

    def __init__(self, config: CriticConfig):
        super().__init__(config)
        self.min_length = config.params.get("min_length", 10)
        self.max_length = config.params.get("max_length", 100)

    def validate(self, text: str) -> bool:
        """Validate the text length."""
        return self.min_length <= len(text) <= self.max_length

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Improve the text length."""
        if len(text) < self.min_length:
            return text + " " + "Additional content to reach minimum length." * (
                (self.min_length - len(text)) // 40 + 1
            )
        if len(text) > self.max_length:
            return text[:self.max_length - 3] + "..."
        return text

    def critique(self, text: str) -> CriticMetadata[str]:
        """Critique the text length."""
        if len(text) < self.min_length:
            return CriticMetadata(
                score=0.5,
                feedback="Text is too short",
                issues=[f"Text length ({len(text)}) is below minimum ({self.min_length})"],
                suggestions=["Add more content to reach the minimum length"]
            )
        if len(text) > self.max_length:
            return CriticMetadata(
                score=0.5,
                feedback="Text is too long",
                issues=[f"Text length ({len(text)}) exceeds maximum ({self.max_length})"],
                suggestions=["Reduce content to stay within the maximum length"]
            )
        return CriticMetadata(
            score=1.0,
            feedback="Text length is good",
            issues=[],
            suggestions=[]
        )

# Create the critic
critic = LengthCritic(
    CriticConfig(
        name="length_critic",
        description="Ensures text is the right length",
        params={"min_length": 20, "max_length": 100}
    )
)

# Use the critic
text = "This is a test"
if not critic.validate(text):
    metadata = critic.critique(text)
    print(f"Issues: {metadata.issues}")
    improved_text = critic.improve(text, [{"issue": "too_short"}])
    print(f"Improved text: {improved_text}")
```

### Using Critics with Chains

Critics are typically used in chains to improve text that fails validation:

```python
from sifaka.chain import create_simple_chain
from sifaka.models.openai import create_openai_chat_provider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import create_prompt_critic

# Create components
model = create_openai_chat_provider(model_name="gpt-4")
rule = create_length_rule(min_chars=50, max_chars=200)
critic = create_prompt_critic(
    system_prompt="You are an expert editor. Improve the text to meet the length requirements."
)

# Create a chain
chain = create_simple_chain(
    model=model,
    rules=[rule],
    critic=critic
)

# Run the chain
result = chain.run("Write a short description of a sunset.")
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
```

## Implementation Details

Critics in Sifaka follow a standardized implementation pattern:

1. **State Management**: Critics use the `_state_manager` pattern for managing state
2. **Configuration**: Critics use `CriticConfig` for configuration
3. **Factory Functions**: Critics provide factory functions for easy instantiation
4. **Interfaces**: Critics implement the `TextValidator`, `TextImprover`, and `TextCritic` interfaces

### State Management

Critics use the `_state_manager` pattern for managing state:

```python
from pydantic import PrivateAttr
from sifaka.critics.base import BaseCritic, CriticConfig, create_critic_state

class MyCritic(BaseCritic[str, str]):
    """Custom critic implementation."""

    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(self, config: CriticConfig):
        super().__init__(config)
        # Initialize any critic-specific attributes

    def warm_up(self):
        """Initialize expensive resources."""
        state = self._state_manager.get_state()
        if not state.initialized:
            # Initialize state
            state.initialized = True

    def validate(self, text: str) -> bool:
        """Validate the text."""
        state = self._state_manager.get_state()
        # Use state for validation
        return True
```

## Best Practices

1. **Use factory functions** for creating critics
2. **Use standardized state management** with `_state_manager`
3. **Implement all required interfaces** (`TextValidator`, `TextImprover`, `TextCritic`)
4. **Handle empty text gracefully** in all methods
5. **Include detailed metadata** in critique results
6. **Use appropriate temperature settings** for language models
7. **Set reasonable max_attempts** to prevent infinite loops
8. **Document prompt templates** in docstrings
9. **Use system prompts** to guide language model behavior
10. **Implement warm_up()** for lazy initialization of expensive resources

## Error Handling

Critics implement several error handling patterns:

### Handling Empty Text

```python
def validate(self, text: str) -> bool:
    """Validate the text."""
    if not text:
        return True  # Empty text is valid by default

    # Normal validation logic
    return len(text) > 10
```

### Handling Model Errors

```python
def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
    """Improve the text."""
    try:
        # Try to improve with model
        return self._improve_with_model(text, violations)
    except ModelError:
        # Fall back to simple improvement
        return self._improve_fallback(text, violations)
```

### Handling Timeouts

```python
def critique(self, text: str) -> CriticMetadata[str]:
    """Critique the text."""
    try:
        # Try to critique with timeout
        return self._critique_with_timeout(text, timeout=10.0)
    except TimeoutError:
        # Return basic metadata
        return CriticMetadata(
            score=0.5,
            feedback="Critique timed out",
            issues=["Timeout occurred during critique"],
            suggestions=[]
        )
```

## See Also

- [Critics Component Documentation](../../components/critics.md)
- [Implementation Notes for Critics](../../implementation_notes/prompt_critic.md)
- [Chain API Reference](../chain/README.md)
