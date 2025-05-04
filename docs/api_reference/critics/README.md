# Critics API Reference

Critics are components in Sifaka that provide feedback and suggestions for improving text. They work alongside rules and classifiers to create a complete validation and improvement system.

## Core Classes and Protocols

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

### MultiClassifierCritic

`MultiClassifierCritic` uses multiple classifiers to guide text improvement.

```python
from sifaka.critics.multi_classifier import create_multi_classifier_critic
from sifaka.classifiers.toxicity import create_toxicity_classifier
from sifaka.classifiers.readability import create_readability_classifier
from sifaka.models.openai import create_openai_chat_provider

# Create classifiers
toxicity_classifier = create_toxicity_classifier()
readability_classifier = create_readability_classifier()

# Create a model provider
model = create_openai_chat_provider(model_name="gpt-4")

# Create a multi-classifier critic
critic = create_multi_classifier_critic(
    classifiers=[toxicity_classifier, readability_classifier],
    model=model,
    name="multi_classifier_critic",
    description="Improves text based on toxicity and readability"
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
