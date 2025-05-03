# Sifaka Critics

This package provides components for critiquing, validating, and improving text using language models.

## Architecture

The critics architecture follows the Single Responsibility Principle by breaking down functionality into smaller, focused components:

```
CriticCore
├── PromptManager
├── ResponseParser
├── MemoryManager
└── CritiqueService
```

### Core Components

- **CriticCore**: Main interface that delegates to specialized components
- **PromptManager**: Manages prompt creation and management
- **ResponseParser**: Handles parsing responses from language models
- **MemoryManager**: Handles memory management for critics
- **CritiqueService**: Handles critiquing, validating, and improving text

### Critic Types

- **PromptCritic**: Uses language models to evaluate and improve text
- **ReflexionCritic**: Uses reflections on past feedback to improve text

## Usage

### Basic Usage

```python
from sifaka.critics import create_prompt_critic, create_reflexion_critic

# Create a prompt critic
critic = create_prompt_critic(
    llm_provider=model,
    name="my_critic",
    description="My custom critic",
    system_prompt="You are an expert editor that improves text.",
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
)

# Validate text
is_valid = critic.validate("This is a test.")

# Critique text
critique = critic.critique("This is a test.")
print(f"Score: {critique.score}")
print(f"Feedback: {critique.feedback}")
print(f"Issues: {critique.issues}")
print(f"Suggestions: {critique.suggestions}")

# Improve text
improved = critic.improve_with_feedback("This is a test.", "Make it more formal.")
print(f"Improved: {improved}")
```

### Advanced Usage

```python
from sifaka.critics import CriticCore, CriticConfig
from sifaka.critics.managers import PromptCriticPromptManager, ResponseParser, MemoryManager
from sifaka.critics.services import CritiqueService

# Create configuration
config = CriticConfig(
    name="advanced_critic",
    description="Advanced critic with custom components",
    min_confidence=0.7,
    max_attempts=3,
)

# Create managers
prompt_manager = PromptCriticPromptManager(config)
response_parser = ResponseParser()
memory_manager = MemoryManager(buffer_size=10)

# Create critic
critic = CriticCore(
    config=config,
    llm_provider=model,
    prompt_manager=prompt_manager,
    response_parser=response_parser,
    memory_manager=memory_manager,
)

# Use critic
result = critic.critique("This is a test.")
```

## Extending

To create a custom critic, you can extend the `PromptManager` class to customize prompts:

```python
from sifaka.critics.managers import PromptManager

class MyCustomPromptManager(PromptManager):
    def _create_validation_prompt_impl(self, text: str) -> str:
        return f"""Custom validation prompt:
        
        TEXT: {text}
        
        VALID: [true/false]
        """
    
    def _create_critique_prompt_impl(self, text: str) -> str:
        return f"""Custom critique prompt:
        
        TEXT: {text}
        
        SCORE: [0-1]
        FEEDBACK: [feedback]
        """
    
    def _create_improvement_prompt_impl(self, text: str, feedback: str, reflections=None) -> str:
        return f"""Custom improvement prompt:
        
        TEXT: {text}
        FEEDBACK: {feedback}
        
        IMPROVED_TEXT: [improved]
        """
    
    def _create_reflection_prompt_impl(self, original_text: str, feedback: str, improved_text: str) -> str:
        return f"""Custom reflection prompt:
        
        ORIGINAL: {original_text}
        FEEDBACK: {feedback}
        IMPROVED: {improved_text}
        
        REFLECTION: [reflection]
        """
```

Then use your custom prompt manager with `CriticCore`:

```python
critic = CriticCore(
    config=config,
    llm_provider=model,
    prompt_manager=MyCustomPromptManager(config),
)
```
