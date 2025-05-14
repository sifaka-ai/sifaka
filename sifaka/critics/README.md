# Sifaka Critics

This package provides components for critiquing, validating, and improving text using language models. Critics are central to the feedback and improvement mechanisms in Sifaka.

## Architecture

The critics architecture follows a component-based design for maximum flexibility and extensibility:

```
CriticCore
├── PromptManager (creates and manages prompts)
│   ├── DefaultPromptManager
│   ├── PromptCriticPromptManager
│   └── ReflexionCriticPromptManager
├── ResponseParser (parses model responses)
├── MemoryManager (manages interaction history)
└── CritiqueService (handles critiquing operations)
```

### Core Components

- **CriticCore**: Main component that delegates to specialized sub-components
- **PromptManager**: Creates and manages prompts for different operations
- **ResponseParser**: Parses structured responses from language models
- **MemoryManager**: Manages history and state for stateful critics
- **CritiqueService**: Coordinates critiquing, validation, and improvement operations

### Critic Implementations

- **PromptCritic**: Basic critic that uses prompts to guide LLM responses
- **ReflexionCritic**: Uses reflections on past feedback to improve text
- **ConstitutionalCritic**: Applies constitutional principles for content moderation
- **SelfRefineCritic**: Self-refines its own responses through multiple iterations
- **SelfRAGCritic**: Uses retrieval augmented generation to improve responses
- **FeedbackCritic**: Incorporates external feedback into improvement
- **ValueCritic**: Evaluates content based on specified values or principles
- **LACritic**: Language-action critic for improved goal-oriented evaluation

## Usage

### Basic Usage

```python
from sifaka.critics import create_prompt_critic, create_reflexion_critic
from sifaka.models import OpenAIProvider

# Create a model provider
model = OpenAIProvider("gpt-4")

# Create a prompt critic
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text.",
    min_confidence=0.7,
    max_attempts=3,
)

# Validate text
is_valid = critic.validate("This is a test document.")
print(f"Is valid: {is_valid}")

# Critique text
critique = critic.critique("This is a test document.")
print(f"Score: {critique.score}")
print(f"Feedback: {critique.feedback}")
print(f"Issues: {critique.issues}")
print(f"Suggestions: {critique.suggestions}")

# Improve text with feedback
improved = critic.improve_with_feedback(
    "This is a test document.",
    "The document is too brief and lacks details."
)
print(f"Improved: {improved}")
```

### Using Different Critic Types

```python
from sifaka.critics import (
    create_prompt_critic,
    create_reflexion_critic,
    create_constitutional_critic,
    create_self_refine_critic,
    create_self_rag_critic
)
from sifaka.models import OpenAIProvider

# Create model provider
model = OpenAIProvider("gpt-4")

# Create a reflexion critic
reflexion_critic = create_reflexion_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text.",
    min_confidence=0.7,
    max_attempts=3,
    reflection_count=2
)

# Create a constitutional critic
constitutional_critic = create_constitutional_critic(
    llm_provider=model,
    principles=[
        "Avoid political bias",
        "Maintain factual accuracy",
        "Be respectful and inclusive"
    ]
)

# Create a self-refine critic
self_refine_critic = create_self_refine_critic(
    llm_provider=model,
    refine_iterations=3
)

# Create a self-RAG critic
self_rag_critic = create_self_rag_critic(
    llm_provider=model,
    retrieval_context="Provide factual information about the topic."
)

# Use the critics
text = "This is a draft document about climate change."
result1 = reflexion_critic.improve_with_feedback(text, "Add more scientific context")
result2 = constitutional_critic.improve_with_feedback(text, "Ensure political neutrality")
result3 = self_refine_critic.improve_with_feedback(text, "Improve clarity and precision")
result4 = self_rag_critic.improve_with_feedback(text, "Add factual references")
```

### Advanced Configuration

```python
from sifaka.critics import CriticCore, CriticConfig
from sifaka.critics.managers import ResponseParser
from sifaka.core.managers.memory import BufferMemoryManager
from sifaka.core.managers.prompt import PromptCriticPromptManager
from sifaka.models import OpenAIProvider

# Create model provider
model = OpenAIProvider("gpt-4")

# Create configuration
config = CriticConfig(
    name="advanced_critic",
    description="Advanced critic with custom components",
    min_confidence=0.7,
    max_attempts=3,
    params={
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": "You are an expert technical editor."
    }
)

# Create specialized components
prompt_manager = PromptCriticPromptManager(config=config)
response_parser = ResponseParser()
memory_manager = BufferMemoryManager(buffer_size=10)

# Create critic with custom components
critic = CriticCore(
    config=config,
    llm_provider=model,
    prompt_manager=prompt_manager,
    response_parser=response_parser,
    memory_manager=memory_manager
)

# Use the critic
result = critic.critique("This is a technical document about AI systems.")
```

### Retrieving Statistics

```python
from sifaka.critics import create_prompt_critic
from sifaka.models import OpenAIProvider

# Create critic
model = OpenAIProvider("gpt-4")
critic = create_prompt_critic(llm_provider=model)

# Use the critic multiple times
critic.validate("Document 1")
critic.critique("Document 2")
critic.improve_with_feedback("Document 3", "Make it more concise")

# Get statistics
stats = critic.get_statistics()
print(f"Validation count: {stats['validation_count']}")
print(f"Critique count: {stats['critique_count']}")
print(f"Improvement count: {stats['improvement_count']}")
print(f"Average critique time: {stats['avg_critique_time']:.2f}s")
print(f"Score distribution: {stats['score_distribution']}")
```

## Extending

### Creating a Custom Prompt Manager

The prompt manager is the easiest component to customize for your specific use case:

```python
from sifaka.core.managers.prompt import PromptManager
from sifaka.utils.config.critics import CriticConfig

class CustomPromptManager(PromptManager):
    def __init__(self, config: CriticConfig):
        super().__init__(config)

    def _create_validation_prompt_impl(self, text: str) -> str:
        """Create a prompt for validating text."""
        return f"""
        You are an expert validator. Determine if the following text meets quality standards.

        TEXT: {text}

        VALID: [true/false]
        REASON: [explanation]
        """

    def _create_critique_prompt_impl(self, text: str) -> str:
        """Create a prompt for critiquing text."""
        return f"""
        You are an expert critic. Evaluate the following text and provide detailed feedback.

        TEXT: {text}

        SCORE: [0-1]
        FEEDBACK: [general feedback]
        ISSUES:
        - [issue 1]
        - [issue 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]
        """

    def _create_improvement_prompt_impl(self, text: str, feedback: str, reflections=None) -> str:
        """Create a prompt for improving text based on feedback."""
        reflections_text = ""
        if reflections:
            reflections_text = "REFLECTIONS:\n" + "\n".join(f"- {r}" for r in reflections)

        return f"""
        You are an expert editor. Improve the following text based on the provided feedback.

        TEXT: {text}

        FEEDBACK: {feedback}

        {reflections_text}

        IMPROVED_TEXT: [improved version]
        """
```

### Using the Custom Prompt Manager

```python
from sifaka.critics import CriticCore, CriticConfig
from sifaka.models import OpenAIProvider

# Create config and model
config = CriticConfig(name="custom_critic")
model = OpenAIProvider("gpt-4")

# Create custom prompt manager
from my_module import CustomPromptManager
prompt_manager = CustomPromptManager(config)

# Create critic with custom prompt manager
critic = CriticCore(
    config=config,
    llm_provider=model,
    prompt_manager=prompt_manager
)

# Use the critic
result = critic.critique("This is a test document.")
```
