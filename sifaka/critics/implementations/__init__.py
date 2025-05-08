"""
Critic implementations for the composition over inheritance pattern.

This package provides implementations of the CriticImplementation protocol
for use with the CompositionCritic class. These implementations follow the
composition over inheritance pattern.

## Available Implementations

1. **PromptCriticImplementation**
   - Uses language models to evaluate and improve text
   - Configurable with system prompts and model parameters
   - Supports memory for tracking improvements

2. **ReflexionCriticImplementation**
   - Uses language models with memory to evaluate and improve text
   - Maintains reflections on past feedback to improve future generations
   - Configurable with memory buffer size and reflection depth

3. **SelfRefineCriticImplementation**
   - Uses language models to iteratively critique and revise text
   - Implements the Self-Refine approach for progressive improvement
   - Configurable with iteration count and specialized prompts

4. **SelfRAGCriticImplementation**
   - Uses language models with retrieval to evaluate and improve text
   - Implements the Self-RAG approach for retrieval-augmented generation
   - Configurable with retrieval settings and specialized prompts

## Usage

```python
from sifaka.critics.implementations import PromptCriticImplementation
from sifaka.critics.base import CompositionCritic, create_composition_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create an implementation
implementation = PromptCriticImplementation(
    config=PromptCriticConfig(
        name="my_critic",
        description="A critic for improving technical documentation",
        system_prompt="You are an expert technical writer.",
        temperature=0.7,
        max_tokens=1000
    ),
    llm_provider=provider
)

# Create a critic with the implementation
critic = create_composition_critic(
    name="my_critic",
    description="A critic for improving technical documentation",
    implementation=implementation
)
```
"""

from .prompt_implementation import PromptCriticImplementation
from .reflexion_implementation import ReflexionCriticImplementation
from .self_refine_implementation import SelfRefineCriticImplementation
from .self_rag_implementation import SelfRAGCriticImplementation

__all__ = [
    "PromptCriticImplementation",
    "ReflexionCriticImplementation",
    "SelfRefineCriticImplementation",
    "SelfRAGCriticImplementation",
]
