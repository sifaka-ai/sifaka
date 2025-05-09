# Implementation Notes: Self-RAG Critic

This document provides implementation details and notes for the Self-RAG Critic in the Sifaka project.

## Overview

The Self-RAG Critic implements the Self-Reflective Retrieval-Augmented Generation approach, which enables language models to decide when and what to retrieve, and reflect on the relevance and utility of the retrieved information.

Based on: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

## Architecture

The SelfRAGCritic follows a component-based architecture with retrieval augmentation:

1. **Core Components**
   - **SelfRAGCritic**: Main class that implements the critic interfaces
   - **Retriever**: Component that retrieves relevant information
   - **PromptManager**: Creates prompts for different stages of the process
   - **ResponseParser**: Parses and validates model responses
   - **MemoryManager**: Manages history of retrievals and reflections

## Implementation Details

### State Management

The SelfRAGCritic uses direct state management with a `CriticState` object:

```python
# Initialize state
self._state = CriticState()

# Store components in state
self._state.model = llm_provider
self._state.retriever = retriever
self._state.cache = {
    "retrieval_threshold": config.retrieval_threshold,
    "retrieval_prompt_template": config.retrieval_prompt_template,
    "generation_prompt_template": config.generation_prompt_template,
    "reflection_prompt_template": config.reflection_prompt_template,
    "system_prompt": config.system_prompt,
    "temperature": config.temperature,
    "max_tokens": config.max_tokens,
    "reflection_enabled": config.reflection_enabled,
}
self._state.initialized = True
```

### Core Methods

The SelfRAGCritic implements these core methods:

1. **Validation and Critique**:
```python
def validate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool
def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
async def avalidate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool
async def acritique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
```

2. **Text Improvement**:
```python
def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str
def improve_with_feedback(self, text: str, feedback: str) -> str
async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str
```

3. **Core Process**:
```python
def run(self, task: str, response: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
async def arun(self, task: str, response: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
```

### Factory Function

The SelfRAGCritic provides a factory function for easy creation:

```python
def create_self_rag_critic(
    llm_provider: Any,
    retriever: Retriever,
    name: str = "self_rag_critic",
    description: str = "Improves text through self-reflective retrieval-augmented generation",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    retrieval_threshold: float = None,
    retrieval_prompt_template: Optional[str] = None,
    generation_prompt_template: Optional[str] = None,
    reflection_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], SelfRAGCriticConfig]] = None,
    **kwargs: Any,
) -> SelfRAGCritic
```

## Usage Example

```python
from sifaka.critics.implementations.self_rag import create_self_rag_critic
from sifaka.models.providers import OpenAIProvider
from sifaka.retrieval import SimpleRetriever

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a retriever with some documents
documents = [
    "Health insurance claims must be filed within 90 days of service.",
    "To file a claim, you need to submit the claim form and receipts.",
    "Claims can be submitted online or by mail."
]
retriever = SimpleRetriever(documents=documents)

# Create a Self-RAG critic
critic = create_self_rag_critic(
    llm_provider=provider,
    retriever=retriever
)

# Use the critic to improve text
task = "What are the steps to file a claim for health reimbursement?"
result = critic.run(task, response=None)
print(f"Response: {result['response']}")
print(f"Reflection: {result['reflection']}")
```

## Error Handling

The SelfRAGCritic handles these error cases:

1. **Initialization Errors**
   - Missing required parameters
   - Invalid provider type
   - Invalid retriever type
   - Invalid configuration values

2. **Validation Errors**
   - Empty text
   - Missing task in metadata
   - Uninitialized critic

3. **Generation Errors**
   - Model provider failures
   - Retrieval failures
   - Invalid prompt formatting
   - Response parsing errors

## Testing

The SelfRAGCritic includes comprehensive tests that verify:

1. Initialization with different configurations
2. Retrieval functionality
3. Text generation
4. Reflection generation
5. Async method behavior
6. Error handling
7. Memory management

## Future Improvements

Potential future improvements for the SelfRAGCritic include:

1. Adding support for more sophisticated retrieval strategies
2. Implementing parallel processing for multiple retrievals
3. Adding support for custom retrieval templates
4. Implementing more advanced memory management
5. Adding support for streaming responses

## References

- [Sifaka Critics Documentation](../components/critics.md)
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
