# Self-RAG Critic

The Self-RAG Critic implements the Self-Reflective Retrieval-Augmented Generation approach, which enables language models to decide when and what to retrieve, and reflect on the relevance and utility of the retrieved information.

Based on the Self-RAG paper: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

## Key Concepts

| Component               | Description                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Adaptive Retrieval**  | Before retrieving, the model reflects: *"Do I need external info?"* If yes, it defines the retrieval query.        |
| **Reflection Tokens**   | Markers in the prompt/response (e.g., `<reflect>`, `<retrieve>`) that segment when reflection or retrieval occurs. |
| **Post-Hoc Evaluation** | After generation, a critic reviews whether retrieval was necessary, sufficient, and used correctly.                |

## Usage

```python
from sifaka.critics.self_rag import create_self_rag_critic
from sifaka.models.providers import OpenAIProvider
from sifaka.retrieval import create_simple_retriever

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a retriever with a document collection
documents = {
    "health insurance": "To file a claim for health reimbursement, follow these steps: 1) Complete the claim form...",
    "travel insurance": "For travel insurance claims, you need to provide: 1) Proof of travel 2) Incident report..."
}
retriever = create_simple_retriever(documents=documents)

# Create a self-rag critic
critic = create_self_rag_critic(
    llm_provider=provider,
    retriever=retriever,
    name="insurance_rag_critic",
    description="A critic for insurance-related queries",
    system_prompt="You are an expert at retrieving and using insurance information.",
    temperature=0.7,
    max_tokens=1000
)

# Use the critic to process a task
task = "What are the steps to file a claim for health reimbursement?"
result = critic.run(task)

# Access the results
print(f"Response: {result['response']}")
print(f"Retrieval Query: {result['retrieval_query']}")
print(f"Retrieved Context: {result['retrieved_context']}")
print(f"Reflection: {result['reflection']}")
```

## Configuration

The `SelfRAGCriticConfig` class provides configuration options for the Self-RAG critic:

```python
from sifaka.critics.self_rag import SelfRAGCriticConfig

config = SelfRAGCriticConfig(
    name="custom_rag_critic",
    description="Custom RAG critic",
    system_prompt="You are an expert at deciding when to retrieve information.",
    temperature=0.8,
    max_tokens=1500,
    retrieval_threshold=0.7,
    retrieval_prompt_template="Custom retrieval prompt template...",
    generation_prompt_template="Custom generation prompt template...",
    reflection_prompt_template="Custom reflection prompt template..."
)
```

## Process Flow

The Self-RAG critic follows a four-step process:

1. **Retrieval Decision**: The model decides whether external information is needed and formulates a retrieval query.
2. **Information Retrieval**: If retrieval is needed, the retriever fetches relevant information.
3. **Response Generation**: The model generates a response using the retrieved information (if any).
4. **Self-Reflection**: The model reflects on the quality of its response and the utility of the retrieved information.

## Customization

### Custom Retriever

You can create a custom retriever by implementing the `RetrieverImplementation` protocol and using the `Retriever` class:

```python
from sifaka.retrieval import Retriever, RetrieverImplementation, RetrieverConfig

class MyCustomRetrieverImplementation:
    """Implementation of custom retrieval logic."""

    def __init__(self, config: RetrieverConfig):
        self.api_key = config.params.get("api_key")
        self.index_name = config.params.get("index_name")
        # Initialize your retrieval system

    def retrieve_impl(self, query: str) -> str:
        # Implement your retrieval logic
        # Return relevant information as a string
        return "Retrieved information..."

    def warm_up_impl(self) -> None:
        # Initialize any resources needed
        pass

# Create a factory function for your custom retriever
def create_my_custom_retriever(api_key: str, index_name: str, **kwargs):
    # Prepare params
    params = kwargs.pop("params", {})
    params.update({
        "api_key": api_key,
        "index_name": index_name,
    })

    # Create config
    config = RetrieverConfig(params=params)

    # Create implementation
    implementation = MyCustomRetrieverImplementation(config)

    # Create and return retriever
    return Retriever(
        name=kwargs.pop("name", "custom_retriever"),
        description=kwargs.pop("description", "Custom retriever implementation"),
        config=config,
        implementation=implementation,
        **kwargs,
    )
```

### Custom Prompt Templates

You can customize the prompt templates used by the Self-RAG critic:

```python
critic = create_self_rag_critic(
    llm_provider=provider,
    retriever=retriever,
    retrieval_prompt_template=(
        "Given the following task, determine if you need external information. "
        "If yes, provide a specific search query. If no, say 'No retrieval needed.'\n\n"
        "Task: {task}"
    ),
    generation_prompt_template=(
        "Use the following context (if available) to complete the task.\n\n"
        "Context: {context}\n\n"
        "Task: {task}\n\n"
        "Response:"
    ),
    reflection_prompt_template=(
        "Evaluate your response to the task. Consider:\n"
        "1. Did you use the retrieved information effectively?\n"
        "2. Is your response accurate and complete?\n"
        "3. What could be improved?\n\n"
        "Task: {task}\n\n"
        "Retrieved Context: {context}\n\n"
        "Your Response: {response}\n\n"
        "Reflection:"
    )
)
```

## Integration with Chains

The Self-RAG critic can be integrated into a chain for more complex workflows:

```python
from sifaka.chain import create_simple_chain
from sifaka.models.providers import OpenAIProvider
from sifaka.critics.self_rag import create_self_rag_critic
from sifaka.retrieval import create_simple_retriever

# Create components
provider = OpenAIProvider(api_key="your-api-key")
retriever = create_simple_retriever(documents=documents)
critic = create_self_rag_critic(llm_provider=provider, retriever=retriever)

# Create a chain with the critic
chain = create_simple_chain(
    model_provider=provider,
    critic=critic,
    max_attempts=3
)

# Run the chain
result = chain.run("What are the steps to file a claim for health reimbursement?")
```

## Best Practices

1. **Document Collection**: Ensure your document collection is relevant and well-structured for the tasks you expect to handle.

2. **Prompt Engineering**: Customize the prompt templates to guide the model's retrieval decisions and reflections.

3. **System Prompt**: Use the system prompt to set the tone and expertise level of the model.

4. **Retrieval Threshold**: Adjust the retrieval threshold based on your use case - higher values make the model more selective about when to retrieve.

5. **Error Handling**: Implement proper error handling for cases where retrieval fails or returns irrelevant information.

## Limitations

1. The quality of responses depends on the quality of the retrieval system and document collection.

2. The model may sometimes decide to retrieve when it's not necessary, or vice versa.

3. The reflection process adds computational overhead and token usage.

4. The current implementation does not support streaming responses.

## See Also

- [Retrieval Module](../retrieval/index.md)
- [Chain Module](../chain/index.md)
- [Model Providers](../models/index.md)
