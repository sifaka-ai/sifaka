# Sifaka: A Framework for Reliable LLM Applications

Sifaka is a powerful framework for building reliable, robust, and responsible language model applications. It provides a modular architecture for text generation, validation, improvement, and evaluation with built-in guardrails.

## Core Concept

Sifaka's core concept is the **Chain** consisting of a **Thought** container, a **Model**, **Validators**, and **Critics**:

```
Chain = [Thought, Model, [Validators], [Critics]]
```

The Chain orchestrates the process of:

1. Creating a Thought with a Prompt
2. Retrieving relevant context for the prompt (optional)
3. Generating text using a language model with the prompt & context
4. Validating the generated text against specified criteria
5. If validation fails, improving the text using specialized critics
6. Repeating until validation passes or max attempts reached

## How Sifaka Works

### The Chain Process

```mermaid
flowchart LR
    Prompt([Prompt]) --> Model
    Model --> Validators
    Validators -->|Pass| Output([Final Output])
    Validators -->|Fail| Critics
    Critics --> Model

    %% Retriever connections
    Retriever([Retriever])
    Model <-.-> Retriever
    Critics <-.-> Retriever
```

1. **Input**: A prompt is provided to the Chain
2. **Model Generation**: The Model generates text based on the prompt and any retrieved context
3. **Validation**: Validators check if the generated text meets specified criteria
4. **Decision Point**:
   - If validation passes → Return the generated text as output
   - If validation fails → Send to Critics for improvement
5. **Critic Improvement**: Critics analyze the text, provide feedback, and suggest improvements
6. **Iteration**: The improved text is sent back to the Model for regeneration
7. **Repeat**: Steps 2-6 repeat until validation passes or max iterations are reached

### The Thought Container

The Thought container holds all information as it flows through the Chain:

- **Prompt**: The original user query or instruction
- **Context**: Information retrieved from external sources
  - **Pre-generation Context**: Retrieved before text generation (used by the Model)
  - **Post-generation Context**: Retrieved after text generation (used by Validators and Critics)
- **Generated Text**: The text produced by the Model
- **Validation Results**: Pass/fail status and details from each Validator
- **Critic Feedback**: Issues identified and suggestions from Critics
- **History**: Record of previous iterations

### Retriever Access

- **Both Models and Critics can directly call Retrievers**
- Models use Retrievers to get context before or during text generation
- Critics use Retrievers to get additional context when analyzing and improving text

## Key Components

- **Thought**: The central state container that passes information between all components
- **Chain**: The main orchestrator that coordinates the generation, validation, and improvement flow
- **Model**: Interface for text generation models (OpenAI, Anthropic, etc.)
  - Models can directly call retrievers to get additional context
- **Validators**: Components that check if text meets specific criteria (length, content, format, etc.)
- **Critics**: Specialized components that analyze and improve text quality
  - Critics can directly call retrievers to get additional context
  - **ReflexionCritic**: Uses reflection to improve text based on past feedback
  - **SelfRAGCritic**: Uses retrieval-augmented generation to improve text with external knowledge
  - **SelfRefineCritic**: Iteratively refines text through self-critique
  - **ConstitutionalCritic**: Ensures text adheres to specified principles
  - **PromptCritic**: General-purpose critic with customizable instructions
  - **NCriticsCritic**: Ensemble of specialized critics for comprehensive feedback
- **Retrievers**: Components that find relevant documents for context
  - Available to both models and critics
  - **InMemoryRetriever**: Simple in-memory document retrieval
  - **VectorDBRetriever**: Retrieval from vector databases (Milvus, Elasticsearch)

## Installation

```bash
pip install sifaka
pip install python-dotenv  # For loading environment variables
```

## Environment Setup

Sifaka requires API keys for the language models you want to use. You can set these as environment variables in your shell or use a `.env` file.

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

The examples will automatically load these environment variables using `python-dotenv`.

## Quick Start

```python
import os
from dotenv import load_dotenv
from sifaka.chain import Chain
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.critics.base import ReflexionCritic
from sifaka.models.base import create_model
from sifaka.retrievers.base import MockRetriever
from sifaka.core.thought import Thought

# Load environment variables from .env file if it exists
load_dotenv()

# Get API key from environment variables
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Create a model
model = create_model("openai:gpt-4", api_key=api_key)

# Create validators and critics
length_validator = LengthValidator(min_length=50, max_length=1000)
content_validator = RegexValidator(
    forbidden_patterns=["violent", "harmful"]
)
critic = ReflexionCritic(model=model)

# Create a retriever
retriever = MockRetriever()

# Create a chain with model, prompt, retriever, validators and critics
# The Chain orchestrates ALL retrieval automatically
prompt = "Write a short story about a robot."
chain = Chain(
    model=model,
    prompt=prompt,
    retriever=retriever,  # Chain handles all retrieval
    max_improvement_iterations=3,
    apply_improvers_on_validation_failure=True,
)

chain.validate_with(length_validator)
chain.validate_with(content_validator)
chain.improve_with(critic)

# Run the chain - it handles all retrieval automatically
result = chain.run()

# Check the result
print(f"Generated text: {result.text}")

# Access validation results
for name, validation_result in result.validation_results.items():
    print(f"{name}: {'Passed' if validation_result.passed else 'Failed'}")
    if validation_result.issues:
        print(f"Issues: {validation_result.issues}")
```

## Working with the Thought Container

The Thought container is the central state container in Sifaka. It passes information between all components and maintains the history of iterations.

```python
from sifaka.core.thought import Thought, Document, CriticFeedback, ValidationResult
from datetime import datetime

# Create a basic thought
thought = Thought(
    prompt="Write a short story about a robot.",
    system_prompt="You are a creative writer."
)

# Add pre-generation context
thought = thought.add_pre_generation_context([
    Document(
        text="Robots are machines that can be programmed to perform tasks.",
        metadata={"source": "definition"},
        score=0.95
    ),
    Document(
        text="Asimov's Three Laws of Robotics are rules for robots in his science fiction.",
        metadata={"source": "literature"},
        score=0.85
    )
])

# Set generated text
thought = thought.set_text("Once upon a time, there was a robot named R2D2...")

# Add validation results
thought = thought.add_validation_result(
    "LengthValidator",
    ValidationResult(
        passed=True,
        message="Text meets length requirements",
        score=1.0
    )
)

# Add critic feedback
thought = thought.add_critic_feedback(
    CriticFeedback(
        critic_name="ReflexionCritic",
        issues=["The story lacks character development"],
        suggestions=["Add more details about the robot's personality"]
    )
)

# Create the next iteration
next_thought = thought.next_iteration()
print(f"Current iteration: {thought.iteration}")
print(f"Next iteration: {next_thought.iteration}")
print(f"History count: {len(next_thought.history)}")
```

## Working with Retrievers

Retrievers find relevant documents for a query. The Chain orchestrates all retrieval automatically - you just provide the retriever to the Chain and it handles everything:

```python
from sifaka.core.thought import Thought
from sifaka.retrievers.base import InMemoryRetriever
from sifaka.models.base import create_model
from sifaka.critics.base import ReflexionCritic
from sifaka.chain import Chain

# Create a retriever
retriever = InMemoryRetriever()

# Add documents to the retriever
retriever.add_document("doc1", "Robots are machines that can be programmed to perform tasks.")
retriever.add_document("doc2", "Asimov's Three Laws of Robotics are rules for robots in his science fiction.")
retriever.add_document("doc3", "Machine learning allows robots to learn from data and improve over time.")

# Create a model and critic (no retriever needed - Chain handles it)
model = create_model("mock:default")
critic = ReflexionCritic(model=model)

# Create a Chain with the retriever - it orchestrates ALL retrieval
chain = Chain(
    model=model,
    prompt="Write a short story about a robot that learns.",
    retriever=retriever,  # Chain handles all retrieval automatically
    pre_generation_retrieval=True,   # Retrieve before generation
    post_generation_retrieval=True,  # Retrieve after generation
    critic_retrieval=True,           # Retrieve for critics
)

chain.improve_with(critic)

# Run the chain - it handles all retrieval automatically:
# 1. Pre-generation retrieval: Gets context before text generation
# 2. Text generation: Model uses the retrieved context
# 3. Post-generation retrieval: Gets context after text generation
# 4. Critic retrieval: Gets fresh context for critics during improvement
result = chain.run()

# Print the retrieved context
print("Pre-generation context:")
for doc in result.pre_generation_context:
    print(f"- {doc.text} (score: {doc.score})")

print("\nPost-generation context:")
for doc in result.post_generation_context:
    print(f"- {doc.text} (score: {doc.score})")
```

### Available Retrievers

Sifaka provides several retriever implementations:

- **MockRetriever**: Returns predefined documents for testing
- **InMemoryRetriever**: Simple keyword-based retrieval from in-memory documents
- **RedisRetriever**: Redis-based caching retriever that can wrap other retrievers for performance

### Redis Caching for Performance

The RedisRetriever provides powerful caching capabilities to improve performance:

```python
from sifaka.retrievers.redis import RedisRetriever, create_redis_retriever
from sifaka.retrievers.base import InMemoryRetriever

# Create a base retriever
base_retriever = InMemoryRetriever()
base_retriever.add_document("doc1", "Python is excellent for AI development.")
base_retriever.add_document("doc2", "Machine learning frameworks like TensorFlow use Python.")

# Wrap with Redis caching (recommended approach)
cached_retriever = RedisRetriever(
    base_retriever=base_retriever,
    redis_host="localhost",  # Default
    redis_port=6379,         # Default
    cache_ttl=300,           # 5 minutes cache
)

# Use in chain for automatic caching
chain = Chain(
    model=model,
    prompt="Explain Python's role in AI development.",
    retriever=cached_retriever,  # Automatic caching for all retrievals
)

# First run: cache miss, retrieves from base_retriever and caches
result1 = chain.run()

# Second run: cache hit, much faster retrieval
result2 = chain.run()

# Check cache statistics
stats = cached_retriever.get_cache_stats()
print(f"Cached queries: {stats['cached_queries']}")
```

You can also use RedisRetriever as a standalone document store:

```python
# Standalone Redis document store
redis_store = RedisRetriever(cache_ttl=3600)  # 1 hour cache

# Add documents directly
redis_store.add_document("ai_doc", "Artificial intelligence is transforming technology.")
redis_store.add_document("ml_doc", "Machine learning enables computers to learn from data.")

# Retrieve documents
results = redis_store.retrieve("artificial intelligence machine learning")
print(f"Found {len(results)} relevant documents")
```

## Working with Critics

Critics analyze text, identify issues, and provide suggestions for improvement. The Chain orchestrates retrieval for critics automatically:

```python
from sifaka.core.thought import Thought
from sifaka.critics.base import ReflexionCritic
from sifaka.models.base import create_model
from sifaka.retrievers.base import InMemoryRetriever
from sifaka.chain import Chain

# Create a retriever
retriever = InMemoryRetriever()
retriever.add_document("doc1", "Quantum computing uses quantum bits or qubits.")
retriever.add_document("doc2", "Superposition allows qubits to exist in multiple states simultaneously.")
retriever.add_document("doc3", "Quantum entanglement connects qubits in ways that classical bits cannot be connected.")

# Create a model and critic (no retriever needed - Chain handles it)
model = create_model("mock:default")
critic = ReflexionCritic(model=model)

# Create a Chain that orchestrates retrieval for critics
chain = Chain(
    model=model,
    prompt="Explain quantum computing to a high school student.",
    retriever=retriever,  # Chain handles all retrieval for critics
    critic_retrieval=True,  # Enable retrieval for critics
)

chain.improve_with(critic)

# Run the chain - it automatically:
# 1. Generates initial text
# 2. Retrieves context for critics
# 3. Critics use the retrieved context to provide better feedback
result = chain.run()

# Print the final result
print(f"Final text: {result.text}")
print(f"Validation results: {result.validation_results}")
print(f"Retrieved context documents: {len(result.post_generation_context)}")
```

## Multi-Retriever Support

Sifaka supports using different retrievers for different stages of the chain. This is powerful for use cases like fact-checking, where you want models to use recent context (like Twitter posts) but critics to use authoritative sources (like Wikipedia).

```python
from sifaka.chain import Chain
from sifaka.models.base import create_model
from sifaka.critics.base import ReflexionCritic
from sifaka.retrievers.specialized import TwitterRetriever, FactualDatabaseRetriever

# Create specialized retrievers
twitter_retriever = TwitterRetriever()      # Recent context for model
factual_retriever = FactualDatabaseRetriever()  # Authoritative sources for critics

# Create model and critic
model = create_model("openai:gpt-4", api_key=api_key)
critic = ReflexionCritic(model=model)

# Create Chain with different retrievers for different stages
chain = Chain(
    model=model,
    prompt="Write a news summary about recent AI developments and their implications.",
    model_retriever=twitter_retriever,    # Model uses recent Twitter/news context
    critic_retriever=factual_retriever,   # Critics use factual database for verification
    pre_generation_retrieval=True,        # Get recent context before generation
    post_generation_retrieval=True,       # Get more recent context after generation
    critic_retrieval=True,                # Get factual context for critics
)

chain.improve_with(critic)

# Run the chain - it automatically:
# 1. Uses TwitterRetriever for pre/post-generation context (model gets recent info)
# 2. Uses FactualDatabaseRetriever for critic context (critics verify against facts)
result = chain.run()

print("Recent context used by model:")
for doc in result.pre_generation_context:
    print(f"- {doc.text}")

print("\nFactual context used by critics:")
for doc in result.post_generation_context:
    print(f"- {doc.text}")
```

### Retriever Fallback Logic

- If `model_retriever` is not provided, falls back to `retriever`
- If `critic_retriever` is not provided, falls back to `retriever`
- If no retrievers are provided, the Chain works without retrieval

```python
# Example: Using default retriever for all stages
chain = Chain(
    model=model,
    prompt="Write about AI",
    retriever=default_retriever,  # Used for all stages
)

# Example: Mixed retrievers
chain = Chain(
    model=model,
    prompt="Write about AI",
    retriever=default_retriever,        # Fallback
    critic_retriever=factual_retriever, # Specific for critics
    # model_retriever not provided - uses default_retriever
)
```

## Persistence Options

The Thought container can be persisted in various ways:

```python
from sifaka.core.thought import Thought
import json

# Create a thought
thought = Thought(prompt="Write a short story about a robot.")

# Serialize to JSON
thought_json = thought.model_dump_json()

# Save to file
with open("thought.json", "w") as f:
    f.write(thought_json)

# Load from JSON
with open("thought.json", "r") as f:
    loaded_json = f.read()
    loaded_thought = Thought.model_validate_json(loaded_json)

print(f"Loaded thought prompt: {loaded_thought.prompt}")
```

Sifaka also supports Redis for caching and performance optimization:
- **Redis caching**: Available now via RedisRetriever for fast retrieval caching
- **Vector databases**: Planned support for Milvus and Elasticsearch for semantic search
- **PostgreSQL**: Planned support for relational storage with history tracking

## Documentation


## Development

### Code Formatting

Sifaka uses automated code formatting to maintain consistent code style. We use the following tools:

- **Black**: Code formatting
- **isort**: Import sorting
- **autoflake**: Removing unused imports
- **Ruff**: Linting with automatic fixes
- **mypy**: Type checking

To set up the development environment:

```bash
# Install development dependencies
make install-dev

# Format code
make format

# Run linting checks
make lint

# Run tests
make test
```

The CI pipeline will automatically format code in pull requests, so you don't need to worry about formatting issues.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

[MIT License](LICENSE)
