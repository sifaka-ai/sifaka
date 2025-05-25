# Sifaka API Reference

This document provides a comprehensive reference for the Sifaka framework's API.

## Core Components

### Thought

The central data container that flows through the chain.

```python
from sifaka.core.thought import Thought

# Create a new thought
thought = Thought(prompt="Your prompt here")

# Access properties
thought.prompt          # The original prompt
thought.text           # Generated text (if any)
thought.iteration      # Current iteration number
thought.validation_results  # List of validation results
thought.critic_feedback     # List of critic feedback
```

#### Key Methods

- `add_pre_generation_context(documents)` - Add context before generation
- `add_post_generation_context(documents)` - Add context after generation
- `add_validation_result(result)` - Add validation result
- `add_critic_feedback(feedback)` - Add critic feedback
- `to_dict()` - Convert to dictionary
- `from_dict(data)` - Create from dictionary

### Chain

The main orchestrator that coordinates models, validators, and critics.

```python
from sifaka.core.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators.base import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic

# Create chain components
model = create_model("mock:default")
validator = LengthValidator(min_length=50)
critic = ReflexionCritic(model=model)

# Create and configure chain
chain = Chain(
    model=model,
    prompt="Write about AI",
    model_retrievers=[model_retriever],    # Optional: retrievers for model context
    critic_retrievers=[critic_retriever],  # Optional: retrievers for critic context
    max_improvement_iterations=3
)
chain.validate_with(validator)
chain.improve_with(critic)

# Or using fluent API
chain = Chain(model=model, prompt="Write about AI") \
    .with_model_retrievers([model_retriever]) \
    .with_critic_retrievers([critic_retriever]) \
    .validate_with(validator) \
    .improve_with(critic)

# Run the chain
result = chain.run()
```

#### Configuration Options

- `max_improvement_iterations` - Maximum number of improvement iterations (default: 3)
- `always_apply_critics` - Apply critics even when validation passes (default: False)
- `apply_improvers_on_validation_failure` - Apply critics when validation fails (default: False)
- `model_retrievers` - Optional list of retrievers for model context (pre-generation)
- `critic_retrievers` - Optional list of retrievers for critic context (post-generation)

## Models

### Base Model Protocol

```python
from sifaka.models.base import Model

class CustomModel(Model):
    def generate_with_thought(self, thought: Thought) -> tuple[str, str]:
        # Return (generated_text, model_name)
        pass
```

### Available Models

#### MockModel
```python
from sifaka.models.base import create_model

model = create_model("mock:default")
```

#### OpenAI Models
```python
from sifaka.models.base import create_model

model = create_model("openai:gpt-4")
# API key from OPENAI_API_KEY environment variable
```

#### Anthropic Models
```python
from sifaka.models.base import create_model

model = create_model("anthropic:claude-3-sonnet")
# API key from ANTHROPIC_API_KEY environment variable
```

## Validators

### Base Validator Protocol

```python
from sifaka.validators.base import Validator, ValidationResult

class CustomValidator(Validator):
    def validate(self, text: str) -> ValidationResult:
        # Return ValidationResult
        pass
```

### Available Validators

#### Length Validator
```python
from sifaka.validators.base import LengthValidator

validator = LengthValidator(
    min_length=10,
    max_length=1000
)
```

#### Content Validator
```python
from sifaka.validators.content import ContentValidator

validator = ContentValidator(
    required_phrases=["machine learning"],
    prohibited_phrases=["spam"]
)
```

#### Regex Validator
```python
from sifaka.validators.base import RegexValidator

validator = RegexValidator(
    pattern=r"machine learning|ML"
)
```

#### Format Validators
```python
from sifaka.validators.format import FormatValidator

json_validator = FormatValidator(format_type="json")
markdown_validator = FormatValidator(format_type="markdown")
```

#### Classifier Validators
```python
from sifaka.validators.classifier import ClassifierValidator
from sifaka.classifiers.sentiment import SentimentClassifier

classifier = SentimentClassifier()
validator = ClassifierValidator(
    classifier=classifier,
    threshold=0.7,
    valid_labels=["positive", "neutral"]
)
```

#### GuardrailsAI Validator
```python
from sifaka.validators.guardrails import GuardrailsValidator

validator = GuardrailsValidator(
    guard_name="pii_detection"
    # API key from GUARDRAILS_API_KEY environment variable
)
```

## Critics

### Base Critic Protocol

```python
from sifaka.critics.base import Critic, CriticFeedback

class CustomCritic(Critic):
    def critique(self, thought: Thought) -> CriticFeedback:
        # Return CriticFeedback
        pass
```

### Available Critics

#### Reflexion Critic
```python
from sifaka.critics.reflexion import ReflexionCritic

critic = ReflexionCritic(
    model=your_model,
    name="ReflexionCritic"
)
```

#### Self-RAG Critic
```python
from sifaka.critics.self_rag import SelfRAGCritic

critic = SelfRAGCritic(
    model=your_model,
    retriever=your_retriever,
    name="SelfRAGCritic"
)
```

#### Constitutional Critic
```python
from sifaka.critics.constitutional import ConstitutionalCritic

critic = ConstitutionalCritic(
    model=your_model,
    principles=["Be helpful", "Be harmless"],
    name="ConstitutionalCritic"
)
```

#### N-Critics
```python
from sifaka.critics.n_critics import NCriticsCritic

critic = NCriticsCritic(
    model=your_model,
    num_critics=3,
    name="NCriticsCritic"
)
```

## Retrievers

### Base Retriever Protocol

```python
from sifaka.retrievers.base import Retriever

class CustomRetriever(Retriever):
    def retrieve(self, query: str) -> List[str]:
        # Return list of document texts
        pass

    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        # Return thought with added context
        pass
```

### Available Retrievers

#### Mock Retriever
```python
from sifaka.retrievers import MockRetriever

retriever = MockRetriever()
```

#### In-Memory Retriever
```python
from sifaka.retrievers import InMemoryRetriever

retriever = InMemoryRetriever()
# Add documents
retriever.add_document("doc1", "This is about AI")
retriever.add_document("doc2", "This is about ML")
```

#### Cached Retriever (3-Tier Storage)
```python
from sifaka.storage import SifakaStorage
from sifaka.retrievers import InMemoryRetriever

# Create storage manager
storage = SifakaStorage(redis_config=redis_config, milvus_config=milvus_config)

# Wrap any retriever with caching
base_retriever = InMemoryRetriever()
cached_retriever = storage.get_retriever_cache(base_retriever)
```

## Classifiers

### Text Classifiers

#### Sentiment Classifier
```python
from sifaka.classifiers.sentiment import SentimentClassifier, CachedSentimentClassifier

classifier = SentimentClassifier()
cached_classifier = CachedSentimentClassifier(cache_size=128)
```

#### Toxicity Classifier
```python
from sifaka.classifiers.toxicity import ToxicityClassifier, CachedToxicityClassifier

classifier = ToxicityClassifier()
cached_classifier = CachedToxicityClassifier(cache_size=128)
```

#### Spam Classifier
```python
from sifaka.classifiers.spam import SpamClassifier

classifier = SpamClassifier(threshold=0.7)
```

## Persistence

### JSON Persistence
```python
from sifaka.persistence.json import JSONPersistence

persistence = JSONPersistence(file_path="thoughts.json")

# Save thought
persistence.save_thought(thought)

# Load thoughts
thoughts = persistence.load_thoughts()
```

## Utilities

### Logging
```python
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Your message here")
```

### Error Handling
```python
from sifaka.utils.error_handling import error_context, SifakaError

with error_context(
    component="YourComponent",
    operation="your_operation",
    error_class=SifakaError,
    message_prefix="Operation failed"
):
    # Your code here
    pass
```

## Error Recovery

### Circuit Breaker

```python
from sifaka.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# Configure circuit breaker
config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30.0,
    expected_exception=ConnectionError
)

breaker = CircuitBreaker("redis-service", config)

# Use as decorator
@breaker.protect
def call_external_service():
    # Your service call here
    pass

# Use as context manager
with breaker:
    # Your service call here
    pass
```

### Retry Manager

```python
from sifaka.utils.retry import RetryConfig, retry_with_backoff, BackoffStrategy

# Configure retry behavior
config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
    jitter=True
)

# Use as decorator
@retry_with_backoff(config)
def unreliable_function():
    # Your code here
    pass
```

### Resilient Wrappers

```python
from sifaka.models.resilient import ResilientModel
from sifaka.retrievers.resilient import ResilientRetriever

# Create resilient model
resilient_model = ResilientModel(
    primary_model=primary_model,
    fallback_models=[fallback_model],
    circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3),
    retry_config=RetryConfig(max_attempts=3)
)

# Create resilient retriever
resilient_retriever = ResilientRetriever(
    primary_retriever=primary_retriever,
    fallback_retrievers=[fallback_retriever]
)
```

## Performance Monitoring

### Performance Monitor

```python
from sifaka.utils.performance import PerformanceMonitor, timer, time_operation

# Get singleton instance
monitor = PerformanceMonitor.get_instance()

# Use timer decorator
@timer("my_operation")
def my_function():
    # Your code here
    pass

# Use context manager
with time_operation("database_query"):
    # Your database operation
    pass

# Get performance stats
stats = monitor.get_stats()
summary = monitor.get_summary()

# Clear performance data
monitor.clear()
```

### Chain Performance Methods

```python
from sifaka.chain import Chain
from sifaka.utils.performance import PerformanceMonitor

# Create and run chain
chain = Chain(model=model, prompt="Test prompt")
result = chain.run()

# Get performance data from monitor
monitor = PerformanceMonitor.get_instance()
stats = monitor.get_stats()
summary = monitor.get_summary()

print(f"Performance summary: {summary}")

# Clear performance data
monitor.clear()
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GUARDRAILS_API_KEY` - GuardrailsAI API key
- `REDIS_URL` - Redis connection URL

### Installation

```bash
# Install from source (development)
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka
pip install -e .

# Install dependencies for specific features
pip install openai anthropic         # For OpenAI/Anthropic models
pip install redis                    # For Redis caching
pip install guardrails-ai            # For GuardrailsAI validation
```
