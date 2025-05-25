# Sifaka Examples Documentation

This document provides comprehensive examples demonstrating how to use the Sifaka framework.

## Quick Start

### Basic Chain Example

The simplest way to get started with Sifaka:

```python
from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.models.mock import MockModel
from sifaka.validators.length import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic

# Create components
model = MockModel(name="example-model")
validators = [LengthValidator(min_length=50, max_length=500)]
critics = [ReflexionCritic()]

# Create and configure chain
chain = Chain(
    model=model,
    validators=validators,
    critics=critics,
    max_iterations=3
)

# Run the chain
prompt = "Write a comprehensive paragraph about artificial intelligence."
result = chain.run(prompt)

print(f"Final text: {result.text}")
print(f"Iterations: {result.iteration}")
```

**Run this example**: `python examples/mock/basic_chain.py`

## Model Examples

### Using OpenAI Models

```python
from sifaka.models.openai import OpenAIModel
import os

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create OpenAI model
model = OpenAIModel(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=500
)

# Use in chain
chain = Chain(model=model, validators=[], critics=[])
result = chain.run("Explain quantum computing")
```

### Using Anthropic Models

```python
from sifaka.models.anthropic import AnthropicModel
import os

# Set your API key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"

# Create Anthropic model
model = AnthropicModel(
    model_name="claude-3-sonnet-20240229",
    max_tokens=500
)

# Use in chain
chain = Chain(model=model, validators=[], critics=[])
result = chain.run("Write a poem about nature")
```

## Validation Examples

### Content Validation

```python
from sifaka.validators.content import ContentValidator
from sifaka.validators.regex import RegexValidator
from sifaka.validators.length import LengthValidator

# Require specific content
content_validator = ContentValidator(
    required_phrases=["machine learning", "artificial intelligence"],
    prohibited_phrases=["spam", "advertisement"],
    name="ContentCheck"
)

# Pattern matching
regex_validator = RegexValidator(
    pattern=r"\b(AI|ML|artificial intelligence|machine learning)\b",
    name="TechTermsCheck"
)

# Length constraints
length_validator = LengthValidator(
    min_length=100,
    max_length=1000,
    name="LengthCheck"
)

# Use multiple validators
validators = [content_validator, regex_validator, length_validator]
chain = Chain(model=model, validators=validators, critics=[])
```

### Format Validation

```python
from sifaka.validators.format import JSONValidator, MarkdownValidator, EmailValidator

# JSON format validation
json_validator = JSONValidator(name="JSONFormat")

# Markdown format validation
markdown_validator = MarkdownValidator(name="MarkdownFormat")

# Email format validation
email_validator = EmailValidator(name="EmailFormat")

# Test JSON validation
text = '{"name": "John", "age": 30}'
result = json_validator.validate(text)
print(f"Valid JSON: {result.passed}")
```

### Classifier-Based Validation

```python
from sifaka.validators.classifier import ClassifierValidator
from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.classifiers.toxicity import ToxicityClassifier

# Sentiment validation (require positive sentiment)
sentiment_classifier = SentimentClassifier()
sentiment_validator = ClassifierValidator(
    classifier=sentiment_classifier,
    threshold=0.7,
    valid_labels=["positive"],
    name="PositiveSentiment"
)

# Toxicity validation (block toxic content)
toxicity_classifier = ToxicityClassifier()
toxicity_validator = ClassifierValidator(
    classifier=toxicity_classifier,
    threshold=0.8,
    invalid_labels=["toxic", "severe_toxic", "threat"],
    name="ToxicityFilter"
)

validators = [sentiment_validator, toxicity_validator]
```

## Critic Examples

### Reflexion Critic

```python
from sifaka.critics.reflexion import ReflexionCritic

# Create reflexion critic
reflexion_critic = ReflexionCritic(
    model=your_model,
    name="ReflexionCritic"
)

# Use in chain
chain = Chain(
    model=model,
    validators=[],
    critics=[reflexion_critic],
    always_apply_critics=True  # Apply even when validation passes
)
```

### Constitutional AI Critic

```python
from sifaka.critics.constitutional import ConstitutionalCritic

# Define constitutional principles
principles = [
    "Be helpful and informative",
    "Avoid harmful or offensive content",
    "Provide accurate information",
    "Be respectful and professional"
]

constitutional_critic = ConstitutionalCritic(
    model=your_model,
    principles=principles,
    name="ConstitutionalCritic"
)
```

### N-Critics Ensemble

```python
from sifaka.critics.n_critics import NCriticsCritic

# Use multiple critics for ensemble feedback
n_critics = NCriticsCritic(
    model=your_model,
    num_critics=3,
    focus_areas=["clarity", "accuracy", "engagement"],
    name="EnsembleCritic"
)
```

**Run critic examples**: `python examples/mock/critics_example.py`

## Retrieval Examples

### Redis Retrieval

```python
from sifaka.retrievers.redis import RedisRetriever

# Create Redis retriever
redis_retriever = RedisRetriever(
    host="localhost",
    port=6379,
    db=0,
    max_results=5
)

# Add documents
redis_retriever.add_document("doc1", "AI is transforming industries...")
redis_retriever.add_document("doc2", "Machine learning enables...")

# Use with chain
chain = Chain(
    model=model,
    validators=[],
    critics=[],
    retriever=redis_retriever
)
```

**Run Redis example**: `python examples/mock/redis_retriever_example.py`

### Vector Database Retrieval

```python
from sifaka.retrievers.milvus import MilvusRetriever

# Create Milvus retriever
milvus_retriever = MilvusRetriever(
    collection_name="knowledge_base",
    embedding_model="BAAI/bge-m3",
    dimension=384,
    max_results=3
)

# Add documents with embeddings
milvus_retriever.add_document(
    doc_id="ai_overview",
    text="Artificial intelligence is a broad field...",
    metadata={"category": "overview", "topic": "AI"}
)

# Use with chain for context-aware generation
chain = Chain(
    model=model,
    validators=[],
    critics=[],
    retriever=milvus_retriever
)
```

## Advanced Examples

### Self-RAG with Retrieval

```python
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.retrievers.redis import RedisRetriever

# Set up retriever with knowledge base
retriever = RedisRetriever()
retriever.add_document("fact1", "The capital of France is Paris.")
retriever.add_document("fact2", "Python was created by Guido van Rossum.")

# Create Self-RAG critic
self_rag_critic = SelfRAGCritic(
    model=your_model,
    retriever=retriever,
    name="SelfRAGCritic"
)

# Use in chain
chain = Chain(
    model=model,
    validators=[],
    critics=[self_rag_critic],
    retriever=retriever  # Also use for pre-generation context
)
```

### Multi-Modal Validation

```python
from sifaka.validators.length import LengthValidator
from sifaka.validators.content import ContentValidator
from sifaka.validators.classifier import ClassifierValidator
from sifaka.classifiers.sentiment import SentimentClassifier

# Combine multiple validation types
validators = [
    LengthValidator(min_length=50, max_length=500),
    ContentValidator(required_phrases=["evidence", "research"]),
    ClassifierValidator(
        classifier=SentimentClassifier(),
        threshold=0.6,
        valid_labels=["neutral", "positive"]
    )
]

chain = Chain(model=model, validators=validators, critics=[])
```

### Always Apply Critics

```python
# Apply critics even when validation passes
chain = Chain(
    model=model,
    validators=[LengthValidator(min_length=50)],
    critics=[ReflexionCritic()],
    always_apply_critics=True,  # Key setting
    max_iterations=5
)

result = chain.run("Write about renewable energy")
```

### Storage Example

```python
from sifaka.storage import SifakaStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

# Configure unified storage
redis_config = MCPServerConfig(
    name="redis-server",
    transport_type=MCPTransportType.STDIO,
    url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379"
)

milvus_config = MCPServerConfig(
    name="milvus-server",
    transport_type=MCPTransportType.STDIO,
    url="npx -y @milvus-io/mcp-server-milvus"
)

# Create storage manager
storage = SifakaStorage(redis_config, milvus_config)
thought_storage = storage.get_thought_storage()

# Run chain and save results
result = chain.run("Explain blockchain technology")
thought_storage.save_thought(result)

# Find similar thoughts using vector search
similar_thoughts = thought_storage.find_similar_thoughts_by_text("blockchain", limit=5)
print(f"Found {len(similar_thoughts)} similar thoughts")
```

**Run storage example**: `python examples/mock/storage_example.py`

## Performance Monitoring Examples

### Basic Performance Monitoring

```python
from sifaka.core.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators.base import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic

# Create chain with performance monitoring
model = create_model("openai:gpt-4", api_key="your-api-key")
validator = LengthValidator(min_length=50, max_length=500)
critic = ReflexionCritic(model=model)

chain = Chain(
    model=model,
    prompt="Write about the future of AI in healthcare."
)
chain.validate_with(validator)
chain.improve_with(critic)

# Clear previous performance data
chain.clear_performance_data()

# Run with automatic performance monitoring
result = chain.run()

# Get performance summary
performance = chain.get_performance_summary()
print(f"Total execution time: {performance['total_time']:.2f}s")
print(f"Number of operations: {performance['total_operations']}")

# Identify bottlenecks
bottlenecks = chain.get_performance_bottlenecks()
for bottleneck in bottlenecks:
    print(f"Bottleneck: {bottleneck}")

# Export performance data
import json
with open("performance_report.json", "w") as f:
    json.dump(performance, f, indent=2)
```

### Custom Performance Monitoring

```python
from sifaka.utils.performance import timer, time_operation, PerformanceMonitor

# Use timer decorator
@timer("custom_operation")
def my_custom_function():
    # Your code here
    time.sleep(0.1)  # Simulate work
    return "result"

# Use context manager
with time_operation("database_query"):
    # Simulate database operation
    time.sleep(0.05)

# Get performance stats
monitor = PerformanceMonitor.get_instance()
stats = monitor.get_stats()
summary = monitor.get_summary()

print(f"Custom operation avg time: {stats['custom_operation']['avg_time']:.3f}s")
print(f"Database query avg time: {stats['database_query']['avg_time']:.3f}s")
```

**Run performance example**: `python examples/performance/performance_demo.py`

## Error Recovery Examples

### Circuit Breaker Pattern

```python
from sifaka.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# Configure circuit breaker
config = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    expected_exception=ConnectionError
)

breaker = CircuitBreaker("external-service", config)

# Use as decorator
@breaker.protect
def call_external_api():
    # Simulate API call that might fail
    import random
    if random.random() < 0.3:  # 30% failure rate
        raise ConnectionError("Service unavailable")
    return "API response"

# Test circuit breaker
for i in range(10):
    try:
        result = call_external_api()
        print(f"Call {i+1}: {result}")
    except Exception as e:
        print(f"Call {i+1}: Failed - {e}")
```

### Retry with Backoff

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

@retry_with_backoff(config)
def unreliable_function():
    import random
    if random.random() < 0.7:  # 70% failure rate
        raise ConnectionError("Temporary failure")
    return "Success!"

# Test retry mechanism
try:
    result = unreliable_function()
    print(f"Result: {result}")
except Exception as e:
    print(f"Failed after retries: {e}")
```

### Resilient Model with Fallbacks

```python
from sifaka.models.resilient import ResilientModel
from sifaka.models.base import create_model
from sifaka.utils.circuit_breaker import CircuitBreakerConfig
from sifaka.utils.retry import RetryConfig
from sifaka.utils.fallback import FallbackConfig

# Create primary and fallback models
primary_model = create_model("openai:gpt-4", api_key="your-openai-key")
fallback_model = create_model("anthropic:claude-3-sonnet", api_key="your-anthropic-key")

# Create resilient model
resilient_model = ResilientModel(
    primary_model=primary_model,
    fallback_models=[fallback_model],
    circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3),
    retry_config=RetryConfig(max_attempts=3, backoff_factor=2.0),
    fallback_config=FallbackConfig(max_fallbacks=2)
)

# Use in chain - automatic error recovery
chain = Chain(
    model=resilient_model,
    prompt="Write about AI safety and alignment."
)

# Run with automatic fallback on failures
result = chain.run()
print(f"Generated text: {result.text}")
print(f"Model used: {result.model_name}")  # Shows which model was actually used
```

### Enhanced Retrievers with Retry/Fallback

```python
from sifaka.retrievers import InMemoryRetriever, MockRetriever

# Create fallback retriever
fallback_retriever = MockRetriever(max_results=3)

# Create primary retriever with retry/fallback capabilities
primary_retriever = InMemoryRetriever(
    max_retries=3,
    fallback_retrievers=[fallback_retriever]
)

# Add documents
primary_retriever.add_document("doc1", "Primary knowledge base content...")

# Use in chain
chain = Chain(
    model=model,
    prompt="Explain machine learning concepts.",
    retrievers=[primary_retriever]
)

result = chain.run()
```

**Run error recovery example**: `python examples/error_recovery/enhanced_error_recovery_example.py`

## Testing Examples

### Validator Testing

```python
def test_length_validator():
    validator = LengthValidator(min_length=10, max_length=100)

    # Test valid text
    result = validator.validate("This is a valid length text.")
    assert result.passed

    # Test too short
    result = validator.validate("Short")
    assert not result.passed
    assert "too short" in result.message.lower()

test_length_validator()
```

### Chain Testing

```python
def test_basic_chain():
    chain = Chain(
        model=MockModel(),
        validators=[LengthValidator(min_length=10)],
        critics=[]
    )

    result = chain.run("Test prompt")
    assert result.text is not None
    assert result.iteration >= 0

test_basic_chain()
```

## Configuration Examples

### Environment Variables

```bash
# Set API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GUARDRAILS_API_KEY="your-guardrails-key"

# Redis configuration
export REDIS_URL="redis://localhost:6379/0"
```

### Chain Configuration

```python
# Comprehensive chain configuration
chain = Chain(
    model=OpenAIModel(model_name="gpt-4"),
    validators=[
        LengthValidator(min_length=100, max_length=1000),
        ContentValidator(required_phrases=["research", "evidence"]),
        ClassifierValidator(
            classifier=ToxicityClassifier(),
            threshold=0.9,
            invalid_labels=["toxic"]
        )
    ],
    critics=[
        ReflexionCritic(),
        ConstitutionalCritic(principles=["Be accurate", "Be helpful"])
    ],
    retriever=RedisRetriever(max_results=5),
    max_iterations=3,
    always_apply_critics=False
)
```

## Running Examples

All examples are located in the `examples/` directory:

```bash
# Basic examples (work without API keys)
python examples/mock/basic_chain.py
python examples/mock/critics_example.py
python examples/mock/validators_example.py
python examples/mock/storage_example.py

# Performance monitoring examples
python examples/performance/performance_demo.py
python examples/performance/pii_detection_demo.py  # Requires OpenAI + Guardrails API keys

# Error recovery examples
python examples/error_recovery/enhanced_error_recovery_example.py

# Advanced examples (require API keys)
python examples/openai/always_apply_critics_example.py
python examples/mixed/advanced_chain_example.py
```

## Best Practices

1. **Start Simple**: Begin with mock models and basic validators
2. **Iterate Gradually**: Add complexity step by step
3. **Test Components**: Test validators and critics independently
4. **Monitor Performance**: Use performance monitoring to identify bottlenecks
5. **Plan for Failures**: Use resilient wrappers for production deployments
6. **Use Appropriate Models**: Match model capabilities to task complexity
7. **Configure Thoughtfully**: Set reasonable iteration limits and thresholds
8. **Enable Error Recovery**: Use circuit breakers and retry mechanisms for external services
9. **Track Metrics**: Export performance data for analysis and optimization
10. **Implement Fallbacks**: Always have backup models and retrievers for critical applications
