# Sifaka API Reference

This document provides comprehensive API documentation for all Sifaka components.

## Table of Contents

- [Core Components](#core-components)
  - [Chain](#chain)
  - [Thought](#thought)
  - [Core Interfaces](#core-interfaces)
- [Models](#models)
  - [Factory Functions](#model-factory-functions)
  - [Model Implementations](#model-implementations)
- [Validators](#validators)
  - [Base Validators](#base-validators)
  - [Content Validators](#content-validators)
  - [Format Validators](#format-validators)
  - [Classifier Validators](#classifier-validators)
  - [Guardrails Validators](#guardrails-validators)
- [Critics](#critics)
- [Classifiers](#classifiers)
- [Storage](#storage)
- [Utilities](#utilities)

---

## Core Components

### Chain

The main orchestrator for text generation, validation, and improvement workflows.

#### Import

```python
from sifaka.core.chain import Chain
```

#### Constructor

```python
Chain(
    model: Optional[Model] = None,
    storage: Optional[Storage] = None,
    checkpoint_storage: Optional[Storage] = None
)
```

**Parameters:**
- `model`: Language model for text generation
- `storage`: Storage backend for thoughts
- `checkpoint_storage`: Storage backend for checkpoints

#### Configuration Methods

##### `with_model(model: Union[Model, str]) -> Chain`

Set the language model for text generation.

```python
# Using model instance
chain.with_model(OpenAIModel(api_key="your-key"))

# Using model specification string
chain.with_model("openai:gpt-4")
```

##### `with_prompt(prompt: str) -> Chain`

Set the prompt for text generation.

```python
chain.with_prompt("Write a short story about AI.")
```

##### `with_storage(storage: Storage) -> Chain`

Set the storage backend for thoughts.

```python
from sifaka.storage.redis import RedisStorage
chain.with_storage(RedisStorage())
```

##### `with_checkpoint_storage(storage: Storage) -> Chain`

Set the storage backend for checkpoints.

```python
chain.with_checkpoint_storage(FileStorage("./checkpoints"))
```

##### `with_model_retrievers(*retrievers: Retriever) -> Chain`

Add retrievers for pre-generation context.

```python
chain.with_model_retrievers(retriever1, retriever2)
```

##### `with_critic_retrievers(*retrievers: Retriever) -> Chain`

Add retrievers for post-generation context.

```python
chain.with_critic_retrievers(retriever1, retriever2)
```

##### `validate_with(validator: Validator) -> Chain`

Add a validator to check generated text.

```python
from sifaka.validators.base import LengthValidator
chain.validate_with(LengthValidator(min_length=50, max_length=500))
```

##### `improve_with(critic: Critic) -> Chain`

Add a critic to improve generated text.

```python
from sifaka.critics.reflexion import ReflexionCritic
chain.improve_with(ReflexionCritic(model=model))
```

##### `with_options(**options: Any) -> Chain`

Set execution options for the chain.

```python
chain.with_options(
    max_attempts=5,
    run_critics_on_valid=True,
    temperature=0.7
)
```

#### Execution Methods

##### `run() -> Thought`

Execute the complete chain workflow.

```python
result = chain.run()
print(f"Generated text: {result.text}")
print(f"Iterations: {result.iteration}")
```

##### `run_with_recovery() -> Thought`

Execute the chain with automatic checkpointing and recovery.

```python
result = chain.run_with_recovery()
```

---

### Thought

Central state container that tracks the generation process.

#### Import

```python
from sifaka.core.thought import Thought
```

#### Constructor

```python
Thought(
    prompt: str,
    text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model_prompt: Optional[str] = None,
    pre_generation_context: Optional[List[Document]] = None,
    post_generation_context: Optional[List[Document]] = None,
    validation_results: Optional[Dict[str, ValidationResult]] = None,
    critic_feedback: Optional[List[CriticFeedback]] = None,
    history: Optional[List[ThoughtReference]] = None,
    parent_id: Optional[str] = None,
    id: str = "",
    iteration: int = 0,
    timestamp: Optional[datetime] = None,
    chain_id: Optional[str] = None,
    metadata: Dict[str, Any] = {}
)
```

#### Key Properties

- `prompt`: The original prompt
- `text`: Generated text
- `iteration`: Current iteration number
- `validation_results`: Results from validators
- `critic_feedback`: Feedback from critics
- `history`: Previous iterations

#### Methods

##### `next_iteration() -> Thought`

Create a new Thought for the next iteration.

```python
next_thought = current_thought.next_iteration()
```

##### `add_validation_result(name: str, result: ValidationResult) -> Thought`

Add a validation result.

```python
thought = thought.add_validation_result("length", validation_result)
```

##### `add_critic_feedback(feedback: CriticFeedback) -> Thought`

Add critic feedback.

```python
thought = thought.add_critic_feedback(feedback)
```

##### `set_text(text: str) -> Thought`

Set the generated text.

```python
thought = thought.set_text("Generated text here")
```

---

### Core Interfaces

#### Model Protocol

```python
from sifaka.core.interfaces import Model
```

**Methods:**
- `generate(prompt: str, **options: Any) -> str`: Generate text from prompt
- `count_tokens(text: str) -> int`: Count tokens in text

#### Validator Protocol

```python
from sifaka.core.interfaces import Validator
```

**Methods:**
- `validate(thought: Thought) -> ValidationResult`: Validate text

#### Critic Protocol

```python
from sifaka.core.interfaces import Critic
```

**Methods:**
- `critique(thought: Thought) -> Dict[str, Any]`: Critique text
- `improve(thought: Thought) -> str`: Improve text based on critique

#### Retriever Protocol

```python
from sifaka.core.interfaces import Retriever
```

**Methods:**
- `retrieve(query: str, limit: int = 10) -> List[Document]`: Retrieve relevant documents

---

## Models

### Model Factory Functions

#### `create_model(model_spec: str, **kwargs: Any) -> Model`

Create a model instance from a specification string.

```python
from sifaka.models.base import create_model

# OpenAI models
model = create_model("openai:gpt-4", api_key="your-key")
model = create_model("openai:gpt-3.5-turbo")

# Anthropic models
model = create_model("anthropic:claude-3-sonnet", api_key="your-key")

# Google Gemini models
model = create_model("gemini:gemini-1.5-flash", api_key="your-key")
model = create_model("gemini:gemini-1.5-pro")

# Ollama models
model = create_model("ollama:llama2")

# Mock models for testing
model = create_model("mock:gpt-4")
```



---

## Validators

### Base Validators

#### LengthValidator

Validates text length requirements.

```python
from sifaka.validators.base import LengthValidator

validator = LengthValidator(
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_sentences: Optional[int] = None,
    max_sentences: Optional[int] = None
)
```

**Example:**
```python
# Character-based validation
validator = LengthValidator(min_length=100, max_length=1000)

# Word-based validation
validator = LengthValidator(min_words=50, max_words=200)

# Sentence-based validation
validator = LengthValidator(min_sentences=3, max_sentences=10)
```

#### RegexValidator

Validates text against regex patterns.

```python
from sifaka.validators.base import RegexValidator

validator = RegexValidator(
    patterns: List[Union[str, Pattern]],
    require_all: bool = True,
    case_sensitive: bool = True
)
```

**Example:**
```python
# Require email format
validator = RegexValidator(
    patterns=[r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
    require_all=True
)

# Prohibit certain patterns
validator = RegexValidator(
    patterns=[r'\b(password|secret)\b'],
    require_all=False  # Fails if ANY pattern matches
)
```

### Content Validators

#### ContentValidator

Validates against prohibited content.

```python
from sifaka.validators.content import ContentValidator

validator = ContentValidator(
    prohibited_terms: List[str],
    case_sensitive: bool = False,
    whole_words_only: bool = True
)
```

#### Factory Functions

```python
from sifaka.validators.content import create_content_validator, prohibited_content

# Using factory function
validator = create_content_validator(["spam", "scam", "phishing"])

# Using predefined prohibited content
validator = prohibited_content()
```

### Format Validators

#### FormatValidator

Validates text format (JSON, Markdown, custom).

```python
from sifaka.validators.format import FormatValidator

validator = FormatValidator(
    format_type: str,
    strict: bool = True,
    custom_validator: Optional[Callable] = None
)
```

#### Factory Functions

```python
from sifaka.validators.format import (
    create_format_validator,
    json_format,
    markdown_format,
    custom_format
)

# JSON validation
validator = json_format()

# Markdown validation
validator = markdown_format()

# Custom format validation
def validate_csv(text: str) -> bool:
    # Custom validation logic
    return "," in text

validator = custom_format(validate_csv)
```

### Classifier Validators

#### ClassifierValidator

Uses ML classifiers for validation.

```python
from sifaka.validators.classifier import ClassifierValidator

validator = ClassifierValidator(
    classifier: Classifier,
    threshold: float = 0.5,
    target_label: Optional[str] = None,
    invert: bool = False
)
```

#### Factory Functions

```python
from sifaka.validators.classifier import create_classifier_validator, classifier_validator
from sifaka.classifiers.toxicity import ToxicityClassifier

# Using factory function
classifier = ToxicityClassifier()
validator = create_classifier_validator(classifier, threshold=0.8)

# Using decorator-style function
validator = classifier_validator(classifier, threshold=0.8)
```

### Guardrails Validators

#### GuardrailsValidator

Integrates with GuardrailsAI for advanced validation.

```python
from sifaka.validators.guardrails import GuardrailsValidator

validator = GuardrailsValidator(
    guard_name: str,
    api_key: Optional[str] = None,  # Uses GUARDRAILS_API_KEY env var if not provided
    **guard_kwargs: Any
)
```

#### Factory Functions

```python
from sifaka.validators.guardrails import create_guardrails_validator, guardrails_validator

# PII detection
validator = create_guardrails_validator("detect_pii")

# Custom guardrail
validator = guardrails_validator("custom_guard", threshold=0.9)
```

---

## Critics

All critics implement the `Critic` protocol with `critique()` and `improve()` methods.

### ReflexionCritic

Implements the Reflexion approach for self-improvement.

```python
from sifaka.critics.reflexion import ReflexionCritic

critic = ReflexionCritic(
    model: Model,
    reflection_prompt: Optional[str] = None,
    improvement_prompt: Optional[str] = None
)
```

### SelfRAGCritic

Implements Self-RAG (Self-Reflective Retrieval-Augmented Generation).

```python
from sifaka.critics.self_rag import SelfRAGCritic

critic = SelfRAGCritic(
    model: Model,
    retriever: Optional[Retriever] = None,
    reflection_prompt: Optional[str] = None
)
```

### SelfRefineCritic

Implements the Self-Refine approach.

```python
from sifaka.critics.self_refine import SelfRefineCritic

critic = SelfRefineCritic(
    model: Model,
    refinement_prompt: Optional[str] = None,
    max_refinements: int = 3
)
```

### ConstitutionalCritic

Implements Constitutional AI principles.

```python
from sifaka.critics.constitutional import ConstitutionalCritic

critic = ConstitutionalCritic(
    model: Model,
    principles: List[str],
    critique_prompt: Optional[str] = None,
    revision_prompt: Optional[str] = None
)
```

**Example:**
```python
principles = [
    "Be helpful and informative",
    "Avoid harmful or offensive content",
    "Provide accurate information"
]
critic = ConstitutionalCritic(model=model, principles=principles)
```

### MetaRewardingCritic

Implements Meta-Rewarding approach with two-stage judgment process.

```python
from sifaka.critics.meta_rewarding import MetaRewardingCritic

critic = MetaRewardingCritic(
    model: Model,
    base_critic: Optional[BaseCritic] = None,
    meta_judge_model: Optional[Model] = None,
    meta_judge_model_name: Optional[str] = None,
    judgment_criteria: Optional[List[str]] = None,
    meta_judgment_criteria: Optional[List[str]] = None,
    use_scoring: bool = True,
    score_range: tuple[int, int] = (1, 10)
)
```

**Example:**
```python
# Basic usage
critic = MetaRewardingCritic(model=model)

# With base critic for initial judgment
base_critic = ConstitutionalCritic(model=model)
meta_critic = MetaRewardingCritic(
    model=model,
    base_critic=base_critic,
    meta_judge_model_name="openai:gpt-4"
)

# Custom judgment criteria
judgment_criteria = [
    "Accuracy and factual correctness",
    "Helpfulness and relevance",
    "Clarity and coherence"
]
meta_judgment_criteria = [
    "Quality of evaluation",
    "Thoroughness of feedback",
    "Constructiveness of suggestions"
]
critic = MetaRewardingCritic(
    model=model,
    judgment_criteria=judgment_criteria,
    meta_judgment_criteria=meta_judgment_criteria
)
```

**Note:** This is a simplified implementation of the Meta-Rewarding approach from the paper "Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge" (https://arxiv.org/abs/2407.19594). The implementation uses prompt-based meta-judgment rather than the specialized training procedures described in the original paper.

### SelfConsistencyCritic

Implements Self-Consistency approach with multiple critique generation and consensus.

```python
from sifaka.critics.self_consistency import SelfConsistencyCritic

critic = SelfConsistencyCritic(
    model: Model,
    base_critic: Optional[BaseCritic] = None,
    num_iterations: int = 5,
    consensus_threshold: float = 0.6,
    aggregation_method: str = "majority_vote",
    use_chain_of_thought: bool = True,
    similarity_threshold: float = 0.7
)
```

**Example:**
```python
# Basic usage
critic = SelfConsistencyCritic(model=model, num_iterations=5)

# With base critic for individual critiques
base_critic = ConstitutionalCritic(model=model)
consistency_critic = SelfConsistencyCritic(
    model=model,
    base_critic=base_critic,
    num_iterations=7,
    consensus_threshold=0.7
)

# Custom configuration
critic = SelfConsistencyCritic(
    model=model,
    num_iterations=3,
    consensus_threshold=0.5,
    use_chain_of_thought=True,
    aggregation_method="majority_vote"
)
```

**Note:** This is an adaptation of the Self-Consistency approach from the paper "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (https://arxiv.org/abs/2203.11171). The original paper focuses on improving reasoning accuracy through multiple generations and majority voting of final answers. This implementation applies the same principle to text critique, generating multiple critiques and using consensus to determine the most reliable feedback.

### PromptCritic

Simple prompt-based criticism.

```python
from sifaka.critics.prompt import PromptCritic

critic = PromptCritic(
    model: Model,
    critique_prompt: str,
    improvement_prompt: str
)
```

### NCriticsCritic

Implements the N-Critics ensemble approach.

```python
from sifaka.critics.n_critics import NCriticsCritic

critic = NCriticsCritic(
    critics: List[Critic],
    aggregation_method: str = "majority",
    min_agreement: float = 0.5
)
```

---

## Classifiers

### Base Classes

#### TextClassifier

Abstract base class for all text classifiers.

```python
from sifaka.classifiers.base import TextClassifier, ClassificationResult

class CustomClassifier(TextClassifier):
    def classify(self, text: str) -> ClassificationResult:
        # Implementation here
        pass
```

#### CachedTextClassifier

Base class with LRU caching for expensive classifiers.

```python
from sifaka.classifiers.base import CachedTextClassifier

class CachedCustomClassifier(CachedTextClassifier):
    def _classify_uncached(self, text: str) -> ClassificationResult:
        # Implementation here (will be automatically cached)
        pass
```

### Available Classifiers

#### BiasClassifier

Detects potential bias in text using machine learning.

```python
from sifaka.classifiers.bias import BiasClassifier, create_bias_validator

classifier = BiasClassifier()
result = classifier.classify("This text might contain bias.")

# Create validator from classifier
validator = create_bias_validator(threshold=0.7)
```

#### LanguageClassifier

Detects the language of text.

```python
from sifaka.classifiers.language import LanguageClassifier, create_language_validator

classifier = LanguageClassifier()
result = classifier.classify("Hello world")  # Returns "en"

# Validate for specific language
validator = create_language_validator(expected_language="en")
```

#### ProfanityClassifier

Detects profanity and inappropriate language.

```python
from sifaka.classifiers.profanity import ProfanityClassifier, create_profanity_validator

classifier = ProfanityClassifier()
validator = create_profanity_validator(threshold=0.8)
```

#### SentimentClassifier

Analyzes sentiment (positive, negative, neutral).

```python
from sifaka.classifiers.sentiment import SentimentClassifier, create_sentiment_validator

classifier = SentimentClassifier()
result = classifier.classify("I love this!")  # Returns positive sentiment

# Validate for positive sentiment
validator = create_sentiment_validator(target_sentiment="positive", threshold=0.7)
```

#### SpamClassifier

Detects spam content.

```python
from sifaka.classifiers.spam import SpamClassifier, create_spam_validator

classifier = SpamClassifier()
validator = create_spam_validator(threshold=0.9)
```

#### ToxicityClassifier

Detects toxic or harmful content.

```python
from sifaka.classifiers.toxicity import ToxicityClassifier, create_toxicity_validator

classifier = ToxicityClassifier()
validator = create_toxicity_validator(threshold=0.8)
```

---

## Storage

### Storage Protocol

All storage implementations follow the same interface.

```python
from sifaka.storage.protocol import Storage

# Common methods for all storage backends
storage.get(key: str) -> Optional[Any]
storage.set(key: str, value: Any) -> None
storage.search(query: str, limit: int = 10) -> List[Any]
storage.clear() -> None
```

### Memory Storage

In-memory storage for development and testing.

```python
from sifaka.storage.memory import MemoryStorage

storage = MemoryStorage()
```

### File Storage

File-based storage for persistence.

```python
from sifaka.storage.file import FileStorage

storage = FileStorage(
    file_path: str = "sifaka_storage.json",
    auto_save: bool = True
)
```

### Redis Storage

Redis-based storage with MCP integration.

```python
from sifaka.storage.redis import RedisStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

config = MCPServerConfig(
    name="redis-server",
    transport_type=MCPTransportType.STDIO,
    url="uv run --directory /path/to/mcp-redis src/main.py"
)

storage = RedisStorage(
    mcp_config: MCPServerConfig,
    key_prefix: str = "sifaka:",
    ttl: Optional[int] = None
)
```

### Milvus Storage

Vector storage with Milvus and MCP integration.

```python
from sifaka.storage.milvus import MilvusStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

config = MCPServerConfig(
    name="milvus-server",
    transport_type=MCPTransportType.STDIO,
    url="uv run --directory /path/to/mcp-server-milvus src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530"
)

storage = MilvusStorage(
    mcp_config: MCPServerConfig,
    collection_name: str = "sifaka_thoughts",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
)
```

### Cached Storage

Wraps storage backends with caching layers.

```python
from sifaka.storage.cached import CachedStorage
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.redis import RedisStorage

# Memory cache with Redis persistence
storage = CachedStorage(
    cache=MemoryStorage(),
    persistence=RedisStorage(mcp_config=redis_config)
)
```

---

## Utilities

### Error Handling

```python
from sifaka.utils.error_handling import (
    SifakaError,
    ConfigurationError,
    ValidationError,
    ModelError,
    StorageError,
    validation_context,
    model_context,
    storage_context
)

# Context managers for better error handling
with validation_context("MyValidator", "validation", "Failed to validate"):
    # Validation code here
    pass
```

### Logging

```python
from sifaka.utils.logging import get_logger, configure_logging

# Get logger for your module
logger = get_logger(__name__)

# Configure logging globally
configure_logging(level="INFO", format="detailed")
```

### Factory Utilities

```python
from sifaka.utils.factory_utils import (
    parse_model_spec,
    create_http_client,
    get_api_key
)

# Parse model specifications
provider, model_name = parse_model_spec("openai:gpt-4")

# Get API keys from environment
api_key = get_api_key("OPENAI_API_KEY", "openai")
```

### HTTP Utilities

```python
from sifaka.utils.http_utils import AsyncHTTPClient

# Async HTTP client with retry logic
async with AsyncHTTPClient() as client:
    response = await client.post("/api/endpoint", json=data)
```

### Performance Utilities

```python
from sifaka.utils.performance import timer, memory_usage

# Time function execution
with timer() as t:
    # Some operation
    pass
print(f"Operation took {t.elapsed:.2f} seconds")

# Monitor memory usage
usage = memory_usage()
print(f"Memory usage: {usage.current_mb:.1f} MB")
```

---

## Complete Example

Here's a comprehensive example showing how to use multiple Sifaka components together:

```python
from sifaka.core.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators.base import LengthValidator
from sifaka.validators.classifier import create_classifier_validator
from sifaka.classifiers.toxicity import ToxicityClassifier
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.storage.redis import RedisStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

# Create model
model = create_model("openai:gpt-4")

# Configure MCP servers
redis_config = MCPServerConfig(
    name="redis-server",
    transport_type=MCPTransportType.STDIO,
    url="uv run --directory /path/to/mcp-redis src/main.py"
)

# Create storage
redis_config = MCPServerConfig(
    transport_type=MCPTransportType.STDIO,
    command=["python", "-m", "mcp_redis"]
)
storage = RedisStorage(mcp_config=redis_config)

# Create validators
length_validator = LengthValidator(min_words=50, max_words=200)
toxicity_classifier = ToxicityClassifier()
toxicity_validator = create_classifier_validator(
    toxicity_classifier,
    threshold=0.8
)

# Create critic
critic = ReflexionCritic(model=model)

# Build and run chain
result = (Chain(model=model, storage=storage)
    .with_prompt("Write a helpful guide about AI safety.")
    .validate_with(length_validator)
    .validate_with(toxicity_validator)
    .improve_with(critic)
    .with_options(max_attempts=3, run_critics_on_valid=True)
    .run())

print(f"Generated text: {result.text}")
print(f"Iterations: {result.iteration}")
print(f"Validation results: {result.validation_results}")
```

## Quick Reference

### Common Import Patterns

```python
# Core components
from sifaka.core.chain import Chain
from sifaka.core.thought import Thought

# Models
from sifaka.models.base import create_model
from sifaka.models.openai import OpenAIModel
from sifaka.models.anthropic import AnthropicModel
from sifaka.models.gemini import GeminiModel

# Validators
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.classifier import create_classifier_validator

# Critics
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.meta_rewarding import MetaRewardingCritic
from sifaka.critics.self_consistency import SelfConsistencyCritic

# Classifiers
from sifaka.classifiers.toxicity import ToxicityClassifier
from sifaka.classifiers.sentiment import SentimentClassifier

# Storage
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.redis import RedisStorage
from sifaka.storage.milvus import MilvusStorage
```

### Environment Variables

Sifaka uses environment variables for API keys and configuration:

```bash
# Model API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export HUGGINGFACE_API_KEY="your-huggingface-key"

# Guardrails API key
export GUARDRAILS_API_KEY="your-guardrails-key"

# Storage configuration
export REDIS_URL="redis://localhost:6379"
export MILVUS_HOST="localhost"
export MILVUS_PORT="19530"
```

For more examples and detailed usage patterns, see the [examples directory](../examples/) in the repository.
