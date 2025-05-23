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
from sifaka.models.mock import MockModel
from sifaka.validators.length import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic

# Create chain components
model = MockModel()
validators = [LengthValidator(min_length=50)]
critics = [ReflexionCritic()]

# Create and configure chain
chain = Chain(
    model=model,
    validators=validators,
    critics=critics,
    max_iterations=3
)

# Run the chain
result = chain.run("Write about AI")
```

#### Configuration Options

- `max_iterations` - Maximum number of iterations (default: 3)
- `always_apply_critics` - Apply critics even when validation passes (default: False)
- `retriever` - Optional retriever for context

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
from sifaka.models.mock import MockModel

model = MockModel(name="test-model")
```

#### OpenAI Models
```python
from sifaka.models.openai import OpenAIModel

model = OpenAIModel(
    model_name="gpt-4",
    api_key="your-api-key"  # or set OPENAI_API_KEY env var
)
```

#### Anthropic Models
```python
from sifaka.models.anthropic import AnthropicModel

model = AnthropicModel(
    model_name="claude-3-sonnet-20240229",
    api_key="your-api-key"  # or set ANTHROPIC_API_KEY env var
)
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
from sifaka.validators.length import LengthValidator

validator = LengthValidator(
    min_length=10,
    max_length=1000,
    name="LengthCheck"
)
```

#### Content Validator
```python
from sifaka.validators.content import ContentValidator

validator = ContentValidator(
    required_phrases=["machine learning"],
    prohibited_phrases=["spam"],
    name="ContentCheck"
)
```

#### Regex Validator
```python
from sifaka.validators.regex import RegexValidator

validator = RegexValidator(
    pattern=r"machine learning|ML",
    name="MLMentionCheck"
)
```

#### Format Validators
```python
from sifaka.validators.format import JSONValidator, MarkdownValidator, EmailValidator

json_validator = JSONValidator()
markdown_validator = MarkdownValidator()
email_validator = EmailValidator()
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
    guard_name="pii_detection",
    api_key="your-api-key"  # or set GUARDRAILS_API_KEY env var
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
from sifaka.retrievers.mock import MockRetriever

retriever = MockRetriever(max_results=3)
```

#### Redis Retriever
```python
from sifaka.retrievers.redis import RedisRetriever

retriever = RedisRetriever(
    host="localhost",
    port=6379,
    db=0,
    max_results=5
)
```

#### Milvus Retriever
```python
from sifaka.retrievers.milvus import MilvusRetriever

retriever = MilvusRetriever(
    collection_name="documents",
    embedding_model="BAAI/bge-m3",
    max_results=3
)
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

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key  
- `GUARDRAILS_API_KEY` - GuardrailsAI API key
- `REDIS_URL` - Redis connection URL

### Installation

```bash
# Basic installation
pip install sifaka

# With specific extras
pip install sifaka[models]           # OpenAI/Anthropic models
pip install sifaka[retrievers]       # Vector database retrievers
pip install sifaka[classifiers]      # ML classifiers
pip install sifaka[all]             # Everything
```
