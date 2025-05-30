# Sifaka Quick Reference

## Chain Selection

### 🚀 PydanticAI Chain (Recommended for New Projects)
```python
from sifaka.agents import create_pydantic_chain
from pydantic_ai import Agent

agent = Agent("openai:gpt-4")
@agent.tool_plain
def search(query: str) -> str:
    return f"Results for: {query}"

chain = create_pydantic_chain(
    agent=agent,
    validators=[LengthValidator(50, 500)],
    critics=[ReflexionCritic(model)],
    always_apply_critics=True
)
```

**Best for**: Tool calling, modern patterns, extensible workflows, type safety

### 🏗️ Traditional Chain (Deprecated)
```python
# ⚠️ Deprecated - Use PydanticAI Chain for new projects
from sifaka import Chain

chain = Chain(
    model=create_model("openai:gpt-4"),
    prompt="Your prompt here",
    max_improvement_iterations=3,
    always_apply_critics=True
)
chain = chain.validate_with(validator).improve_with(critic)
```

**Best for**: Pipeline workflows, pre-built features, configuration-driven development

## Quick Start Patterns

### Development (Fast)
```python
from sifaka.quickstart import QuickStart
chain = QuickStart.for_development()
result = chain.run("Your prompt")
```

### Production (Validated)
```python
chain = QuickStart.for_production(
    "openai:gpt-4",
    "Your prompt",
    validators=["length", "content"],
    critics=["reflexion"]
)
```

### Research (Comprehensive)
```python
chain = QuickStart.for_research(
    "anthropic:claude-3-sonnet",
    "Research prompt"
)
```

## Common Validators

```python
from sifaka.validators import (
    LengthValidator,
    RegexValidator,
    ContentValidator,
    BiasValidator,
    ReadingLevelValidator
)

# Length constraints
LengthValidator(min_length=50, max_length=500)

# Content requirements
RegexValidator(required_patterns=[r"\b\d{4}\b"])  # Requires year
ContentValidator(required_keywords=["AI", "machine learning"])

# Quality checks
BiasValidator(threshold=0.7)
ReadingLevelValidator(min_level=6, max_level=12)
```

## Common Critics

```python
from sifaka.critics import (
    ReflexionCritic,
    SelfRefineCritic,
    ConstitutionalCritic,
    SelfRAGCritic,
    NCriticsCritic
)

# Basic improvement
ReflexionCritic(model=model)
SelfRefineCritic(model=model)

# Constitutional AI
ConstitutionalCritic(model=model)

# Advanced critics
SelfRAGCritic(model=model, retriever=retriever)
NCriticsCritic(model=model, num_critics=3)
```

## Storage Options

```python
from sifaka.storage import (
    MemoryStorage,
    FileStorage,
    RedisStorage,    # ⚠️ Currently broken
    MilvusStorage    # ⚠️ Currently broken
)

# Development (Working)
MemoryStorage()
FileStorage("thoughts.json")

# Production (Currently broken - use Memory/File for now)
# RedisStorage(host="localhost", port=6379)    # MCP issues
# MilvusStorage(host="localhost", port=19530)  # MCP issues
```

## Model Creation

```python
from sifaka.models import create_model

# OpenAI
create_model("openai:gpt-4")
create_model("openai:gpt-4o-mini")

# Anthropic
create_model("anthropic:claude-3-sonnet")
create_model("anthropic:claude-3-5-haiku-latest")

# Google
create_model("google:gemini-1.5-flash")

# HuggingFace
create_model("huggingface:microsoft/DialoGPT-medium")

# Ollama (local)
create_model("ollama:llama2")

# Mock (testing)
create_model("mock:test-model")
```

## Configuration Patterns

### Always Apply Critics
```python
chain = Chain(
    model=model,
    prompt=prompt,
    always_apply_critics=True,  # Run critics even if validation passes
    max_improvement_iterations=3
)
```

### Validation-Driven Improvement
```python
chain = Chain(
    model=model,
    prompt=prompt,
    apply_improvers_on_validation_failure=True,  # Only improve if validation fails
    max_improvement_iterations=2
)
```

### Hybrid Storage
```python
from sifaka.storage import CachedStorage

# Currently limited to Memory + File due to MCP issues
storage = CachedStorage(
    cache=MemoryStorage(),     # Fast access
    persistence=FileStorage()  # Persistence
)
```

## Error Handling

```python
try:
    result = chain.run()
except ValidationError as e:
    print(f"Validation failed: {e}")
except ModelError as e:
    print(f"Model error: {e}")
except StorageError as e:
    print(f"Storage error: {e}")
```

## Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
chain = Chain(model=model, prompt=prompt, debug=True)
```

## Performance Tips

1. **Use appropriate models**: Smaller models for critics, larger for generation
2. **Limit iterations**: Set reasonable `max_improvement_iterations`
3. **Cache storage**: Use Redis for frequently accessed data
4. **Async patterns**: Use PydanticAI chains for better async performance
5. **Batch operations**: Process multiple prompts together when possible

## Common Patterns

### Content Generation with Validation
```python
chain = Chain(model=model, prompt=prompt)
chain = chain.validate_with(LengthValidator(100, 1000))
chain = chain.validate_with(ContentValidator(required_keywords=["AI"]))
chain = chain.improve_with(ReflexionCritic(model))
```

### Research with Retrieval
```python
chain = Chain(
    model=model,
    prompt=prompt,
    model_retrievers=[vector_retriever],
    critic_retrievers=[context_retriever]
)
```

### Multi-Stage Improvement
```python
chain = Chain(model=model, prompt=prompt)
chain = chain.improve_with(SelfRefineCritic(model))
chain = chain.improve_with(ConstitutionalCritic(model))
chain = chain.improve_with(ReflexionCritic(model))
```

## Troubleshooting Quick Fixes

- **Import errors**: Install extras `pip install sifaka[openai,anthropic]`
- **API errors**: Check environment variables `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- **Timeout issues**: Use smaller models or reduce iterations
- **Memory issues**: Use file storage instead of memory storage
- **Validation failures**: Check validator thresholds and requirements
- **Storage issues**: Use Memory or File storage (Redis/Milvus MCP currently broken)
