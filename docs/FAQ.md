# Frequently Asked Questions (FAQ)

## General Questions

### What is Sifaka?

Sifaka is a Python library that improves AI-generated text through iterative critique using research-backed techniques. It provides a transparent feedback loop where AI systems validate and improve their own outputs, with complete observability of the improvement process.

### How does Sifaka differ from other text improvement tools?

Unlike simple prompt engineering or one-shot generation:
- **Research-backed**: Implements proven techniques like Reflexion, Constitutional AI, and Self-Refine
- **Iterative**: Continuously improves text through multiple rounds
- **Observable**: Provides complete audit trails of all improvements
- **Pluggable**: Easily extend with custom critics, validators, and storage

### What LLM providers does Sifaka support?

Sifaka currently supports:
- OpenAI (GPT-3.5, GPT-4, GPT-4 Turbo)
- Anthropic (Claude 2, Claude 3)
- Google (Gemini Pro, Gemini Ultra)

You can easily switch between providers or use different providers for generation and critique.

### How much does it cost to use Sifaka?

Sifaka itself is free and open-source. However, you'll need to pay for API calls to your chosen LLM provider. Costs depend on:
- Number of iterations
- Length of text
- Model selection
- Provider pricing

Use the `pricing` module to estimate costs before running improvements.

## Common Integration Patterns

### How do I use Sifaka in a web application?

```python
from sifaka import improve_async
from fastapi import FastAPI

app = FastAPI()

@app.post("/improve")
async def improve_text(text: str, max_iterations: int = 3):
    result = await improve_async(
        text,
        critics=["reflexion"],
        max_iterations=max_iterations
    )
    return {
        "original": result.original_text,
        "improved": result.final_text,
        "iterations": result.iterations,
        "cost": result.total_cost
    }
```

### How can I save results to a database?

Use the storage backend system:

```python
from sifaka import improve_sync, register_storage_backend
from sifaka.storage.base import StorageBackend
import asyncpg

class PostgresBackend(StorageBackend):
    def __init__(self, connection_string):
        self.conn_string = connection_string

    async def save(self, result):
        conn = await asyncpg.connect(self.conn_string)
        await conn.execute('''
            INSERT INTO sifaka_results (id, original, final, data)
            VALUES ($1, $2, $3, $4)
        ''', result.id, result.original_text, result.final_text, result.json())
        await conn.close()
        return result.id

# Register and use
register_storage_backend("postgres", PostgresBackend)
result = improve_sync(text, storage_backend="postgres")
```

### How do I implement retry logic for API failures?

Sifaka includes built-in retry logic, but you can customize it:

```python
from sifaka import improve_sync
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def improve_with_retry(text):
    return improve_sync(
        text,
        retry_config={
            "max_retries": 3,
            "backoff_factor": 2.0,
            "retry_on": [429, 500, 502, 503, 504]
        }
    )
```

### How can I use Sifaka with streaming responses?

```python
from sifaka import improve_stream

async def stream_improvements(text):
    async for event in improve_stream(text, critics=["reflexion"]):
        if event.type == "critique":
            print(f"Critique: {event.data}")
        elif event.type == "improvement":
            print(f"Improved: {event.data}")
        elif event.type == "complete":
            print(f"Final: {event.data.final_text}")
```

## Performance Tuning Tips

### How can I reduce API costs?

1. **Use cheaper models for critique**:
   ```python
   improve_sync(text, generation_model="gpt-4", critique_model="gpt-3.5-turbo")
   ```

2. **Limit iterations**:
   ```python
   improve_sync(text, max_iterations=2)  # Default is 3
   ```

3. **Use caching**:
   ```python
   from sifaka.middleware import CacheMiddleware

   improve_sync(
       text,
       middleware=[CacheMiddleware(ttl=3600)]
   )
   ```

4. **Batch similar requests**:
   ```python
   results = await improve_batch_async(
       texts=["text1", "text2", "text3"],
       critics=["reflexion"]
   )
   ```

### How can I speed up improvements?

1. **Use async operations**:
   ```python
   # Process multiple texts concurrently
   import asyncio

   texts = ["text1", "text2", "text3"]
   tasks = [improve_async(text) for text in texts]
   results = await asyncio.gather(*tasks)
   ```

2. **Disable unnecessary validators**:
   ```python
   improve_sync(text, validators=None)  # Skip validation
   ```

3. **Use simpler critics for drafts**:
   ```python
   # Quick improvement for drafts
   draft_result = improve_sync(text, critics=["prompt"], max_iterations=1)

   # Thorough improvement for final version
   final_result = improve_sync(
       draft_result.final_text,
       critics=["reflexion", "constitutional"],
       max_iterations=3
   )
   ```

### How do I handle rate limits?

Sifaka automatically handles rate limits with exponential backoff. You can customize:

```python
improve_sync(
    text,
    rate_limit_config={
        "requests_per_minute": 20,
        "tokens_per_minute": 40000,
        "concurrent_requests": 5
    }
)
```

## Troubleshooting Guide

### Why is my text not improving?

Common causes:
1. **Validators too strict**: Relax validation rules
2. **Max iterations too low**: Increase `max_iterations`
3. **Prompt issues**: Check critic prompts are appropriate
4. **Model limitations**: Try a more capable model

Debug with verbose logging:
```python
import logging
logging.getLogger("sifaka").setLevel(logging.DEBUG)

result = improve_sync(text, debug=True)
print(result.improvement_history)
```

### How do I handle "context too long" errors?

```python
from sifaka.middleware import ChunkingMiddleware

# Automatically chunk long texts
result = improve_sync(
    long_text,
    middleware=[ChunkingMiddleware(max_chunk_size=2000)]
)

# Or manually chunk
chunks = [long_text[i:i+2000] for i in range(0, len(long_text), 2000)]
results = [improve_sync(chunk) for chunk in chunks]
final_text = " ".join(r.final_text for r in results)
```

### Why am I getting "API key not found" errors?

Check your API key setup:

```python
# Method 1: Environment variable
os.environ["OPENAI_API_KEY"] = "your-key"

# Method 2: Direct parameter
improve_sync(text, api_key="your-key")

# Method 3: Config file (~/.sifaka/config.json)
{
    "openai_api_key": "your-key",
    "anthropic_api_key": "your-key"
}
```

### How do I debug critic behavior?

```python
# Get detailed critique information
result = improve_sync(text, return_critique_details=True)

for iteration in result.improvement_history:
    print(f"\nIteration {iteration['iteration']}:")
    print(f"Critique: {iteration['critique']}")
    print(f"Improvement: {iteration['improved_text']}")
    print(f"Tokens: {iteration['tokens_used']}")
```

## Model Selection Guidance

### Which model should I use?

**For general text improvement:**
- GPT-4: Best quality, higher cost
- GPT-3.5 Turbo: Good balance of quality/cost
- Claude 3 Sonnet: Strong reasoning, good for complex texts

**For specific use cases:**
- Academic writing: GPT-4 or Claude 3 Opus
- Creative writing: Claude 3 or GPT-4
- Technical docs: GPT-4 or Gemini Pro
- Quick drafts: GPT-3.5 Turbo

### Can I use different models for different critics?

Yes! Configure per-critic models:

```python
result = improve_sync(
    text,
    critic_configs={
        "reflexion": {"model": "gpt-4"},
        "constitutional": {"model": "claude-3-sonnet"},
        "self_refine": {"model": "gpt-3.5-turbo"}
    }
)
```

### How do I use local models?

Sifaka supports any OpenAI-compatible API:

```python
from sifaka.core.config import ModelConfig

# Use with LocalAI, Ollama, etc.
config = ModelConfig(
    provider="openai",
    model_name="llama2",
    api_key="not-needed",
    base_url="http://localhost:8080/v1"
)

result = improve_sync(text, model_config=config)
```

## Advanced Topics

### How do I create custom improvement strategies?

```python
from sifaka.core.interfaces import ImprovementStrategy

class MyStrategy(ImprovementStrategy):
    def should_continue(self, iteration, result):
        # Custom logic for when to stop
        return iteration < 5 and "perfect" not in result.final_text

    def select_next_critic(self, history):
        # Custom critic selection logic
        if len(history) == 0:
            return "reflexion"
        elif "clarity" in history[-1].critique:
            return "self_refine"
        return "constitutional"

result = improve_sync(text, strategy=MyStrategy())
```

### Can I use Sifaka for non-English text?

Yes, but performance varies by language and model:

```python
# Explicitly set language
result = improve_sync(
    text="Bonjour le monde",
    language="fr",
    critics=["reflexion"],
    critic_kwargs={"language": "French"}
)

# Best models for non-English:
# - GPT-4 (most languages)
# - Claude 3 (major languages)
# - Gemini Pro (100+ languages)
```

### How do I integrate with existing ML pipelines?

```python
from sifaka import improve_sync
import mlflow

# Track improvements in MLflow
with mlflow.start_run():
    result = improve_sync(text)

    mlflow.log_param("critics", ["reflexion"])
    mlflow.log_metric("iterations", result.iterations)
    mlflow.log_metric("improvement_score", result.metadata.get("score", 0))
    mlflow.log_text(result.final_text, "improved_text.txt")
```

## Getting Help

### Where can I find more examples?

- `/examples` directory in the repository
- [API documentation](https://sifaka.ai/docs/api)
- [Community examples](https://github.com/sifaka-ai/sifaka/discussions)

### How do I report bugs or request features?

- Bug reports: [GitHub Issues](https://github.com/sifaka-ai/sifaka/issues)
- Feature requests: [GitHub Discussions](https://github.com/sifaka-ai/sifaka/discussions)
- Security issues: security@sifaka.ai

### Is there a community forum?

Yes! Join our community:
- [Discord server](https://discord.gg/sifaka)
- [GitHub Discussions](https://github.com/sifaka-ai/sifaka/discussions)
- [Stack Overflow tag](https://stackoverflow.com/questions/tagged/sifaka)
