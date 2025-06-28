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

### What is the default critic when none is specified?

When you don't specify any critics, Sifaka automatically uses **"reflexion"** as the default critic. This provides general-purpose text improvement suitable for most use cases.

```python
# These two are equivalent:
result = await improve("Your text")  # Uses reflexion by default
result = await improve("Your text", critics=["reflexion"])
```

## Common Integration Patterns

### How do I use Sifaka in a web application?

```python
from sifaka import improve
from fastapi import FastAPI

app = FastAPI()

@app.post("/improve")
async def improve_text(text: str, max_iterations: int = 3):
    # Note: reflexion is the default critic if none specified
    result = await improve(
        text,
        max_iterations=max_iterations
    )
    return {
        "original": result.original_text,
        "improved": result.final_text,
        "iterations": result.iteration,
    }
```

### How can I save results to a database?

Use the storage backend system:

```python
from sifaka import improve_sync
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
        ''', result.id, result.original_text, result.final_text, result.model_dump_json())
        await conn.close()
        return result.id

# Use custom storage
storage = PostgresBackend("postgresql://...")
result = improve_sync(text, storage=storage)
```

### How do I implement retry logic for API failures?

Sifaka includes built-in retry logic, but you can customize it:

```python
from sifaka import improve_sync, Config

# Sifaka includes built-in retry logic via Config
config = Config(
    retry_max_attempts=3,
    retry_initial_delay=1.0,
    retry_exponential_base=2.0
)

def improve_with_retry(text):
    return improve_sync(text, config=config)
```

### How can I monitor improvement progress?

```python
from sifaka import improve

async def monitor_improvements(text):
    result = await improve(text, critics=["reflexion"])

    # Access improvement history
    print(f"Total iterations: {result.iteration}")
    print(f"Processing time: {result.processing_time:.2f}s")

    # Review each critique
    for critique in result.critiques:
        print(f"{critique.critic}: {critique.feedback}")
```

## Performance Tuning Tips

### How can I reduce API costs?

1. **Use hybrid model approach** (default in Sifaka):
   ```python
   from sifaka import Config
   config = Config(
       model="gpt-4o-mini",           # Quality for generation
       critic_model="gpt-3.5-turbo"  # Speed for criticism (default)
   )
   improve_sync(text, config=config)
   ```

2. **Limit iterations**:
   ```python
   improve_sync(text, max_iterations=2)  # Default is 3
   ```

3. **Use file storage for caching**:
   ```python
   from sifaka.storage.file import FileStorage

   storage = FileStorage("./cache")
   improve_sync(text, storage=storage)
   ```

4. **Process multiple texts concurrently**:
   ```python
   import asyncio

   async def batch_improve(texts):
       # Uses reflexion (default) critic for each text
       tasks = [improve(text) for text in texts]
       return await asyncio.gather(*tasks)
   ```

### How can I speed up improvements?

1. **Use async operations**:
   ```python
   # Process multiple texts concurrently
   import asyncio

   texts = ["text1", "text2", "text3"]
   tasks = [improve(text) for text in texts]
   results = await asyncio.gather(*tasks)
   ```

2. **Use fewer critics**:
   ```python
   # Fast single critic (reflexion is default)
   improve_sync(text, max_iterations=1)
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

Sifaka automatically handles rate limits with exponential backoff via retry configuration:

```python
from sifaka import Config

config = Config(
    retry_max_attempts=5,
    retry_initial_delay=2.0,
    retry_exponential_base=2.0
)

improve_sync(text, config=config)
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

config = Config(show_improvement_prompt=True)
result = improve_sync(text, config=config)
print(f"Iterations: {result.iteration}")
for critique in result.critiques:
    print(f"{critique.critic}: {critique.feedback}")
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
result = improve_sync(text, config=Config(show_improvement_prompt=True))

print(f"Total iterations: {result.iteration}")
for i, critique in enumerate(result.critiques):
    print(f"\nIteration {i+1}:")
    print(f"Critic: {critique.critic}")
    print(f"Feedback: {critique.feedback}")
    print(f"Tokens used: {critique.tokens_used}")
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
    mlflow.log_metric("iterations", result.iteration)
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
