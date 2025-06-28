# Advanced Usage Guide

This guide covers advanced features and patterns for getting the most out of Sifaka.

## Async and Sync APIs

Sifaka provides both async and sync APIs:

```python
# Async (recommended)
from sifaka import improve

async def improve_text():
    result = await improve("Your text")
    return result

# Sync (for simpler scripts)
from sifaka import improve_sync

result = improve_sync("Your text")
```

## Multiple Critics Strategy

Combine different critics for comprehensive improvement:

```python
# Sequential critic chain
result = await improve(
    text,
    critics=["reflexion", "self_rag", "constitutional"],
    max_iterations=3
)

# Custom critic instances
from sifaka.critics.n_critics import NCriticsCritic

perspectives_critic = NCriticsCritic(
    perspectives={
        "Domain Expert": "Ensure technical accuracy",
        "Editor": "Improve clarity and flow",
        "Fact Checker": "Verify all claims"
    }
)

result = await improve(
    text,
    critics=["self_refine", perspectives_critic, "meta_rewarding"]
)
```

## Complex Validation Rules

Build sophisticated validation logic:

```python
from sifaka.validators.composable import Validator

# Research paper validator
research_validator = (
    # Structure requirements
    Validator.create("research_paper")
    .length(3000, 8000)
    .sentences(100, 400)
    .contains(["abstract", "introduction", "methodology", "results", "conclusion"], mode="all")
    .matches(r"\[\d+\]", "citations")
    .build()
)

# Style requirements
style_validator = (
    Validator.create("academic_style")
    .contains(["however", "therefore", "furthermore", "moreover"], mode="any")
    .matches(r"[A-Z]\w+\set\sal\.\s\(\d{4}\)", "author_citations")
    .build()
)

# Combined requirements
paper_validator = research_validator & style_validator

result = await improve(
    draft,
    validators=[paper_validator],
    max_iterations=5
)
```

## Error Handling and Recovery

Robust error handling for production use:

```python
from sifaka import improve, SifakaError, ModelProviderError
from sifaka.core.exceptions import ValidationError, TimeoutError

async def improve_with_fallback(text: str):
    try:
        # Try primary provider
        result = await improve(
            text,
            provider="openai",
            model="gpt-4o",
            timeout=30.0
        )
        return result

    except TimeoutError:
        # Fallback to faster model
        return await improve(
            text,
            provider="openai",
            model="gpt-4o-mini",
            max_iterations=1
        )

    except ModelProviderError as e:
        # Fallback to different provider
        if "rate_limit" in str(e):
            return await improve(
                text,
                provider="anthropic",
                model="claude-3-haiku-20240307"
            )
        raise

    except ValidationError as e:
        # Handle validation failures
        print(f"Validation failed: {e}")
        # Return original or partially improved text
        return e.partial_result if hasattr(e, 'partial_result') else text
```

## Streaming and Progress Tracking

Monitor improvement progress:

```python
from sifaka import improve, Config
from sifaka.core.models import SifakaResult

class ProgressTracker:
    async def on_iteration(self, iteration: int, result: SifakaResult):
        print(f"Iteration {iteration}: Confidence {result.critiques[-1].confidence:.2f}")

# Future API (example)
result = await improve(
    text,
    config=Config(max_iterations=5),
    progress_callback=ProgressTracker().on_iteration
)
```

## Batch Processing

Process multiple texts efficiently:

```python
import asyncio
from typing import List

async def batch_improve(texts: List[str]) -> List[SifakaResult]:
    # Process in parallel with concurrency limit
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

    async def improve_with_limit(text: str):
        async with semaphore:
            return await improve(text)

    tasks = [improve_with_limit(text) for text in texts]
    return await asyncio.gather(*tasks, return_exceptions=True)

# Usage
texts = ["Text 1", "Text 2", "Text 3", ...]
results = await batch_improve(texts)

# Handle results and errors
for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Text {i} failed: {result}")
    else:
        print(f"Text {i} improved: {len(result.final_text)} chars")
```

## Custom Storage Backend

Implement custom storage for caching:

```python
from sifaka.storage.base import StorageBackend
from typing import Optional, Any
import json

class RedisStorage(StorageBackend):
    def __init__(self, redis_client):
        self.redis = redis_client

    async def get(self, key: str) -> Optional[Any]:
        value = await self.redis.get(key)
        return json.loads(value) if value else None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        await self.redis.set(
            key,
            json.dumps(value),
            ex=ttl
        )

    async def delete(self, key: str) -> None:
        await self.redis.delete(key)

    async def clear(self) -> None:
        await self.redis.flushdb()

# Register and use
from sifaka.core.plugins import register_storage_backend

register_storage_backend("redis", RedisStorage)
```

## Performance Optimization

Tips for optimal performance:

```python
# 1. Use appropriate models
# Faster for simple tasks
quick_result = await improve(text, model="gpt-4o-mini")

# Better for complex tasks
quality_result = await improve(text, model="gpt-4o")

# 2. Optimize iterations
config = Config(
    max_iterations=2,  # Often sufficient
    min_quality_score=0.75  # Stop early if good enough
)

# 3. Cache results
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_improvement(text_hash: str):
    # Cache based on text hash
    pass

# 4. Use batch APIs when available
# Process multiple texts in one request
```

## Integration Examples

### FastAPI Integration
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sifaka import improve

app = FastAPI()

class ImproveRequest(BaseModel):
    text: str
    critics: List[str] = ["self_refine"]
    max_iterations: int = 3

@app.post("/improve")
async def improve_endpoint(request: ImproveRequest):
    try:
        result = await improve(
            request.text,
            critics=request.critics,
            max_iterations=request.max_iterations
        )
        return {
            "original": result.original_text,
            "improved": result.final_text,
            "confidence": result.critiques[-1].confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Gradio Interface
```python
import gradio as gr
from sifaka import improve_sync

def improve_text_ui(text, critic_type, max_iterations):
    result = improve_sync(
        text,
        critics=[critic_type],
        max_iterations=int(max_iterations)
    )
    return result.final_text, f"Confidence: {result.critiques[-1].confidence:.2f}"

interface = gr.Interface(
    fn=improve_text_ui,
    inputs=[
        gr.Textbox(lines=10, label="Input Text"),
        gr.Dropdown(["reflexion", "self_rag", "constitutional"], label="Critic"),
        gr.Slider(1, 5, value=3, label="Max Iterations")
    ],
    outputs=[
        gr.Textbox(lines=10, label="Improved Text"),
        gr.Text(label="Final Confidence")
    ]
)

interface.launch()
```

## Debugging and Monitoring

Enable detailed logging:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Get insights into the improvement process
result = await improve(text, config=Config(debug=True))

# Inspect critique history
for i, critique in enumerate(result.critiques):
    print(f"Iteration {i+1}:")
    print(f"  Critic: {critique.critic}")
    print(f"  Feedback: {critique.feedback[:100]}...")
    print(f"  Confidence: {critique.confidence}")
```

## Best Practices

1. **Start simple**: Begin with basic usage and add complexity as needed
2. **Monitor performance**: Track API costs and latency
3. **Handle errors gracefully**: Always have fallback strategies
4. **Cache when possible**: Reuse results for identical inputs
5. **Choose critics wisely**: Different critics for different use cases
6. **Test thoroughly**: Validate your validators and critic combinations
