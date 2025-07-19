# Frequently Asked Questions

## General Questions

### What is Sifaka?

Sifaka is a Python library for improving text using AI-powered critics. It implements research-backed critique methods to iteratively refine text for better quality, clarity, and effectiveness.

### What makes Sifaka different from other text improvement tools?

Sifaka focuses on:
- **Research-backed methods**: Implements critique techniques from academic papers
- **Iterative improvement**: Uses multiple rounds of critique and refinement
- **Flexibility**: Supports multiple LLM providers and custom critics
- **Observability**: Complete visibility into the improvement process
- **Simplicity**: Single function API (`improve()`) for ease of use

### Which LLM providers are supported?

Sifaka supports:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Any OpenAI-compatible API

### Do I need API keys for all providers?

No, you only need an API key for the provider you want to use. Sifaka will automatically use the available provider based on your environment variables.

## Usage Questions

### How do I get started?

```python
import asyncio
from sifaka import improve

async def main():
    result = await improve("Your text here")
    print(result.final_text)

asyncio.run(main())
```

### Can I use multiple critics at once?

Yes! You can combine multiple critics:

```python
from sifaka import improve
from sifaka.core.types import CriticType

result = await improve(
    "Your text",
    critics=[CriticType.SELF_REFINE, CriticType.REFLEXION]
)
```

### How do I save results?

Use the storage parameter:

```python
from sifaka import improve
from sifaka.storage.file import FileStorage

result = await improve(
    "Your text",
    storage=FileStorage("./results")
)
```

### Can I set a timeout?

Yes, use the `timeout_seconds` parameter:

```python
result = await improve(
    "Your text",
    timeout_seconds=30.0  # 30 second timeout
)
```

### How do I validate the improved text?

Use validators:

```python
from sifaka import improve
from sifaka.validators import LengthValidator

result = await improve(
    "Your text",
    validators=[LengthValidator(min_length=100, max_length=500)]
)
```

## Critic Questions

### What critics are available?

Built-in critics include:
- **SELF_REFINE**: General-purpose improvement
- **REFLEXION**: Learning from previous attempts
- **CONSTITUTIONAL**: Principle-based evaluation
- **SELF_CONSISTENCY**: Consensus-based improvement
- **SELF_RAG**: Fact-checking with retrieval
- **STYLE**: Style and tone adjustment
- **META_REWARDING**: Self-evaluating critique
- **N_CRITICS**: Multiple perspective evaluation

### Which critic should I use?

- **General improvement**: `SELF_REFINE`
- **Academic/technical**: `REFLEXION` or `SELF_RAG`
- **Marketing/creative**: `STYLE`
- **Safety-critical**: `CONSTITUTIONAL`
- **Balanced perspectives**: `SELF_CONSISTENCY` or `N_CRITICS`

### Can I create custom critics?

Yes! Implement the `CriticPlugin` interface:

```python
from sifaka.plugins import CriticPlugin
from sifaka.core.models import CritiqueResult

class MyCritic(CriticPlugin):
    async def critique(self, text: str, result):
        # Your critique logic
        return CritiqueResult(
            critic="my_critic",
            feedback="Your feedback",
            suggestions=["Suggestion 1"],
            needs_improvement=True,
            confidence=0.8
        )
```

## Performance Questions

### How can I improve performance?

1. **Use faster models**: Gemini Flash or GPT-3.5-turbo
2. **Reduce iterations**: Set `max_iterations=1` or `2`
3. **Use connection pooling**: Automatically enabled
4. **Batch processing**: Process multiple texts concurrently

### Does Sifaka cache results?

Not by default, but you can:
- Use `FileStorage` or `RedisStorage` to save results
- Implement custom caching in your application

### How much does it cost?

Costs depend on:
- LLM provider and model choice
- Text length
- Number of iterations
- Number of critics

Typical improvements cost $0.001-0.01 per text with efficient models.

## Troubleshooting

### Why am I getting timeout errors?

Common causes:
- Slow LLM responses
- Too many iterations
- Complex critics

Solutions:
- Increase `timeout_seconds`
- Reduce `max_iterations`
- Use faster models

### Why isn't my text improving?

Check:
- Temperature settings (try 0.7-0.9)
- Critic selection (try different critics)
- Model choice (larger models often perform better)
- Input text quality

### How do I debug issues?

Enable logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use Logfire integration for detailed tracing.

## Integration Questions

### Can I use Sifaka in production?

Yes! Sifaka is designed for production use with:
- Comprehensive error handling
- Timeout support
- Connection pooling
- Storage backends
- Monitoring integration

### Does Sifaka work with async frameworks?

Yes, Sifaka is fully async and works with:
- FastAPI
- aiohttp
- Django (with async views)
- Any async Python framework

### Can I use Sifaka with Jupyter notebooks?

Yes:
```python
import nest_asyncio
nest_asyncio.apply()

# Now you can use await in Jupyter
result = await improve("Your text")
```

### Is there a synchronous API?

Not built-in, but you can wrap it:
```python
import asyncio

def improve_sync(text, **kwargs):
    return asyncio.run(improve(text, **kwargs))
```

## Contributing

### How can I contribute?

1. Check out our [GitHub repository](https://github.com/sifaka-ai/sifaka)
2. Read the [Developer Setup](development/DEVELOPER_SETUP.md) guide
3. Submit issues, PRs, or feature requests

### Where can I get help?

- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and ideas
- Documentation: This site

### Is Sifaka open source?

Yes! Sifaka is MIT licensed and open source.
