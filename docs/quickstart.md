# Sifaka Quick Start Guide

Get up and running with Sifaka in 5 minutes.

## Installation

> **Note**: Sifaka will be available on PyPI in October 2025. Until then, install from source:

```bash
# Clone the repository
git clone https://github.com/sifaka-ai/sifaka
cd sifaka

# Install with uv (recommended)
uv pip install -e .

# Or with standard pip
pip install -e .
```

## Basic Setup

1. **Set your API key** (required for real usage):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

2. **Basic improvement**:
   ```python
   import asyncio
   from sifaka import improve

   async def main():
       result = await improve("Write about renewable energy")
       print(result.final_text)

   asyncio.run(main())
   ```

## Common Use Cases

### 1. Content Writing
Improve blog posts, articles, and marketing copy:

```python
result = await improve(
    "Write a blog post about sustainable living tips",
    critics=["reflexion", "constitutional"],
    max_iterations=3
)
```

### 2. Academic Writing
Enhance research papers and academic content:

```python
from sifaka.validators.basic import LengthValidator

result = await improve(
    "Write an abstract for machine learning research",
    critics=["n_critics", "self_rag"],
    validators=[LengthValidator(min_length=150, max_length=300)]
)
```

### 3. Technical Documentation
Improve clarity and completeness:

```python
result = await improve(
    "Explain how to set up a REST API",
    critics=["self_refine", "prompt"],
    max_iterations=4
)
```

### 4. Creative Writing
Enhance narratives and creative content:

```python
from sifaka import Config

config = Config(temperature=0.8)  # More creative
result = await improve(
    "Write a short story about time travel",
    critics=["reflexion", "self_consistency"],
    config=config
)
```

## Configuration Examples

### Fast Processing
```python
# Quick turnaround
config = Config(model="gpt-4o-mini", max_iterations=2)
result = await improve(text, config=config)
```

### Quality Focus
```python
# High-quality output
config = Config(model="gpt-4o", max_iterations=5)
result = await improve(
    text,
    critics=["reflexion", "constitutional", "n_critics"],
    config=config
)
```

### Complete Example
```python
from sifaka import improve, Config
from sifaka.validators.basic import LengthValidator

config = Config(
    model="gpt-4o-mini",
    max_iterations=3,
    show_improvement_prompt=True
)

result = await improve(
    "Write about climate change impacts",
    critics=["reflexion"],
    validators=[LengthValidator(min_length=300, max_length=800)],
    config=config
)
```


## Validation Examples

### Length Requirements
```python
from sifaka.validators.basic import LengthValidator

result = await improve(
    "Short text that needs expansion",
    validators=[LengthValidator(min_length=200)],
    max_iterations=3
)
```

### Content Requirements
```python
from sifaka.validators.basic import ContentValidator

result = await improve(
    "Write about research methodology",
    validators=[ContentValidator(
        required_terms=["hypothesis", "data", "analysis"]
    )],
    critics=["self_rag"]  # Good for factual content
)
```

## Storage Examples

### Save Results
```python
from sifaka.storage.file import FileStorage

storage = FileStorage("./my_results")

result = await improve(
    "Important content to save",
    storage=storage
)

print(f"Saved as: {result.id}")
```

### Load Previous Results
```python
# Load by ID
loaded = await storage.load(result.id)
print(f"Original: {loaded.original_text}")
print(f"Final: {loaded.final_text}")

# List all results
all_results = await storage.list_results()
for result_id in all_results:
    r = await storage.load(result_id)
    print(f"{result_id}: {r.original_text[:50]}...")
```

## Error Handling

```python
from sifaka import improve, Config
from sifaka.core.exceptions import TimeoutError

config = Config(timeout_seconds=30)  # Short timeout

try:
    result = await improve(
        text,
        config=config
    )
except TimeoutError:
    print("Operation timed out")
except Exception as e:
    print(f"Other error: {e}")
```

## Analyzing Results

```python
result = await improve(text, critics=["reflexion", "constitutional"])

# Basic metrics
print(f"Iterations: {result.iteration}")
print(f"Confidence: {result.confidence:.2f}")

# Critique analysis
for critique in result.critiques:
    print(f"\nCritic: {critique.critic}")
    print(f"Needs improvement: {critique.needs_improvement}")
    print(f"Confidence: {critique.confidence:.2f}")
    print(f"Suggestions: {len(critique.suggestions)}")

# Generation history
for i, gen in enumerate(result.generations):
    print(f"\nGeneration {i}")
    print(f"Text: {gen.text[:100]}...")
```

## Tips for Success

1. **Start Small**: Begin with simple examples and low iteration limits
2. **Choose the Right Critics**: Match critics to your content type
3. **Set Reasonable Limits**: Balance quality needs with iterations
4. **Use Validation**: Enforce hard requirements with validators
5. **Save Important Results**: Use file storage for valuable outputs

## Next Steps

- Read the full [API documentation](API.md)
- Explore [example scripts](../examples/) in the examples directory
- Check out the [research papers](README.md#critics) behind each critic
- Join the community for tips and best practices

## Troubleshooting

### "No module named 'sifaka'"
```bash
# Install from source (current method)
git clone https://github.com/sifaka-ai/sifaka
cd sifaka
uv pip install -e .
```

### "Authentication failed"
```bash
export OPENAI_API_KEY="your-actual-api-key"
```


### "Operation too slow"
Try faster settings:
```python
config = Config(model="gpt-4o-mini", max_iterations=1)
result = await improve(
    text,
    critics=["reflexion"],
    config=config
)
```

## Getting Help

- **Documentation**: Full API reference in [API.md](API.md)
- **Examples**: See the [examples directory](../examples/) for working code samples
- **Issues**: Report bugs on GitHub
- **Community**: Join discussions for tips and use cases

Happy improving! 🚀
