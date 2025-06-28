# Critics Quick Guide

This guide provides a quick reference for using critics in Sifaka. For detailed documentation including parameters and customization options, see the [comprehensive critics overview](../critics/overview.md).

## What are Critics?

Critics are AI-powered text analyzers that provide feedback to improve your content. Each critic uses different techniques from AI research papers.

> **Default Critic**: When no critics are specified, Sifaka uses **"reflexion"** by default.

## Basic Usage

```python
from sifaka import improve

# Use default critic (reflexion)
result = await improve("Your text here")

# Specify critics explicitly
result = await improve("Your text here", critics=["self_rag", "constitutional"])

# Use multiple critics
result = await improve(
    "Your text here",
    critics=["reflexion", "self_refine", "meta_rewarding"],
    max_iterations=3
)
```

## Available Critics at a Glance

| Critic | Best For | Key Feature |
|--------|----------|-------------|
| **reflexion** (default) | General improvement | Uses memory of previous feedback |
| **self_rag** | Fact-checking | Can use web search for verification |
| **constitutional** | Ethics & safety | Applies customizable principles |
| **n_critics** | Multiple perspectives | Evaluates from different viewpoints |
| **self_consistency** | Technical accuracy | Consensus from multiple evaluations |
| **meta_rewarding** | Quality optimization | Three-stage evaluation process |
| **self_refine** | Polish & style | Six quality dimensions |
| **style** | Voice matching | Transform to match reference style |

## Quick Examples

### Fact-Checking with Self-RAG
```python
from sifaka import Config

config = Config(enable_tools=True)
result = await improve(
    "The Eiffel Tower was built in 1850 by Napoleon.",
    critics=["self_rag"],
    config=config
)
```

### Multiple Perspectives with N-Critics
```python
from sifaka.critics.n_critics import NCriticsCritic

critic = NCriticsCritic(perspectives={
    "Technical": "Focus on accuracy",
    "Reader": "Focus on clarity"
})
result = await improve("Technical content", critics=[critic])
```

### Ethical Review with Constitutional
```python
config = Config(
    constitutional_principles=[
        "Be helpful and harmless",
        "Avoid bias"
    ]
)
result = await improve(text, critics=["constitutional"], config=config)
```

### Style Matching
```python
from sifaka.critics.style import StyleCritic

critic = StyleCritic(
    reference_text="Your brand voice examples...",
    style_description="Friendly and conversational"
)
result = await improve("Formal text", critics=[critic])
```

## Choosing Critics by Task

### For Blog Posts
```python
critics = ["reflexion", "self_refine"]  # Clarity and polish
```

### For Technical Documentation
```python
critics = ["self_rag", "self_consistency"]  # Accuracy and consistency
```

### For Marketing Copy
```python
critics = ["meta_rewarding", "n_critics"]  # Engagement and perspectives
```

### For Academic Writing
```python
critics = ["self_rag", "constitutional", "self_consistency"]  # Accuracy and ethics
```

## Advanced Usage

For detailed information about:
- Critic parameters and customization
- Performance optimization
- Creating custom critics
- Configuration options

See the [comprehensive critics documentation](../critics/overview.md).

## Related Guides

- [Basic Usage](basic-usage.md) - Getting started with Sifaka
- [Configuration](configuration.md) - Detailed configuration options
- [Validators](validators.md) - Additional quality checks
- [Advanced Usage](advanced-usage.md) - Complex workflows
