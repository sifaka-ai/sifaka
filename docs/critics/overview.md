# Critics Overview

Critics are the heart of Sifaka's text improvement system. Each critic implements a different research-backed technique for analyzing and improving text.

## Available Critics

### Reflexion
**Best for**: General purpose improvement, clarity, and coherence

The Reflexion critic uses self-reflection to identify areas for improvement. It's the default critic and works well for most text types.

```python
result = await improve(text, critics=["reflexion"])
```

### Self-RAG (Retrieval-Augmented Generation)
**Best for**: Fact-checking, accuracy, research-based content

Self-RAG can use web search tools to verify claims and add supporting information.

```python
# With tools enabled
config = Config(enable_tools=True)
result = await improve(text, critics=["self_rag"], config=config)
```

### Constitutional AI
**Best for**: Ethical considerations, bias reduction, safety

Applies constitutional principles to ensure text aligns with specified values.

```python
config = Config(
    constitutional_principles=[
        "Be helpful and harmless",
        "Avoid bias and stereotypes",
        "Provide balanced perspectives"
    ]
)
result = await improve(text, critics=["constitutional"], config=config)
```

### Meta-Rewarding
**Best for**: Quality optimization, engagement, persuasiveness

Uses reward modeling to optimize for specific quality metrics.

```python
result = await improve(text, critics=["meta_rewarding"])
```

### Self-Consistency
**Best for**: Technical accuracy, logical consistency

Generates multiple versions and finds the most consistent elements.

```python
config = Config(self_consistency_num_samples=5)
result = await improve(text, critics=["self_consistency"], config=config)
```

### Self-Refine
**Best for**: Iterative refinement, polish, style

Focuses on incremental improvements to style and clarity.

```python
result = await improve(text, critics=["self_refine"])
```

## Choosing Critics

### By Content Type

| Content Type | Recommended Critics |
|-------------|-------------------|
| Blog Posts | reflexion, self_refine |
| Technical Docs | self_consistency, self_rag |
| Academic | self_rag, constitutional |
| Marketing | meta_rewarding, self_refine |
| Code Documentation | reflexion, self_consistency |

### By Goal

| Goal | Recommended Critics |
|------|-------------------|
| Accuracy | self_rag, self_consistency |
| Clarity | reflexion, self_refine |
| Engagement | meta_rewarding |
| Ethics | constitutional |
| Completeness | self_rag, reflexion |

## Using Multiple Critics

Critics can be combined for comprehensive improvement:

```python
# Accuracy + Clarity
result = await improve(
    text,
    critics=["self_rag", "reflexion", "self_refine"]
)

# Full review
result = await improve(
    text,
    critics=["reflexion", "constitutional", "self_consistency", "meta_rewarding"]
)
```

## Performance Considerations

- **Single Critic**: ~5-10 seconds
- **Multiple Critics**: ~10-30 seconds per iteration
- **With Tools**: Add ~5-10 seconds for web search

## Custom Critics

You can create custom critics by extending the base class:

```python
from sifaka.critics.core import BaseCritic

class CustomCritic(BaseCritic):
    @property
    def name(self) -> str:
        return "custom"
    
    async def _create_messages(self, text, result):
        # Your critique logic here
        pass
```

See the [Plugin Development Guide](../dev/plugin-development.md) for details.

## Next Steps

- Read detailed guides for each critic
- Learn about [critic selection strategies](selection-guide.md)
- Explore [advanced usage patterns](../guide/advanced-usage.md)