# Critics Guide

Critics are the core of Sifaka's text improvement system. They analyze text and provide feedback based on different methodologies from AI research.

## Available Critics

### Reflexion
Learn from iterative attempts by reflecting on previous improvements.

```python
from sifaka import improve

result = await improve(
    "Your text here",
    critics=["reflexion"],
    max_iterations=3
)
```

### Self-RAG
Analyze factual accuracy and identify where external information would help.

```python
result = await improve(
    "The Earth orbits the Moon every 24 hours.",
    critics=["self_rag"]
)
```

### Constitutional AI
Apply ethical principles and guidelines to improve text quality.

```python
result = await improve(
    "Write persuasive marketing copy",
    critics=["constitutional"]
)
```

### N-Critics
Get feedback from multiple perspectives simultaneously.

```python
from sifaka.critics.n_critics import NCriticsCritic

critic = NCriticsCritic(
    perspectives={
        "Technical Expert": "Focus on accuracy",
        "General Reader": "Focus on clarity"
    }
)

result = await improve(
    "Technical documentation",
    critics=[critic]
)
```

### Self-Consistency
Evaluate consistency across multiple evaluations.

```python
result = await improve(
    "Complex argument",
    critics=["self_consistency"]
)
```

### Meta-Rewarding
Self-evaluate and reward high-quality improvements.

```python
result = await improve(
    "Draft content",
    critics=["meta_rewarding"]
)
```

### Self-Refine
General purpose improvement through iterative refinement.

```python
result = await improve(
    "Any text",
    critics=["self_refine"]
)
```

## Combining Critics

You can use multiple critics together:

```python
result = await improve(
    text,
    critics=["reflexion", "constitutional", "n_critics"],
    max_iterations=3
)
```

## Custom Critics

Create your own critic by extending `BaseCritic`:

```python
from sifaka.critics.core.base import BaseCritic

class MyCritic(BaseCritic):
    @property
    def name(self) -> str:
        return "my_critic"

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # Your critique logic here
        pass
```

## Choosing the Right Critic

| Use Case | Recommended Critics |
|----------|-------------------|
| Technical accuracy | self_rag, constitutional |
| Creative writing | reflexion, self_refine |
| Multiple viewpoints | n_critics |
| Consistency | self_consistency |
| General improvement | self_refine, meta_rewarding |

See the [Critic Selection Guide](/critics/selection-guide/) for detailed recommendations.
