# Critics Overview

Critics are the heart of Sifaka's text improvement system. Each critic implements a specific evaluation strategy based on cutting-edge AI research.

## What Are Critics?

Critics analyze text and provide structured feedback for improvement. They:
- Evaluate text quality
- Identify specific issues
- Suggest improvements
- Guide iterative refinement

## Available Critics

### Core Critics

| Critic | Purpose | Best For |
|--------|---------|----------|
| **SELF_REFINE** | General improvement | All-purpose text enhancement |
| **REFLEXION** | Learning from attempts | Complex reasoning tasks |
| **CONSTITUTIONAL** | Ethical evaluation | Safety-critical content |
| **SELF_CONSISTENCY** | Consensus building | Balanced perspectives |

### Advanced Critics

| Critic | Purpose | Best For |
|--------|---------|----------|
| **SELF_RAG** | Fact verification | Academic/factual content |
| **META_REWARDING** | Quality assurance | High-stakes content |
| **N_CRITICS** | Multiple perspectives | Comprehensive review |
| **STYLE** | Tone adaptation | Audience-specific content |

## How Critics Work

1. **Analysis**: Critic examines the text
2. **Evaluation**: Identifies strengths and weaknesses
3. **Feedback**: Provides specific improvement suggestions
4. **Iteration**: Process repeats until satisfaction

```python
# Basic critic flow
from sifaka import improve
from sifaka.core.types import CriticType

result = await improve(
    "Your text",
    critics=[CriticType.SELF_REFINE],
    max_iterations=3
)

# Access critique history
for critique in result.critiques:
    print(f"Critic: {critique.critic}")
    print(f"Feedback: {critique.feedback}")
    print(f"Confidence: {critique.confidence}")
```

## Choosing Critics

### By Content Type

**Technical/Academic:**
- REFLEXION - Deep analysis
- SELF_RAG - Fact checking
- SELF_CONSISTENCY - Balanced claims

**Creative/Marketing:**
- STYLE - Tone adaptation
- SELF_REFINE - General polish
- N_CRITICS - Multiple perspectives

**Business/Professional:**
- SELF_REFINE - Clarity and professionalism
- CONSTITUTIONAL - Ethical considerations
- META_REWARDING - Quality assurance

### By Goal

**Improve Clarity:**
```python
critics=[CriticType.SELF_REFINE]
```

**Ensure Accuracy:**
```python
critics=[CriticType.SELF_RAG, CriticType.REFLEXION]
```

**Maintain Safety:**
```python
critics=[CriticType.CONSTITUTIONAL]
```

**Adapt Style:**
```python
critics=[CriticType.STYLE]
```

## Combining Critics

Critics can work together for comprehensive improvement:

```python
# Accuracy + Safety
result = await improve(
    text,
    critics=[
        CriticType.SELF_RAG,        # Verify facts
        CriticType.CONSTITUTIONAL   # Ensure safety
    ]
)

# Style + Quality
result = await improve(
    text,
    critics=[
        CriticType.STYLE,          # Match tone
        CriticType.SELF_REFINE,    # Polish
        CriticType.META_REWARDING  # Verify quality
    ]
)
```

## Research Foundation

Each critic is based on peer-reviewed research:

- **SELF_REFINE**: [Madaan et al., 2023](https://arxiv.org/abs/2303.17651)
- **REFLEXION**: [Shinn et al., 2023](https://arxiv.org/abs/2303.11366)
- **CONSTITUTIONAL**: [Bai et al., 2022](https://arxiv.org/abs/2212.08073)
- **SELF_CONSISTENCY**: [Wang et al., 2022](https://arxiv.org/abs/2203.11171)
- **SELF_RAG**: [Asai et al., 2023](https://arxiv.org/abs/2310.11511)
- **META_REWARDING**: [Wu et al., 2024](https://arxiv.org/abs/2407.19594)

## Understanding Critique Results

Each critique provides:

```python
class CritiqueResult:
    critic: str              # Which critic
    feedback: str            # Detailed feedback
    suggestions: list[str]   # Specific improvements
    needs_improvement: bool  # Continue or stop
    confidence: float        # 0.0-1.0 confidence
```

Example output:
```python
CritiqueResult(
    critic="self_refine",
    feedback="The text lacks specific examples and data.",
    suggestions=[
        "Add concrete examples",
        "Include relevant statistics",
        "Clarify the main argument"
    ],
    needs_improvement=True,
    confidence=0.75
)
```

## Performance Characteristics

### Speed

**Fast** (< 2s per iteration):
- SELF_REFINE
- STYLE

**Medium** (2-5s per iteration):
- REFLEXION
- CONSTITUTIONAL
- META_REWARDING

**Slower** (5-10s per iteration):
- SELF_CONSISTENCY (multiple samples)
- N_CRITICS (multiple perspectives)
- SELF_RAG (retrieval required)

### Quality Impact

**High Impact:**
- REFLEXION (for reasoning)
- N_CRITICS (for comprehensiveness)
- META_REWARDING (for quality)

**Targeted Impact:**
- STYLE (for tone)
- CONSTITUTIONAL (for safety)
- SELF_RAG (for accuracy)

## Custom Critics

Create domain-specific critics:

```python
from sifaka.plugins import CriticPlugin
from sifaka.core.models import CritiqueResult

class SEOCritic(CriticPlugin):
    """Critic for SEO optimization."""

    async def critique(self, text: str, result):
        # Analyze for SEO factors
        issues = []
        if len(text.split()) < 300:
            issues.append("Add more content (300+ words)")
        if not any(keyword in text.lower()
                  for keyword in ["keyword", "phrases"]):
            issues.append("Include target keywords")

        return CritiqueResult(
            critic="seo_critic",
            feedback="SEO analysis complete",
            suggestions=issues,
            needs_improvement=len(issues) > 0,
            confidence=0.8
        )
```

## Best Practices

1. **Start Simple**: Begin with SELF_REFINE
2. **Add Specificity**: Layer specialized critics
3. **Consider Context**: Match critics to content type
4. **Balance Speed/Quality**: More critics = better but slower
5. **Monitor Confidence**: High confidence = diminishing returns
6. **Test Combinations**: Find what works for your use case

## Next Steps

- Read the [detailed Critics Guide](../guide/critics.md)
- Try the [examples](https://github.com/sifaka-ai/sifaka/tree/main/examples/critics)
- Learn about [custom critics](../plugin_development.md)
