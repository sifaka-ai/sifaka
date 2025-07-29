# Critics Guide

Critics are the core of Sifaka's text improvement system. They analyze text and provide structured feedback for iterative refinement.

## Overview

Each critic implements a specific evaluation strategy based on academic research:

| Critic | Best For | Research Paper |
|--------|----------|----------------|
| SELF_REFINE | General improvement | [Self-Refine (2023)](https://arxiv.org/abs/2303.17651) |
| REFLEXION | Learning from mistakes | [Reflexion (2023)](https://arxiv.org/abs/2303.11366) |
| CONSTITUTIONAL | Safety & principles | [Constitutional AI (2022)](https://arxiv.org/abs/2212.08073) |
| SELF_CONSISTENCY | Balanced perspectives | [Self-Consistency (2022)](https://arxiv.org/abs/2203.11171) |
| SELF_RAG | Fact-checking | [Self-RAG (2023)](https://arxiv.org/abs/2310.11511) |
| META_REWARDING | Self-evaluation | [Meta-Rewarding (2024)](https://arxiv.org/abs/2407.19594) |
| N_CRITICS | Multiple perspectives | [N-Critics (2023)](https://arxiv.org/abs/2310.18679) |
| SELF_TAUGHT_EVALUATOR | Comparative analysis | [Self-Taught Evaluator (2024)](https://arxiv.org/abs/2408.02666) |
| AGENT4DEBATE | Debate dynamics | [Agent4Debate (2024)](https://arxiv.org/abs/2408.04472) |
| STYLE | Tone & style | Custom implementation |

## Using Critics

### Single Critic

```python
from sifaka import improve
from sifaka.core.types import CriticType

result = await improve(
    "Your text here",
    critics=[CriticType.SELF_REFINE]
)
```

### Multiple Critics

```python
result = await improve(
    "Your text here",
    critics=[CriticType.SELF_REFINE, CriticType.REFLEXION]
)
```

### Custom Configuration

```python
from sifaka.core.config import Config, CriticConfig

config = Config(
    critic=CriticConfig(
        critics=[CriticType.CONSTITUTIONAL],
        confidence_threshold=0.8
    )
)

result = await improve("Your text", config=config)
```

## Critic Details

### SELF_REFINE

General-purpose improvement focusing on clarity, coherence, and completeness.

**Use when:**
- You need balanced, general improvement
- Working with any type of content
- Starting point for text refinement

**Example:**
```python
result = await improve(
    "AI is important for business.",
    critics=[CriticType.SELF_REFINE],
    max_iterations=3
)
```

### REFLEXION

Learns from previous attempts by reflecting on what worked and what didn't.

**Use when:**
- Complex reasoning tasks
- Technical explanations
- Content requiring deep analysis

**Example:**
```python
result = await improve(
    "Explain quantum computing simply.",
    critics=[CriticType.REFLEXION],
    max_iterations=4  # Benefits from more iterations
)
```

### CONSTITUTIONAL

Evaluates text against constitutional principles for safety and ethics.

**Use when:**
- Content needs ethical review
- Safety-critical applications
- Ensuring balanced perspectives

**Principles evaluated:**
- Harmlessness
- Helpfulness
- Honesty
- Accuracy
- Nuance

**Example:**
```python
result = await improve(
    "Guide on pest control methods",
    critics=[CriticType.CONSTITUTIONAL]
)
```

### SELF_CONSISTENCY

Generates multiple perspectives and finds consensus.

**Use when:**
- Controversial topics
- Need balanced viewpoints
- Avoiding bias

**Example:**
```python
result = await improve(
    "Analysis of renewable energy policies",
    critics=[CriticType.SELF_CONSISTENCY]
)
```

### SELF_RAG

Fact-checks and retrieves information to verify claims.

**Use when:**
- Factual content
- Academic writing
- News or reports

**Requires tools:**
```python
# Tools must be configured separately
result = await improve(
    "The Great Wall of China facts",
    critics=[CriticType.SELF_RAG]
)
```

### META_REWARDING

Evaluates its own critique quality through meta-evaluation.

**Use when:**
- High-stakes content
- Need confidence in improvements
- Quality assurance

**Example:**
```python
result = await improve(
    "Medical advice disclaimer",
    critics=[CriticType.META_REWARDING]
)
```

### N_CRITICS

Uses multiple critical perspectives in parallel.

**Use when:**
- Comprehensive review needed
- Multiple stakeholder perspectives
- Final quality check

**Default perspectives:**
- Technical Expert
- General Audience
- Subject Matter Expert
- Editor
- Skeptic

**Example:**
```python
result = await improve(
    "Product launch announcement",
    critics=[CriticType.N_CRITICS]
)
```

### SELF_TAUGHT_EVALUATOR

Evaluates text by generating contrasting versions and reasoning traces.

**Use when:**
- Need transparent evaluation reasoning
- Complex comparative analysis
- Understanding trade-offs between approaches
- Educational evaluation contexts

**Key features:**
- Generates 2-3 contrasting text versions
- Provides detailed reasoning traces
- Learns from evaluation history
- No training data required

**Example:**
```python
result = await improve(
    "Technical documentation that needs clarity",
    critics=[CriticType.SELF_TAUGHT_EVALUATOR],
    max_iterations=3
)

# Access contrasting outputs and reasoning
for critique in result.critiques:
    if critique.critic == "self_taught_evaluator":
        print(f"Reasoning: {critique.metadata.get('reasoning_trace', '')}")
        print(f"Alternatives: {critique.metadata.get('contrasting_outputs', [])}")
```

### AGENT4DEBATE

Uses multi-agent competitive debate to evaluate improvement strategies.

**Use when:**
- Weighing complex trade-offs
- Need adversarial testing of ideas
- High-stakes content decisions
- Exploring competing approaches

**Key features:**
- Simulates debate between different perspectives
- Reveals trade-offs explicitly
- Competitive argumentation
- Judge-based decision making

**Example:**
```python
result = await improve(
    "Strategic business proposal",
    critics=[CriticType.AGENT4DEBATE],
    max_iterations=2
)

# The critic will debate approaches like:
# - Conservative: Minimal changes
# - Transformative: Major rewrites
# - Balanced: Selective improvements
```

### STYLE

Transforms text style and tone.

**Use when:**
- Adapting content for audiences
- Brand voice consistency
- Style transformation

**Configuration:**
```python
from sifaka.critics.style import StyleCritic

critic = StyleCritic(
    style_description="Casual and friendly",
    style_examples=[
        "Hey there! Let me explain...",
        "No worries, we've got you covered!"
    ]
)
```

## Combining Critics

Critics work well together:

```python
# Technical accuracy + readability
result = await improve(
    text,
    critics=[CriticType.REFLEXION, CriticType.STYLE]
)

# Safety + factual accuracy
result = await improve(
    text,
    critics=[CriticType.CONSTITUTIONAL, CriticType.SELF_RAG]
)

# Comprehensive review
result = await improve(
    text,
    critics=[
        CriticType.SELF_REFINE,
        CriticType.N_CRITICS,
        CriticType.META_REWARDING
    ]
)
```

## Performance Considerations

### Speed vs Quality

**Fast critics:**
- SELF_REFINE
- STYLE

**Thorough critics:**
- REFLEXION
- N_CRITICS
- META_REWARDING

**Resource intensive:**
- SELF_RAG (requires tools)
- SELF_CONSISTENCY (multiple samples)

### Optimization Tips

1. **Start simple**: Use SELF_REFINE first
2. **Add specificity**: Layer on specialized critics
3. **Limit iterations**: 2-3 for most use cases
4. **Use appropriate models**: Smaller models for simple critics

## Custom Critics

Create your own critic:

```python
from sifaka.plugins import CriticPlugin
from sifaka.core.models import CritiqueResult

class DomainExpertCritic(CriticPlugin):
    def __init__(self, domain: str):
        self.domain = domain

    async def critique(self, text: str, result):
        # Your critique logic
        feedback = f"From a {self.domain} perspective..."

        return CritiqueResult(
            critic=f"domain_expert_{self.domain}",
            feedback=feedback,
            suggestions=["Add more domain-specific details"],
            needs_improvement=True,
            confidence=0.75
        )

# Use it
result = await improve(
    text,
    critics=[DomainExpertCritic("medical")]
)
```

## Best Practices

1. **Match critic to content**: Use domain-appropriate critics
2. **Start broad, get specific**: SELF_REFINE → specialized critics
3. **Consider your audience**: Use STYLE for audience adaptation
4. **Verify facts**: Use SELF_RAG for factual content
5. **Ensure safety**: Use CONSTITUTIONAL for public content
6. **Iterate wisely**: More isn't always better (2-4 iterations usually sufficient)

## Troubleshooting

**Text not improving?**
- Try different critics
- Increase temperature (0.7-0.9)
- Add more specific critics

**Too many changes?**
- Reduce iterations
- Lower temperature
- Use more focused critics

**Inconsistent results?**
- Use SELF_CONSISTENCY
- Lower temperature for stability
- Set specific style guidelines with STYLE critic
