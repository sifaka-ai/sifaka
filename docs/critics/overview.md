# Critics Overview

Critics are the heart of Sifaka's text improvement system. Each critic implements a different research-backed technique for analyzing and improving text.

> **Note**: When no critics are specified, Sifaka uses **"reflexion"** as the default critic.

## Quick Start

```python
from sifaka import improve

# Using default critic (reflexion)
result = await improve("Your text here")

# Specifying critics explicitly
result = await improve("Your text here", critics=["self_rag", "constitutional"])
```

## Available Critics

### Reflexion (Default)
**Best for**: General purpose improvement, clarity, and coherence

The Reflexion critic uses self-reflection and episodic memory to identify areas for improvement. It maintains context from previous iterations to avoid repetitive feedback.

```python
result = await improve(text, critics=["reflexion"])
```

**Parameters**: None beyond standard (uses standard model/temperature settings)

### Self-RAG (Retrieval-Augmented Generation)
**Best for**: Fact-checking, accuracy, research-based content

Self-RAG evaluates content using reflection tokens (ISREL, ISSUP, ISUSE) and can use web search tools to verify claims when enabled.

```python
# With tools enabled for fact-checking
config = Config(enable_tools=True)
result = await improve(text, critics=["self_rag"], config=config)
```

**Parameters**:
- Tool usage controlled via `config.enable_tools` or `config.critic_tool_settings`

### Constitutional AI
**Best for**: Ethical considerations, bias reduction, safety

Applies constitutional principles to ensure text aligns with specified values.

```python
# With custom principles
from sifaka.critics.constitutional import ConstitutionalCritic

critic = ConstitutionalCritic(principles=[
    "Be helpful and harmless",
    "Avoid bias and stereotypes",
    "Provide balanced perspectives"
])
result = await improve(text, critics=[critic])

# Or via config
config = Config(
    constitutional_principles=[
        "Be helpful and harmless",
        "Avoid bias and stereotypes"
    ]
)
result = await improve(text, critics=["constitutional"], config=config)
```

**Parameters**:
- `principles`: List of constitutional principles (optional)
- Default principles include safety, accuracy, clarity, respect, and professionalism

### N-Critics
**Best for**: Multiple viewpoints, comprehensive review

Evaluates text from multiple perspectives simultaneously.

```python
from sifaka.critics.n_critics import NCriticsCritic

# With custom perspectives
critic = NCriticsCritic(perspectives={
    "Technical Expert": "Focus on accuracy and precision",
    "General Reader": "Focus on clarity and accessibility",
    "Editor": "Focus on grammar and style"
})

# Or auto-generate perspectives
critic = NCriticsCritic(
    auto_generate_perspectives=True,
    perspective_count=5
)

result = await improve(text, critics=[critic])
```

**Parameters**:
- `perspectives`: List of critical perspectives (optional)
- `auto_generate_perspectives`: Generate context-appropriate perspectives (default: False)
- `perspective_count`: Number of perspectives when auto-generating (default: 4)
- Default perspectives: Clarity, Accuracy, Completeness, Style

### Self-Consistency
**Best for**: Technical accuracy, logical consistency

Generates multiple independent evaluations and finds consensus.

```python
from sifaka.critics.self_consistency import SelfConsistencyCritic

# With custom sample count
critic = SelfConsistencyCritic(num_samples=5)
result = await improve(text, critics=[critic])

# Or via config
config = Config(self_consistency_num_samples=5)
result = await improve(text, critics=["self_consistency"], config=config)
```

**Parameters**:
- `num_samples`: Number of independent evaluations (default: 3)

### Meta-Rewarding
**Best for**: Quality optimization, engagement, persuasiveness

Uses a three-stage evaluation process: initial critique → meta-evaluation → refined critique.

```python
result = await improve(text, critics=["meta_rewarding"])
```

**Parameters**: None beyond standard

### Self-Refine
**Best for**: Iterative refinement, polish, style

Focuses on incremental improvements across six quality dimensions: clarity, completeness, coherence, engagement, accuracy, and conciseness.

```python
result = await improve(text, critics=["self_refine"])
```

**Parameters**: None beyond standard

### Style
**Best for**: Matching writing styles, brand voice, audience adaptation

Transforms text to match a specific writing style, voice, or tone by analyzing reference text and applying its characteristics.

```python
from sifaka.critics.style import StyleCritic

# With reference text
critic = StyleCritic(
    reference_text="Sample text in target style...",
    style_description="Conversational and friendly",
    style_examples=["Hey there!", "You'll love this..."]
)
result = await improve(text, critics=[critic])

# Or use the string name with config
config = Config(
    style_reference_text="Sample text...",
    style_description="Professional executive tone"
)
result = await improve(text, critics=["style"], config=config)
```

**Parameters**:
- `reference_text`: Text exemplifying the target style (optional)
- `style_description`: Description of the desired style (optional)
- `style_examples`: List of example phrases in target style (optional)
- Outputs `alignment_score` (0.0-1.0) indicating style match

## Common Configuration Options

All critics support these base parameters:

```python
from sifaka import Config

config = Config(
    # Model settings
    model="gpt-4o-mini",              # Default model for generation
    temperature=0.7,                   # Generation temperature
    critic_model="gpt-3.5-turbo",     # Model specifically for critics
    critic_temperature=0.7,            # Temperature for critic operations

    # Critic behavior
    critic_base_confidence=0.7,        # Base confidence level
    critic_context_window=3,           # Previous critiques to consider
    critic_timeout_seconds=30.0,       # Timeout for operations

    # Tool settings (for Self-RAG)
    enable_tools=False,                # Enable tool usage
    tool_timeout=5.0,                  # Tool call timeout

    # Critic-specific settings
    constitutional_principles=[...],    # For Constitutional AI
    self_consistency_num_samples=3,    # For Self-Consistency
)
```

## Using Multiple Critics

Critics can be combined for comprehensive improvement:

```python
# Accuracy + Clarity
result = await improve(
    text,
    critics=["self_rag", "reflexion", "self_refine"],
    max_iterations=3
)

# Full review with custom critics
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.n_critics import NCriticsCritic

constitutional = ConstitutionalCritic(principles=["Be concise", "Be accurate"])
n_critics = NCriticsCritic(auto_generate_perspectives=True)

result = await improve(
    text,
    critics=["reflexion", constitutional, n_critics, "meta_rewarding"]
)
```

## Choosing Critics

### By Content Type

| Content Type | Recommended Critics | Why |
|-------------|-------------------|-----|
| Blog Posts | reflexion, self_refine | General improvement and polish |
| Technical Docs | self_consistency, self_rag | Accuracy and fact-checking |
| Academic Papers | self_rag, constitutional | Citations and ethical considerations |
| Marketing Copy | meta_rewarding, style | Engagement and brand voice |
| Code Documentation | reflexion, self_consistency | Clarity and technical accuracy |
| Creative Writing | style, n_critics | Voice and multiple perspectives |
| Brand Content | style, constitutional | Consistent voice and values |

### By Goal

| Goal | Recommended Critics | Configuration Tips |
|------|-------------------|-------------------|
| Accuracy | self_rag, self_consistency | Enable tools, increase samples |
| Clarity | reflexion, self_refine | Default settings work well |
| Engagement | meta_rewarding, n_critics | Use audience-specific perspectives |
| Ethics | constitutional | Define clear principles |
| Completeness | self_rag, reflexion | Enable tools for research |
| Consistency | self_consistency | Increase num_samples for complex text |
| Style Match | style | Provide clear reference text |

## Performance Considerations

- **Single Critic**: ~5-10 seconds per iteration
- **Multiple Critics**: ~10-30 seconds per iteration
- **With Tools**: Add ~5-10 seconds for web search
- **Self-Consistency**: Multiply by num_samples

Tips for optimization:
- Use `critic_model="gpt-3.5-turbo"` for faster, cheaper critiques
- Limit `max_iterations` to control total time
- Use specific critics rather than all critics

## Custom Critics

Create custom critics by extending the base class:

```python
from sifaka.critics.core.base import BaseCritic
from sifaka.core.models import CritiqueResult, SifakaResult

class CustomCritic(BaseCritic):
    @property
    def name(self) -> str:
        return "custom"

    async def _create_messages(self, text: str, result: SifakaResult) -> list:
        return [
            {"role": "system", "content": "You are a helpful critic."},
            {"role": "user", "content": f"Critique this text: {text}"}
        ]
```

## Next Steps

- Try the [quickstart guide](../quickstart.md) to get started
- Learn about [validators](../guide/validators.md) for additional quality checks
- Explore [advanced usage patterns](../guide/advanced-usage.md)
- See [examples](https://github.com/sifaka-ai/sifaka/tree/main/examples) for real-world usage
