# Sifaka Design Decisions

This document captures key architectural decisions made during Sifaka's development, including the rationale, trade-offs considered, and implementation approaches chosen.

## Decision 1: PydanticAI Integration Strategy

**Date**: May 2025
**Status**: Implemented
**Decision**: Hybrid composition-based integration with PydanticAI rather than replacing Sifaka's core architecture.

### Context

With PydanticAI emerging as a leading agent framework, we needed to decide how to integrate with it while preserving Sifaka's unique value propositions around validation, criticism, and observability.

### Options Considered

1. **Full Migration**: Replace Sifaka chains with pure PydanticAI agents
2. **Wrapper Approach**: Wrap PydanticAI agents as Sifaka models
3. **Hybrid Composition**: Orchestrate PydanticAI agents with Sifaka components
4. **Parallel Development**: Maintain separate implementations

### Decision

**Chosen**: Hybrid Composition (#3)

**Rationale**:
- Preserves Sifaka's research-grade validation and criticism capabilities
- Leverages PydanticAI's tool calling and type safety
- Enables gradual migration path for users
- Maintains academic rigor while embracing modern patterns

### Implementation

```python
# PydanticAI Chain - Hybrid approach
from sifaka.agents import create_pydantic_chain
from pydantic_ai import Agent

agent = Agent("openai:gpt-4", system_prompt="You are a helpful assistant")
chain = create_pydantic_chain(
    agent=agent,                    # PydanticAI handles generation + tools
    validators=[LengthValidator()], # Sifaka handles validation
    critics=[ReflexionCritic()],    # Sifaka handles criticism
)
```

### Outcomes

- ✅ Preserved research-grade capabilities
- ✅ Gained PydanticAI's tool calling and type safety
- ✅ Maintained backward compatibility
- ✅ Enabled modern development patterns

---

## Decision 2: Retriever Architecture - Current vs Tool-Based

**Date**: May 2025
**Status**: Under Consideration
**Decision**: Hybrid approach - maintain current architecture as primary, add optional tool-based retrieval.

### Context

Should Sifaka's retrievers be implemented as PydanticAI tools to leverage agent-driven retrieval, or maintain the current deterministic retrieval phases?

### Current Architecture

```python
# Deterministic retrieval phases
chain = create_pydantic_chain(
    agent=agent,
    model_retrievers=[retriever],   # Pre-generation context
    critic_retrievers=[retriever],  # Critic-specific context
)
```

### Tool-Based Alternative

```python
# Agent-driven retrieval
@agent.tool_plain
def search_documents(query: str) -> str:
    """Search for relevant documents."""
    return retriever.retrieve(query)
```

### Trade-offs Analysis

#### Benefits of Current Architecture
- **Deterministic**: Guaranteed retrieval at specific phases
- **Rich Metadata**: Structured `Document` objects with scores, sources
- **Performance**: No additional token usage, better caching control
- **Specialized Retrieval**: Different strategies for models vs critics

#### Benefits of Tool-Based Approach
- **Agent-Driven**: Dynamic, context-aware retrieval decisions
- **Natural Language Queries**: Agent generates sophisticated queries
- **Flexible Strategies**: Multiple retrieval calls with different queries
- **PydanticAI Integration**: Consistent with tool-calling patterns

### Decision

**Chosen**: Hybrid Approach

**Implementation Strategy**:
1. **Keep current architecture as default** for production reliability
2. **Add optional tool-based retrieval** for agent-driven applications
3. **Provide utilities** to convert existing retrievers to tools
4. **Clear documentation** on when to use each approach

```python
# Option 1: Current deterministic approach (recommended for production)
chain = create_pydantic_chain(
    agent=agent,
    model_retrievers=[retriever],  # Guaranteed pre-generation context
)

# Option 2: Tool-based approach (for dynamic retrieval)
@agent.tool_plain
def search_knowledge(query: str) -> str:
    return retriever.retrieve(query)

chain = create_pydantic_chain(
    agent=agent,  # Agent calls search_knowledge as needed
    enable_retriever_tools=True
)
```

### Rationale

- **Production Systems**: Need deterministic, high-performance retrieval
- **Research Applications**: Benefit from rich metadata and structured context
- **Agent-Driven Apps**: Can leverage dynamic retrieval when appropriate
- **Migration Path**: Users can choose the right approach for their use case

---

## Decision 3: Validator and Critic Architecture - Orchestration vs Tools

**Date**: May 2025
**Status**: Under Consideration
**Decision**: Hybrid approach - maintain orchestrated architecture as primary, add optional tool-based quality control.

### Context

Should validators and critics be implemented as PydanticAI tools to enable agent-driven quality control, or maintain the current orchestrated execution?

### Current Architecture

```python
# Orchestrated quality control
chain = create_pydantic_chain(
    agent=agent,
    validators=[LengthValidator(50, 500)],  # Deterministic validation
    critics=[ReflexionCritic(model)],       # Research-grade criticism
    max_improvement_iterations=3
)
```

### Tool-Based Alternative

```python
# Agent-driven quality control
@agent.tool_plain
def validate_length(text: str) -> str:
    """Check if text meets length requirements."""
    # Validation logic here

@agent.tool_plain
def improve_with_reflexion(text: str, prompt: str) -> str:
    """Improve text using reflexion methodology."""
    # Improvement logic here
```

### Trade-offs Analysis

#### Benefits of Current Architecture
- **Guaranteed Quality Control**: Validation after every generation
- **Rich Metadata**: Structured `ValidationResult` objects with scores, issues
- **Academic Rigor**: Faithful implementation of research papers
- **Audit Trails**: Complete validation/criticism history in `Thought` objects
- **Performance**: No additional token usage for quality control

#### Benefits of Tool-Based Approach
- **Agent-Driven**: Dynamic quality control decisions
- **Integrated Workflow**: Quality control as part of generation process
- **Flexible Strategies**: Agent chooses which validators/critics to apply
- **Natural Reasoning**: Agent can reason about quality issues

#### Critical Considerations
- **Research Fidelity**: Current critics implement complex multi-step processes (e.g., Reflexion's critique→reflect→improve)
- **Production Reliability**: Deterministic validation crucial for enterprise applications
- **Observability**: Rich metadata essential for debugging and analysis

### Decision

**Chosen**: Hybrid Approach with Current Architecture as Primary

**Rationale**:
- **Validators**: Should remain deterministic for production reliability
- **Critics**: Research-grade implementations too valuable to simplify
- **Quality Assurance**: Guaranteed validation essential for enterprise use
- **Academic Value**: Faithful paper implementations maintain research credibility

**Implementation Strategy**:
```python
# Primary approach: Orchestrated quality control
chain = create_pydantic_chain(
    agent=agent,
    validators=[LengthValidator(50, 500)],    # Guaranteed validation
    critics=[ReflexionCritic(model)],         # Research-grade improvement
    always_apply_critics=False                # Only if validation fails
)

# Optional: Agent-driven quality tools
@agent.tool_plain
def check_quality(text: str) -> str:
    """Lightweight quality check for agent use."""
    # Simplified validation logic

chain = create_pydantic_chain(
    agent=agent,
    validators=[LengthValidator(50, 500)],    # Still guaranteed
    enable_quality_tools=True,                # Agent can also self-check
)
```

### Key Insight

Unlike retrievers (which are naturally tool-like), validators and critics are **fundamental quality control mechanisms** that benefit from:
- Deterministic execution for reliability
- Rich metadata for observability
- Research-grade implementations for academic rigor

The hybrid approach preserves these strengths while enabling agent-driven quality control where appropriate.

---

## Decision 4: Thought-Centric Architecture

**Date**: May 2025
**Status**: Implemented
**Decision**: Maintain immutable `Thought` containers as the central state management system.

### Context

Should Sifaka continue using `Thought` objects as central state containers, or adopt simpler state management approaches?

### Decision

**Chosen**: Keep Thought-Centric Architecture

**Rationale**:
- **Complete Observability**: Every iteration, validation, and critique tracked
- **Immutable State**: Proper versioning and history management
- **Audit Trails**: Essential for debugging, compliance, and research
- **Research Value**: Enables analysis of AI behavior patterns
- **Unique Value**: No other framework provides this level of observability

### Implementation

```python
class Thought(BaseModel):
    prompt: str
    text: Optional[str] = None
    validation_results: Optional[Dict[str, ValidationResult]] = None
    critic_feedback: Optional[List[CriticFeedback]] = None
    history: Optional[List[ThoughtReference]] = None
    iteration: int = 0
    # ... complete state tracking
```

**Benefits Realized**:
- Enterprise-grade audit trails
- Rich debugging capabilities
- Research-grade observability
- Compliance-ready documentation

---

## Future Decisions

### Under Active Consideration

1. **MCP Storage Integration**: Fixing Redis and Milvus storage via MCP
2. **Feedback Summarization**: Enhanced summarization using local and API models
3. **Multi-Modal Support**: Extending validation and criticism to images, audio
4. **Tool Registration**: Automatic conversion of Sifaka components to PydanticAI tools

### Decision Framework

When evaluating architectural decisions, we consider:

1. **Research Fidelity**: Does it maintain academic rigor?
2. **Production Readiness**: Is it reliable for enterprise use?
3. **PydanticAI Alignment**: Does it leverage PydanticAI's strengths?
4. **Migration Path**: Can users adopt incrementally?
5. **Unique Value**: Does it preserve Sifaka's differentiators?

---

## References

- [Vision Document](VISION.md) - Strategic direction and philosophy
- [Architecture Document](ARCHITECTURE.md) - Technical implementation details
- [PydanticAI Integration Guide](guides/pydantic-ai-integration.md)
- [Chain Selection Guide](guides/chain-selection.md)
