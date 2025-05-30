# Chain Selection Guide: PydanticAI vs Traditional

> **🚨 Version 0.2.0 Update**: PydanticAI Chain is now the primary and recommended approach. Traditional Chain is deprecated but still available.

Sifaka offers two distinct chain implementations. **PydanticAI Chain is now recommended for all new projects** with Traditional Chain in maintenance mode for existing codebases.

## Overview

| Aspect | PydanticAI Chain | Traditional Chain |
|--------|------------------|-------------------|
| **Primary Use Case** | Tool-heavy applications | Pipeline-based workflows |
| **Architecture** | Modern, async-first | Mature, orchestrated |
| **Tool Integration** | Native, type-safe | Manual implementation |
| **Built-in Features** | Fewer (but extensible via tools) | Many (pre-built components) |
| **Extensibility** | High (via tool system) | High (via configuration) |
| **Development Speed** | Fast (for tool-based apps) | Moderate (more setup) |
| **Learning Curve** | Gentle | Steeper |

## 🚀 PydanticAI Chain

### When to Choose PydanticAI Chain

✅ **Perfect for:**
- **Tool-heavy applications** (web search, APIs, calculations)
- **Modern development** with type safety and validation
- **Rapid prototyping** and quick setup
- **Structured outputs** and data validation
- **Extensible workflows** via tool composition

❌ **Consider Traditional Chain for:**
- **Pre-built advanced features** (if you need them immediately)
- **Complex pipeline orchestration** (if you prefer configuration over tools)
- **Existing Traditional Chain workflows** (migration may not be worth it)

### Key Features

#### 🛠️ **Native Tool Integration**
```python
from sifaka.agents import create_pydantic_chain
from pydantic_ai import Agent

agent = Agent("openai:gpt-4")

@agent.tool_plain
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@agent.tool_plain
def calculate(expression: str) -> float:
    """Safely evaluate mathematical expressions."""
    return eval(expression)  # Use safe_eval in production

chain = create_pydantic_chain(
    agent=agent,
    validators=[LengthValidator(min_length=50, max_length=500)],
    critics=[ReflexionCritic(model=create_model("openai:gpt-4"))],
    always_apply_critics=True
)
```

#### 🎯 **Type Safety**
- Automatic validation of tool inputs/outputs
- Structured data generation with Pydantic models
- Compile-time error detection

#### ⚡ **Performance**
- Async-first architecture
- Efficient tool execution
- Minimal overhead

### Architecture

```
User Prompt → PydanticAI Agent → Tool Calls → Sifaka Validation → Critics → Improved Output
```

## 🏗️ Traditional Chain

### When to Choose Traditional Chain

✅ **Perfect for:**
- **Pipeline-based workflows** with complex orchestration
- **Pre-built advanced features** (retrieval, error recovery, etc.)
- **Configuration-driven** development patterns
- **Existing Traditional Chain** codebases
- **Complex multi-stage** document processing

❌ **Consider PydanticAI Chain for:**
- **Tool-heavy applications** (PydanticAI excels here)
- **Modern development patterns** (async-first, type-safe)
- **Rapid prototyping** (simpler setup)

### Key Features

#### 🔍 **Advanced Retrieval**
```python
from sifaka import Chain
from sifaka.retrievers import InMemoryRetriever
from sifaka.storage import RedisStorage, MilvusStorage

# Multi-stage retrieval
pre_retriever = InMemoryRetriever()  # Fast lookup
post_retriever = MilvusStorage()     # Semantic search
critic_retriever = RedisStorage()    # Cached context

chain = Chain(
    model=create_model("openai:gpt-4"),
    prompt="Analyze complex document",
    model_retrievers=[pre_retriever],
    critic_retrievers=[critic_retriever],
    max_improvement_iterations=5,
    always_apply_critics=True
)
```

#### 🛡️ **Production Features**
- Checkpoint and recovery systems
- Advanced error handling
- Comprehensive monitoring
- Multiple storage backends

#### 🎛️ **Maximum Flexibility**
- Custom orchestration logic
- Pluggable components
- Fine-grained configuration

### Architecture

```
User Prompt → Pre-Retrieval → Model → Post-Retrieval → Validation → Critics → Improvement Loop → Final Output
```

## Decision Matrix

### Choose PydanticAI Chain if you answer "Yes" to most:

- [ ] Do you need tool calling (APIs, search, calculations)?
- [ ] Do you prefer simple, clean APIs?
- [ ] Are you building a modern application?
- [ ] Do you value type safety and validation?
- [ ] Is rapid development important?
- [ ] Are your workflows relatively straightforward?

### Choose Traditional Chain if you answer "Yes" to most:

- [ ] Do you need advanced retrieval systems?
- [ ] Is this a production system with high reliability requirements?
- [ ] Do you need maximum flexibility and customization?
- [ ] Are you building complex, multi-stage workflows?
- [ ] Do you need advanced error recovery?
- [ ] Are you working with large-scale document processing?

## Migration Path

### From Traditional to PydanticAI

```python
# Traditional Chain
chain = Chain(model=model, prompt=prompt)
chain = chain.validate_with(validator).improve_with(critic)

# PydanticAI Chain equivalent
agent = Agent(model_name, system_prompt="...")
chain = create_pydantic_chain(
    agent=agent,
    validators=[validator],
    critics=[critic]
)
```

### From PydanticAI to Traditional

```python
# PydanticAI Chain
chain = create_pydantic_chain(agent=agent, validators=validators)

# Traditional Chain equivalent
chain = Chain(
    model=create_model(model_name),
    prompt=prompt,
    max_improvement_iterations=2
)
for validator in validators:
    chain = chain.validate_with(validator)
for critic in critics:
    chain = chain.improve_with(critic)
```

## Performance Comparison

| Metric | PydanticAI Chain | Traditional Chain |
|--------|------------------|-------------------|
| **Setup Time** | ~50ms | ~200ms |
| **Tool Call Overhead** | ~10ms (native) | N/A (manual implementation) |
| **Memory Usage** | Lower (simpler) | Higher (more features) |
| **Async Performance** | Excellent | Excellent |
| **Scalability** | High | High |

## Best Practices

### PydanticAI Chain
- Use type hints for all tools
- Keep tool functions simple and focused
- Leverage structured outputs
- Use async patterns consistently

### Traditional Chain
- Design retrieval strategy carefully
- Implement proper error handling
- Use appropriate storage backends
- Monitor performance metrics

## Conclusion

**As of Sifaka 0.2.0**:

- **✅ Use PydanticAI Chain** for all new projects - it's now the primary, recommended approach with full feature parity
- **⚠️ Traditional Chain is deprecated** - still available for existing projects but in maintenance mode
- **🔄 Consider migrating** existing Traditional Chain projects to PydanticAI Chain for better long-term support

**PydanticAI Chain now has full feature parity** including retriever support, making it suitable for all use cases that Traditional Chain previously handled.

Both chains can coexist during migration, allowing you to gradually transition existing workflows.
