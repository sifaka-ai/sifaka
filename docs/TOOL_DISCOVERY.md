# Tool Discovery for Critics

This document explains how critics can automatically discover and use tools in Sifaka, making them aware of all existing and future tools without manual configuration.

## Overview

Critics in Sifaka can now automatically discover and use tools through several mechanisms:

1. **Manual Tool Configuration** (existing approach)
2. **Automatic Tool Discovery** (new approach)
3. **Tool Registry Management**
4. **Category-based Tool Filtering**

## Quick Start

### Automatic Tool Discovery

The easiest way to make critics aware of all tools:

```python
from sifaka.critics.self_rag import SelfRAGCritic

# Automatically discover and use ALL available tools
critic = SelfRAGCritic(
    model_name="anthropic:claude-3-5-haiku-latest",
    auto_discover_tools=True  # This is the magic flag!
)

# The critic now has access to all registered tools
print(f"Critic has {len(critic.retrieval_tools)} tools available")
```

### Category-based Tool Discovery

For more control, filter tools by category:

```python
# Only use retrieval tools
critic = SelfRAGCritic(
    auto_discover_tools=True,
    tool_categories=["retrieval"]  # Only retrieval tools
)

# Use multiple categories
critic = SelfRAGCritic(
    auto_discover_tools=True,
    tool_categories=["retrieval", "web_search"]
)
```

### Convenience Functions

Use helper functions for common patterns:

```python
from sifaka.tools import discover_all_tools, discover_retrieval_tools

# Get all available tools
all_tools = discover_all_tools()

# Get only retrieval tools
retrieval_tools = discover_retrieval_tools()

# Use with any critic
critic = SelfRAGCritic(retrieval_tools=retrieval_tools)
```

## Detailed Usage

### 1. Manual Tool Configuration (Current Approach)

```python
from sifaka.tools import create_web_search_tools
from sifaka.critics.self_rag import SelfRAGCritic

# Manually create tools
web_tools = create_web_search_tools(providers=["duckduckgo"])

# Manually pass to critic
critic = SelfRAGCritic(
    retrieval_tools=web_tools  # Explicit tool list
)
```

**Pros:**
- Explicit control over which tools are used
- No surprises about tool availability
- Fine-grained configuration

**Cons:**
- Requires manual setup for each critic
- Must update code when new tools are added
- More boilerplate code

### 2. Automatic Tool Discovery (New Approach)

```python
# Automatically discover all tools
critic = SelfRAGCritic(auto_discover_tools=True)

# Discover tools by category
critic = SelfRAGCritic(
    auto_discover_tools=True,
    tool_categories=["retrieval", "web_search"]
)

# Combine manual and automatic
manual_tools = create_custom_tools()
critic = SelfRAGCritic(
    retrieval_tools=manual_tools,  # Manual tools
    auto_discover_tools=True       # Plus auto-discovered tools
)
```

**Pros:**
- Automatic access to all available tools
- Future-proof (new tools automatically available)
- Less boilerplate code
- Consistent tool availability across critics

**Cons:**
- Less explicit control
- May include tools you don't want
- Potential for unexpected tool usage

### 3. Tool Registry Management

```python
from sifaka.tools import tool_registry

# List available tools
available_tools = tool_registry.list_tools()
print(f"Available tools: {available_tools}")

# List tool categories
categories = tool_registry.list_categories()
print(f"Categories: {categories}")

# Get all tools from registry
all_tools = tool_registry.get_all_tools()

# Get tools by category
retrieval_tools = tool_registry.get_tools_by_category("retrieval")
```

## Tool Categories

Tools are organized into categories for easy filtering:

- **`retrieval`**: Storage and data retrieval tools
- **`web_search`**: Web search providers (DuckDuckGo, Tavily)
- **`common`**: PydanticAI common tools
- **`custom`**: User-defined tools

## All Critics Support Auto-Discovery

All Sifaka critics support the new auto-discovery parameters:

```python
from sifaka.critics import (
    SelfRAGCritic,
    ReflexionCritic,
    ConstitutionalCritic,
    SelfRefineCritic
)

# All critics support these parameters
critics = [
    SelfRAGCritic(auto_discover_tools=True),
    ReflexionCritic(auto_discover_tools=True, tool_categories=["retrieval"]),
    ConstitutionalCritic(auto_discover_tools=True),
    SelfRefineCritic(auto_discover_tools=True, tool_categories=["web_search"])
]
```

## Best Practices

### 1. Use Auto-Discovery for Development

During development, use auto-discovery to get access to all tools:

```python
# Development: Get all tools
critic = SelfRAGCritic(auto_discover_tools=True)
```

### 2. Use Category Filtering for Production

In production, be more specific about tool categories:

```python
# Production: Only retrieval tools
critic = SelfRAGCritic(
    auto_discover_tools=True,
    tool_categories=["retrieval"]
)
```

### 3. Combine Manual and Automatic

For maximum flexibility, combine both approaches:

```python
# Custom tools + auto-discovered tools
custom_tools = create_my_custom_tools()
critic = SelfRAGCritic(
    retrieval_tools=custom_tools,    # Manual tools
    auto_discover_tools=True,        # Plus auto-discovered
    tool_categories=["retrieval"]    # Filter auto-discovered
)
```

### 4. Check Tool Availability

Always check what tools are actually available:

```python
critic = SelfRAGCritic(auto_discover_tools=True)
print(f"Critic has {len(critic.retrieval_tools)} tools:")
for i, tool in enumerate(critic.retrieval_tools):
    print(f"  {i+1}. {tool.name}")
```

## Migration Guide

### From Manual to Auto-Discovery

**Before:**
```python
# Old way: Manual tool configuration
web_tools = create_web_search_tools(["duckduckgo"])
storage_tools = create_storage_retrieval_tools(storage)
all_tools = web_tools + storage_tools

critic = SelfRAGCritic(retrieval_tools=all_tools)
```

**After:**
```python
# New way: Auto-discovery
critic = SelfRAGCritic(auto_discover_tools=True)
```

### Gradual Migration

You can migrate gradually by combining approaches:

```python
# Step 1: Keep existing manual tools, add auto-discovery
manual_tools = create_web_search_tools(["duckduckgo"])
critic = SelfRAGCritic(
    retrieval_tools=manual_tools,
    auto_discover_tools=True,
    tool_categories=["retrieval"]  # Only add retrieval tools automatically
)

# Step 2: Eventually remove manual tools
critic = SelfRAGCritic(auto_discover_tools=True)
```

## Troubleshooting

### No Tools Found

If auto-discovery finds no tools:

```python
from sifaka.tools import tool_registry

# Check what's registered
print(f"Registered tools: {tool_registry.list_tools()}")
print(f"Tool instances: {len(tool_registry._instances)}")

# Make sure tools are properly registered
# Tools are auto-registered when their modules are imported
```

### Wrong Tools Discovered

If auto-discovery includes unwanted tools:

```python
# Use category filtering
critic = SelfRAGCritic(
    auto_discover_tools=True,
    tool_categories=["retrieval"]  # Only specific categories
)

# Or use manual configuration for precise control
specific_tools = create_web_search_tools(["duckduckgo"])
critic = SelfRAGCritic(retrieval_tools=specific_tools)
```

### Tool Registration Issues

If tools aren't being discovered:

```python
# Make sure tool modules are imported
import sifaka.tools.retrieval  # This registers retrieval tools
import sifaka.tools.common     # This registers common tools

# Check registration
from sifaka.tools import tool_registry
print(f"Available tools: {tool_registry.list_tools()}")
```

## Future Enhancements

Planned improvements to tool discovery:

1. **Plugin System**: Automatic discovery of external tool plugins
2. **Configuration Files**: Tool discovery configuration via files
3. **Environment-based Discovery**: Different tool sets for different environments
4. **Tool Dependencies**: Automatic resolution of tool dependencies
5. **Tool Metadata**: Rich metadata for better tool selection
