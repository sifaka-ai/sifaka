# Sifaka Examples

This directory contains examples organized by complexity and use case:

## 📚 Learning Path

### 1. **`basic/`** - Start Here
- `simple_style.py` - Basic text improvement
- `validators_example.py` - Using built-in validators

### 2. **`critics/`** - Core Critics
- `constitutional_example.py` - Constitutional AI critic
- `reflexion_example.py` - Reflexion-based improvement
- `self_refine_example.py` - Self-refinement critic
- `self_consistency_example.py` - Self-consistency critic

### 3. **`advanced/`** - Advanced Features
- `meta_rewarding_example.py` - Meta-rewarding critic
- `n_critics_example.py` - Multiple critic orchestration
- `safety_meta_rewarding_critic.py` - Safety-focused critic
- `style_critic_example.py` - Style-specific improvements
- `self_rag_builtin_tools.py` - Self-RAG with tools

### 4. **`integrations/`** - External Systems
- `redis_thoughts_example.py` - Redis storage backend

### 5. **`plugins/`** - Plugin Development
- `example_critic_plugin.py` - Custom critic plugin
- `example_validator_plugin.py` - Custom validator plugin

## 🚀 Running Examples

```bash
# Set up environment
uv pip install -e ".[dev]"

# Run basic examples
uv run examples/basic/simple_style.py

# Run specific critic examples
uv run examples/critics/constitutional_example.py

# Run plugin examples
uv run examples/plugins/example_critic_plugin.py
```

## 📖 Documentation

For detailed guides, see the [documentation](../docs/).
