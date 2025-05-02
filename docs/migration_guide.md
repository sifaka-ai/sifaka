# Sifaka Migration Guide

This guide explains the recent structural changes to the Sifaka codebase and how to migrate your code to the new structure.

## Overview of Changes

The Sifaka codebase has undergone a significant reorganization to improve maintainability and reduce duplication. Key changes include:

1. **Creation of adapters directory**
   - New `/sifaka/adapters/` directory for integration code
   - Moved integration files to appropriate adapter modules
   - Added rule adapters in `/sifaka/adapters/rules/`

2. **Model provider consolidation**
   - Consolidated Anthropic implementations into `models/anthropic.py`
   - Standardized model provider interfaces

3. **Removal of incomplete modules**
   - Removed the incomplete `/sifaka/domain` directory
   - Kept domain-specific rules in `/sifaka/rules/domain/`

4. **Import path changes**
   - Updated import paths to reflect the new structure
   - Fixed self-references in various modules

## Detailed Changes

### Moved Files

| Old Path | New Path | Notes |
|----------|----------|-------|
| `sifaka/integrations/langchain.py` | `sifaka/adapters/langchain.py` | Functionality unchanged |
| `sifaka/integrations/langgraph.py` | `sifaka/adapters/langgraph.py` | Functionality unchanged |
| `sifaka/integrations/anthropic.py` | `sifaka/models/anthropic.py` | Consolidated with existing Anthropic code |

### Deleted Files

The following files were deleted as part of the restructuring:

- `/sifaka/domain/*` - Incomplete domain implementation
- `/sifaka/integrations/anthropic.py` - Consolidated into models/anthropic.py
- `/sifaka/integrations/langchain.py` - Moved to adapters/langchain.py
- `/sifaka/integrations/langgraph.py` - Moved to adapters/langgraph.py

### New Files

The following files were created:

- `/sifaka/adapters/__init__.py` - Adapter module initialization
- `/sifaka/adapters/rules/__init__.py` - Rule adapters initialization
- `/sifaka/adapters/rules/base.py` - Base functionality for rule adapters
- `/sifaka/adapters/rules/classifier.py` - Adapter for classifier-based rules
- `/sifaka/adapters/rules/guardrails_adapter.py` - Guardrails integration

## How to Migrate Your Code

### Import Path Updates

Update your import statements as follows:

```python
# Old imports
from sifaka.integrations.langchain import LangChainAdapter
from sifaka.integrations.langgraph import LangGraphAdapter
from sifaka.integrations.anthropic import AnthropicProvider

# New imports
from sifaka.adapters.langchain import LangChainAdapter
from sifaka.adapters.langgraph import LangGraphAdapter
from sifaka.models.anthropic import AnthropicProvider
```

### Domain References

If you were using the incomplete domain module:

```python
# Old code (will no longer work)
from sifaka.domain import Domain, DomainConfig

# New code options:

# Option 1: Use domain-specific rules
from sifaka.rules.domain import DomainRules

# Option 2: Use classifier adapters
from sifaka.adapters.rules import ClassifierAdapter

# Option 3 (Recommended): Use the new Chain architecture
from sifaka.chain import create_validation_chain, create_improvement_chain
```

### Using Chain Architecture

The recommended approach to replace Domain functionality is to use the new Chain architecture:

```python
# Old domain-based approach
from sifaka.domain import Domain, DomainConfig
from sifaka.rules.formatting.length import create_length_rule

domain_config = DomainConfig(
    name="example_domain",
    rules=[create_length_rule(min_chars=10, max_chars=100)]
)
domain = Domain(config=domain_config)
result = domain.validate("test text")

# New chain-based approach
from sifaka.chain import create_validation_chain
from sifaka.rules.formatting.length import create_length_rule

chain = create_validation_chain(
    name="validation_chain",
    rules=[create_length_rule(min_chars=10, max_chars=100)]
)
result = chain.process("test text")
```

For more details on the Chain architecture, see the [Chain Architecture Guide](chain_architecture.md).

### ClassifierAdapter Usage

If you were using classifiers directly with rules:

```python
# Old approach (still works but not recommended)
from sifaka.classifiers import ToxicityClassifier
from sifaka.rules import create_rule_from_classifier

# New recommended approach
from sifaka.classifiers import ToxicityClassifier
from sifaka.adapters.rules import ClassifierAdapter

classifier = ToxicityClassifier()
rule = ClassifierAdapter(classifier=classifier, name="toxicity_rule")
```

### Anthropic Model Usage

If you were using Anthropic models:

```python
# Old approach (multiple implementations)
from sifaka.integrations.anthropic import AnthropicProvider
# or
from sifaka.models.anthropic import AnthropicModel

# New consolidated approach
from sifaka.models.anthropic import AnthropicProvider
```

## Additional Notes

### Version Compatibility

These changes were made in version 0.1.0 and do not maintain backward compatibility with pre-release versions. Future versions will maintain stricter compatibility guarantees.

### Testing Your Migration

After updating your import paths, run your tests to ensure everything is working correctly. If you encounter any issues, check that:

1. All import paths have been updated
2. Any direct instantiation of moved classes uses the correct paths
3. Code that relied on the domain module has been updated to use appropriate alternatives

## Example Migration

### Before

```python
from sifaka.integrations.langchain import LangChainAdapter
from sifaka.integrations.anthropic import AnthropicProvider
from sifaka.domain import Domain, DomainConfig

# Create model
model = AnthropicProvider(model="claude-2")

# Create chain
chain = LangChainAdapter.create_chain(model)

# Create domain
domain_config = DomainConfig(name="example_domain")
domain = Domain(config=domain_config)

# Use domain with chain
result = chain.run("Generate content about AI")
validated = domain.validate(result)
```

### After

```python
from sifaka.adapters.langchain import LangChainAdapter
from sifaka.models.anthropic import AnthropicProvider
from sifaka.chain import create_validation_chain
from sifaka.rules.formatting.length import create_length_rule

# Create model
model = AnthropicProvider(model="claude-3-sonnet")

# Create LangChain adapter
langchain_adapter = LangChainAdapter.create_chain(model)

# Create validation chain
validation_chain = create_validation_chain(
    name="content_validation",
    rules=[create_length_rule(min_chars=10, max_chars=1000)]
)

# Use the chains
result = langchain_adapter.run("Generate content about AI")
validation_result = validation_chain.process(result)
print(f"Validation passed: {validation_result.all_passed}")
```

## Getting Help

If you encounter any issues migrating to the new structure:

1. Check the updated documentation
2. Review the examples in the `/examples` directory
3. File an issue on the repository with details about your problem