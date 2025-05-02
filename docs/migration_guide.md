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

# New code
from sifaka.rules.domain import DomainRules
# Or, depending on your use case:
from sifaka.adapters.rules import ClassifierAdapter
```

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
from sifaka.rules.domain import create_domain_rule

# Create model
model = AnthropicProvider(model="claude-3-sonnet")

# Create chain
chain = LangChainAdapter.create_chain(model)

# Create domain rules
domain_rule = create_domain_rule(name="example_domain")

# Use domain rule with chain
result = chain.run("Generate content about AI")
validated = domain_rule.validate(result)
```

## Getting Help

If you encounter any issues migrating to the new structure:

1. Check the updated documentation
2. Review the examples in the `/examples` directory
3. File an issue on the repository with details about your problem