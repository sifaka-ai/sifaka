# Import Migration Plan

## Overview

This document outlines the detailed plan for updating imports throughout the codebase to use the new modular directory structure for the following refactored modules:

1. `sifaka/utils/config.py` → `sifaka/utils/config/` directory
2. `sifaka/utils/errors.py` → `sifaka/utils/errors/` directory
3. `sifaka/core/dependency.py` → `sifaka/core/dependency/` directory

## Critical Requirements

**ABSOLUTELY NO BACKWARD COMPATIBILITY IS ALLOWED**

- Original files MUST be deleted after refactoring
- All imports MUST be updated to use the new module structure
- No re-export layers or compatibility shims are permitted
- Zero tolerance for backward compatibility code

## Implementation Steps

### 1. Delete Original Files

Immediately delete the following files:
- `sifaka/utils/config.py`
- `sifaka/utils/errors.py`
- `sifaka/core/dependency.py`

### 2. Update Config Imports

For each file importing from `sifaka.utils.config`:

| Original Import | New Import |
|-----------------|------------|
| `from sifaka.utils.config import BaseConfig` | `from sifaka.utils.config.base import BaseConfig` |
| `from sifaka.utils.config import ModelConfig, OpenAIConfig` | `from sifaka.utils.config.models import ModelConfig, OpenAIConfig` |
| `from sifaka.utils.config import RuleConfig, RulePriority` | `from sifaka.utils.config.rules import RuleConfig, RulePriority` |
| `from sifaka.utils.config import CriticConfig, PromptCriticConfig` | `from sifaka.utils.config.critics import CriticConfig, PromptCriticConfig` |
| `from sifaka.utils.config import ChainConfig, EngineConfig` | `from sifaka.utils.config.chain import ChainConfig, EngineConfig` |
| `from sifaka.utils.config import ClassifierConfig` | `from sifaka.utils.config.classifiers import ClassifierConfig` |
| `from sifaka.utils.config import RetrieverConfig` | `from sifaka.utils.config.retrieval import RetrieverConfig` |
| `from sifaka.utils.config import standardize_model_config` | `from sifaka.utils.config.models import standardize_model_config` |
| `from sifaka.utils.config import standardize_rule_config` | `from sifaka.utils.config.rules import standardize_rule_config` |
| `from sifaka.utils.config import standardize_critic_config` | `from sifaka.utils.config.critics import standardize_critic_config` |
| `from sifaka.utils.config import standardize_chain_config` | `from sifaka.utils.config.chain import standardize_chain_config` |
| `from sifaka.utils.config import standardize_classifier_config` | `from sifaka.utils.config.classifiers import standardize_classifier_config` |
| `from sifaka.utils.config import standardize_retriever_config` | `from sifaka.utils.config.retrieval import standardize_retriever_config` |

### 3. Update Error Imports

For each file importing from `sifaka.utils.errors`:

| Original Import | New Import |
|-----------------|------------|
| `from sifaka.utils.errors import SifakaError` | `from sifaka.utils.errors.base import SifakaError` |
| `from sifaka.utils.errors import ValidationError, ConfigurationError` | `from sifaka.utils.errors.base import ValidationError, ConfigurationError` |
| `from sifaka.utils.errors import ChainError, ModelError` | `from sifaka.utils.errors.component import ChainError, ModelError` |
| `from sifaka.utils.errors import RuleError, CriticError` | `from sifaka.utils.errors.component import RuleError, CriticError` |
| `from sifaka.utils.errors import ClassifierError, RetrievalError` | `from sifaka.utils.errors.component import ClassifierError, RetrievalError` |
| `from sifaka.utils.errors import handle_error, try_operation` | `from sifaka.utils.errors.handling import handle_error, try_operation` |
| `from sifaka.utils.errors import ErrorResult, create_error_result` | `from sifaka.utils.errors.results import ErrorResult, create_error_result` |
| `from sifaka.utils.errors import safely_execute_component_operation` | `from sifaka.utils.errors.safe_execution import safely_execute_component_operation` |
| `from sifaka.utils.errors import log_error` | `from sifaka.utils.errors.logging import log_error` |

### 4. Update Dependency Imports

For each file importing from `sifaka.core.dependency`:

| Original Import | New Import |
|-----------------|------------|
| `from sifaka.core.dependency import DependencyProvider` | `from sifaka.core.dependency.provider import DependencyProvider` |
| `from sifaka.core.dependency import DependencyScope` | `from sifaka.core.dependency.scopes import DependencyScope` |
| `from sifaka.core.dependency import SessionScope, RequestScope` | `from sifaka.core.dependency.scopes import SessionScope, RequestScope` |
| `from sifaka.core.dependency import DependencyInjector` | `from sifaka.core.dependency.injector import DependencyInjector` |
| `from sifaka.core.dependency import inject_dependencies` | `from sifaka.core.dependency.injector import inject_dependencies` |
| `from sifaka.core.dependency import provide_dependency` | `from sifaka.core.dependency.utils import provide_dependency` |
| `from sifaka.core.dependency import provide_factory` | `from sifaka.core.dependency.utils import provide_factory` |
| `from sifaka.core.dependency import get_dependency` | `from sifaka.core.dependency.utils import get_dependency` |
| `from sifaka.core.dependency import get_dependency_by_type` | `from sifaka.core.dependency.utils import get_dependency_by_type` |
| `from sifaka.core.dependency import create_session_scope` | `from sifaka.core.dependency.utils import create_session_scope` |
| `from sifaka.core.dependency import create_request_scope` | `from sifaka.core.dependency.utils import create_request_scope` |
| `from sifaka.core.dependency import clear_dependencies` | `from sifaka.core.dependency.utils import clear_dependencies` |

## Implementation Strategy

### 1. Identify Files to Update

Use the following command to identify files that import from the refactored modules:

```bash
grep -r "from sifaka.utils.config import" --include="*.py" .
grep -r "from sifaka.utils.errors import" --include="*.py" .
grep -r "from sifaka.core.dependency import" --include="*.py" .
```

### 2. Update Imports in Batches

Update imports in the following order:

1. Core modules first (interfaces, base classes)
2. Utility modules next (common, logging)
3. Component implementations last (models, rules, critics)

### 3. Test After Each Batch

Run tests after updating each batch to catch and fix issues early.

## Verification Steps

After updating all imports:

1. Run all tests to ensure they pass
2. Verify that the application works correctly
3. Check for any remaining references to the old import paths
4. Ensure no backward compatibility code remains

## Timeline

- **COMPLETED**: Delete original files and update imports in core modules
- **COMPLETED**: Update imports in utility modules
- **COMPLETED**: Update imports in component implementations
- **COMPLETED**: Run tests and fix any issues
- **COMPLETED**: Final verification and documentation updates

## Success Criteria

- All original files are deleted
- All imports use the new module structure
- All tests pass
- No backward compatibility code remains
- Documentation reflects the new import structure
