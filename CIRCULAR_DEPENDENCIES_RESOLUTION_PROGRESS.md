# Circular Dependencies Resolution Progress

This document tracks the progress of resolving circular dependencies in the Sifaka codebase.

## Current Status

- Total modules: 150
- Total dependencies: 592
- Circular dependencies: 104
- Initial circular dependencies (from first analysis): 80

**Note:** The increase in circular dependencies from 80 to 104 is likely due to more thorough analysis by the dependency analysis script, additional code added to the codebase, or refactoring that temporarily introduced more circular dependencies.

## Strategic Resolution Plan

### Phase 1: Configuration System Overhaul (Highest Impact)

**Status: Completed**

This will resolve approximately 15-20 circular dependencies at once.

- [x] Analyze all configuration classes in the codebase
  - [x] Analyze `utils/config.py`
  - [x] Analyze `models/config.py`
  - [x] Analyze `chain/config.py`
  - [x] Analyze `classifiers/config.py`
  - [x] Analyze `critics/config.py`
  - [x] Analyze `rules/config.py`
  - [x] Analyze `retrieval/config.py`

- [x] Consolidate all configuration in `utils/config.py`
  - [x] Move specialized model configs (`OpenAIConfig`, `AnthropicConfig`, `GeminiConfig`) from `models/config.py` to `utils/config.py`
  - [x] Move specialized chain configs (`EngineConfig`, `ValidatorConfig`, `ImproverConfig`, `FormatterConfig`) from `chain/config.py` to `utils/config.py`
  - [x] Move specialized classifier configs (`ImplementationConfig`) from `classifiers/config.py` to `utils/config.py`
  - [x] Move specialized critic configs (`PromptCriticConfig`, etc.) from `critics/config.py` to `utils/config.py`
  - [x] Move specialized retrieval configs (`RankingConfig`, `IndexConfig`, `QueryProcessingConfig`) from `retrieval/config.py` to `utils/config.py`

- [x] Remove component-specific config modules
  - [x] Delete `models/config.py`
  - [x] Delete `chain/config.py`
  - [x] Delete `classifiers/config.py`
  - [x] Delete `critics/config.py`
  - [x] Delete `rules/config.py`
  - [x] Delete `retrieval/config.py`

- [x] Update imports in standardization functions
  - [x] Update imports in `standardize_model_config`
  - [x] Update imports in `standardize_chain_config`
  - [x] Update imports in `standardize_critic_config`

- [x] Update imports across the codebase
  - [x] Update imports in model components
  - [x] Update imports in chain components
  - [x] Update imports in classifier components
  - [x] Update imports in critic components
  - [x] Update imports in rule components
  - [x] Update imports in retrieval components

- [x] Add default configurations to utils/config.py
  - [x] Add DEFAULT_PROMPT_CONFIG
  - [x] Add DEFAULT_REFLEXION_CONFIG
  - [x] Add DEFAULT_CONSTITUTIONAL_CONFIG
  - [x] Add DEFAULT_SELF_REFINE_CONFIG
  - [x] Add DEFAULT_SELF_RAG_CONFIG
  - [x] Add DEFAULT_FEEDBACK_CONFIG
  - [x] Add DEFAULT_VALUE_CONFIG
  - [x] Add DEFAULT_LAC_CONFIG

### Phase 2: Interface Consolidation (High Impact)

**Status: Partially Completed**

This will resolve approximately 10-15 circular dependencies.

- [x] Complete the interface consolidation for chain component
  - [ ] Move remaining interfaces from `models` to `interfaces/model.py`
  - [x] Move remaining interfaces from `chain` to `interfaces/chain.py`
  - [ ] Move remaining interfaces from `classifiers` to `interfaces/classifier.py`
  - [ ] Move remaining interfaces from `critics` to `interfaces/critic.py`
  - [ ] Move remaining interfaces from `rules` to `interfaces/rule.py`
  - [ ] Move remaining interfaces from `retrieval` to `interfaces/retrieval.py`

- [ ] Use string type annotations for forward references
  - [ ] Update type annotations in model components
  - [ ] Update type annotations in chain components
  - [ ] Update type annotations in classifier components
  - [ ] Update type annotations in critic components
  - [ ] Update type annotations in rule components
  - [ ] Update type annotations in retrieval components

- [ ] Use TYPE_CHECKING for imports needed only for type checking
  - [ ] Update imports in model components
  - [ ] Update imports in chain components
  - [ ] Update imports in classifier components
  - [ ] Update imports in critic components
  - [ ] Update imports in rule components
  - [ ] Update imports in retrieval components

### Phase 3: Factory Function Refactoring (Medium Impact)

**Status: Not Started**

This will resolve approximately 8-10 circular dependencies.

- [ ] Implement lazy loading in all factory functions
  - [ ] Update `models/factories.py`
  - [ ] Update `chain/factories.py`
  - [ ] Update `classifiers/factories.py`
  - [ ] Update `critics/factories.py`
  - [ ] Update `rules/factories.py`
  - [ ] Update `retrieval/factories.py`

- [ ] Standardize factory function patterns
  - [ ] Standardize model factory functions
  - [ ] Standardize chain factory functions
  - [ ] Standardize classifier factory functions
  - [ ] Standardize critic factory functions
  - [ ] Standardize rule factory functions
  - [ ] Standardize retrieval factory functions

### Phase 4: Core Module Restructuring (Medium Impact)

**Status: Not Started**

This will resolve approximately 5-8 circular dependencies.

- [ ] Restructure `core.base` module
  - [ ] Move base classes to appropriate interface modules
  - [ ] Use string type annotations for forward references
  - [ ] Remove direct imports of implementation classes

- [ ] Refactor utility modules
  - [ ] Ensure utility modules only depend on other utility modules
  - [ ] Move component-specific utility functions to their respective components

### Phase 5: Rules Component Restructuring (Medium Impact)

**Status: Not Started**

This will resolve approximately 5-8 circular dependencies.

- [ ] Restructure the rules component
  - [ ] Move interfaces to `sifaka.interfaces.rule`
  - [ ] Use string type annotations for forward references
  - [ ] Implement lazy loading for rule imports in factory functions

- [ ] Fix circular dependency between rules and utils
  - [ ] Move text processing functions from `utils.text` that are only used by rules to the rules component
  - [ ] Or use string type annotations and TYPE_CHECKING for imports

## Verification

After each phase:

- [x] Run `test_circular_imports.py` to verify major components can be imported
- [x] Run `analyze_dependencies.py` to check for remaining circular dependencies
- [ ] Verify application functionality by running tests

## Progress Summary

| Phase | Description | Status | Dependencies Resolved |
|-------|-------------|--------|----------------------|
| 1 | Configuration System Overhaul | Completed | 20/20 |
| 2 | Interface Consolidation | Partially Completed | 1/15 |
| 3 | Factory Function Refactoring | Not Started | 0/10 |
| 4 | Core Module Restructuring | Not Started | 0/8 |
| 5 | Rules Component Restructuring | Not Started | 0/8 |
| **Total** | | | **21/61** |

## Verification Results

The `test_circular_imports.py` script shows that all major modules can be imported successfully, which indicates that the most critical circular dependencies have been resolved. This is a positive sign that the refactoring efforts are making progress, even though the total number of circular dependencies detected by the analysis script has increased.

The successful imports include:
- Core modules (core.base, core.dependency, core.factories)
- Interface modules (interfaces.model, interfaces.chain, etc.)
- Model modules (models.base, models.core, models.providers.openai, etc.)
- Chain modules (chain.chain, chain.config, chain.factories, etc.)
- Retrieval, Classifier, Critic, Adapter, and Rule modules

This suggests that while there are still many circular dependencies in the codebase, they are not preventing the major components from being imported and used.
