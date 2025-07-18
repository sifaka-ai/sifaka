# Sifaka Improvement TODO List

This document tracks the major improvements identified in the codebase review. Each item includes specific tasks, acceptance criteria, and progress tracking.

## 1. Configuration Refactoring ✅

**Goal**: Break up the monolithic Config class into focused, composable configuration objects.

### Tasks:
- [x] Create `LLMConfig` class for LLM-related settings
  - [x] Extract: model, temperature, provider, api_key, timeout
  - [x] Add validation for provider-specific constraints
- [x] Create `ValidationConfig` class for validation settings
  - [x] Extract: validators, validator-specific settings
  - [x] Add validator registry integration
- [x] Create `CriticConfig` class for critic settings
  - [x] Extract: critics, critic_tool_settings
  - [x] Add critic registry integration
- [x] Create `EngineConfig` class for engine behavior
  - [x] Extract: max_iterations, min_quality_score, enable_middleware
  - [x] Add performance tuning options
- [x] Create `StorageConfig` class for storage settings
  - [x] Extract: storage_backend, storage_path
  - [x] Add connection pooling options
- [x] Update main `Config` class to compose these configs
  - [x] Maintain backward compatibility with deprecation warnings
  - [x] Add migration guide
- [x] Add factory methods for common configurations
  - [x] `Config.minimal()` - bare minimum settings
  - [x] `Config.development()` - good for testing
  - [x] `Config.production()` - optimized for production
  - [x] `Config.research()` - all critics, detailed logging

### Acceptance Criteria:
- Old Config API still works with deprecation warnings
- New configs have focused responsibilities
- Type safety improved with specific config types
- Documentation updated with examples

## 2. Plugin System Extension 🔧

**Goal**: Extend the plugin system to support critics and validators, not just storage backends.

### Tasks:
- [ ] Create `PluginInterface` base class
  - [ ] Define common plugin lifecycle methods
  - [ ] Add metadata requirements (name, version, author)
- [ ] Implement `CriticPlugin` interface
  - [ ] Extend BaseCritic with plugin capabilities
  - [ ] Add discovery mechanism
  - [ ] Create registration system
- [ ] Implement `ValidatorPlugin` interface
  - [ ] Extend BaseValidator with plugin capabilities
  - [ ] Add discovery mechanism
  - [ ] Create registration system
- [ ] Create plugin loader utility
  - [ ] Scan for plugins in specified directories
  - [ ] Handle plugin dependencies
  - [ ] Add plugin validation
- [ ] Add plugin management CLI commands
  - [ ] `sifaka plugins list`
  - [ ] `sifaka plugins install <name>`
  - [ ] `sifaka plugins validate`
- [ ] Create plugin templates
  - [ ] Cookiecutter template for critic plugins
  - [ ] Cookiecutter template for validator plugins
  - [ ] Example plugins with tests
- [ ] Document plugin development
  - [ ] Plugin development guide
  - [ ] API reference for plugin interfaces
  - [ ] Best practices guide

### Acceptance Criteria:
- Plugins can be loaded from external packages
- Plugin discovery works automatically
- Templates make creating new plugins easy
- Documentation is comprehensive

## 3. Documentation Consolidation 📚

**Goal**: Create a single source of truth for documentation with clear learning paths.

### Tasks:
- [ ] Restructure documentation hierarchy
  - [ ] Create `docs/` directory structure
  - [ ] Move scattered docs to organized locations
- [ ] Create documentation sections:
  - [ ] Getting Started (progressive tutorials)
  - [ ] User Guide (how-to guides)
  - [ ] API Reference (auto-generated)
  - [ ] Architecture Guide (design decisions)
  - [ ] Plugin Development
  - [ ] Troubleshooting
- [ ] Add learning paths
  - [ ] Beginner path: simple text improvement
  - [ ] Intermediate path: custom validators
  - [ ] Advanced path: custom critics and plugins
- [ ] Create example progression
  - [ ] Number examples (01_basic.py, 02_validators.py, etc.)
  - [ ] Add README to examples directory
  - [ ] Create example test suite
- [ ] Set up documentation site
  - [ ] Use MkDocs with Material theme
  - [ ] Auto-deploy to GitHub Pages
  - [ ] Add search functionality
- [ ] Add architecture documentation
  - [ ] Create ADRs (Architecture Decision Records)
  - [ ] Document key design patterns
  - [ ] Add system diagrams

### Acceptance Criteria:
- Single documentation site with all content
- Clear navigation and search
- Examples follow logical progression
- Architecture decisions documented

## 4. Type Safety Improvements 🔒

**Goal**: Replace magic strings with enums and strengthen type annotations throughout.

### Tasks:
- [x] Create enums for string constants
  - [x] `CriticType` enum for critic names
  - [x] `ValidatorType` enum for validator names
  - [x] `StorageType` enum for storage backends
  - [x] `Provider` enum already exists - use consistently
- [x] Strengthen type annotations
  - [ ] Replace `Any` with specific types where possible
  - [ ] Add generic types for plugin system
  - [x] Use Protocol types for interfaces
  - [ ] Add overloads for functions with multiple signatures
- [ ] Create type stubs for dynamic components
  - [ ] Types for plugin interfaces
  - [ ] Types for middleware
  - [ ] Types for result objects
- [ ] Configure stricter mypy settings
  - [ ] Enable `strict` mode gradually
  - [ ] Fix all type errors
  - [ ] Add type checking to pre-commit
- [ ] Add runtime type validation
  - [ ] Use Pydantic for all public APIs
  - [ ] Add helpful error messages for type errors
  - [ ] Create type guards for common patterns

### Acceptance Criteria:
- No magic strings in public APIs
- Mypy strict mode passes
- Runtime type errors have helpful messages
- Type stubs available for all public APIs

## 5. Performance Optimization 🚀

**Goal**: Improve performance through connection pooling, parallel processing, and profiling.

### Tasks:
- [ ] Implement LLM client connection pooling
  - [ ] Create `LLMClientPool` class
  - [ ] Add connection lifecycle management
  - [ ] Configure pool size limits
  - [ ] Add connection health checks
- [ ] Enable parallel critic evaluation
  - [ ] Add `parallel` flag to Config
  - [ ] Implement concurrent critic execution
  - [ ] Handle result aggregation
  - [ ] Add concurrency limits
- [ ] Optimize file storage
  - [ ] Replace sync I/O with proper async
  - [ ] Add file operation batching
  - [ ] Implement caching layer
- [ ] Add performance monitoring
  - [ ] Create performance benchmarks
  - [ ] Add timing to all major operations
  - [ ] Create performance regression tests
  - [ ] Add performance dashboard
- [ ] Profile and optimize hot paths
  - [ ] Profile with cProfile/py-spy
  - [ ] Identify bottlenecks
  - [ ] Optimize JSON serialization
  - [ ] Cache compiled validators

### Acceptance Criteria:
- Connection pooling reduces latency
- Parallel evaluation improves throughput
- No performance regressions
- Performance metrics available

## Progress Tracking

### Completed ✅
- [x] Created TODO.md file
- [x] Configuration Refactoring
  - Created focused config classes (LLM, Critic, Engine, Storage, Validation)
  - Maintained backward compatibility
  - Added new factory methods
  - Created migration guide
- [x] Type Safety Improvements (Partial)
  - Created enums for string constants (CriticType, ValidatorType, StorageType)
  - Updated configs to use enums with backward compatibility
  - Created Protocol types for interfaces
  - Added type-safe example

### In Progress 🚧
- Type Safety Improvements (remaining tasks)
  - [ ] Replace remaining `Any` types
  - [ ] Configure stricter mypy settings
  - [ ] Add runtime type validation

### Not Started ⏸️
- Plugin System Extension
- Documentation Consolidation
- Performance Optimization

## Priority Order

1. **Configuration Refactoring** - Foundation for other improvements
2. **Type Safety Improvements** - Makes other changes safer
3. **Plugin System Extension** - Enables community contributions
4. **Documentation Consolidation** - Improves adoption
5. **Performance Optimization** - Fine-tuning once structure is solid

## Notes

- Maintain backward compatibility where possible
- Add deprecation warnings for breaking changes
- Update tests alongside each change
- Document decisions as we go
- Consider creating a v2.0 branch for major changes
