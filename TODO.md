# Sifaka TODO List

Based on the comprehensive codebase review, this TODO list prioritizes improvements to elevate Sifaka from B- (74/100) to A- quality for production readiness.

## 🚨 CRITICAL PRIORITY (Must Fix)

### 1. Consolidate Critics and Classifiers
**Target: Single file per component type for readability**

- [ ] **Merge all critics into `sifaka/critics.py`**
  - Consolidate: `constitutional.py`, `reflexion.py`, `self_refine.py`, `self_consistency.py`, `n_critics.py`, `meta_rewarding.py`, `prompt.py`, `self_rag.py`
  - Keep base classes and protocols separate
  - Maintain clear class separation within the single file
  - Update all imports across the codebase

- [ ] **Merge all classifiers into `sifaka/classifiers.py`**
  - Consolidate: `emotion.py`, `intent.py`, `language.py`, `readability.py`, `sentiment.py`, `spam.py`, `toxicity.py`
  - Keep base classes separate
  - Maintain clear class separation within the single file
  - Update all imports across the codebase

### 2. Refactor Monolithic Functions
**Target: Functions under 50 lines, single responsibility**

- [ ] **Break down `SifakaEngine.think()` (currently 130+ lines)**
  ```python
  # Split into:
  async def think(self, prompt: str, max_iterations: int = 3) -> SifakaThought:
      thought = self._validate_and_create_thought(prompt, max_iterations)
      thought = await self._check_cache_or_execute(thought)
      self._record_metrics_and_log(thought)
      return thought
  ```

- [ ] **Refactor `_validate_preset_params()` in presets.py (75+ lines)**
  - Split into separate validation functions per parameter type
  - Create `ParameterValidator` class

- [ ] **Simplify complex error handling in `utils/errors.py`**
  - Reduce `ErrorMessageBuilder` complexity
  - Consolidate similar error types

### 3. Fix Critical Consistency Issues
**Target: Uniform patterns across codebase**

- [ ] **Standardize naming conventions**
  - `SifakaThought` → `Thought` (shorter, clearer)
  - `ValidationResult` → `ValidationResult` (keep)
  - `CritiqueResult` → `CritiqueResult` (keep)
  - Consistent method naming: `async def process_*()` for all async operations

- [ ] **Unify async/sync patterns**
  - Make ALL public APIs async (remove sync methods)
  - Remove mixed patterns like `get_timing_stats()` (sync) alongside async methods
  - Consistent error handling: always raise exceptions, never return error messages

- [ ] **Consolidate configuration approaches**
  - Remove duplicate configuration methods
  - Single source of truth: `SifakaConfig` class only
  - Deprecate direct `SifakaDependencies` usage in favor of config

## 🔥 HIGH PRIORITY (Production Blockers)

### 4. Simplify Architecture
**Target: Reduce complexity, improve maintainability**

- [ ] **Replace complex dependency injection with simple registry**
  ```python
  # Replace SifakaDependencies with:
  class ComponentRegistry:
      def register_critic(self, name: str, critic_class: Type[Critic]) -> None
      def register_validator(self, name: str, validator_class: Type[Validator]) -> None
      def get_critic(self, name: str) -> Critic
  ```

- [ ] **Eliminate tight coupling between graph nodes**
  - Make nodes independent of specific dependency types
  - Use interfaces/protocols only
  - Remove hard-coded component references

- [ ] **Simplify preset system**
  - Reduce from 6 presets to 3: `quick()`, `balanced()`, `thorough()`
  - Remove redundant aliases (`academic = academic_writing`)
  - Clear parameter inheritance hierarchy

### 5. Improve Error Handling
**Target: Consistent, actionable error messages**

- [ ] **Standardize exception hierarchy**
  - All exceptions inherit from `SifakaError`
  - Consistent constructor signatures
  - Always include actionable suggestions

- [ ] **Fix error message inconsistencies**
  - Remove functions that return error messages instead of raising
  - Standardize error context information
  - Improve error message clarity

- [ ] **Add proper error recovery**
  - Graceful degradation when components fail
  - Retry mechanisms for transient failures
  - Better error propagation in graph execution

## 📈 MEDIUM PRIORITY (Quality Improvements)

### 6. Test Suite Improvements
**Target: Simpler, more reliable tests**

- [ ] **Reduce test complexity**
  - Minimize mocking in favor of real component testing
  - Create test utilities for common setup
  - Remove overly complex test scenarios

- [ ] **Add missing integration tests**
  - End-to-end workflow testing
  - Real API integration tests (with proper mocking)
  - Performance regression tests

- [ ] **Improve test organization**
  - Clear separation: unit vs integration vs performance
  - Consistent test naming and structure
  - Better test data management

### 7. Performance Optimization
**Target: Production-ready performance**

- [ ] **Profile and optimize hot paths**
  - Graph execution performance
  - Memory usage optimization
  - Async operation efficiency

- [ ] **Fix memory leaks**
  - Proper cleanup in long-running processes
  - Cache size management
  - Resource disposal

- [ ] **Add performance monitoring**
  - Built-in metrics collection
  - Performance regression detection
  - Resource usage tracking

### 8. Documentation Updates
**Target: Current, accurate documentation**

- [ ] **Update all examples to current API**
  - Remove deprecated patterns
  - Consistent code style
  - Working examples only

- [ ] **Add missing guides**
  - Performance optimization guide
  - Advanced configuration examples
  - Migration guide for API changes

- [ ] **Improve troubleshooting**
  - Common error scenarios
  - Debug mode documentation
  - Performance tuning tips

## 🔧 LOW PRIORITY (Polish & Enhancement)

### 9. Code Quality Improvements
**Target: Clean, maintainable code**

- [ ] **Reduce code duplication**
  - Extract common patterns
  - Create utility functions
  - Consolidate similar logic

- [ ] **Improve type safety**
  - Add missing type hints
  - Fix mypy warnings
  - Better generic type usage

- [ ] **Code style consistency**
  - Consistent docstring format
  - Uniform import organization
  - Standard code formatting

### 10. Feature Cleanup
**Target: Remove unused/confusing features**

- [ ] **Remove unused code paths**
  - Dead code elimination
  - Unused configuration options
  - Deprecated features

- [ ] **Simplify complex features**
  - Reduce configuration options
  - Streamline API surface
  - Remove edge case handling

- [ ] **Improve user experience**
  - Better error messages
  - Clearer API documentation
  - Simplified getting started guide

## 📊 Success Metrics

### Code Quality Targets
- [ ] All functions under 50 lines
- [ ] Test coverage > 85%
- [ ] MyPy strict mode passing
- [ ] No code duplication > 10 lines
- [ ] All public APIs async-only

### Performance Targets
- [ ] Graph execution < 100ms overhead
- [ ] Memory usage < 100MB for typical workflows
- [ ] No memory leaks in 24h runs
- [ ] API response time < 5s for complex workflows

### User Experience Targets
- [ ] Getting started in < 5 minutes
- [ ] Clear error messages with solutions
- [ ] Working examples for all features
- [ ] Single configuration approach

## 🗓️ Timeline Estimate

**Phase 1 (Critical Priority): 3-4 weeks**
- Critics/classifiers consolidation: 1 week
- Function refactoring: 1.5 weeks
- Consistency fixes: 1 week

**Phase 2 (High Priority): 3-4 weeks**
- Architecture simplification: 2 weeks
- Error handling standardization: 1.5 weeks

**Phase 3 (Medium Priority): 2-3 weeks**
- Test improvements: 1.5 weeks
- Performance optimization: 1 week
- Documentation updates: 1 week

**Phase 4 (Low Priority): 1-2 weeks**
- Code quality polish: 1 week
- Feature cleanup: 1 week

**Total: 9-13 weeks to A- quality**

---

## 🎯 Definition of Done

A task is complete when:
1. ✅ Code passes all existing tests
2. ✅ New tests added for new functionality
3. ✅ Documentation updated
4. ✅ No new mypy/ruff warnings
5. ✅ Performance impact measured and acceptable
6. ✅ Breaking changes documented in CHANGELOG.md
