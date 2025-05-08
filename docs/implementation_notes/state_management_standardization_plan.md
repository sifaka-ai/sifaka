# State Management Standardization Plan

This document outlines the implementation plan for standardizing state management across the Sifaka codebase.

## Goals

1. Standardize all implementation classes to use `_state_manager` instead of direct `_state`
2. Ensure consistent initialization timing across components
3. Standardize state access patterns to always use `_state_manager.get_state()`
4. Update documentation to address the current inconsistencies

## 1. Analysis and Preparation

### 1.1 Identify Affected Components
- [x] Create a comprehensive list of all implementation classes that use direct `_state` instead of `_state_manager`
- [x] Identify components with inconsistent initialization timing
- [x] Document all variations of state access patterns

### 1.2 Create Reference Implementation
- [x] Develop a reference implementation for each component type (Classifier, Critic, Rule, Chain)
- [x] Document the standardized patterns for:
  - State declaration
  - Initialization timing
  - State access
  - Caching approach

### 1.3 Define Testing Strategy
- [x] Determine how to verify that refactored components maintain the same behavior
- [x] Create test cases that exercise state initialization, access, and modification
- [x] Establish criteria for successful refactoring

## 2. Implementation: Standardize Implementation Classes

### 2.1 Classifier Implementations
- [x] Update `ToxicityClassifierImplementation` to use `_state_manager` (completed)
- [x] Update `SentimentClassifierImplementation` to use `_state_manager` (completed)
- [x] Update `BiasDetectorImplementation` to use `_state_manager` (completed)
- [x] Update `SpamClassifierImplementation` to use `_state_manager` (completed)
- [x] Update `ProfanityClassifierImplementation` to use `_state_manager` (completed)
- [x] Update `NERClassifierImplementation` to use `_state_manager` (completed)
- [x] Update `GenreClassifierImplementation` to use `_state_manager` (completed)
- [x] Update `ReadabilityClassifierImplementation` to use `_state_manager` (completed)
- [x] Ensure consistent initialization in either `__init__` or `warm_up()` (completed)
- [x] Standardize state access through `_state_manager.get_state()` (completed)

Example refactoring for `ToxicityClassifierImplementation`:
```python
# Before
def __init__(self, config: ClassifierConfig) -> None:
    self.config = config
    self._state = ClassifierState()
    self._state.initialized = False
    self._state.cache = {}

# After
_state_manager = PrivateAttr(default_factory=create_classifier_state)

def __init__(self, config: ClassifierConfig) -> None:
    self.config = config
    # State is managed by StateManager, no need to initialize here
```

### 2.2 Critic Implementations
- [x] Update `PromptCriticImplementation` to use `_state_manager` (completed)
- [x] Update other critic implementations (reflexion, self-refine, constitutional, etc.) (completed)
- [x] Standardize initialization timing (completed)
- [x] Ensure consistent state access (completed)

### 2.3 Rule Implementations
- [x] Update rule validators to use `_state_manager` where appropriate
- [x] Standardize rule initialization
- [x] Ensure consistent state access

### 2.4 Chain Implementations
- [x] Verify all chain implementations use `_state_manager` consistently
- [x] Standardize initialization timing
- [x] Ensure consistent state access

## 3. Implementation: Standardize Initialization Timing

### 3.1 Define Standard Initialization Pattern
- [x] Decide on the standard initialization pattern (either in `__init__` or `warm_up()`)
- [x] Document the chosen pattern with clear examples

The standard initialization pattern for Sifaka components is:

1. **Basic initialization in `__init__()`**:
   - Initialize the state manager with `_state_manager = PrivateAttr(default_factory=create_X_state)`
   - Set `state.initialized = False` at the beginning of initialization
   - Store configuration and lightweight components in state
   - Set `state.initialized = True` at the end of basic initialization

2. **Resource-intensive initialization in `warm_up()`**:
   - Check if already initialized with `if not state.initialized:`
   - Initialize expensive resources (models, large data structures)
   - Handle initialization errors with try/except blocks
   - Set `state.initialized = True` after successful initialization

This pattern ensures:
- Components are usable immediately after `__init__()` for basic operations
- Resource-intensive operations are deferred until `warm_up()` is called
- Consistent state management across all components
- Proper error handling for initialization failures

### 3.2 Update Components
- [x] Update all components to follow the standard initialization pattern
- [x] Ensure initialization checks for `state.initialized` before performing initialization
- [x] Add appropriate error handling for initialization failures

Example standardized initialization:
```python
def warm_up(self) -> None:
    """Initialize resources if not already initialized."""
    state = self._state_manager.get_state()
    if not state.initialized:
        try:
            # Initialize resources
            state.model = self._load_model()
            state.initialized = True
        except Exception as e:
            state.error = f"Initialization failed: {e}"
            raise RuntimeError(f"Failed to initialize: {e}")
```

## 4. Implementation: Standardize State Access Patterns

### 4.1 Define Standard Access Pattern
- [x] Document the standard state access pattern using `_state_manager.get_state()`
- [x] Create helper methods for common state access patterns if needed

The standard state access pattern for Sifaka components is:

1. **Get state at the beginning of methods**:
   ```python
   def some_method(self, input_data):
       # Get state at the beginning of the method
       state = self._state_manager.get_state()

       # Use state throughout the method
       if not state.initialized:
           self.warm_up()

       # Access cached data
       if input_data in state.cache:
           return state.cache[input_data]
   ```

2. **Check initialization status**:
   ```python
   def validate(self, text):
       state = self._state_manager.get_state()
       if not state.initialized:
           raise RuntimeError("Component not initialized")
   ```

3. **Use state.cache for temporary data**:
   ```python
   def process(self, input_data):
       state = self._state_manager.get_state()

       # Check cache first
       if input_data in state.cache:
           return state.cache[input_data]

       # Process and cache result
       result = self._compute_result(input_data)
       state.cache[input_data] = result
       return result
   ```

### 4.2 Update Direct State Access
- [x] Replace all direct `self._state` accesses with `self._state_manager.get_state()`
- [x] Update all state modifications to use the state manager
- [x] Standardize error handling for state access

Example refactoring for `SelfRAGCriticImplementation`:
```python
# Before
if not self._state.initialized:
    raise RuntimeError("SelfRAGCriticImplementation not properly initialized")

retrieval_template = self._state.cache.get("retrieval_prompt_template")
retrieval_query = self._state.model.generate(...)

# After
state = self._state_manager.get_state()
if not state.initialized:
    raise RuntimeError("SelfRAGCriticImplementation not properly initialized")

retrieval_template = state.cache.get("retrieval_prompt_template")
retrieval_query = state.model.generate(...)
```

### 4.3 Standardize Caching
- [x] Define standard caching approach using `state.cache`
- [x] Update all components to use the standard caching approach
- [x] Ensure cache management is consistent (initialization, access, cleanup)

The standard caching approach for Sifaka components is:

1. **Initialize cache in `__init__()`**:
   ```python
   def __init__(self, config):
       # Get state
       state = self._state_manager.get_state()

       # Initialize cache
       state.cache = {}

       # Store configuration in cache
       state.cache["config"] = config
   ```

2. **Access cache with get() for safety**:
   ```python
   def process(self, input_data):
       state = self._state_manager.get_state()

       # Safely access cache with default values
       model = state.cache.get("model")
       threshold = state.cache.get("threshold", 0.5)
   ```

3. **Use cache for temporary data and expensive computations**:
   ```python
   def compute_embeddings(self, text):
       state = self._state_manager.get_state()

       # Check cache first
       cache_key = f"embeddings_{hash(text)}"
       if cache_key in state.cache:
           return state.cache[cache_key]

       # Compute and cache
       embeddings = self._compute_embeddings(text)
       state.cache[cache_key] = embeddings
       return embeddings
   ```

## 5. Documentation Updates

### 5.1 Update State Management Documentation
- [x] Update `state_management.md` to address current inconsistencies
- [x] Add clear examples of the standardized patterns
- [x] Document the rationale for the chosen patterns

### 5.2 Update Component Documentation
- [x] Update docstrings for all refactored components
- [x] Add examples of proper state management in component documentation
- [x] Ensure consistency between code and documentation

Example of updated docstring for SelfRAGCriticImplementation:
```python
class SelfRAGCriticImplementation:
    """
    Implementation of a Self-RAG critic using language models with retrieval.

    This class implements the CriticImplementation protocol for a Self-RAG critic
    that uses language models and retrieval to evaluate, validate, and improve text.

    ## Lifecycle Management

    The SelfRAGCriticImplementation manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up language model provider
       - Sets up retriever
       - Initializes state using StateManager

    2. **Operation**
       - Decides whether to retrieve
       - Retrieves relevant information
       - Generates responses
       - Reflects on responses

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)
```

### 5.3 Create Migration Guide
- [x] Document the process for migrating legacy components to the standardized approach
- [x] Provide before/after examples for each component type
- [x] Include troubleshooting guidance for common issues

## 6. Testing and Validation

### 6.1 Unit Testing
- [ ] Update unit tests to verify proper state management
- [ ] Add tests specifically for state initialization, access, and modification
- [ ] Ensure test coverage for error handling

### 6.2 Integration Testing
- [ ] Test components together to ensure they work correctly after refactoring
- [ ] Verify that state is properly managed across component interactions
- [ ] Test edge cases and error conditions

### 6.3 Performance Testing
- [ ] Verify that the standardized approach doesn't introduce performance regressions
- [ ] Test with large datasets to ensure efficient state management
- [ ] Optimize if necessary

## 7. Rollout Strategy

### 7.1 Phased Implementation
- [ ] Start with one component type (e.g., Classifiers) and standardize all implementations
- [ ] Move to the next component type after successful validation
- [ ] Prioritize components based on usage and complexity

### 7.2 Code Review Process
- [ ] Establish clear review criteria for state management
- [ ] Create a checklist for reviewers to verify standardization
- [ ] Document common issues and solutions

### 7.3 Backward Compatibility
- [ ] Ensure refactored components maintain the same public API
- [ ] Document any breaking changes and provide migration guidance
- [ ] Consider providing compatibility layers if necessary

## 8. Timeline and Resources

### 8.1 Timeline
- Week 1: Analysis and preparation
- Week 2-3: Standardize classifier implementations
- Week 4-5: Standardize critic implementations
- Week 6: Standardize rule implementations
- Week 7: Standardize chain implementations
- Week 8: Documentation updates
- Week 9: Testing and validation
- Week 10: Final review and cleanup

### 8.2 Resources Required
- 1-2 developers familiar with the Sifaka codebase
- Code reviewers for each component type
- Documentation writer to update documentation
- QA resources for testing and validation

## 9. Success Criteria

- All implementation classes use `_state_manager` instead of direct `_state`
- Consistent initialization timing across all components
- All state access uses `_state_manager.get_state()`
- Updated documentation that accurately reflects the standardized patterns
- All tests pass with the refactored code
- No regressions in functionality or performance

## 10. Monitoring and Maintenance

### 10.1 Linting Rules
- [ ] Implement linting rules to enforce the standardized patterns
- [ ] Add CI checks to verify compliance
- [ ] Document the linting rules and how to apply them

### 10.2 Ongoing Maintenance
- [ ] Establish a process for reviewing new components
- [ ] Create templates for new components with proper state management
- [ ] Schedule periodic reviews to ensure continued compliance
