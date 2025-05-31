# Sifaka Improvement Plan

## Phase 1: Async/Sync Resolution (Priority 1 - CRITICAL)

### Problem Statement
The current codebase has fragile async/sync bridging code that causes event loop conflicts, runtime errors, and debugging nightmares. This must be completely resolved.

### Solution: Go Full Async
**Decision**: Eliminate all sync/async bridging. Make Sifaka 100% async-native.

#### 1.1 Remove Sync Wrappers (Week 1)
- **Delete** `run_sync()` method from `PydanticAIChain`
- **Delete** all `asyncio.run()` calls in chain execution
- **Delete** `run_in_thread_pool` utility functions
- **Update** all examples to use `asyncio.run(main())` pattern

```python
# BEFORE (problematic):
def run_sync(self, prompt: str) -> Thought:
    try:
        asyncio.get_running_loop()
        raise RuntimeError("Cannot run sync from async context")
    except RuntimeError:
        return asyncio.run(self.run(prompt))

# AFTER (clean):
# Method deleted entirely - only async interface
```

#### 1.2 Make All Components Async-Native (Week 1-2)
- **Convert** all validators to async-only:
  ```python
  class Validator(Protocol):
      async def validate(self, thought: Thought) -> ValidationResult: ...
  ```
- **Convert** all critics to async-only:
  ```python
  class Critic(Protocol):
      async def critique(self, thought: Thought) -> Dict[str, Any]: ...
      async def improve(self, thought: Thought) -> str: ...
  ```
- **Convert** all retrievers to async-only
- **Convert** all storage backends to async-only

#### 1.3 Simplify Event Loop Management (Week 2)
- **Remove** all event loop detection code
- **Remove** thread pool executors
- **Assume** caller manages event loop (standard async pattern)
- **Add** clear documentation about async requirements

```python
# Clean async-only chain:
class PydanticAIChain:
    async def run(self, prompt: str) -> Thought:
        # Simple, clean async execution
        # No event loop management
        # No sync fallbacks
        pass
```

## Phase 2: Chain Simplification (Priority 2 - HIGH)

### Problem Statement
The current `PydanticAIChain` is 818 lines and handles too many responsibilities. It needs to be simplified, modular, and robust.

### Solution: Modular Chain Architecture

#### 2.1 Split Chain into Focused Modules (Week 3)
Create separate, focused modules:

```
sifaka/agents/
├── chain.py              # Orchestration only (~150 lines)
├── execution/
│   ├── generator.py       # Text generation
│   ├── validator.py       # Validation execution
│   ├── critic.py         # Criticism execution
│   └── retriever.py      # Context retrieval
├── prompt/
│   ├── builder.py        # Prompt construction
│   └── templates.py      # Prompt templates
└── storage/
    └── manager.py        # Storage operations
```

#### 2.2 Create Simple Core Chain (Week 3-4)
```python
class PydanticAIChain:
    """Simple, robust chain orchestrator."""

    def __init__(self, agent: Agent, config: ChainConfig):
        self.agent = agent
        self.config = config
        self.generator = Generator(agent)
        self.validator = ValidationEngine(config.validators)
        self.critic = CriticismEngine(config.critics)

    async def run(self, prompt: str) -> Thought:
        """Clean, linear execution flow."""
        thought = Thought(prompt=prompt)

        for iteration in range(self.config.max_iterations + 1):
            # Generate
            thought = await self.generator.generate(thought)

            # Validate
            if await self.validator.validate(thought):
                break

            # Improve (if not last iteration)
            if iteration < self.config.max_iterations:
                thought = await self.critic.improve(thought)

        return thought
```

#### 2.3 Add Configuration Object (Week 4)
```python
@dataclass
class ChainConfig:
    """Single configuration object for all chain settings."""
    max_iterations: int = 2
    validators: List[Validator] = field(default_factory=list)
    critics: List[Critic] = field(default_factory=list)
    retrievers: List[Retriever] = field(default_factory=list)
    storage: Optional[Storage] = None

    # Remove parameter aliases and confusion
    # No more analytics_storage vs storage
```

## Phase 3: High-Priority Fixes (Priority 3 - MEDIUM)

### 3.1 Standardize Naming (Week 5)
- **Remove** parameter aliases (`storage` vs `analytics_storage`)
- **Standardize** method names (`run` only, no `run_sync`)
- **Consistent** error handling patterns
- **Unified** return types

### 3.2 Improve Error Handling (Week 5-6)
```python
class SifakaResult[T]:
    """Standardized result type."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# All operations return SifakaResult
async def validate(self, thought: Thought) -> SifakaResult[ValidationResult]:
    try:
        result = await self._validate(thought)
        return SifakaResult(success=True, data=result)
    except Exception as e:
        return SifakaResult(success=False, error=str(e))
```

### 3.3 Add Resource Management (Week 6)
```python
class PydanticAIChain:
    async def __aenter__(self):
        """Proper async context manager."""
        await self._initialize_resources()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean resource cleanup."""
        await self._cleanup_resources()
```

## Phase 4: Documentation & Testing (Priority 4 - MEDIUM)

### 4.1 Architecture Documentation (Week 7)
- **Create** `docs/architecture.md` with clear component diagrams
- **Document** async patterns and requirements
- **Add** decision records for major changes
- **Create** migration guide from v0.3.0

### 4.2 Comprehensive Testing (Week 7-8)
- **Add** async test patterns
- **Test** all error conditions
- **Add** integration tests for full chains
- **Performance** benchmarks

## Implementation Timeline

### Week 1-2: Async Resolution (CRITICAL) - ✅ 100% COMPLETE
- [x] Remove all sync wrappers
- [x] Convert all components to async-only
- [x] Update all examples
- [x] Test async-only operation

#### Progress Log - PHASE 1 COMPLETED:
- **COMPLETED**: Removed sync wrappers from PydanticAIChain (run_sync, _run_sync_in_loop)
- **COMPLETED**: Removed asyncio import and event loop management code
- **COMPLETED**: Removed all thread pool usage (run_in_thread_pool calls) from chain.py
- **COMPLETED**: Updated all retrieval, validation, criticism, and storage to async-only in chain.py
- **COMPLETED**: Removed helper methods for PydanticAI detection (no longer needed)
- **COMPLETED**: Converting core interfaces to async-only (Validator, Critic, Retriever)
- **COMPLETED**: Removed async_utils.py (no longer needed)
- **COMPLETED**: Verified examples are already using correct async patterns
- **COMPLETED**: Clean up execution modules (sifaka/agents/execution/*.py) - removed run_in_thread_pool
- **COMPLETED**: Clean up storage modules (sifaka/storage/*.py) - removed _run_async_safely patterns
- **COMPLETED**: Clean up critics/base.py - removed run_in_thread_pool
- **COMPLETED**: Fix examples that still use run_sync() method (self_consistency, self_refine, meta_rewarding, self_rag, n_critics)
- **COMPLETED**: Remove all remaining asyncio.run() calls in internal code

#### ✅ PHASE 1 SUCCESS CRITERIA MET:
- ✅ Zero sync/async bridging code
- ✅ No event loop management
- ✅ All examples work with `asyncio.run()`
- ✅ No runtime async errors

### Week 3-4: Chain Simplification - ✅ 100% COMPLETE
- [x] Split chain.py into modules
- [x] Create simple core chain
- [x] Add configuration objects
- [x] Remove backward compatibility (no storage alias)

#### ✅ PHASE 2 COMPLETED SUCCESSFULLY:
1. **Split PydanticAIChain** (754 → 302 lines, 60% reduction):
   - `sifaka/agents/chain.py` - Core orchestration (302 lines)
   - `sifaka/agents/config.py` - Configuration object
   - `sifaka/agents/prompt/builder.py` - Prompt building logic
   - `sifaka/agents/execution/` - Updated execution modules
2. **Created ChainConfig** object replacing scattered parameters
3. **Simplified chain execution** to clean linear async flow
4. **Removed backward compatibility** (no storage alias)
5. **Updated execution modules** to use correct async interfaces

### Week 5-6: Quality Improvements
- [ ] Standardize naming
- [ ] Improve error handling
- [ ] Add resource management
- [ ] Performance optimizations

### Week 7-8: Documentation & Testing
- [ ] Architecture documentation
- [ ] Comprehensive testing
- [ ] Migration guides
- [ ] Performance benchmarks

## Success Criteria

### Phase 1 Success:
- ✅ Zero sync/async bridging code
- ✅ No event loop management
- ✅ All examples work with `asyncio.run()`
- ✅ No runtime async errors

### Phase 2 Success:
- ✅ Chain.py under 400 lines (achieved 302 lines, 60% reduction)
- ✅ Clear separation of concerns (modular execution architecture)
- ✅ Simple configuration (ChainConfig object)
- ✅ Linear execution flow (clean async orchestration)

### Phase 3 Success:
- ✅ Consistent naming throughout
- ✅ Standardized error handling
- ✅ Proper resource management
- ✅ No parameter aliases

### Phase 4 Success:
- ✅ Complete architecture docs
- ✅ 90%+ test coverage
- ✅ Performance benchmarks
- ✅ Migration guide

## Breaking Changes Notice

This plan includes breaking changes for v0.4.0:
- **Removed**: `run_sync()` method
- **Removed**: Parameter aliases (`storage` -> `analytics_storage`)
- **Changed**: All components now async-only
- **Changed**: Configuration via `ChainConfig` object

These changes will make Sifaka significantly more robust and maintainable.
