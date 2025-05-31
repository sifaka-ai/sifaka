# Sifaka v0.3.0 - Complete Async Migration Plan

## 🚨 CRITICAL ISSUES IDENTIFIED

### Current Status: Phase 2 Complete, Phase 3 URGENT ⚠️

After completing the chain simplification, testing revealed **critical async interface mismatches** that prevent examples from working. This plan addresses comprehensive async migration.

### Issues Found:
1. **Critics**: Missing `_perform_critique_async` methods (SelfConsistencyCritic fails)
2. **Storage**: Chain expects `save_thought_async`/`save_async` but storage only has `set()`
3. **Validators**: Some still use sync-only interfaces
4. **Retrievers**: Mixed sync/async support
5. **Examples**: Failing due to async interface mismatches

## Phase 3: Complete Async Migration (URGENT) 🔥

### 3.1 Fix Critics Async Interface (Priority 1)
**Problem**: Critics have `_perform_critique()` but execution expects `_perform_critique_async()`

**Files to Fix**:
- `sifaka/critics/self_consistency.py` - Add `_perform_critique_async()`
- `sifaka/critics/constitutional.py` - Add `_perform_critique_async()`
- `sifaka/critics/prompt.py` - Add `_perform_critique_async()`
- `sifaka/critics/self_refine.py` - Add `_perform_critique_async()`
- `sifaka/critics/self_rag.py` - Add `_perform_critique_async()`
- `sifaka/critics/meta_rewarding.py` - Add `_perform_critique_async()`
- `sifaka/critics/n_critics.py` - Add `_perform_critique_async()`

**Solution**:
```python
# Add to each critic class:
async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
    """Async version of critique logic."""
    # For now, wrap sync version - later optimize for true async
    return self._perform_critique(thought)
```

### 3.2 Fix Storage Interface (Priority 1)
**Problem**: Chain expects `save_thought_async()` but storage only has `set()`

**File to Fix**: `sifaka/agents/chain.py`

**Solution**:
```python
# Replace in _save_thought_for_analytics():
if hasattr(self.config.analytics_storage, "save_thought_async"):
    await self.config.analytics_storage.save_thought_async(thought)
elif hasattr(self.config.analytics_storage, "save_async"):
    await self.config.analytics_storage.save_async(thought.id, thought)
else:
    # Use standard storage protocol
    await self.config.analytics_storage.set(thought.id, thought)
```

### 3.3 Fix Validators Async Interface (Priority 2)
**Problem**: Some validators may not have proper async support

**Files to Check**:
- `sifaka/validators/base.py` - Ensure `validate_async()` exists
- `sifaka/validators/length.py` - Check async implementation
- `sifaka/validators/regex.py` - Check async implementation
- All other validator implementations

### 3.4 Fix Retrievers Async Interface (Priority 2)
**Problem**: Mixed sync/async support in retrievers

**Files to Check**:
- `sifaka/retrievers/` - All retriever implementations
- `sifaka/agents/execution/retrieval.py` - Already updated to handle multiple interfaces

### 3.5 Update Examples (Priority 3)
**Problem**: Examples failing due to async interface issues

**Files to Fix**:
- `examples/self_consistency_example.py` - Test after critic fixes
- All other examples in `examples/` directory

## Implementation Priority

### Week 1: Critical Fixes
1. **Day 1**: Fix all critics async interface
2. **Day 2**: Fix storage interface in chain
3. **Day 3**: Test self_consistency_example.py
4. **Day 4**: Fix any remaining validator issues
5. **Day 5**: Test all examples

### Week 2: Comprehensive Testing
1. **Day 1-2**: Run all examples and fix issues
2. **Day 3-4**: Add async tests for all components
3. **Day 5**: Documentation updates

## Success Criteria

### Phase 3 Complete When:
- ✅ All critics have `_perform_critique_async()` methods
- ✅ Storage interface works correctly with chain
- ✅ All validators support async operations
- ✅ All retrievers support async operations
- ✅ All examples run successfully
- ✅ No async/sync interface mismatches
- ✅ Clean error messages (no "object has no attribute" errors)

## Breaking Changes for v0.3.0

### Removed (Clean API):
- No backward compatibility aliases
- No sync wrapper methods
- No mixed sync/async patterns

### Added (Async-First):
- Complete async interface for all components
- Proper error handling for async operations
- Clean async-only examples

## Post-Phase 3: Quality Improvements

### Phase 4: Advanced Features (Future)
- Enhanced error handling
- Performance optimizations
- Advanced async patterns
- Comprehensive documentation
- Integration tests

## Notes

This plan addresses the critical async migration issues that were missed in the initial Phase 2 planning. The focus is on making Sifaka truly async-native and ensuring all components work together seamlessly.

The modular architecture from Phase 2 provides a solid foundation for these async fixes.
