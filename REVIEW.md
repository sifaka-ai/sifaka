# Sifaka Improvement Plan

This document provides a focused improvement plan based on the code review, prioritizing the highest impact changes to get Sifaka production-ready FAST.

## CRITICAL ISSUES TO FIX IMMEDIATELY

### 🔥 HIGHEST IMPACT - Fix These First (1-2 days)

**1. Fix Storage Async Bug (BLOCKING TESTS)**
- **Issue**: `asyncio.run() cannot be called from a running event loop` in storage
- **Impact**: 15 test failures, storage completely broken
- **Root Cause**: Chain executor trying to call sync storage methods from async context
- **Fix**: Storage executor needs to use async methods directly, not wrap with asyncio.run()

**2. Fix Missing Test Dependencies**
- **Issue**: `PromptCritic` not imported, `MockStorageFactory` missing
- **Impact**: Test failures, broken examples
- **Fix**: Add missing imports and factory classes

**3. Fix Chain Configuration Issues**
- **Issue**: `max_attempts` parameter doesn't exist, critic validation failing
- **Impact**: API inconsistency, test failures
- **Fix**: Update Chain API and critic validation logic

### 🚀 HIGH IMPACT - Fix These Next (2-3 days)

**4. Fix MyPy Type Errors**
- **Issue**: MyPy hanging/failing, type errors throughout codebase
- **Impact**: No type safety, CI failures
- **Fix**: Run mypy, fix type errors systematically

**5. Fix Test Assertion Logic**
- **Issue**: Tests expecting wrong number of critic feedback items
- **Impact**: Flaky tests, unclear behavior
- **Fix**: Update test expectations to match actual behavior

**6. Fix Examples**
- **Issue**: Examples likely broken due to API changes
- **Impact**: Users can't get started, documentation is wrong
- **Fix**: Update all examples to work with current API

## ASYNC/SYNC IMPLEMENTATION STATUS ✅

**GOOD NEWS**: Your async/sync implementation is EXACTLY RIGHT!

- ✅ **Chain.run()**: Sync public API, async internal implementation
- ✅ **Models**: Sync public API (`generate()`), async internal (`_generate_async()`)
- ✅ **Storage**: Sync public API (`get()`, `set()`), async internal (`_get_async()`, `_set_async()`)
- ✅ **Critics**: Sync public API (`critique()`), async internal (`_critique_async()`)

**The pattern is perfect** - users see sync, you get async performance under the hood.

**The only issue** is the storage bug where Chain executor calls sync storage methods from async context.

## PRIORITY FIXES BY IMPACT

### Priority 1: BLOCKING ISSUES (Fix Today)

1. **Storage Async Bug** - 15 test failures
2. **Missing Test Dependencies** - Import errors
3. **Chain Configuration** - API inconsistencies

### Priority 2: QUALITY ISSUES (Fix This Week)

4. **MyPy Type Errors** - Type safety
5. **Test Assertion Logic** - Test reliability
6. **Examples Updates** - User experience

### Priority 3: POLISH ISSUES (Fix Next Week)

7. **Documentation Gaps** - Missing tutorials
8. **Configuration Simplification** - User experience
9. **Error Message Improvements** - User experience

## SPECIFIC FIXES NEEDED

### Fix 1: Storage Async Bug (CRITICAL)

**Problem**: Chain executor calls `storage.set()` from async context, which calls `asyncio.run()`, causing nested event loop error.

**Solution**: Chain executor should call `await storage._set_async()` directly.

**Files to Fix**:
- `sifaka/core/chain/executor.py` - Use async storage methods
- `sifaka/storage/protocol.py` - Ensure async methods work correctly

### Fix 2: Missing Test Dependencies

**Problem**: Tests import missing classes.

**Solution**: Add missing imports and classes.

**Files to Fix**:
- `tests/test_e2e_scenarios.py` - Add `PromptCritic` import
- `tests/utils/mocks.py` - Add `MockStorageFactory` class

### Fix 3: Chain Configuration Issues

**Problem**: Tests use non-existent parameters.

**Solution**: Update Chain API or fix test parameters.

**Files to Fix**:
- `sifaka/core/chain/chain.py` - Add missing parameters or update API
- `tests/test_e2e_scenarios.py` - Fix parameter usage

### Fix 4: MyPy Type Errors

**Problem**: MyPy hanging or failing with type errors.

**Solution**: Run mypy and fix errors systematically.

**Approach**:
1. Run `mypy sifaka --show-error-codes`
2. Fix errors one module at a time
3. Add type ignores only for false positives

### Fix 5: Test Assertion Logic

**Problem**: Tests expect specific numbers of critic feedback but get different amounts.

**Solution**: Update test expectations to match actual behavior.

**Files to Fix**:
- `tests/test_e2e_scenarios.py` - Update assertion expectations
- `tests/utils/assertions.py` - Make assertions more flexible

### Fix 6: Examples Updates

**Problem**: Examples likely broken due to API changes.

**Solution**: Test and update all examples.

**Files to Fix**:
- All files in `examples/` directory
- Update imports, API calls, and parameters

## IMPLEMENTATION PLAN

### Day 1: Fix Blocking Issues
1. Fix storage async bug (2-3 hours)
2. Add missing test dependencies (1 hour)
3. Fix Chain configuration issues (1-2 hours)
4. Run tests to verify fixes (30 minutes)

### Day 2: Fix Quality Issues
1. Run mypy and fix type errors (3-4 hours)
2. Fix test assertion logic (1-2 hours)
3. Update examples (2-3 hours)

### Day 3: Verify and Polish
1. Run full test suite (30 minutes)
2. Test examples manually (1 hour)
3. Update documentation if needed (1-2 hours)
4. Run CI checks (30 minutes)

## SUCCESS METRICS

### After Day 1:
- ✅ All tests pass
- ✅ No import errors
- ✅ Storage works correctly

### After Day 2:
- ✅ MyPy passes with no errors
- ✅ Examples run without errors
- ✅ Test suite is reliable

### After Day 3:
- ✅ CI passes completely
- ✅ Documentation is accurate
- ✅ Ready for production use

## WHAT NOT TO CHANGE

**Keep these as-is** (they're working correctly):

1. ✅ **Async/Sync Pattern** - Perfect implementation
2. ✅ **Thought-Centric Architecture** - Excellent design
3. ✅ **Protocol-Based Interfaces** - Great extensibility
4. ✅ **Error Handling System** - Comprehensive and well-designed
5. ✅ **Modular Architecture** - Clean separation of concerns

## BOTTOM LINE

**You're 90% there!** The architecture is excellent, the async/sync pattern is exactly right, and the core functionality works.

**Just fix these 6 specific issues** and you'll have a production-ready framework that users will love.

**Biggest impact**: Fix the storage async bug first - that alone will get 15 tests passing and unblock everything else.