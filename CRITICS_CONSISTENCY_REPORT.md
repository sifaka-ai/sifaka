# Critics Consistency Report

## Critical Issues Found

### 1. **Method Interface Mismatch** 🚨
- **BaseCritic expects**: `_create_messages(text, result) -> List[Dict[str, str]]`
- **All critics implement**: `_generate_critique(text, result) -> str`
- This is a fundamental interface mismatch that will cause runtime errors

### 2. **Config Class Mismatch** 🚨
- **Critics use**: `CriticConfig` from `.core.config`
- **Should use**: Unified `Config` class from `core.config`
- Critics are using a config class that may not exist or be outdated

### 3. **Non-existent Methods Referenced** ⚠️
- Critics call `_parse_json_response()` which doesn't exist in BaseCritic
- Critics expect config attributes that don't exist (context_weight, depth_weight, domain_weight)

### 4. **Examples Using Non-existent API** 🚨
- **All examples use**: `Runner` class that doesn't exist
- **Should use**: `improve()` function from the API
- Fixed all 7 examples to use correct API

## Files That Need Updating

### Critics (8 files):
1. `constitutional.py` - Wrong method name, wrong config
2. `meta_rewarding.py` - Wrong method name, wrong config
3. `n_critics.py` - Wrong method name, wrong config
4. `prompt.py` - Wrong method name, wrong config
5. `reflexion.py` - Wrong method name, wrong config
6. `self_consistency.py` - Wrong method name, wrong config, complex async
7. `self_rag.py` - Wrong method name, wrong config
8. `self_refine.py` - Wrong method name, wrong config

### Examples (7 files) - ✅ FIXED:
1. `constitutional_example.py` - Now uses `improve()`
2. `meta_rewarding_example.py` - Now uses `improve()` with validators
3. `n_critics_example.py` - Now uses `improve()`
4. `reflexion_example.py` - Now uses `improve()`
5. `self_consistency_example.py` - Now uses `improve()`
6. `self_rag_example.py` - Now uses `improve()` with storage
7. `self_refine_example.py` - Now uses `improve()`

## Required Actions

### 1. Update All Critics
Each critic needs to:
- Rename `_generate_critique()` to `_create_messages()`
- Return `List[Dict[str, str]]` instead of `str`
- Remove references to `_parse_json_response()`
- Use the unified Config class
- Remove custom config parameters

### 2. Update CriticConfig
Either:
- Update CriticConfig to match what critics expect, OR
- Remove CriticConfig and use unified Config class

### 3. Test Everything
After updates:
- Run all examples to ensure they work
- Test each critic individually
- Verify the interface is consistent

## Impact Assessment

- **High Risk**: Critics won't work at all with current BaseCritic
- **User Impact**: Any code using critics will fail
- **Fix Complexity**: Medium - need to update 8 critic files

This is a critical issue that prevents the critics from functioning properly.