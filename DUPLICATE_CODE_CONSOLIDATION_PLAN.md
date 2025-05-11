# Duplicate Code Consolidation Plan

This document outlines the plan for consolidating duplicate code implementations in the Sifaka codebase, as identified in the `REVIEW.md` file under the task "**Consolidate Duplicate Code**: Consolidate duplicate implementations (e.g., memory managers)."

## Important Note

**NO BACKWARD COMPATIBILITY IS REQUIRED**. This is a critical priority. All duplicate code should be removed completely, and imports should be updated to use the consolidated implementations directly.

## Overview of Duplicate Code

Based on our analysis, we've identified several areas of duplicate code that need to be consolidated:

1. **Memory Managers**:
   - `chain/managers/memory.py` contains `ChainMemoryManager` and `ChainBufferMemoryManager`
   - `core/managers/memory.py` contains `KeyValueMemoryManager` and `BufferMemoryManager`

2. **Prompt Managers**:
   - `core/managers/prompt.py` contains the main prompt manager implementations
   - `critics/managers/prompt.py` doesn't exist, but there might be duplicate implementations elsewhere

3. **Result Classes**:
   - `chain/result.py` contains `ChainResult`
   - `chain/formatters/result.py` doesn't exist, but there might be duplicate result classes elsewhere
   - Various components have their own result classes that could be standardized

4. **Error Handling**:
   - `utils/errors.py` and `utils/error_patterns.py` contain overlapping functionality
   - Component-specific error handling could be consolidated

5. **Interfaces**:
   - `interfaces/` directory contains consolidated interfaces
   - Some components still have their own interface definitions (e.g., `critics/interfaces/`)

## Current Status

### Memory Managers
- ✅ `critics/managers/memory.py` doesn't exist, and the critics module already imports memory managers from the core module
- ✅ `chain/managers/memory.py` has been removed, and the chain module now imports memory managers from the core module

### Prompt Managers
- ✅ `critics/managers/prompt.py` doesn't exist, and the critics module already imports prompt managers from the core module
- ✅ No duplicate prompt manager implementations found in other components

### Result Classes
- ✅ `chain/result.py` has been removed, and the chain module now imports result classes from the core module
- ✅ Most result classes have been consolidated in `core/results.py`
- ⚠️ `models/result.py` and `retrieval/result.py` still exist but contain specialized implementations that extend the core result classes

### Error Handling
- ✅ Consolidated all error handling functionality in `utils/errors.py` and removed `utils/error_patterns.py`
- ✅ Updated component-specific error handling to use the consolidated functions

### Interfaces
- ✅ All interface definitions have been moved to the main `interfaces/` directory
- ✅ Empty component-specific interface directories have been removed
- ✅ All components now import interfaces from the main `interfaces/` directory

## Detailed Plan

### 1. Consolidate Memory Managers

1. **Move all memory manager implementations to `core/managers/memory.py`**
   - Keep `KeyValueMemoryManager` and `BufferMemoryManager` as the base implementations
   - Remove `ChainMemoryManager` and `ChainBufferMemoryManager` from `chain/managers/memory.py`
   - Update imports in all files that use these classes

2. **Create factory functions in `core/managers/memory.py`**
   - Implement `create_memory_manager` and `create_buffer_memory_manager` with component-specific parameters
   - These factory functions should replace the component-specific factory functions

3. **Remove `chain/managers/memory.py` completely**
   - Do NOT maintain backward compatibility
   - Update all imports to use core memory managers directly

### 2. Consolidate Error Handling

1. **Consolidate error handling in `utils/errors.py`**
   - Move all error handling functionality from `utils/error_patterns.py` to `utils/errors.py`
   - Ensure consistent error handling patterns across all components

2. **Create standardized error handling functions**
   - Implement `safely_execute` with component-specific parameters
   - These functions should replace the component-specific error handling functions

3. **Update imports in all files that use error handling**
   - Replace imports from `utils/error_patterns.py` with imports from `utils/errors.py`

### 3. Consolidate Result Classes

1. **Move all result base classes to `utils/base_results.py`**
   - Keep `BaseResult` and `BaseRuleResult` as the base implementations
   - Ensure all component-specific result classes inherit from these base classes

2. **Standardize component-specific result classes**
   - Ensure `ChainResult`, `ClassificationResult`, `RuleResult`, etc. follow a consistent pattern
   - Remove any duplicate result classes

3. **Create factory functions in `utils/results.py`**
   - Implement `create_result` with component-specific parameters
   - These factory functions should replace the component-specific factory functions

4. **Create a consolidated result module in `core/results.py`**
   - Move all result classes to this central location
   - Provide standardized factory functions for all result types
   - Update imports across the codebase to use this consolidated module

### 4. Consolidate Interfaces

1. **Ensure all interfaces are in the `interfaces/` directory**
   - Move any remaining interfaces from component-specific directories to `interfaces/`
   - Update imports in all files that use these interfaces

2. **Remove duplicate interface definitions**
   - Remove duplicate interfaces from `critics/interfaces/`, etc.
   - Update imports in all files that use these interfaces

## Progress Tracking
**NO BACKWARD COMPATIBILITY ALLOWED**. This is a critical priority. DO NOT FORGET THIS!!!!!!
| Task | Status | Notes |
|------|--------|-------|
| **Memory Managers** | ✅ Completed | |
| Move memory manager implementations to core | ✅ Completed | Core implementations already existed |
| Create factory functions in core | ✅ Completed | Updated factory functions to support component-specific defaults |
| Remove chain/managers/memory.py completely | ✅ Completed | Removed duplicate file and updated imports |
| **Error Handling** | ✅ Completed | |
| Consolidate error handling in utils/errors.py | ✅ Completed | Moved all functionality from utils/error_patterns.py to utils/errors.py |
| Create standardized error handling functions | ✅ Completed | Added safely_execute_component and other functions to utils/errors.py |
| Update imports in all files | ✅ Completed | Updated imports in critics/utils.py, rules/validators.py, and rules/base.py |
| **Result Classes** | ✅ Completed | |
| Move result base classes to utils/base_results.py | ✅ Completed | Moved CriticMetadata from utils/config.py to utils/base_results.py |
| Standardize component-specific result classes | ✅ Completed | Updated ChainResult, ClassificationResult, and RuleResult to inherit from BaseResult |
| Create factory functions in utils/results.py | ✅ Completed | Added create_generic_result function to utils/results.py |
| Create consolidated result module in core/results.py | ✅ Completed | Created core/results.py with standardized result classes and factory functions |
| Update imports to use consolidated result module | ✅ Completed | Updated all imports across the codebase to use core/results.py |
| **Interfaces** | ✅ Completed | |
| Ensure all interfaces are in interfaces/ directory | ✅ Completed | Moved all interfaces from component-specific directories to main interfaces directory |
| Remove duplicate interface definitions | ✅ Completed | Removed duplicate interface directories and updated imports |
| Remove empty interface directories | ✅ Completed | Removed empty component-specific interface directories |

## Next Steps

1. ✅ Consolidate memory managers (COMPLETED)
2. ✅ Consolidate error handling (COMPLETED)
3. ✅ Consolidate result classes (COMPLETED)
   - ✅ Created consolidated result module in core/results.py
   - ✅ Updated all imports across the codebase to use the consolidated module
   - ✅ Removed duplicate result class files (chain/result.py, classifiers/result.py, rules/result.py)
   - ⚠️ Note: `models/result.py` and `retrieval/result.py` still exist but contain specialized implementations that extend the core result classes
4. ✅ Consolidate interfaces (COMPLETED)
5. ✅ Complete all duplicate code consolidation tasks (COMPLETED)
