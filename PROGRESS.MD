# Sifaka Refactoring Progress

## Important Note: NO Backward Compatibility

**This refactoring will NOT maintain backward compatibility with previous versions.**
We are prioritizing clean architecture, proper dependency management, and code quality over backward compatibility.
Breaking changes will be introduced as needed to solve structural issues in the codebase.

## Key Focus Areas

### 1. Circular Import Issues

Current Status: ✅ Successfully implemented a registry-based solution to break circular dependencies in the factory system. This initial implementation provides a blueprint for addressing other circular dependency issues.

#### Plan:
- [x] Identify all circular import patterns in the codebase
- [ ] Map dependency graph of key modules
- [x] Categorize circular dependencies by type/cause
- [x] Prioritize critical circular imports to resolve
- [x] Develop architectural solutions rather than workarounds
- [x] Implement solutions incrementally (started with factory system)
- [ ] Verify each change with comprehensive tests

#### Progress:
- Initial code review done, confirmed circular imports are being handled primarily through lazy loading
- Key modules with circular dependencies include:
  - `sifaka/interfaces/chain/__init__.py`
  - `sifaka/core/factories.py`
  - `sifaka/chain/factories.py`
- **[COMPLETED]** Created a component registry system in `sifaka/core/registry.py` to break circular dependencies in the factory system
- **[COMPLETED]** Refactored all factory functions in `sifaka/core/factories.py` to use the registry:
  - `create_critic`
  - `create_rule`
  - `create_classifier`
  - `create_model_provider`
  - `create_retriever`
- **[COMPLETED]** Created a registry initialization system:
  - Created `sifaka/core/initialize_registry.py`
  - Updated `sifaka/__init__.py` to initialize registry at startup
  - Added environment variable toggle for registry initialization

#### Implementation Details for Factory Registry Pattern:

1. **Created Component Registry System**:
   - Created a new `sifaka/core/registry.py` module with registry storage
   - Implemented registration functions for different component types
   - Implemented retrieval functions to get factory functions by type and name
   - Added helper functions for common component types

2. **Updated Factory Functions**:
   - Replaced direct imports with registry lookups in all factory functions
   - Added fallback mechanisms for backward compatibility during transition
   - Standardized error handling for missing factories

3. **Registration Mechanism**:
   - Added registration code to the critics implementation
   - Placed registration at module level to ensure it happens on import
   - Registration happens only once per module, avoiding duplicate registrations

4. **Automatic Initialization**:
   - Created initialization module to import all implementation files
   - Added startup code in package `__init__.py` to ensure registry is populated
   - Added environment variable to disable automatic initialization if needed

#### Analysis of Circular Import Patterns:

1. **TYPE_CHECKING Pattern**:
   - In `sifaka/interfaces/chain/__init__.py`, circular imports are addressed by defining placeholder classes during type checking and using runtime imports elsewhere.
   - Example:
     ```python
     if TYPE_CHECKING:
         class Model(Protocol): ...
     else:
         try:
             from .model import Model
         except ImportError:
             Model = None
     ```

2. **Lazy Imports in Factory Functions**:
   - In `sifaka/core/factories.py`, imports were placed inside functions to avoid circular dependencies.
   - Example:
     ```python
     def create_critic(critic_type: str, ...):
         # Import critic factory functions lazily to avoid circular dependencies
         from sifaka.critics.implementations import (
             create_prompt_critic,
             create_reflexion_critic,
             # ...
         )
     ```
   - **[FIXED]** Replaced with registry-based approach that avoids direct imports

3. **Dependency Injection Workarounds**:
   - Using dependency injection to resolve dependencies at runtime rather than import time.
   - Example in `sifaka/chain/factories.py`:
     ```python
     if model is None:
         from sifaka.core.dependency.provider import DependencyProvider
         provider = DependencyProvider()
         model = provider.get("model_provider", ...)
     ```

#### Root Causes of Circular Dependencies:

1. **Cross-Module Component References**:
   - Components in different modules reference each other, creating circular dependencies.
   - For example, `Chain` depends on `Model`, `Validator`, and `Improver` components, which themselves may reference `Chain`.

2. **Factory Functions and Component Creation**:
   - Factory functions need to import specific implementations, but these implementations import the factory functions.
   - **[FIXED]** Implemented a registry system to break this circular dependency

3. **Interfaces vs. Implementations**:
   - Interfaces import implementations for type checking, and implementations import interfaces for compliance.

#### Next Steps for Circular Import Resolution:

1. Complete the registry pattern implementation:
   - **[COMPLETED]** Update all critic implementations to register with the registry
   - **[COMPLETED]** Apply the same pattern to rules, classifiers, and other components
     - **[COMPLETED]** Applied registry pattern to classifiers implementations
     - **[COMPLETED]** Updated classifier factories to register with the registry
     - **[COMPLETED]** Ensured classifier factory registry is initialized properly
   - **[COMPLETED]** Update all factory functions to use the registry

2. Address chain component dependencies:
   - **[COMPLETED]** Apply similar registry patterns to chain components
   - **[COMPLETED]** Move interface definitions to a separate, dependency-free package

3. Address interface-implementation dependencies:
   - **[COMPLETED]** Implement cleaner separation between interfaces and implementations
   - **[COMPLETED]** Use TYPE_CHECKING more consistently to avoid runtime circular imports

### 2. Interface and Implementation Separation

Current Status: ✅ Successfully implemented interface separation with a dedicated interfaces directory and proper Protocol naming.

#### Plan:
- [x] Review current interface structure in `sifaka/interfaces/`
- [x] Map which implementations use which interfaces
- [x] Identify inconsistencies in interface usage
- [x] Standardize interface definitions and naming conventions
- [x] Ensure implementations properly implement interfaces
- [ ] Introduce static type checking for interface adherence
- [ ] Update documentation to reflect interface design

#### Interface Standardization Completed:
- Standardized all interface names to consistently use the `Protocol` suffix
- Renamed `Rule` to `RuleProtocol`, `Validatable` to `ValidatableProtocol`, etc.
- Renamed `Retriever` to `RetrieverProtocol`, `DocumentStore` to `DocumentStoreProtocol`, etc.
- Renamed `LanguageModelProtocol` to `LanguageModelProviderProtocol` for clarity
- Renamed `RuleProtocol` to `SimpleRuleProtocol` to avoid naming conflicts
- Updated imports in `__init__.py` to reflect new naming conventions
- Removed confusing aliases to improve code clarity
- Reinforced strict no-backward-compatibility stance by removing compatibility modules and aliases
- [x] Ensure implementations properly implement interfaces

#### Implementation Updates Completed:
- Fixed imports in key files that were affected by interface renaming
- Updated type annotations in manager classes to use new interface names
- Created a new dedicated `interfaces/core.py` module to replace the deleted `core/interfaces.py`
- Updated imports across the codebase to use the new interface names
- Fixed circular import issues between core and interfaces packages
- Removed backward compatibility aliases in __init__.py files as per project policy
- Deleted backward compatibility files (classifiers/interfaces.py)
- Successfully fixed all import issues - the package now imports cleanly
- Fixed PluginProtocol to include required component_type property
- Implemented a basic Engine class to fix broken imports in classifiers module
- Fixed type checking errors with mypy:
  - Fixed TypeVar covariance issues in core protocols
  - Updated Chain interface to use new core protocols
  - Updated imports to use Protocol interfaces consistently

#### Progress:
- Initial review shows a dedicated interfaces directory
- Current interfaces include component, model, rule, critic, chain, classifier, and retrieval protocols
- Examined model provider implementations to understand how interfaces are used

#### Analysis of Interface Structure:

1. **Protocol-Based Interfaces**:
   - The code uses Python's Protocol class for structural typing (duck typing).
   - Example:
     ```python
     @runtime_checkable
     class ModelProviderProtocol(Protocol):
         def generate(self, prompt: str, **kwargs: Any) -> str: ...
     ```

2. **Interface Hierarchy**:
   - The interfaces have a clear hierarchy:
     - `ComponentProtocol` as the base interface
     - Specific interfaces like `ModelProviderProtocol`, `RuleProtocol`, etc.
   - There's separation between chain-specific interfaces and general interfaces.

3. **Interface Versioning**:
   - Some interfaces have different versions, indicated by naming patterns like `ChainValidatorProtocol` vs `ValidatorProtocol`.
   - This suggests interface evolution over time.

4. **Adapter Pattern**:
   - The code uses adapter classes to adapt various implementations to the required interfaces.
   - Example in `sifaka/chain/factories.py`:
     ```python
     adapted_model = model if isinstance(model, Model) else ModelAdapter(model)
     ```

#### Key Implementations Using Interfaces:

1. **ModelProviderCore (sifaka/models/core/provider.py)**:
   - Implements `ModelProviderProtocol`
   - Delegates to specialized components for better separation of concerns
   - Used as a base class for specific provider implementations (e.g., OpenAIProvider)

2. **OpenAIProvider (sifaka/models/providers/openai.py)**:
   - Extends `ModelProviderCore` to provide OpenAI-specific functionality
   - Implements concrete functionality for abstract methods

3. **Model Provider Base Class (sifaka/models/base/provider.py)**:
   - Abstract base class implementation of `ModelProviderProtocol`
   - Provides common functionality for all model providers

#### Current Implementation Issues:

1. **Inconsistent Interface Naming**:
   - Some interfaces use `Protocol` suffix (e.g., `ModelProviderProtocol`)
   - Others don't (e.g., `Rule`, `Validator`)
   - Inconsistent use of Protocol vs Abstract Base Classes

2. **Duplicate Interface Definitions**:
   - Some interfaces seem to be defined in multiple places
   - Example: Chain-specific vs. general interfaces

3. **Ambiguous Interface Compliance**:
   - Runtime type checking used inconsistently
   - Static type checking not uniformly applied
   - Mix of structural typing (Protocol) and nominal typing (inheritance)

#### Standardization Plan:

1. **Consistent Naming Conventions**:
   - Use `Protocol` suffix for all protocol interfaces
   - Use descriptive names without suffix for abstract base classes
   - Standardize implementation naming patterns

2. **Unified Interface Location**:
   - All interfaces should be in the `sifaka/interfaces/` directory
   - Group related interfaces in meaningful subdirectories
   - Remove duplicate interface definitions

3. **Type Checking Enforcement**:
   - Use runtime checkable protocols where appropriate
   - Enforce static type checking with mypy
   - Provide type checking documentation

### 3. Dependency Injection System

Current Status: A dependency injection system exists with mechanisms to detect circular dependencies.

#### Plan:
- [x] Review the current dependency injection implementation
- [x] Assess how it relates to the circular import issues
- [x] Identify strengths and weaknesses of current approach
- [ ] Plan improvements to make DI more robust
- [ ] Consider whether DI can help solve circular import issues

#### Progress:
- Initial code review identified a dependency provider system in `sifaka/core/dependency/provider.py`
- The system includes cycle detection for dependencies
- Dependency injection is used as a workaround for circular imports in factory functions
- The DI system has dependency scope management (singleton, session, request)

#### Analysis of Dependency Injection System:

1. **Implementation**:
   - Singleton pattern for the DependencyProvider
   - Support for different scopes (singleton, session, request)
   - Factory function support
   - Cycle detection during registration

2. **Usage Patterns**:
   - Used primarily in factory functions to resolve dependencies
   - Used to manage component lifecycles
   - Used as a workaround for circular imports

3. **Strengths**:
   - Comprehensive scope management
   - Dependency graph validation
   - Flexible factory function support
   - Proper error handling

4. **Weaknesses**:
   - Used as a workaround rather than solving root causes
   - Complex implementation with multiple responsibilities
   - Not consistently used across the codebase
   - May introduce runtime errors that could be caught at compile time

### 4. Response Parsing Issues

Current Status: Multiple response parsers throughout the codebase may have issues with extracting information from model outputs.

#### Plan:
- [ ] Identify all response parsing implementations
- [ ] Check for consistency in parsing approaches
- [ ] Test parsers with various model response formats
- [ ] Develop a unified parsing strategy
- [ ] Consolidate parsers where appropriate

#### Progress:
- Response parsing logic found in:
  - `sifaka/core/managers/response.py`
  - `sifaka/critics/managers/response.py`
  - `sifaka/critics/strategies/improvement.py`