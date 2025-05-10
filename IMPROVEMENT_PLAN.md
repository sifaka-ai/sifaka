# Sifaka Chain System Improvement Plan

This document outlines the plan for refactoring the Sifaka chain system to create a more maintainable, extensible, and user-friendly architecture without maintaining backward compatibility.

## 1. New Architecture Design

### 1.1 Core Design Principles

- **Simplicity**: Minimize the number of components and dependencies
- **Modularity**: Clear separation of concerns with well-defined interfaces
- **Extensibility**: Easy to extend with plugins and custom components
- **Discoverability**: Self-documenting API with clear extension points
- **Consistency**: Uniform patterns for state management, error handling, and configuration

### 1.2 Component Architecture

The new architecture will consist of the following core components:

```
Chain
├── Engine (core execution logic)
│   ├── Executor (handles execution flow)
│   └── StateTracker (centralized state management)
├── Components (pluggable components)
│   ├── Model (text generation)
│   ├── Validator (rule-based validation)
│   ├── Improver (output improvement)
│   └── Formatter (result formatting)
└── Plugins (extension mechanism)
    ├── PluginRegistry (plugin discovery and registration)
    └── PluginLoader (dynamic plugin loading)
```

#### 1.2.1 Chain

The main user-facing class that provides a simple interface for running chains. It delegates to the Engine for execution and manages the overall configuration.

#### 1.2.2 Engine

The core execution engine that coordinates the flow between components. It consists of:

- **Executor**: Handles the execution flow, including retries and error handling
- **StateTracker**: Centralizes state management across all components

#### 1.2.3 Components

Pluggable components that implement specific functionality:

- **Model**: Generates text based on prompts
- **Validator**: Validates outputs against rules
- **Improver**: Improves outputs based on validation results
- **Formatter**: Formats results for presentation

#### 1.2.4 Plugins

Extension mechanism for adding custom components:

- **PluginRegistry**: Discovers and registers plugins
- **PluginLoader**: Dynamically loads plugins at runtime

### 1.3 Execution Flow

The simplified execution flow will be:

1. **Initialization**: Create chain with components and configuration
2. **Execution**: Run chain with input prompt
   - Format prompt
   - Generate output using model
   - Validate output
   - Improve output if needed
   - Format result
3. **Result**: Return standardized result object

### 1.4 State Management

State management will be centralized in the StateTracker, which will:

- Track state across all components
- Provide a consistent interface for state access
- Support state snapshots for debugging
- Enable state persistence for long-running chains

### 1.5 Error Handling

Error handling will be standardized with:

- Consistent error hierarchy
- Centralized error handling in the Engine
- Detailed error information for debugging
- Graceful degradation when possible

### 1.6 Configuration

Configuration will be simplified with:

- Single configuration object for the entire chain
- Component-specific configuration sections
- Sensible defaults for most options
- Validation of configuration values

## 2. Core Interfaces

### 2.1 Chain Interface

```python
class Chain:
    """Main interface for running chains."""
    
    def __init__(
        self,
        model: Model,
        validators: List[Validator] = None,
        improver: Improver = None,
        formatter: Formatter = None,
        config: ChainConfig = None,
    ):
        """Initialize the chain with components and configuration."""
        
    def run(self, prompt: str) -> ChainResult:
        """Run the chain on the given prompt."""
        
    def run_async(self, prompt: str) -> Awaitable[ChainResult]:
        """Run the chain asynchronously."""
```

### 2.2 Component Interfaces

#### 2.2.1 Model Interface

```python
class Model(Protocol):
    """Interface for text generation models."""
    
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        
    async def generate_async(self, prompt: str) -> str:
        """Generate text asynchronously."""
```

#### 2.2.2 Validator Interface

```python
class Validator(Protocol):
    """Interface for output validators."""
    
    def validate(self, output: str) -> ValidationResult:
        """Validate an output."""
        
    async def validate_async(self, output: str) -> ValidationResult:
        """Validate an output asynchronously."""
```

#### 2.2.3 Improver Interface

```python
class Improver(Protocol):
    """Interface for output improvers."""
    
    def improve(
        self, 
        output: str, 
        validation_results: List[ValidationResult]
    ) -> str:
        """Improve an output based on validation results."""
        
    async def improve_async(
        self, 
        output: str, 
        validation_results: List[ValidationResult]
    ) -> str:
        """Improve an output asynchronously."""
```

#### 2.2.4 Formatter Interface

```python
class Formatter(Protocol):
    """Interface for result formatters."""
    
    def format(
        self, 
        output: str, 
        validation_results: List[ValidationResult]
    ) -> ChainResult:
        """Format a result."""
        
    async def format_async(
        self, 
        output: str, 
        validation_results: List[ValidationResult]
    ) -> ChainResult:
        """Format a result asynchronously."""
```

### 2.3 Plugin Interface

```python
class Plugin(Protocol):
    """Interface for plugins."""
    
    @property
    def name(self) -> str:
        """Get the plugin name."""
        
    @property
    def version(self) -> str:
        """Get the plugin version."""
        
    @property
    def component_type(self) -> str:
        """Get the component type this plugin provides."""
        
    def create_component(self, config: Dict[str, Any]) -> Any:
        """Create a component instance."""
```

## 3. Migration Path

### 3.1 Phase 1: Core Implementation

1. Create the new package structure
2. Implement the core interfaces
3. Implement the Chain and Engine classes
4. Implement the StateTracker
5. Create basic implementations of components

### 3.2 Phase 2: Component Implementation

1. Implement Model adapters for existing model providers
2. Implement Validator adapters for existing rules
3. Implement Improver adapters for existing critics
4. Implement Formatter for standardized results

### 3.3 Phase 3: Plugin System

1. Implement the PluginRegistry
2. Implement the PluginLoader
3. Create example plugins
4. Document the plugin API

### 3.4 Phase 4: Migration Utilities

1. Create migration guide for users
2. Implement utility functions to help with migration
3. Create examples showing how to migrate from old to new system

### 3.5 Phase 5: Documentation and Testing

1. Create comprehensive documentation
2. Write unit tests for all components
3. Write integration tests for the entire system
4. Create example applications

## 4. Implementation Timeline

### Week 1: Core Architecture

- Day 1-2: Design and planning
- Day 3-4: Core interfaces and base classes
- Day 5: State management and error handling

### Week 2: Components and Plugins

- Day 1-2: Component implementations
- Day 3-4: Plugin system
- Day 5: Testing and refinement

### Week 3: Documentation and Examples

- Day 1-2: API documentation
- Day 3-4: Migration guide and examples
- Day 5: Final testing and release preparation

## 5. Key Benefits

1. **Simplified API**: Easier to understand and use
2. **Reduced Complexity**: Fewer components and dependencies
3. **Improved Extensibility**: Clear extension points and plugin system
4. **Better Error Handling**: Standardized error handling and reporting
5. **Enhanced Testability**: Easier to test individual components
6. **Clearer Documentation**: Self-documenting API with comprehensive docs
7. **Future-Proof Design**: Designed for future extensions and improvements
