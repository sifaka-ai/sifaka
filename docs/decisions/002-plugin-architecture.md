# ADR-002: Plugin Architecture for Critics and Validators

## Status
Accepted

## Context
Sifaka needs to support multiple text improvement strategies (critics) and quality validators. The system should be extensible to allow:
- Adding new critics without modifying core code
- Custom validators for specific use cases
- Third-party extensions
- Easy configuration of which critics to use

We considered several approaches:
1. Hard-coded critics in the core library
2. Registry-based plugin system
3. Entry point-based plugin discovery
4. Configuration-driven plugin loading

## Decision
We will implement a registry-based plugin system with both automatic discovery and manual registration.

```python
# Automatic registration (in plugin modules)
from sifaka.critics import register_critic

@register_critic("my_critic")
class MyCritic(BaseCritic):
    # implementation

# Manual registration
from sifaka.critics import CriticRegistry
CriticRegistry.register("custom_critic", CustomCritic)

# Usage
result = await improve("text", critics=["my_critic", "custom_critic"])
```

## Rationale
1. **Extensibility**: Easy to add new critics without core changes
2. **Modularity**: Critics can be developed independently
3. **Discoverability**: Registry allows listing available critics
4. **Configuration**: Users can easily choose which critics to use
5. **Testing**: Each critic can be tested in isolation

## Design Principles
- **Protocol-based**: Critics implement a common interface
- **Composable**: Multiple critics can be combined
- **Configurable**: Each critic can have its own configuration
- **Isolated**: Critics should not depend on each other

## Implementation Details

### Base Classes
```python
class BaseCritic(ABC):
    @abstractmethod
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
```

### Registry System
```python
class CriticRegistry:
    _critics: Dict[str, Type[BaseCritic]] = {}

    @classmethod
    def register(cls, name: str, critic_class: Type[BaseCritic]):
        cls._critics[name] = critic_class

    @classmethod
    def get(cls, name: str) -> Type[BaseCritic]:
        return cls._critics.get(name)
```

### Plugin Discovery
- Automatic registration through decorators
- Manual registration for dynamic loading
- Configuration-based critic selection
- Validation of critic implementations

## Consequences

### Positive
- Easy to extend with new critics
- Clean separation of concerns
- Testable and maintainable
- Supports both built-in and custom critics
- Enables community contributions

### Negative
- Additional complexity in the core system
- Plugin discovery overhead
- Potential version compatibility issues
- Need for plugin documentation standards

### Mitigation
- Provide clear base classes and interfaces
- Document plugin development guidelines
- Implement plugin validation
- Version compatibility checking
- Comprehensive examples and templates

## Built-in Critics
The system will include several built-in critics:
- **ReflexionCritic**: Self-reflection based improvement
- **ConstitutionalCritic**: Principle-based evaluation
- **SelfRefineCritic**: Iterative refinement
- **SelfRAGCritic**: Retrieval-augmented critique
- **NCriticsCritic**: Ensemble of multiple critics

## Related Decisions
- [ADR-001: Single Function API](001-single-function-api.md)
- [ADR-003: Memory Management](003-memory-management.md)
- [ADR-004: Error Handling](004-error-handling.md)
