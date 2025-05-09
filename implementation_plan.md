# Implementation Plan

## 1. Pydantic 2 Migration (1 week)

### Step 1: Create Base Model Configuration
```python
# sifaka/core/models.py
from pydantic import BaseModel, ConfigDict

class BaseModelConfig:
    """Base configuration for all Pydantic models."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True
    )

class BaseModel(BaseModel):
    """Base model for all Sifaka models."""

    model_config = BaseModelConfig.model_config
```

### Step 2: Update Core Models
```python
# sifaka/core/config.py
from .models import BaseModel

class CoreConfig(BaseModel):
    """Core configuration for Sifaka."""

    debug: bool = False
    log_level: str = "INFO"
    trace_enabled: bool = True
```

### Step 3: Update Feature Models
```python
# sifaka/rules/config.py
from ..core.models import BaseModel

class RuleConfig(BaseModel):
    """Configuration for rules."""

    priority: str = "MEDIUM"
    cache_size: int = 0
    cost: int = 1
    params: Dict[str, Any] = Field(default_factory=dict)
```

### Step 4: Remove Legacy Code
- Remove all Pydantic 1.x imports
- Remove compatibility layers
- Remove legacy model configurations
- Update type hints

## 2. Configuration System (1 week)

### Step 1: Create Base Configuration
```python
# sifaka/core/config.py
from .models import BaseModel
from typing import Dict, Any, Optional, TypeVar, Generic

T = TypeVar("T", bound=BaseModel)

class BaseConfig(BaseModel):
    """Base configuration for all components."""

    params: Dict[str, Any] = Field(default_factory=dict)

    def with_params(self, **params: Any) -> "BaseConfig":
        """Create new config with updated params."""
        return self.model_copy(update={"params": {**self.params, **params}})

    def with_options(self, **options: Any) -> "BaseConfig":
        """Create new config with updated options."""
        return self.model_copy(update=options)

class Configurable(Generic[T]):
    """Type-safe configuration handling."""

    def __init__(self, config_class: Type[T]):
        self.config_class = config_class

    def create_config(self, **kwargs: Any) -> T:
        """Create type-safe configuration."""
        return self.config_class(**kwargs)
```

### Step 2: Update Component Configurations
```python
# sifaka/rules/config.py
from ..core.config import BaseConfig

class RuleConfig(BaseConfig):
    """Configuration for rules."""

    priority: str = "MEDIUM"
    cache_size: int = 0
    cost: int = 1

# sifaka/critics/config.py
class CriticConfig(BaseConfig):
    """Configuration for critics."""

    min_confidence: float = 0.8
    max_attempts: int = 3
```

### Step 3: Update Factory Functions
```python
# sifaka/rules/factories.py
from ..core.config import Configurable

def create_rule(
    rule_type: Type[Rule],
    config: Optional[RuleConfig] = None,
    **kwargs: Any
) -> Rule:
    """Create a rule with standardized configuration."""
    configurable = Configurable(RuleConfig)
    config = config or configurable.create_config(**kwargs)
    return rule_type(config=config)
```

### Step 4: Remove Legacy Configuration
- Remove old configuration classes
- Remove configuration compatibility layers
- Update all components to use new system
- Update documentation

## 3. State Management (1 week)

### Step 1: Create State Manager
```python
# sifaka/core/state.py
from typing import Dict, Any, List, Optional
from .models import BaseModel

class State(BaseModel):
    """Immutable state container."""

    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StateManager:
    """Unified state management for all components."""

    def __init__(self):
        self._state: State = State()
        self._history: List[State] = []

    def update(self, key: str, value: Any) -> None:
        """Update state with history tracking."""
        self._history.append(self._state)
        self._state = self._state.model_copy(
            update={"data": {**self._state.data, key: value}}
        )

    def rollback(self) -> None:
        """Rollback to previous state."""
        if self._history:
            self._state = self._history.pop()

    def get(self, key: str) -> Any:
        """Get state value."""
        return self._state.data.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self._state = self._state.model_copy(
            update={"metadata": {**self._state.metadata, key: value}}
        )
```

### Step 2: Create Stateful Component Base
```python
# sifaka/core/component.py
from .state import StateManager

class StatefulComponent:
    """Base class for stateful components."""

    def __init__(self):
        self.state = StateManager()

    def update_state(self, key: str, value: Any) -> None:
        """Update component state."""
        self.state.update(key, value)

    def get_state(self, key: str) -> Any:
        """Get component state."""
        return self.state.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set component metadata."""
        self.state.set_metadata(key, value)
```

### Step 3: Update Components
```python
# sifaka/rules/base.py
from ..core.component import StatefulComponent

class Rule(StatefulComponent):
    """Base class for rules."""

    def __init__(self, config: RuleConfig):
        super().__init__()
        self.config = config
        self.set_metadata("rule_type", self.__class__.__name__)
```

### Step 4: Remove Legacy State Management
- Remove old state management code
- Remove state compatibility layers
- Update all components to use new system
- Update documentation

## Implementation Order

1. **Week 1: Pydantic 2 Migration**
   - Day 1: Create base model configuration
   - Day 2: Update core models
   - Day 3: Update feature models
   - Day 4: Remove legacy code
   - Day 5: Testing and documentation

2. **Week 2: Configuration System**
   - Day 1: Create base configuration
   - Day 2: Update component configurations
   - Day 3: Update factory functions
   - Day 4: Remove legacy configuration
   - Day 5: Testing and documentation

3. **Week 3: State Management**
   - Day 1: Create state manager
   - Day 2: Create stateful component base
   - Day 3: Update components
   - Day 4: Remove legacy state management
   - Day 5: Testing and documentation

## Testing Strategy

1. **Unit Tests**
   - Test all new base classes
   - Test configuration handling
   - Test state management
   - Test model validation

2. **Integration Tests**
   - Test component interactions
   - Test configuration flow
   - Test state management flow
   - Test error handling

3. **Migration Tests**
   - Test component updates
   - Test configuration updates
   - Test state management updates
   - Test error handling updates

## Documentation Updates

1. **API Documentation**
   - Update model documentation
   - Update configuration documentation
   - Update state management documentation
   - Update factory function documentation

2. **Architecture Documentation**
   - Update model architecture
   - Update configuration architecture
   - Update state management architecture
   - Update component architecture

3. **Migration Guide**
   - Document breaking changes
   - Document new patterns
   - Document best practices
   - Document examples

## Success Criteria

1. **Pydantic 2 Migration**
   - All models use Pydantic 2
   - No legacy code remains
   - All type hints are correct
   - All tests pass

2. **Configuration System**
   - All components use new system
   - No legacy configuration remains
   - All factory functions updated
   - All tests pass

3. **State Management**
   - All components use new system
   - No legacy state management remains
   - All state operations work
   - All tests pass

## Risk Mitigation

1. **Breaking Changes**
   - Document all breaking changes
   - Provide clear migration paths
   - Update all examples
   - Update all documentation

2. **Testing**
   - Comprehensive unit tests
   - Comprehensive integration tests
   - Comprehensive migration tests
   - Regular test runs

3. **Documentation**
   - Clear API documentation
   - Clear architecture documentation
   - Clear migration guide
   - Clear examples

## Next Steps

1. Review and approve implementation plan
2. Set up development environment
3. Create feature branches
4. Begin implementation
5. Regular progress updates