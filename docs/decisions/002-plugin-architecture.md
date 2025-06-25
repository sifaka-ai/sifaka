# ADR-002: Plugin Architecture Design

## Status
Accepted

## Context
Sifaka needs to be extensible to support various use cases without bloating the core library. Users should be able to:
- Add custom storage backends (Redis, PostgreSQL, S3, etc.)
- Create domain-specific critics (academic writing, legal documents, etc.)
- Implement custom validators (compliance checks, format validation, etc.)
- Integrate with existing infrastructure

The plugin system must balance flexibility with simplicity, maintaining Sifaka's core principle of being easy to use.

## Decision
We will implement a plugin architecture based on:

1. **Abstract Base Classes**: Define clear interfaces for each plugin type
2. **Registration System**: Simple API for registering plugins at runtime
3. **Entry Points**: Support automatic discovery via setuptools entry points
4. **Type Safety**: Maintain full type checking for plugin interfaces

### Plugin Types

#### Storage Backends
```python
class StorageBackend(ABC):
    async def save(self, result: SifakaResult) -> str
    async def load(self, result_id: str) -> Optional[SifakaResult]
    async def list(self, limit: int, offset: int) -> List[str]
    async def delete(self, result_id: str) -> bool
    async def search(self, query: str, limit: int) -> List[str]
```

#### Critics
```python
class BaseCritic(ABC):
    def _build_critique_prompt(self, original_text: str, current_text: str, attempt: int) -> str
    def _build_improvement_prompt(self, original_text: str, current_text: str, critique: str, attempt: int) -> str
```

#### Validators
```python
class BaseValidator(ABC):
    async def validate(self, text: str, metadata: Optional[Dict[str, Any]]) -> ValidationResult
```

### Registration Methods

1. **Manual Registration**:
```python
register_storage_backend("redis", RedisBackend)
register_critic("academic", AcademicCritic)
register_validator("seo", SEOValidator)
```

2. **Automatic Discovery**:
```python
# In setup.py
entry_points={
    "sifaka.storage": ["redis = mypackage:RedisBackend"],
    "sifaka.critics": ["academic = mypackage:AcademicCritic"],
    "sifaka.validators": ["seo = mypackage:SEOValidator"],
}
```

## Consequences

### Positive
- **Extensibility**: Easy to add new functionality without modifying core
- **Separation of Concerns**: Plugins isolated from core logic
- **Type Safety**: Full typing preserved through ABC inheritance
- **Discovery**: Automatic plugin loading via entry points
- **Testing**: Plugins can be tested independently

### Negative
- **Complexity**: Additional abstraction layer
- **Documentation**: Need to maintain plugin development guides
- **Compatibility**: Must maintain backward compatibility for plugin interfaces
- **Validation**: Need to validate plugins meet interface requirements

### Mitigation Strategies
- Comprehensive plugin examples
- Strict versioning for interface changes
- Runtime validation of plugin compliance
- Clear documentation and migration guides

## Implementation Notes

1. Use `pkg_resources` for entry point discovery (with fallback for environments without it)
2. Validate plugin inheritance at registration time
3. Provide factory functions for plugin instantiation
4. Include comprehensive examples for each plugin type
5. Support both sync and async operations where appropriate

## References
- [Setuptools Entry Points](https://setuptools.pypa.io/en/latest/userguide/entry_point.html)
- [Python Plugin Architecture Patterns](https://python-patterns.guide/gang-of-four/abstract-factory/)
- [Type-Safe Plugin Systems](https://mypy.readthedocs.io/en/stable/protocols.html)
