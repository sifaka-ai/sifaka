# API Reference

## Core API

### `improve()`

The main function for improving text using AI critics.

```python
async def improve(
    text: str,
    critics: list[str | CriticType] | None = None,
    validators: list[ValidatorPlugin] | None = None,
    model: str | None = None,
    critic_model: str | None = None,
    temperature: float = 0.7,
    max_iterations: int = 3,
    storage: StoragePlugin | None = None,
    config: Config | None = None,
    timeout_seconds: float = 120.0,
) -> SifakaResult
```

#### Parameters

- **text** (`str`): The text to improve
- **critics** (`list[str | CriticType]`, optional): Critics to use for evaluation. Defaults to `[CriticType.SELF_REFINE]`
- **validators** (`list[ValidatorPlugin]`, optional): Validators to check improved text
- **model** (`str`, optional): LLM model to use for generation
- **critic_model** (`str`, optional): LLM model to use for critique
- **temperature** (`float`): Temperature for text generation (0.0-2.0). Default: 0.7
- **max_iterations** (`int`): Maximum improvement iterations (1-10). Default: 3
- **storage** (`StoragePlugin`, optional): Storage backend for results
- **config** (`Config`, optional): Full configuration object
- **timeout_seconds** (`float`): Timeout for the entire operation. Default: 120.0

#### Returns

`SifakaResult`: Object containing:
- `original_text`: The input text
- `final_text`: The improved text
- `iteration`: Number of iterations performed
- `generations`: List of all generated texts
- `critiques`: List of all critique results
- `validations`: List of validation results
- `processing_time`: Total processing time in seconds
- `id`: Unique identifier for the result
- `created_at`: Timestamp of creation

#### Example

```python
import asyncio
from sifaka import improve
from sifaka.core.types import CriticType

async def main():
    result = await improve(
        "AI is revolutionary technology.",
        critics=[CriticType.SELF_REFINE, CriticType.REFLEXION],
        max_iterations=3
    )
    print(result.final_text)

asyncio.run(main())
```

## Types

### `CriticType`

Enum of available built-in critics:

```python
class CriticType(str, Enum):
    SELF_REFINE = "self_refine"
    REFLEXION = "reflexion"
    CONSTITUTIONAL = "constitutional"
    SELF_CONSISTENCY = "self_consistency"
    SELF_RAG = "self_rag"
    SELF_TAUGHT_EVALUATOR = "self_taught_evaluator"
    AGENT4DEBATE = "agent4debate"
    STYLE = "style"
    META_REWARDING = "meta_rewarding"
    N_CRITICS = "n_critics"
    PROMPT = "prompt"
```

### `SifakaResult`

Result object from text improvement:

```python
class SifakaResult(BaseModel):
    original_text: str
    final_text: str
    iteration: int
    generations: list[Generation]
    critiques: list[CritiqueResult]
    validations: list[ValidationResult]
    processing_time: float
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
```

### `Generation`

Information about a single text generation:

```python
class Generation(BaseModel):
    iteration: int
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

### `CritiqueResult`

Result from a critic evaluation:

```python
class CritiqueResult(BaseModel):
    critic: str
    feedback: str
    suggestions: list[str]
    needs_improvement: bool
    confidence: float
```

### `ValidationResult`

Result from a validator:

```python
class ValidationResult(BaseModel):
    validator: str
    passed: bool
    score: float
    details: str | None = None
```

## Configuration

### `Config`

Main configuration object:

```python
class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
```

### `LLMConfig`

LLM-specific configuration:

```python
class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    critic_model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout_seconds: float = 60.0
```

### `CriticConfig`

Critic configuration:

```python
class CriticConfig(BaseModel):
    critics: list[CriticType] = Field(default_factory=lambda: [CriticType.SELF_REFINE])
    critic_model: str | None = None
    confidence_threshold: float = 0.6
```

### `EngineConfig`

Engine configuration:

```python
class EngineConfig(BaseModel):
    max_iterations: int = 3
    parallel_critics: bool = False
    timeout_seconds: float = 120.0
```

## Storage

### `StoragePlugin`

Base class for storage backends:

```python
class StoragePlugin(ABC):
    @abstractmethod
    async def save(self, result: SifakaResult) -> str:
        """Save a result and return its ID."""

    @abstractmethod
    async def load(self, result_id: str) -> SifakaResult | None:
        """Load a result by ID."""

    @abstractmethod
    async def delete(self, result_id: str) -> bool:
        """Delete a result by ID."""

    @abstractmethod
    async def list_results(self, limit: int = 100) -> list[str]:
        """List available result IDs."""
```

### Built-in Storage

- `MemoryStorage`: In-memory storage (default)
- `FileStorage`: File-based storage
- `RedisStorage`: Redis-based storage
- `MultiStorage`: Multiple backends with fallback

## Validators

### `ValidatorPlugin`

Base class for validators:

```python
class ValidatorPlugin(ABC):
    @abstractmethod
    async def validate(self, text: str) -> ValidationResult:
        """Validate text and return result."""
```

### Built-in Validators

- `LengthValidator`: Validates text length
- `ContentValidator`: Validates required/forbidden terms
- `GuardrailsValidator`: Integration with Guardrails AI

## Critics

### `CriticPlugin`

Base class for custom critics:

```python
class CriticPlugin(ABC):
    @abstractmethod
    async def critique(
        self,
        text: str,
        result: SifakaResult
    ) -> CritiqueResult:
        """Critique text and return feedback."""
```

See the [Plugin Development](../plugin_development.md) guide for creating custom critics.

## Exceptions

### `SifakaError`

Base exception for all Sifaka errors:

```python
class SifakaError(Exception):
    """Base exception for Sifaka errors."""
```

### `ConfigError`

Configuration-related errors:

```python
class ConfigError(SifakaError):
    """Raised when configuration is invalid."""
```

### `CriticError`

Critic-related errors:

```python
class CriticError(SifakaError):
    """Raised when a critic fails."""
```

### `ValidationError`

Validation-related errors:

```python
class ValidationError(SifakaError):
    """Raised when validation fails."""
```

### `TimeoutError`

Timeout errors:

```python
class TimeoutError(SifakaError):
    """Raised when operation times out."""
```
