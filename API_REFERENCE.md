# Sifaka API Reference

## Core Functions

### `improve()`

Asynchronously improve text through iterative critique.

```python
async def improve(
    text: str,
    *,
    critics: Optional[List[str]] = None,
    max_iterations: int = 3,
    validators: Optional[List[Validator]] = None,
    config: Optional[Config] = None,
    storage: Optional[StorageBackend] = None,
) -> SifakaResult
```

**Parameters:**
- `text` (str): The text to improve
- `critics` (List[str], optional): List of critic names. Default: ["reflexion"]
- `max_iterations` (int): Maximum improvement iterations (1-10). Default: 3
- `validators` (List[Validator], optional): Quality validators to apply
- `config` (Config, optional): Advanced configuration options
- `storage` (StorageBackend, optional): Storage backend for results

**Returns:**
- `SifakaResult`: Object containing improved text and complete audit trail

**Example:**
```python
result = await improve(
    "Write about AI benefits",
    critics=["reflexion", "constitutional"],
    max_iterations=3
)
```

### `improve_sync()`

Synchronous wrapper for `improve()`.

```python
def improve_sync(
    text: str,
    *,
    critics: Optional[List[str]] = None,
    max_iterations: int = 3,
    validators: Optional[List[Validator]] = None,
    config: Optional[Config] = None,
    storage: Optional[StorageBackend] = None,
) -> SifakaResult
```

**Parameters:** Same as `improve()`

**Returns:** Same as `improve()`

**Example:**
```python
result = improve_sync("Write about climate change")
print(result.final_text)
```

### `improve_advanced()`

Advanced API with all parameters exposed for power users.

```python
async def improve_advanced(
    text: str,
    *,
    max_iterations: int = 3,
    model: str = "gpt-4o-mini",
    critics: Optional[List[str]] = None,
    validators: Optional[List[Validator]] = None,
    temperature: float = 0.7,
    timeout_seconds: int = 300,
    storage: Optional[StorageBackend] = None,
    force_improvements: bool = False,
    show_improvement_prompt: bool = False,
    critic_model: Optional[str] = None,
    critic_temperature: Optional[float] = None,
    retry_config: Optional[RetryConfig] = None,
) -> SifakaResult
```

**Additional Parameters:**
- `model` (str): LLM model to use. Default: "gpt-4o-mini"
- `temperature` (float): Generation temperature (0.0-2.0). Default: 0.7
- `timeout_seconds` (int): Maximum processing time. Default: 300
- `force_improvements` (bool): Always run critics. Default: False
- `show_improvement_prompt` (bool): Print prompts for debugging. Default: False
- `critic_model` (str, optional): Override model for critics
- `critic_temperature` (float, optional): Override temperature for critics
- `retry_config` (RetryConfig, optional): Retry configuration

## Configuration

### `Config`

Configuration object for advanced options.

```python
class Config:
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    timeout_seconds: int = 300
    critic_model: Optional[str] = None
    critic_temperature: Optional[float] = None
    force_improvements: bool = False
    show_improvement_prompt: bool = False
    max_iterations: int = 3
    critics: List[str] = ["reflexion"]
```

**Example:**
```python
config = Config(
    model="gpt-4",
    temperature=0.8,
    force_improvements=True
)

result = await improve("My text", config=config)
```

### `RetryConfig`

Simple retry configuration.

```python
class RetryConfig:
    max_attempts: int = 3
    delay: float = 1.0
    backoff: float = 2.0
```

**Presets:**
- `RETRY_QUICK`: 2 attempts, 0.5s delay, 1.5x backoff
- `RETRY_STANDARD`: 3 attempts, 1s delay, 2x backoff
- `RETRY_PERSISTENT`: 5 attempts, 1s delay, 2x backoff

## Result Objects

### `SifakaResult`

Complete result with audit trail.

```python
class SifakaResult:
    original_text: str          # Original input text
    final_text: str            # Final improved text
    iteration: int             # Number of iterations performed
    generations: Deque[GenerationResult]  # Text versions (max 10)
    critiques: Deque[CritiqueResult]     # Critic feedback (max 10)
    validations: Deque[ValidationResult]  # Validation results (max 10)
    processing_time: float     # Total processing time in seconds
```

**Properties:**
- `current_text`: Get the most recent text version
- `needs_improvement`: Check if critics suggest improvement

### `CritiqueResult`

Individual critic feedback.

```python
class CritiqueResult:
    critic: str                # Name of the critic
    feedback: str              # Main feedback
    suggestions: List[str]     # Specific suggestions
    needs_improvement: bool    # Whether improvement is needed
    confidence: float          # Confidence score (0.0-1.0)
    metadata: Dict[str, Any]   # Additional data
```

## Critics

### Built-in Critics

- **reflexion**: Self-reflection for iterative improvement
- **constitutional**: Principle-based evaluation
- **self_refine**: Iterative self-improvement
- **n_critics**: Ensemble of diverse perspectives
- **self_rag**: Retrieval-augmented critique
- **meta_rewarding**: Two-stage judgment
- **self_consistency**: Consensus through multiple evaluations

### `register_critic()`

Register a custom critic.

```python
def register_critic(
    name: str,
    critic_class: Type[Critic],
    aliases: Optional[List[str]] = None
) -> None
```

**Example:**
```python
from sifaka import BaseCritic, register_critic

class MyCritic(BaseCritic):
    @property
    def name(self) -> str:
        return "my_critic"
    
    async def _create_messages(self, text: str, result):
        return [
            {"role": "system", "content": "You are a helpful critic."},
            {"role": "user", "content": f"Critique: {text}"}
        ]

register_critic("my_critic", MyCritic)
```

## Validators

### Built-in Validators

#### `LengthValidator`

Validates text length constraints.

```python
LengthValidator(
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
)
```

#### `ContentValidator`

Validates content requirements.

```python
ContentValidator(
    required_terms: Optional[List[str]] = None,
    forbidden_terms: Optional[List[str]] = None
)
```

### Custom Validators

Implement the `Validator` interface:

```python
class MyValidator(Validator):
    @property
    def name(self) -> str:
        return "my_validator"
    
    async def validate(self, text: str) -> ValidationResult:
        passed = len(text) > 100
        return ValidationResult(
            validator=self.name,
            passed=passed,
            message="Text too short" if not passed else "OK"
        )
```

## Storage

### `MemoryStorage`

In-memory storage (default).

```python
storage = MemoryStorage(max_results=100)
```

### `FileStorage`

File-based JSON storage.

```python
storage = FileStorage(
    directory="./results",
    max_results=1000
)
```

## Exceptions

- `SifakaError`: Base exception for all Sifaka errors
- `ConfigurationError`: Invalid configuration
- `ModelProviderError`: LLM provider errors
- `RateLimitError`: Rate limit exceeded
- `AuthenticationError`: Invalid API key
- `CriticError`: Critic execution failed
- `ValidationError`: Validation failed
- `TimeoutError`: Operation timed out
- `StorageError`: Storage operation failed

## Environment Variables

```bash
# Required (one of these)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...

# Optional
SIFAKA_DEFAULT_MODEL=gpt-4o-mini
SIFAKA_DEFAULT_TEMPERATURE=0.7
```

## Complete Examples

### Basic Usage

```python
from sifaka import improve_sync

# Simple one-liner
result = improve_sync("Write about renewable energy")
print(result.final_text)
```

### Advanced Usage

```python
import asyncio
from sifaka import improve, ImproveConfig, LengthValidator

async def main():
    config = ImproveConfig(
        model="gpt-4",
        force_improvements=True,
        show_improvement_prompt=True
    )
    
    validators = [
        LengthValidator(min_length=200, max_length=1000)
    ]
    
    result = await improve(
        "Explain quantum computing",
        critics=["reflexion", "constitutional", "meta_rewarding"],
        max_iterations=4,
        validators=validators,
        config=config
    )
    
    print(f"Improved in {result.iteration} iterations")
    print(f"Final confidence: {result.critiques[-1].confidence}")
    print(f"Result: {result.final_text}")

asyncio.run(main())
```

### Custom Critic with Aliases

```python
from sifaka import BaseCritic, register_critic, improve_sync

class AcademicCritic(BaseCritic):
    @property
    def name(self) -> str:
        return "academic"
    
    async def _create_messages(self, text, result):
        return [{
            "role": "system",
            "content": "You are an academic writing expert."
        }, {
            "role": "user",
            "content": f"Review this text for academic quality: {text}"
        }]

# Register with aliases
register_critic("academic", AcademicCritic, ["scholarly", "research"])

# Use with any alias
result = improve_sync("My research findings", critics=["scholarly"])