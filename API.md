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
from sifaka import improve

result = await improve(
    "Write about AI benefits",
    critics=["reflexion", "constitutional"],
    max_iterations=3
)
print(result.final_text)
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

**Example:**
```python
from sifaka import improve_sync

result = improve_sync("Write about climate change")
print(result.final_text)
```

## Critics

### Available Critics

- **`reflexion`** - Self-reflection and learning from mistakes (default)
- **`constitutional`** - Principle-based ethical evaluation
- **`self_refine`** - Iterative self-improvement
- **`n_critics`** - Multi-perspective ensemble critique
- **`self_rag`** - Retrieval-augmented critique for factual accuracy
- **`meta_rewarding`** - Two-stage judgment with meta-evaluation
- **`self_consistency`** - Consensus-based evaluation
- **`prompt`** - Custom prompt-based critique

### Using Critics

```python
# Single critic
result = await improve(text, critics=["reflexion"])

# Multiple critics
result = await improve(text, critics=["reflexion", "constitutional"])

# Custom critic configuration
from sifaka.critics import PromptCritic

custom_critic = PromptCritic(
    custom_prompt="Evaluate for technical accuracy and clarity"
)
result = await improve(text, critics=[custom_critic])
```

## Validators

### Built-in Validators

#### LengthValidator
```python
from sifaka.validators import LengthValidator

validator = LengthValidator(min_length=100, max_length=1000)
result = await improve(text, validators=[validator])
```

#### ContentValidator
```python
from sifaka.validators import ContentValidator

validator = ContentValidator(
    required_keywords=["methodology", "results"],
    forbidden_words=["maybe", "perhaps"],
    min_sentences=5
)
result = await improve(text, validators=[validator])
```

#### FormatValidator
```python
from sifaka.validators import FormatValidator

validator = FormatValidator(
    require_punctuation=True,
    allow_urls=True,
    max_caps_ratio=0.1  # Max 10% caps
)
result = await improve(text, validators=[validator])
```

#### PatternValidator
```python
from sifaka.validators import PatternValidator

# Validate email format
email_validator = PatternValidator(
    pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    description="Must include valid email"
)
result = await improve(text, validators=[email_validator])
```

#### NumericRangeValidator
```python
from sifaka.validators import NumericRangeValidator

# Validate word count
word_count_validator = NumericRangeValidator(
    min_value=50,
    max_value=100,
    value_extractor=lambda text: len(text.split()),
    description="Word count between 50-100"
)
result = await improve(text, validators=[word_count_validator])
```

### Validator Factories

```python
from sifaka.validators import (
    create_percentage_validator,
    create_price_validator,
    create_age_validator
)

# Validate percentages (0-100%)
percentage_val = create_percentage_validator()

# Validate prices in range
price_val = create_price_validator(min_price=10.0, max_price=1000.0)

# Validate ages
age_val = create_age_validator(min_age=18, max_age=65)
```

## Configuration

### Config Class

```python
from sifaka.core.config import Config

config = Config(
    model="gpt-4",
    temperature=0.7,
    timeout_seconds=60,
    max_iterations=5,
    force_improvements=True,
    show_improvement_prompt=True,
    critic_model="gpt-4o-mini",
    critic_temperature=0.3
)

result = await improve(text, config=config)
```

**Config Parameters:**
- `model` (str): LLM model to use. Default: "gpt-4o-mini"
- `temperature` (float): Generation temperature (0.0-2.0). Default: 0.7
- `timeout_seconds` (int): Maximum processing time. Default: 300
- `max_iterations` (int): Max improvement cycles. Default: 3
- `force_improvements` (bool): Always run critics. Default: False
- `show_improvement_prompt` (bool): Print prompts. Default: False
- `critic_model` (str, optional): Override model for critics
- `critic_temperature` (float, optional): Override temperature for critics

## Storage

### MemoryStorage (Default)
```python
from sifaka.storage import MemoryStorage

storage = MemoryStorage()
result = await improve(text, storage=storage)
```

### FileStorage
```python
from sifaka.storage import FileStorage

storage = FileStorage(storage_dir="./results")
result = await improve(text, storage=storage)

# Load result later
loaded = await storage.load(result.id)
```

## Models

### SifakaResult

The complete result object returned by `improve()`.

**Attributes:**
- `final_text` (str): The final improved text
- `original_text` (str): The original input text
- `iteration` (int): Number of iterations completed
- `generations` (List[Generation]): All text generations
- `critiques` (List[CritiqueResult]): All critique results
- `validations` (List[ValidationResult]): All validation results
- `id` (str): Unique result identifier
- `created_at` (datetime): Creation timestamp
- `processing_time` (float): Total processing time

**Properties:**
- `current_text` (str): Most recent generation or original
- `all_passed` (bool): Whether all validations passed
- `needs_improvement` (bool): Whether critics suggest improvement
- `confidence` (float): Overall confidence score

### CritiqueResult

Individual critic feedback.

**Attributes:**
- `critic` (str): Name of the critic
- `feedback` (str): Detailed feedback text
- `suggestions` (List[str]): Specific improvement suggestions
- `needs_improvement` (bool): Whether improvement is needed
- `confidence` (float): Confidence in the assessment (0.0-1.0)
- `timestamp` (datetime): When critique was created
- `metadata` (dict): Additional critic-specific data

### ValidationResult

Individual validation result.

**Attributes:**
- `validator` (str): Name of the validator
- `passed` (bool): Whether validation passed
- `score` (float): Validation score (0.0-1.0)
- `details` (str): Detailed validation information
- `timestamp` (datetime): When validation was performed

## Error Handling

### Exception Types

```python
from sifaka.core.exceptions import (
    SifakaError,          # Base exception
    ConfigurationError,   # Invalid configuration
    ModelProviderError,   # LLM API issues
    CriticError,         # Critic evaluation failures
    ValidationError,     # Validation failures
    TimeoutError,        # Operation timeouts
    StorageError,        # Storage operation failures
)
```

### Error Handling Example

```python
from sifaka.core.exceptions import TimeoutError, ModelProviderError

try:
    result = await improve(text, max_iterations=5)
except TimeoutError as e:
    print(f"Operation timed out: {e}")
    print(f"Suggestion: {e.suggestion}")
except ModelProviderError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Usage

### Custom Critics

```python
from sifaka.critics.core.base import BaseCritic
from sifaka.core.models import SifakaResult
from typing import List, Dict

class CustomCritic(BaseCritic):
    @property
    def name(self) -> str:
        return "custom"
    
    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are a custom critic."},
            {"role": "user", "content": f"Evaluate this text: {text}"}
        ]
```

### Custom Validators

```python
from sifaka.validators.base import BaseValidator
from sifaka.core.models import SifakaResult

class CustomValidator(BaseValidator):
    @property
    def name(self) -> str:
        return "custom"
    
    async def _perform_validation(
        self, text: str, result: SifakaResult
    ) -> tuple[bool, float, str]:
        # Your validation logic
        passed = len(text) > 50
        score = min(1.0, len(text) / 100)
        details = "Custom validation details"
        return passed, score, details
```

### Batch Processing

```python
texts = ["Text 1", "Text 2", "Text 3"]
results = []

for text in texts:
    result = await improve(
        text,
        critics=["reflexion"],
        max_iterations=2
    )
    results.append(result)

# Analyze results
avg_confidence = sum(r.confidence for r in results) / len(results)
```

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for GPT models
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude models
- `GROQ_API_KEY`: Groq API key for Llama models
- `GEMINI_API_KEY`: Google API key for Gemini models

## Best Practices

1. **Choose Critics Wisely**
   - Use `reflexion` for general improvement
   - Use `constitutional` for ethical content
   - Use `self_rag` for factual accuracy
   - Use `n_critics` for comprehensive analysis

2. **Set Appropriate Limits**
   - Most use cases: `max_iterations=3`
   - Quick drafts: `max_iterations=1-2`
   - Important content: `max_iterations=4-5`

3. **Use Validators**
   - Always validate length requirements
   - Check for required terminology
   - Enforce format constraints

4. **Handle Errors**
   - Always wrap in try/except
   - Provide fallback strategies
   - Log errors for debugging

5. **Optimize Performance**
   - Use faster models for drafts
   - Reduce iterations for speed
   - Batch process when possible