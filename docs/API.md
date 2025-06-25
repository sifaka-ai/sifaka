# Sifaka API Reference

## Main API

### `improve(text, use_case="general", **kwargs) -> str`

Improve text with automatic critic selection based on use case.

**Parameters:**
- `text` (str): The text to improve
- `use_case` (str): One of: general, academic, business, technical, creative, email, social_media
- `**kwargs`: Additional options:
  - `max_iterations` (int): Maximum improvement iterations (default: 3)
  - `validators` (List[Validator]): Validators to apply
  - `model` (str): Override model (default: "gpt-4o-mini")
  - `temperature` (float): Override temperature (default: 0.7)

**Returns:**
- str: The improved text

**Example:**
```python
improved = await improve("My text", use_case="academic")
```

### `improve_sync(text, use_case="general", **kwargs) -> str`

Synchronous version of `improve()`. Same parameters and return value.

## Convenience Functions

### `improve_email(text, max_length=500) -> str`

Improve an email to be professional and concise.

**Parameters:**
- `text` (str): Email text to improve
- `max_length` (int): Maximum character length (default: 500)

### `improve_academic(text, check_facts=True) -> str`

Improve academic text with rigor and optional fact-checking.

**Parameters:**
- `text` (str): Academic text to improve
- `check_facts` (bool): Whether to verify claims (default: True)

### `improve_code_docs(text) -> str`

Improve technical documentation or code comments.

### `quick_improve(text) -> str`

Quick single-pass improvement for any text.

### `improve_with_length(text, min_length=None, max_length=None, use_case="general") -> str`

Improve text with length constraints.

**Parameters:**
- `text` (str): Text to improve
- `min_length` (int, optional): Minimum character length
- `max_length` (int, optional): Maximum character length
- `use_case` (str): Use case for critic selection

### `improve_with_tone(text, tone, use_case="general") -> str`

Improve text with specific tone requirements.

**Parameters:**
- `text` (str): Text to improve
- `tone` (str): Desired tone (professional, casual, formal, friendly, etc.)
- `use_case` (str): Use case for critic selection

## Validators

### `LengthValidator(min_length=None, max_length=None)`

Validates text length.

**Parameters:**
- `min_length` (int, optional): Minimum character count
- `max_length` (int, optional): Maximum character count

### `ContentValidator(tone=None, style=None, audience=None)`

Validates content requirements.

**Parameters:**
- `tone` (str, optional): Required tone
- `style` (str, optional): Required style
- `audience` (str, optional): Target audience

## Constants

### `UseCase`

Enum of available use cases:
- `UseCase.GENERAL`
- `UseCase.ACADEMIC`
- `UseCase.BUSINESS`
- `UseCase.TECHNICAL`
- `UseCase.CREATIVE`
- `UseCase.EMAIL`
- `UseCase.SOCIAL_MEDIA`

## Advanced API

Access advanced features via `get_advanced_api()`:

```python
advanced = get_advanced_api()
```

Returns a dict containing:
- `improve_stream`: Streaming improvement function
- `SifakaResult`: Full result object with metadata
- `configure`: Global configuration function
- `all_critics`: List of available critics

### Streaming

```python
async for partial in advanced['improve_stream'](text, use_case="general"):
    print(partial.text_so_far)
    print(f"Tokens: {partial.tokens_generated}")
```

### Configuration

```python
advanced['configure'](
    model="gpt-4",
    temperature=0.3,
    max_iterations=5,
    logfire_token="your-token"
)
```

### Full Results

```python
from sifaka.api import improve as _improve
result = await _improve(text, critics=["reflexion"])

# Access detailed information
print(result.original_text)
print(result.final_text)
print(result.iterations)
print(result.critiques)  # List of all critic feedback
print(result.get_final_confidence())
```

## Error Handling

All functions may raise:
- `SifakaError`: Base exception with error code
- `InvalidCriticError`: Unknown critic specified
- `InvalidModelError`: Invalid model for provider
- `APIError`: LLM API call failed
- `ValidationError`: Text failed validation
- `ConfigurationError`: Invalid configuration

Example:
```python
from sifaka.exceptions import SifakaError

try:
    result = await improve(text)
except SifakaError as e:
    print(f"Error {e.code}: {e}")
    print(f"Details: {e.details}")
```
