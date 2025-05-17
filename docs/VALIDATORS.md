# Validators Documentation

Validators are components in the Sifaka framework that check if text meets specific criteria, such as length, content, or format requirements. They return validation results that indicate whether the text passed or failed validation, along with details about any issues found.

## Overview

Validators are used by the Chain class to ensure that generated text meets the required criteria before being returned to the user or passed to critics for improvement. They implement the ValidatorProtocol, which defines a consistent interface for all validators.

## Built-in Validators

Sifaka includes several built-in validators that you can use out of the box:

### Length Validator

Checks if the text length (in words or characters) is within specified limits.

```python
from sifaka.validators import length

# Create a validator that checks if the text has between 100 and 500 words
word_length_validator = length(min_words=100, max_words=500)

# Create a validator that checks if the text has between 500 and 2000 characters
char_length_validator = length(min_chars=500, max_chars=2000)

# Create a validator with both word and character constraints
combined_validator = length(min_words=100, max_words=500, min_chars=500, max_chars=2000)
```

### Prohibited Content Validator

Checks if the text contains prohibited terms or phrases.

```python
from sifaka.validators import prohibited_content

# Create a validator that checks if the text contains prohibited terms
content_validator = prohibited_content(prohibited=["violent", "harmful", "offensive"])
```

### JSON Validator

Checks if the text is valid JSON and optionally validates it against a schema.

```python
from sifaka.validators import json_validator

# Create a validator that checks if the text is valid JSON
json_val = json_validator()

# Create a validator that checks if the text is valid JSON and matches a schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "interests": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "age"]
}
json_schema_val = json_validator(schema=schema)
```

### Regex Validator

Checks if the text matches a regular expression pattern.

```python
from sifaka.validators import regex_match

# Create a validator that checks if the text matches an email pattern
email_validator = regex_match(pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
```

### Profanity Validator

Checks if the text contains profanity.

```python
from sifaka.validators import profanity_free

# Create a validator that checks if the text is free of profanity
profanity_validator = profanity_free()
```

## Using Validators with Chain

Validators are typically used with the Chain class to validate generated text:

```python
from sifaka import Chain
from sifaka.validators import length, prohibited_content

# Create a chain with validators
chain = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .validate_with(length(min_words=100, max_words=500))
    .validate_with(prohibited_content(prohibited=["violent", "harmful"]))
)

# Run the chain
result = chain.run()

# Check if validation passed
if result.passed:
    print("Validation passed!")
    print(result.text)
else:
    print("Validation failed!")
    for validation_result in result.validation_results:
        if not validation_result.passed:
            print(f"Failed validation: {validation_result.message}")
```

## Creating Custom Validators

You can create custom validators by inheriting from the BaseValidator class and implementing the _validate method:

```python
from sifaka.validators.base import BaseValidator
from sifaka.results import ValidationResult

class ReadabilityValidator(BaseValidator):
    def __init__(self, max_grade_level: float, name: str = "ReadabilityValidator"):
        super().__init__(name=name)
        self.max_grade_level = max_grade_level
        
    def _validate(self, text: str) -> ValidationResult:
        # Calculate Flesch-Kincaid grade level (simplified example)
        words = len(text.split())
        sentences = len(text.split('.'))
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        if words == 0 or sentences == 0:
            return ValidationResult(
                passed=False,
                message="Text is too short to calculate readability",
                score=0.0
            )
        
        grade_level = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
        
        if grade_level <= self.max_grade_level:
            return ValidationResult(
                passed=True,
                message=f"Text readability is at grade level {grade_level:.1f}, which is within the limit of {self.max_grade_level}",
                score=1.0
            )
        else:
            return ValidationResult(
                passed=False,
                message=f"Text readability is at grade level {grade_level:.1f}, which exceeds the limit of {self.max_grade_level}",
                score=0.0,
                issues=[f"Text is too complex (grade level {grade_level:.1f} > {self.max_grade_level})"],
                suggestions=["Use shorter sentences", "Use simpler words", "Break up complex paragraphs"]
            )
    
    def _count_syllables(self, word: str) -> int:
        # Simplified syllable counting (not accurate for all words)
        word = word.lower()
        if len(word) <= 3:
            return 1
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        return count
```

## Validator Protocol

All validators must implement the ValidatorProtocol, which defines the following interface:

```python
class ValidatorProtocol(Protocol):
    def validate(self, text: str) -> ValidationResult:
        """Validate text against specific criteria."""
        ...
        
    @property
    def name(self) -> str:
        """Get the name of the validator."""
        ...
```

## Validation Results

Validators return ValidationResult objects that contain information about the validation:

```python
@dataclass
class ValidationResult:
    passed: bool  # Whether the validation passed
    message: str  # Human-readable message describing the validation result
    _details: Dict[str, Any] = field(default_factory=dict)  # Additional details
    score: Optional[float] = None  # Normalized score between 0.0 and 1.0
    issues: Optional[List[str]] = None  # List of identified issues
    suggestions: Optional[List[str]] = None  # List of suggestions for improvement
```

You can access these properties to get information about the validation:

```python
result = validator.validate(text)

if result.passed:
    print(f"Validation passed: {result.message}")
    if result.score is not None:
        print(f"Score: {result.score:.2f}")
else:
    print(f"Validation failed: {result.message}")
    if result.issues:
        for issue in result.issues:
            print(f"- Issue: {issue}")
    if result.suggestions:
        for suggestion in result.suggestions:
            print(f"- Suggestion: {suggestion}")
```

## Safe Validation

Sifaka provides a utility function for safely validating text, which ensures that validation errors are properly handled:

```python
from sifaka.validators.base import safe_validate

# Safely validate text
result = safe_validate(validator, text)

# Check the result
if result.passed:
    print("Validation passed!")
else:
    print(f"Validation failed: {result.message}")
```

## Best Practices

1. **Use built-in validators** when possible, as they handle common validation scenarios
2. **Combine multiple validators** to check different aspects of the text
3. **Provide clear error messages** in custom validators to help users understand why validation failed
4. **Include suggestions** for how to fix validation issues
5. **Handle edge cases** like empty text or very short text
6. **Use safe_validate** when you want to ensure that validation always returns a result, even if the validator raises an exception
