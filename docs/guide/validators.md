# Validators Guide

Validators ensure that improved text meets specific criteria before accepting changes.

## Basic Usage

```python
from sifaka import improve
from sifaka.validators import LengthValidator

# Simple length validation
validator = LengthValidator(min_length=100, max_length=500)

result = await improve(
    "Short text",
    validators=[validator],
    max_iterations=3
)
```

## Composable Validators

Build complex validation rules using the fluent API:

```python
from sifaka.validators.composable import Validator

# Create a blog post validator
blog_validator = (
    Validator.create("blog_post")
    .length(500, 2000)
    .sentences(10, 50)
    .contains(["introduction", "conclusion"], mode="all")
    .matches(r"\#{1,3}\s+.+", "headers")
    .build()
)

result = await improve(
    draft,
    validators=[blog_validator]
)
```

## Combining Validators

Use logical operators to combine validators:

```python
# AND: Both conditions must be met
long_and_technical = (
    Validator.length(min_length=1000) &
    Validator.contains(["algorithm", "complexity"], mode="any")
)

# OR: Either condition must be met
short_or_bulleted = (
    Validator.length(max_length=200) |
    Validator.matches(r"^[\*\-]\s+", "bullet_points")
)

# NOT: Invert a condition
not_too_long = ~Validator.length(min_length=5000)
```

## Available Validators

### Length Validator
Control text length:
```python
Validator.length(min_length=100, max_length=1000)
```

### Contains Validator
Ensure specific keywords are present:
```python
# All keywords must be present
Validator.contains(["intro", "body", "conclusion"], mode="all")

# At least one keyword must be present
Validator.contains(["example", "instance", "e.g."], mode="any")
```

### Pattern Validator
Match regular expressions:
```python
# Email validation
Validator.matches(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "email")

# URL validation
Validator.matches(r"https?://[\w\.-]+", "url")
```

### Word/Sentence Count
Control structure:
```python
Validator.words(min_words=50, max_words=200)
Validator.sentences(min_sentences=3, max_sentences=10)
```

## Custom Validators

Create custom validation logic:

```python
from sifaka.validators.base import BaseValidator
from sifaka.core.models import ValidationResult

class ToneValidator(BaseValidator):
    def __init__(self, required_tone: str):
        self.required_tone = required_tone

    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        # Analyze tone (simplified example)
        formal_words = ["therefore", "however", "furthermore"]
        is_formal = any(word in text.lower() for word in formal_words)

        passed = (self.required_tone == "formal") == is_formal

        return ValidationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            validator="tone",
            details=f"Tone: {'formal' if is_formal else 'casual'}"
        )
```

## Validation in Practice

### Academic Writing
```python
academic_validator = (
    Validator.create("academic")
    .length(1500, 5000)
    .sentences(50, 200)
    .contains(["abstract", "introduction", "methodology", "conclusion"], mode="all")
    .matches(r"\[\d+\]", "citations")
    .build()
)
```

### Social Media Post
```python
tweet_validator = (
    Validator.create("tweet")
    .length(max_length=280)
    .contains(["#", "@"], mode="any")
    .build()
)
```

### Technical Documentation
```python
tech_doc_validator = (
    Validator.create("tech_doc")
    .length(min_length=500)
    .contains(["installation", "usage", "api", "examples"], mode="all")
    .matches(r"```[\w]*\n[\s\S]+?\n```", "code_blocks")
    .build()
)
```

## Best Practices

1. **Be specific but not restrictive**: Allow room for creativity while ensuring quality
2. **Combine validators thoughtfully**: Use AND for requirements, OR for alternatives
3. **Test your validators**: Ensure they don't reject good content
4. **Provide clear feedback**: Custom validators should explain why validation failed
5. **Consider the use case**: Different content types need different validation rules
