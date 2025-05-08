# Empty Text Handling in Sifaka

This document describes the standardized approach to handling empty text inputs across all Sifaka components.

## Standard Policy

In Sifaka, empty text inputs are handled consistently across all components with the following standard policy:

1. **Default Behavior**: Empty text inputs **PASS** validation by default
2. **Metadata**: Empty text validation results include standardized metadata
3. **Configuration**: Components can be configured to enforce non-empty text when required

## Implementation Details

### 1. Standard Empty Text Result

When an empty text input is detected, components should return a `RuleResult` with:

```python
RuleResult(
    passed=True,
    message="Empty text validation skipped",
    metadata={"reason": "empty_input"},
)
```

### 2. Empty Text Detection

Empty text is detected using the following check:

```python
if not text or not text.strip():
    # Handle empty text
```

This catches both `None`, empty strings (`""`), and strings containing only whitespace.

### 3. Component-Specific Handling

#### Rules and Validators

Rules and validators use the `handle_empty_text()` method to provide consistent handling:

```python
def handle_empty_text(self, text: str) -> Optional[RuleResult]:
    """Handle empty text validation."""
    if not text.strip():
        return RuleResult(
            passed=True,
            message="Empty text validation skipped",
            metadata={"reason": "empty_input"},
        )
    return None
```

#### Adapters

Adapters follow the same pattern as rules:

```python
def handle_empty_text(self, text: str) -> Optional[RuleResult]:
    """Handle empty or invalid input text."""
    if not text or not text.strip():
        return RuleResult(
            passed=True,
            message="Empty text validation skipped",
            metadata={"reason": "empty_input", "input_length": len(text)},
        )
    return None
```

#### Classifiers

Classifiers return an "unknown" label for empty text:

```python
if isinstance(text, str) and not text.strip():
    return ClassificationResult[R](
        label="unknown",
        confidence=0.0,
        metadata={"reason": "empty_input"}
    )
```

### 4. Configurable Behavior

Some components allow configuring empty text handling:

#### Format Rules

Format rules can be configured to allow or disallow empty text:

```python
# Create a plain text rule that allows empty text
rule = create_plain_text_rule(
    allow_empty=True,
    name="text_validator"
)
```

When `allow_empty=True`, the component will return:

```python
RuleResult(
    passed=True,
    message="Empty text allowed",
    metadata={"reason": "empty_input_allowed"},
)
```

## Usage Examples

### Basic Empty Text Handling

```python
from sifaka.rules.formatting.length import create_length_rule

# Create a rule
rule = create_length_rule(min_chars=10, max_chars=100)

# Validate empty text
result = rule.validate("")
print(result.passed)  # True
print(result.message)  # "Empty text validation skipped"
print(result.metadata)  # {"reason": "empty_input"}
```

### Configuring Empty Text Handling

```python
from sifaka.rules.formatting.format import create_plain_text_rule

# Create a rule that allows empty text
rule_allowing_empty = create_plain_text_rule(
    allow_empty=True,
    name="text_validator_allowing_empty"
)

# Create a rule with standard empty text handling
rule_standard = create_plain_text_rule(
    allow_empty=False,
    name="text_validator_standard"
)

# Validate empty text
result1 = rule_allowing_empty.validate("")
print(result1.passed)  # True
print(result1.message)  # "Empty text allowed"
print(result1.metadata)  # {"reason": "empty_input_allowed"}

result2 = rule_standard.validate("")
print(result2.passed)  # True
print(result2.message)  # "Empty text validation skipped"
print(result2.metadata)  # {"reason": "empty_input"}
```

## Rationale

The decision to make empty text **PASS** validation by default was made for the following reasons:

1. **Simplicity**: It's easier to handle empty text as a special case that passes validation
2. **Flexibility**: Components that need to enforce non-empty text can be configured to do so
3. **Consistency**: Having a single standard approach simplifies the codebase
4. **Metadata**: The metadata clearly indicates that empty text was detected and validation was skipped

This approach ensures that empty text is handled consistently across all components while still allowing for flexibility when needed.
