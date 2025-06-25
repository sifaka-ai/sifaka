# Validator Migration Guide: From Custom to PydanticAI

## Overview

This guide explains how to migrate from Sifaka's custom validators to the new PydanticAI-based validators. The new validators leverage AI for more intelligent, context-aware validation while maintaining backward compatibility.

## Key Benefits of PydanticAI Validators

1. **AI-Powered Intelligence**: Goes beyond simple pattern matching to understand context and meaning
2. **Structured Responses**: Guaranteed structured validation results with detailed feedback
3. **Composability**: Easy to combine validators for complex validation scenarios
4. **Context Awareness**: Can consider the full improvement history when validating
5. **Natural Language Rules**: Define validation rules in plain English
6. **Built-in Observability**: Automatic tracking with Logfire integration

## Migration Examples

### Basic Length Validation

**Old Way:**
```python
from sifaka.validators import LengthValidator

validator = LengthValidator(min_length=100, max_length=500)
result = await validator.validate(text, sifaka_result)
```

**New Way (Direct):**
```python
from sifaka.agents.validators import LengthValidatorAgent

validator = LengthValidatorAgent(
    min_length=100,
    max_length=500,
    count_type="words"  # More flexible: "characters", "words", "sentences", "paragraphs"
)
result = await validator.validate(text)
```

**New Way (With Adapter for Compatibility):**
```python
from sifaka.agents.validators.adapter import create_length_validator

validator = create_length_validator(
    min_length=100,
    max_length=500,
    count_type="words"
)
# Works with existing Sifaka APIs
result = await validator.validate(text, sifaka_result)
```

### Content Validation

**Old Way:**
```python
from sifaka.validators import ContentValidator

validator = ContentValidator(
    required_terms=["AI", "machine learning"],
    forbidden_terms=["deprecated", "obsolete"]
)
```

**New Way:**
```python
from sifaka.agents.validators import ContentValidatorAgent

validator = ContentValidatorAgent(
    required_topics=["AI", "machine learning"],  # AI understands topics, not just terms
    forbidden_topics=["deprecated technology"],
    tone="professional",  # New: tone validation
    target_audience="technical professionals"  # New: audience appropriateness
)
```

### Pattern Validation

**Old Way:**
```python
from sifaka.validators import PatternValidator

validator = PatternValidator(
    required_patterns={"email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"},
    forbidden_patterns={"profanity": r"\b(badword1|badword2)\b"}
)
```

**New Way:**
```python
from sifaka.agents.validators import PatternValidatorAgent

validator = PatternValidatorAgent(
    required_patterns={
        "email": "Valid email addresses",
        "citation": "Academic citations in APA format"
    },
    forbidden_patterns={
        "sensitive_data": "Personal information like SSN or credit cards"
    },
    use_ai_enhancement=True  # AI detects patterns beyond regex
)
```

### Composable Validation

**Old Way:**
```python
from sifaka.validators import ComposableValidator, ValidationRule

rules = [
    ValidationRule(
        name="length",
        check=lambda text: 100 <= len(text.split()) <= 500,
        score_func=lambda text: min(1.0, len(text.split()) / 300),
        detail_func=lambda text: f"{len(text.split())} words"
    )
]
validator = ComposableValidator("custom", rules)
```

**New Way:**
```python
from sifaka.agents.validators import ComposableValidatorAgent

validator = ComposableValidatorAgent(
    name="blog_post_validator",
    validation_rules=[
        "Text must be 100-500 words",
        "Include an engaging introduction",
        "Use clear subheadings",
        "End with a call-to-action",
        "Maintain conversational tone"
    ]
)
```

### Advanced Structured Validation

**New Capability (Not available in old system):**
```python
from sifaka.agents.validators import (
    StructuredValidatorAgent,
    DocumentSpecification
)

# Define document requirements
spec = DocumentSpecification(
    document_type="technical_blog_post",
    required_elements=[
        "problem_statement",
        "solution_overview",
        "implementation_details",
        "results",
        "conclusion"
    ],
    quality_criteria={
        "technical_accuracy": "All code examples must be correct",
        "clarity": "Complex concepts explained simply",
        "completeness": "Cover all aspects of the topic"
    },
    target_audience="Senior developers",
    purpose="Educate about new technology"
)

validator = StructuredValidatorAgent(specification=spec)
```

## New Features Not Available in Old Validators

### 1. Context-Aware Validation
```python
# Validator considers improvement history
result = await validator.validate_with_context(text, sifaka_context)
```

### 2. Multi-Criteria Validation with Weights
```python
from sifaka.agents.validators import MultiCriteriaValidator

validator = MultiCriteriaValidator(
    criteria={
        "clarity": 0.3,
        "completeness": 0.3,
        "accuracy": 0.2,
        "engagement": 0.2
    },
    pass_threshold=0.75
)
```

### 3. Domain-Specific Validators
```python
# Financial data validation
from sifaka.agents.validators import FinancialDataValidator

validator = FinancialDataValidator(
    currency="USD",
    require_sources=True,
    fiscal_year=2024
)

# Scientific data validation
from sifaka.agents.validators import ScientificDataValidator

validator = ScientificDataValidator(
    require_units=True,
    require_uncertainty=True,
    significant_figures=3
)
```

### 4. Template-Based Validation
```python
from sifaka.agents.validators import TemplateValidatorAgent

validator = TemplateValidatorAgent(
    template_name="RFC",
    template_sections=[
        {"name": "Abstract", "required": True, "min_length": 150},
        {"name": "Introduction", "required": True},
        {"name": "Specification", "required": True},
        {"name": "Security Considerations", "required": True},
        {"name": "References", "required": True}
    ],
    strict_conformance=True
)
```

## Migration Strategy

### Phase 1: Compatibility Mode
Use the adapter to run new validators with existing code:

```python
from sifaka.agents.validators.adapter import ValidatorBuilder

# Build a set of validators
validators = (ValidatorBuilder()
    .add_length(min_length=100, max_length=1000)
    .add_content(required_topics=["AI", "ML"])
    .add_format(document_type="blog_post")
    .build())

# Use with existing Sifaka improve() function
result = await improve(text, critics=["reflexion"], validators=validators)
```

### Phase 2: Direct Usage
Use PydanticAI validators directly for new features:

```python
from sifaka.agents.validators import (
    LengthValidatorAgent,
    ContentValidatorAgent,
    create_blog_post_validator
)

# Combine validators
length_validator = LengthValidatorAgent(min_length=500, count_type="words")
content_validator = ContentValidatorAgent(
    required_topics=["AI", "future"],
    tone="inspirational"
)
blog_validator = create_blog_post_validator(seo_optimized=True)

# Use directly
for validator in [length_validator, content_validator, blog_validator]:
    result = await validator.validate(text)
    if not result.is_valid:
        print(f"Validation failed: {result.overall_assessment}")
        print(f"Suggestions: {result.improvement_priority}")
```

### Phase 3: Full Migration
Update your codebase to use the new validator APIs:

```python
# Old
from sifaka.validators import LengthValidator, ContentValidator

# New
from sifaka.agents.validators import (
    LengthValidatorAgent,
    ContentValidatorAgent,
    StructuredValidatorAgent
)
```

## Best Practices

1. **Use Semantic Rules**: Instead of regex patterns, describe what you want to validate in plain English
2. **Leverage Context**: Use `validate_with_context()` when you need history-aware validation
3. **Combine Validators**: Use `ComposableValidatorAgent` for complex validation scenarios
4. **Choose the Right Validator**: Use domain-specific validators (Financial, Scientific, etc.) when applicable
5. **Monitor Performance**: Enable Logfire to track validation performance and accuracy

## Debugging and Monitoring

With PydanticAI validators, you get automatic observability:

```python
# Enable Logfire monitoring
from sifaka.agents.models.config import AgentConfig, LogfireConfig

config = AgentConfig(
    logfire=LogfireConfig(
        enabled=True,
        token="your-logfire-token"
    )
)

validator = LengthValidatorAgent(config=config, min_length=100)
```

## FAQ

**Q: Do I need to migrate immediately?**
A: No, the adapter ensures backward compatibility. Migrate at your own pace.

**Q: Will validation be slower with AI?**
A: Initial validation may have slight latency, but the quality of feedback and intelligent detection of issues more than compensates.

**Q: Can I still use regex patterns?**
A: Yes, `PatternValidatorAgent` supports both AI-enhanced and traditional pattern matching.

**Q: How do I migrate custom validators?**
A: Create a `ComposableValidatorAgent` with your custom rules expressed in natural language, or extend `ValidatorAgent` for complex logic.

## Next Steps

1. Start with the adapter for immediate compatibility
2. Experiment with new validators in development
3. Gradually migrate validators based on your needs
4. Leverage new features like context-awareness and structured validation
5. Monitor validation quality with Logfire

For more examples and advanced usage, see the `examples/validators_pydantic_ai.py` file.
