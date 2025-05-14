# Sifaka Rules System

This package provides a comprehensive validation system for Sifaka, enabling content and format validation through a flexible and extensible rule architecture.

## Architecture

The rules system follows a component-based design with delegation between high-level and low-level components:

```
Rules System
├── Core Components
│   ├── Rule (high-level container defining what to validate)
│   ├── BaseValidator (implements actual validation logic)
│   └── ValidationManager (coordinates validation process)
├── Rule Types
│   ├── Content Rules (validate what the text contains)
│   │   ├── ProhibitedContentRule (forbidden words/phrases)
│   │   ├── ToxicityRule (harmful or offensive content)
│   │   ├── BiasRule (biased language)
│   │   ├── SentimentRule (emotional tone)
│   │   └── LanguageRule (language properties)
│   └── Formatting Rules (validate how the text is structured)
│       ├── LengthRule (text length constraints)
│       ├── StructureRule (organizational patterns)
│       ├── WhitespaceRule (spacing formatting)
│       └── StyleRule (writing style conventions)
└── Integration Components
    ├── FunctionRule (create rules from functions)
    ├── ValidationManager (validate against multiple rules)
    └── Adapters (convert other components to rules)
```

## Available Rules

### Content Rules

- **ProhibitedContentRule**: Checks for disallowed words or phrases
- **ToxicityRule**: Detects harmful, offensive, or inappropriate content
- **BiasRule**: Identifies biased or unfair language
- **SentimentRule**: Evaluates the emotional tone of text
- **HarmfulContentRule**: Detects content that could cause harm
- **LanguageRule**: Validates language properties

### Formatting Rules

- **LengthRule**: Validates text length constraints
- **StructureRule**: Checks organizational patterns and structure
- **WhitespaceRule**: Validates spacing and whitespace usage
- **StyleRule**: Enforces writing style conventions

## Using Rules

The recommended way to create rules is through factory functions:

```python
from sifaka.rules import (
    create_length_rule,
    create_prohibited_content_rule,
    create_toxicity_rule,
    create_sentiment_rule,
    create_structure_rule
)

# Create length validation rule
length_rule = create_length_rule(
    min_chars=10,
    max_chars=1000,
    name="length_rule",
    description="Validates text length is between 10 and 1000 characters"
)

# Create prohibited content rule
prohibited_rule = create_prohibited_content_rule(
    prohibited_terms=["bad", "inappropriate", "offensive"],
    case_sensitive=False
)

# Create toxicity detection rule
toxicity_rule = create_toxicity_rule(
    threshold=0.7,  # Toxicity score threshold
    check_categories=["toxicity", "identity_attack", "threat"]
)

# Create sentiment analysis rule
sentiment_rule = create_sentiment_rule(
    min_score=-0.5,  # Minimum sentiment score (negative)
    max_score=1.0    # Maximum sentiment score (positive)
)

# Create structure validation rule
structure_rule = create_structure_rule(
    required_sections=["introduction", "body", "conclusion"]
)

# Validate text against a rule
text = "This is a sample text that needs validation."
result = length_rule.validate(text)

# Check validation result
print(f"Validation passed: {result.passed}")
if not result.passed:
    print(f"Message: {result.message}")
    print(f"Issues: {result.issues}")
    print(f"Suggestions: {result.suggestions}")
```

## Using Multiple Rules with ValidationManager

When you need to validate against multiple rules, use the ValidationManager:

```python
from sifaka.rules import create_validation_manager, create_length_rule, create_prohibited_content_rule

# Create rules
length_rule = create_length_rule(min_chars=10, max_chars=1000)
prohibited_rule = create_prohibited_content_rule(
    prohibited_terms=["bad", "inappropriate", "offensive"]
)

# Create validation manager
validator = create_validation_manager(
    rules=[length_rule, prohibited_rule],
    fail_fast=True  # Stop on first failure
)

# Validate text
text = "This is a sample text that needs validation."
result = validator.validate(text)

# Check validation result
if result.all_passed:
    print("All validations passed!")
else:
    print("Validation failed:")
    for rule_id, rule_result in result.rule_results.items():
        if not rule_result.passed:
            print(f"- Rule '{rule_id}': {rule_result.message}")
            print(f"  Issues: {rule_result.issues}")
            print(f"  Suggestions: {rule_result.suggestions}")
```

## Integration with Chain

Rules can be integrated with Sifaka's Chain system for comprehensive validation and improvement:

```python
from sifaka.chain import Chain
from sifaka.models import OpenAIProvider
from sifaka.critics import create_prompt_critic
from sifaka.rules import create_length_rule, create_prohibited_content_rule

# Create model provider
model = OpenAIProvider("gpt-4")

# Create rules
rules = [
    create_length_rule(min_chars=50, max_chars=500),
    create_prohibited_content_rule(prohibited_terms=["offensive", "harmful"])
]

# Create critic for improving content
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor who improves text while maintaining its meaning."
)

# Create chain with rules and critic
chain = Chain(
    model=model,
    validators=rules,
    improver=critic,
    max_attempts=3
)

# Generate content with automatic validation and improvement
result = chain.run("Write a short story about friendship.")

print(f"Final output: {result.output}")
print(f"All validations passed: {result.all_passed}")
print(f"Attempts made: {result.attempt_count}")
```

## Creating Custom Rules

You can create custom rules by extending the BaseValidator and Rule classes:

```python
from sifaka.rules import Rule, BaseValidator, RuleResult, create_rule_result

class ReadabilityValidator(BaseValidator):
    """Validator that checks text readability score."""

    def __init__(self, min_score=60, max_score=100):
        """Initialize with readability score range."""
        self.min_score = min_score
        self.max_score = max_score

    def validate(self, text: str) -> RuleResult:
        """Validate text readability."""
        # Handle empty text using standard utility
        if self.is_empty_text(text):
            return self.handle_empty_text()

        # Calculate readability score (example using textstat)
        from textstat import flesch_reading_ease
        score = flesch_reading_ease(text)

        # Check if score is within acceptable range
        if score < self.min_score:
            return create_rule_result(
                passed=False,
                message=f"Text is too complex. Readability score: {score}",
                score=0.0,
                issues=[f"Readability score {score} is below minimum {self.min_score}"],
                suggestions=["Use shorter sentences", "Use simpler words"]
            )
        elif score > self.max_score:
            return create_rule_result(
                passed=False,
                message=f"Text is too simple. Readability score: {score}",
                score=0.0,
                issues=[f"Readability score {score} exceeds maximum {self.max_score}"],
                suggestions=["Use more varied sentence structure", "Include more detailed content"]
            )

        # Validation passed
        return create_rule_result(
            passed=True,
            message=f"Text has appropriate readability. Score: {score}",
            score=1.0
        )

class ReadabilityRule(Rule):
    """Rule for validating text readability."""

    def _create_default_validator(self) -> BaseValidator:
        """Create the default validator for this rule."""
        return ReadabilityValidator(
            min_score=self.config.params.get("min_score", 60),
            max_score=self.config.params.get("max_score", 100)
        )

# Create factory functions
def create_readability_validator(min_score=60, max_score=100):
    """Create a readability validator."""
    return ReadabilityValidator(min_score=min_score, max_score=max_score)

def create_readability_rule(min_score=60, max_score=100, **kwargs):
    """Create a readability rule."""
    # Extract rule-specific parameters
    params = {
        "min_score": min_score,
        "max_score": max_score
    }

    # Create and return rule
    from sifaka.rules.factories import create_rule
    return create_rule(
        rule_class=ReadabilityRule,
        validator=None,  # Use default validator
        params=params,
        **kwargs
    )
```

## Creating Function-Based Rules

For simple cases, you can create rules directly from functions:

```python
from sifaka.rules import FunctionRule, create_rule_result

def check_keyword_presence(text, keywords=None):
    """Check if text contains required keywords."""
    if keywords is None:
        keywords = ["important", "critical", "essential"]

    # Check if any keyword is present
    found_keywords = [keyword for keyword in keywords if keyword.lower() in text.lower()]

    if found_keywords:
        return create_rule_result(
            passed=True,
            message=f"Text contains required keywords: {', '.join(found_keywords)}",
            score=1.0
        )
    else:
        return create_rule_result(
            passed=False,
            message=f"Text doesn't contain any required keywords",
            score=0.0,
            issues=["Missing required keywords"],
            suggestions=[f"Include one or more of these keywords: {', '.join(keywords)}"]
        )

# Create rule from function
keyword_rule = FunctionRule(
    validation_function=check_keyword_presence,
    name="keyword_presence_rule",
    description="Checks if text contains required keywords",
    params={"keywords": ["important", "essential", "critical"]}
)

# Validate text
result = keyword_rule.validate("This is a very important message.")
print(f"Validation passed: {result.passed}")
```

## Best Practices

1. **Use factory functions**: Create rules using provided factory functions whenever possible
2. **Compose rules**: Use ValidationManager to validate against multiple rules
3. **Handle edge cases**: Always handle empty text and special cases explicitly
4. **Provide helpful feedback**: Include actionable suggestions in rule results
5. **Set appropriate priorities**: Use RulePriority to indicate importance of rules
6. **Follow naming conventions**: Use consistent naming for rules and validators
7. **Use standardized results**: Create results using utility functions like create_rule_result
8. **Document validation logic**: Add clear docstrings explaining validation criteria
9. **Consider performance**: Implement optimizations for expensive validation operations
10. **Use error handling**: Wrap validation code in try/except to prevent crashes
