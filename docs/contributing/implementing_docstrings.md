# Implementing Docstring Standards

This guide provides practical steps for implementing the Sifaka docstring standards in the codebase.

## Getting Started

1. **Read the Style Guide**: Familiarize yourself with the [Docstring Style Guide](./docstring_style_guide.md)
2. **Use the Templates**: Copy templates from `docs/templates/docstring_templates.py` for your component type
3. **Start with Public APIs**: Focus on public APIs first, then move to internal components

## Step-by-Step Process

### 1. Identify Component Type

Determine which type of component you're documenting:
- Module
- Rule
- Validator
- Classifier
- Critic
- Chain
- Model Provider
- Factory Function
- Configuration Class
- Method

### 2. Copy the Appropriate Template

Copy the template from `docs/templates/docstring_templates.py` that matches your component type.

### 3. Fill in the Details

Replace the placeholders in the template with specific information about your component:
- Replace `{component_name}` with the actual name
- Fill in descriptions, parameters, return values, etc.
- Add relevant examples
- Remove sections that don't apply

### 4. Add Type Annotations

Ensure all parameters and return values have appropriate type annotations:

```python
def function_name(param1: str, param2: Optional[int] = None) -> bool:
    """
    Function docstring.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
```

### 5. Include Examples

Every public component should have at least one usage example:
- Make examples runnable (include imports)
- Show typical usage patterns
- Demonstrate parameter usage
- Include error handling when relevant

### 6. Review and Refine

Before submitting your changes:
- Check that all placeholders have been replaced
- Ensure examples are correct and runnable
- Verify that type annotations match the docstring
- Check for spelling and grammar errors

## Example: Before and After

### Before

```python
class LengthRule(Rule):
    """Rule for validating text length."""
    
    def __init__(self, min_chars=None, max_chars=None):
        self.min_chars = min_chars
        self.max_chars = max_chars
        
    def validate(self, text):
        # Implementation
```

### After

```python
class LengthRule(Rule[str, RuleResult, LengthValidator, None]):
    """
    Rule for validating text length constraints.
    
    This rule validates that text meets specified length requirements
    in terms of character count or word count.
    
    Lifecycle:
        1. Initialization: Set up with length constraints
        2. Validation: Check text against constraints
        3. Result: Return standardized validation results
    
    Examples:
        ```python
        from sifaka.rules.formatting.length import create_length_rule
        
        # Create a rule using the factory function
        rule = create_length_rule(
            min_chars=10,
            max_chars=100
        )
        
        # Validate text
        result = rule.validate("This is a test")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    
    Attributes:
        min_chars: Minimum number of characters (optional)
        max_chars: Maximum number of characters (optional)
    """
    
    def __init__(
        self,
        validator: LengthValidator,
        name: str = "length_rule",
        description: str = "Validates text length",
        config: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize the length rule.
        
        Args:
            validator: The validator to use for length validation
            name: The name of the rule
            description: Description of the rule
            config: Additional configuration for the rule
            **kwargs: Additional keyword arguments for the rule
        """
        self._length_validator = validator
        super().__init__(name=name, description=description, config=config, **kwargs)
    
    def validate(self, text: str, **kwargs) -> RuleResult:
        """
        Evaluate text against length constraints.
        
        Args:
            text: The text to evaluate
            **kwargs: Additional validation context
            
        Returns:
            RuleResult containing validation results
        """
        # Implementation
```

## Common Mistakes to Avoid

1. **Incomplete docstrings**: Missing sections or parameters
2. **Outdated examples**: Examples that don't match the current API
3. **Inconsistent style**: Mixing different docstring styles
4. **Missing type annotations**: Parameters or return values without types
5. **Placeholder text**: Forgetting to replace template placeholders
6. **Redundant information**: Repeating information that's obvious from the code
7. **Missing edge cases**: Not documenting behavior for edge cases

## Tools for Docstring Validation

Consider using these tools to validate your docstrings:
- `pydocstyle`: Checks compliance with docstring conventions
- `darglint`: Ensures docstring arguments match function signatures
- `sphinx`: Generates documentation from docstrings
- `doctest`: Tests code examples in docstrings

## Next Steps

After implementing docstrings:
1. Generate documentation using Sphinx or a similar tool
2. Review the generated documentation for completeness
3. Get feedback from other developers
4. Iterate and improve based on feedback
