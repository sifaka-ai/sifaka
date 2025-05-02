# Docstring Standardization Guide

This guide provides a practical approach to standardizing docstrings across the Sifaka codebase.

## Overview

Standardizing docstrings is a critical part of improving Sifaka's documentation. This guide outlines a systematic approach to updating docstrings throughout the codebase.

## Prerequisites

Before starting, make sure you're familiar with:
- [Docstring Style Guide](./docstring_style_guide.md)
- [Implementing Docstrings](./implementing_docstrings.md)
- The docstring templates in `docs/templates/docstring_templates.py`

## Standardization Process

### Step 1: Prioritize Components

Focus on standardizing docstrings in this order:
1. Core public APIs and interfaces
2. Commonly used components
3. Factory functions
4. Implementation classes
5. Internal utilities

### Step 2: Audit Current Docstrings

For each module:
1. Check if module docstring exists and follows the standard
2. Identify public classes and functions that need docstring updates
3. Note any inconsistencies or missing information

### Step 3: Update Module Docstrings

Start with module-level docstrings:
1. Add a clear description of the module's purpose
2. List key components provided by the module
3. Include a basic usage example
4. Mention any important concepts or patterns

### Step 4: Update Class Docstrings

For each class:
1. Add a clear description of the class's purpose
2. Include a "Lifecycle" section explaining initialization, usage, and cleanup
3. Add an "Examples" section with runnable code
4. Document attributes and important methods

### Step 5: Update Method Docstrings

For each method:
1. Add a clear description of what the method does
2. Document parameters, return values, and exceptions
3. Include examples for complex methods
4. Note any edge cases or limitations

### Step 6: Update Factory Function Docstrings

For factory functions:
1. Clearly explain what the function creates
2. Document all parameters, including optional ones
3. Include comprehensive examples showing different configurations
4. Document return values and possible exceptions

### Step 7: Review and Test

After updating docstrings:
1. Review for consistency with the style guide
2. Test examples to ensure they work as documented
3. Check for any missing information
4. Verify that type annotations match the docstrings

## Example: Before and After

### Before

```python
def create_length_rule(min_chars=None, max_chars=None):
    """Create a length rule."""
    # Implementation
```

### After

```python
def create_length_rule(
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    rule_id: Optional[str] = None,
    **kwargs,
) -> LengthRule:
    """
    Create a length validation rule with the specified constraints.
    
    This factory function creates a configured LengthRule instance.
    It uses create_length_validator internally to create the validator.
    
    Args:
        min_chars: Minimum number of characters allowed
        max_chars: Maximum number of characters allowed
        min_words: Minimum number of words allowed
        max_words: Maximum number of words allowed
        rule_id: Identifier for the rule (also used as name if provided)
        **kwargs: Additional keyword arguments including:
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation
            - params: Dictionary of additional parameters
            - description: Description of the rule
    
    Returns:
        Configured LengthRule
    
    Examples:
        ```python
        from sifaka.rules.formatting.length import create_length_rule
        
        # Create a basic rule
        rule = create_length_rule(min_chars=10, max_chars=100)
        
        # Create a rule with word count constraints
        rule = create_length_rule(
            min_chars=10,
            max_chars=100,
            min_words=2,
            max_words=20,
            rule_id="comprehensive_length"
        )
        ```
    
    Raises:
        ValueError: If max_chars < min_chars or max_words < min_words
    """
    # Implementation
```

## Tracking Progress

To track progress on docstring standardization:

1. Create a checklist of modules to update
2. Mark modules as they are completed
3. Note any modules that need special attention
4. Track overall completion percentage

## Modules to Standardize

### Core Modules
- [ ] sifaka/rules/base.py
- [ ] sifaka/classifiers/base.py
- [ ] sifaka/critics/base.py
- [ ] sifaka/chain/core.py
- [ ] sifaka/models/base.py

### Rule Modules
- [x] sifaka/rules/formatting/length.py
- [ ] sifaka/rules/formatting/style.py
- [ ] sifaka/rules/content/prohibited.py
- [ ] sifaka/rules/factual/accuracy.py
- [ ] sifaka/rules/domain/legal.py

### Classifier Modules
- [ ] sifaka/classifiers/sentiment.py
- [ ] sifaka/classifiers/toxicity.py
- [ ] sifaka/classifiers/ner.py
- [ ] sifaka/classifiers/readability.py

### Critic Modules
- [ ] sifaka/critics/prompt.py
- [ ] sifaka/critics/reflexion.py
- [ ] sifaka/critics/style.py

### Chain Modules
- [ ] sifaka/chain/managers/prompt.py
- [ ] sifaka/chain/managers/validation.py
- [ ] sifaka/chain/strategies/retry.py
- [ ] sifaka/chain/formatters/result.py

### Model Modules
- [ ] sifaka/models/openai.py
- [ ] sifaka/models/anthropic.py
- [ ] sifaka/models/gemini.py

### Adapter Modules
- [ ] sifaka/adapters/rules/classifier.py
- [ ] sifaka/adapters/langchain/__init__.py
- [ ] sifaka/adapters/langgraph/__init__.py

## Best Practices

1. **Batch Updates**: Update related modules together
2. **Test Examples**: Ensure all examples are runnable
3. **Consistent Terminology**: Use the same terms across docstrings
4. **Cross-References**: Reference related components where appropriate
5. **Progressive Enhancement**: Start with basic improvements and enhance over time

## Conclusion

Standardizing docstrings is an ongoing process. By following this guide, we can systematically improve Sifaka's documentation, making it more maintainable, easier to use, and better documented for all users.
