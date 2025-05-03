# Example Standards

This document defines the standards for creating and maintaining examples in the Sifaka codebase.

## Purpose

Examples demonstrate how to use Sifaka's features and components in practical scenarios. They help users understand how to apply Sifaka to real-world problems and serve as reference implementations.

## Example Types

Sifaka includes several types of examples:

1. **Basic Examples**: Simple demonstrations of core functionality
2. **Component Examples**: Focused on specific components
3. **Integration Examples**: Showing integration with other libraries
4. **End-to-End Examples**: Complete applications using Sifaka
5. **Advanced Examples**: Demonstrating advanced techniques

## Structure

All examples should follow this general structure:

1. **Module Docstring**: Explaining the example's purpose and key concepts
2. **Imports**: Clearly organized imports
3. **Configuration**: Setup and configuration
4. **Component Creation**: Creating and configuring components
5. **Main Logic**: The core example code
6. **Helper Functions**: Any supporting functions
7. **Alternative Approaches**: Optional alternative implementations
8. **Main Block**: Code to run the example

## Content Guidelines

### Module Docstring

The module docstring should include:
- Example title
- Brief description
- Key concepts demonstrated
- Requirements
- Instructions for running the example

Example:
```python
"""
Example: Length Validation with Claude

This example demonstrates how to use Claude with a Length Rule to validate
and improve text length.

Key concepts demonstrated:
1. Configuring Claude as a model provider
2. Setting up length validation rules
3. Using a critic to improve text that fails validation
4. Handling validation results

Requirements:
- Sifaka 0.1.0+
- Anthropic API key

To run this example:
```bash
python -m sifaka.examples.claude_length_validation
```
"""
```

### Code Organization

- Organize code into logical sections with clear comments
- Use descriptive variable and function names
- Include error handling
- Add comments explaining non-obvious code
- Keep examples focused on demonstrating specific features

### Code Style

- Follow PEP 8 style guidelines
- Use type annotations
- Use consistent formatting
- Follow Sifaka coding standards
- Keep lines under 100 characters

### Documentation

- Add comments explaining what the code does
- Document function parameters and return values
- Explain key decisions and patterns
- Include expected output where appropriate

## Example Sections

### Imports Section

```python
# Standard library imports
import os
import json
from typing import Dict, List, Optional

# Third-party imports
from dotenv import load_dotenv

# Sifaka imports
from sifaka.models.anthropic import AnthropicProvider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import create_prompt_critic
from sifaka.chain import create_simple_chain
```

### Configuration Section

```python
# Load environment variables
load_dotenv()

# Get API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY environment variable not set. "
        "Please set it in your environment or .env file."
    )

# Configuration
CONFIG = {
    "model_name": "claude-3-haiku-20240307",
    "temperature": 0.7,
    "max_tokens": 500,
    "min_chars": 100,
    "max_chars": 500
}
```

### Main Logic Section

```python
def main():
    """Run the main example."""
    print("Starting example...")
    
    # Create model provider
    model = AnthropicProvider(
        model_name=CONFIG["model_name"],
        api_key=ANTHROPIC_API_KEY,
        temperature=CONFIG["temperature"],
        max_tokens=CONFIG["max_tokens"]
    )
    
    # Create rule
    rule = create_length_rule(
        min_chars=CONFIG["min_chars"],
        max_chars=CONFIG["max_chars"]
    )
    
    # Create critic
    critic = create_prompt_critic(
        model=model,
        system_prompt="You are an editor who helps adjust text length."
    )
    
    # Create chain
    chain = create_simple_chain(
        model=model,
        rules=[rule],
        critic=critic,
        max_attempts=3
    )
    
    # Run chain
    prompt = "Write a short story about a robot."
    result = chain.run(prompt)
    
    # Display result
    print(f"\nOutput: {result.output}")
    print(f"Validation passed: {result.all_passed}")
    
    print("Example completed successfully!")
```

## Testing

All examples should be tested to ensure they:
- Run without errors
- Demonstrate the intended functionality
- Follow the standards
- Have clear output

## Maintenance

Examples should be maintained to ensure they remain accurate:
- Update examples when Sifaka is updated
- Update examples when APIs change
- Correct any errors or omissions
- Incorporate feedback from users

## Template

Use the [Example Template](../templates/example_template.py) as a starting point for new examples.
