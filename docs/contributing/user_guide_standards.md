# User Guide Standards

This document defines the standards for creating and maintaining user guides in the Sifaka documentation.

## Purpose

User guides provide detailed explanations of Sifaka's features, components, and usage patterns. They help users understand how to use Sifaka effectively and solve common problems.

## Guide Types

Sifaka documentation includes several types of user guides:

1. **Getting Started Guides**: Introduction to Sifaka for new users
2. **Component Guides**: Detailed explanations of specific components
3. **Task-Oriented Guides**: Instructions for accomplishing specific tasks
4. **Concept Guides**: Explanations of important concepts and patterns
5. **Integration Guides**: Instructions for integrating Sifaka with other tools

## Structure

All user guides should follow this general structure:

1. **Title**: Clear, descriptive title
2. **Overview**: Brief introduction to the topic
3. **Prerequisites**: Required knowledge, tools, or setup
4. **Main Content**: Detailed explanation with examples
5. **Best Practices**: Recommended approaches
6. **Troubleshooting**: Solutions to common problems
7. **Related Resources**: Links to related documentation
8. **Next Steps**: Suggestions for further learning

## Content Guidelines

### Writing Style

- Use clear, concise language
- Write in second person (you/your)
- Use active voice
- Explain technical terms
- Be consistent with terminology
- Use present tense

### Code Examples

- Include runnable code examples
- Explain what the code does
- Show complete examples, not just fragments
- Include imports
- Use consistent formatting
- Follow Sifaka coding standards

### Formatting

- Use Markdown formatting consistently
- Use headings to organize content (## for sections, ### for subsections)
- Use code blocks for code examples (```python)
- Use bullet points for lists
- Use bold for emphasis
- Use tables for structured data

## Example Sections

### Prerequisites Section

```markdown
## Prerequisites

Before you begin, make sure you have:
- Sifaka installed (version 0.1.0 or higher)
- Basic understanding of Python
- API keys for any external services (if applicable)
```

### Code Example Section

```markdown
## Creating a Rule

To create a rule, use the appropriate factory function:

```python
from sifaka.rules.formatting.length import create_length_rule

# Create a length rule
rule = create_length_rule(
    min_chars=10,
    max_chars=100,
    rule_id="length_constraint"
)

# Validate text
result = rule.validate("This is a test.")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```

This creates a rule that validates text length, ensuring it's between 10 and 100 characters.
```

### Troubleshooting Section

```markdown
## Troubleshooting

### Rule Validation Always Fails

**Problem**: Your rule validation always fails, even with valid input.

**Solution**: Check that your rule configuration is correct:

```python
# Correct configuration
rule = create_length_rule(
    min_chars=10,  # Minimum characters
    max_chars=100  # Maximum characters
)

# Incorrect configuration
rule = create_length_rule(
    min_chars=100,  # Minimum greater than maximum!
    max_chars=10
)
```
```

## Review Process

All user guides should be reviewed for:

1. **Technical accuracy**: Information is correct and up-to-date
2. **Completeness**: All necessary information is included
3. **Clarity**: Explanations are clear and understandable
4. **Consistency**: Guide follows the standards
5. **Examples**: Code examples are correct and runnable

## Maintenance

User guides should be maintained to ensure they remain accurate:

1. **Version updates**: Update guides when Sifaka is updated
2. **API changes**: Update guides when APIs change
3. **Bug fixes**: Correct any errors or omissions
4. **User feedback**: Incorporate feedback from users

## Template

Use the [User Guide Template](../templates/user_guide_template.md) as a starting point for new guides.
