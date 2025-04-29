# Sifaka Examples

This directory contains example scripts demonstrating various Sifaka capabilities.

## Length Control Examples with Claude

### Prerequisites

Before running the examples, make sure to:

1. Install required dependencies:
   ```bash
   pip install python-dotenv anthropic tiktoken
   ```

2. Create a `.env` file in your project root with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

### Examples

#### 1. Reducing Response Length (`claude_length_critic.py`)

This example demonstrates how to use Claude with a Length Critic to reduce verbose responses. It:
- Configures Claude to generate a comprehensive explanation (typically 500+ words)
- Uses a Length Rule to enforce a 300-word maximum
- Employs a Critic to help Claude condense its response while preserving key information

Run it with:
```bash
python -m sifaka.examples.claude_length_critic
```

#### 2. Expanding Response Length (`claude_expand_length_critic.py`)

This example shows how to use Claude with a Length Critic to expand brief responses. It:
- Asks Claude for a brief explanation (typically 150 words)
- Uses a Length Rule to require 1000-2000 words
- Employs a Critic to guide Claude in expanding its response with relevant details

Run it with:
```bash
python -m sifaka.examples.claude_expand_length_critic
```

## How It Works

These examples demonstrate Sifaka's Chain pattern:

1. A **Model Provider** (Claude) generates text
2. **Rules** (Length Rule) validate the output
3. If validation fails, a **Critic** suggests improvements
4. The chain loops back to the Model with feedback
5. This continues until the output passes all rules or max attempts are reached

This approach creates a self-correcting system where the AI can adapt its outputs to meet specific requirements.

## Customizing

You can customize these examples by:
- Changing the word count limits in the Length Rule configuration
- Modifying the system prompt for the Critic to focus on different aspects of text improvement
- Adjusting the prompt to generate different types of content
- Combining with other Sifaka rules for additional validation