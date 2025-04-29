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

#### 3. ReflexionCritic Example (`reflexion_critic_example.py`)

This example demonstrates how to use the ReflexionCritic to improve text by maintaining a memory of past improvements. It:
- Creates a ReflexionCritic that uses OpenAI's GPT-3.5 model
- Generates text responses for various prompts
- Collects reflections on how texts are improved
- Uses these reflections to guide future improvements
- Maintains a memory buffer of reflections for continuous learning

The example showcases:
- How the ReflexionCritic learns from past improvements
- How it applies those learnings to new text generation tasks
- The reflection generation process and memory management

Run it with:
```bash
python examples/reflexion_critic_example.py
```

## How It Works

These examples demonstrate Sifaka's Chain pattern:

1. A **Model Provider** (Claude or OpenAI) generates text
2. **Rules** (Length Rule) validate the output
3. If validation fails, a **Critic** suggests improvements
4. The chain loops back to the Model with feedback
5. This continues until the output passes all rules or max attempts are reached

The ReflexionCritic adds an additional layer:
- It maintains a memory of past improvement attempts
- It generates reflections on what worked well and what could be improved
- These reflections are used to guide future text improvements
- This creates a learning loop where the critic gets better over time

This approach creates a self-correcting, learning system where the AI can adapt its outputs to meet specific requirements and improve based on past experiences.

## Customizing

You can customize these examples by:
- Changing the word count limits in the Length Rule configuration
- Modifying the system prompt for the Critic to focus on different aspects of text improvement
- Adjusting the prompt to generate different types of content
- Combining with other Sifaka rules for additional validation
- For the ReflexionCritic, you can modify:
  - The memory buffer size to retain more or fewer reflections
  - The reflection depth to control how complex the reflections should be
  - The system prompt to guide the reflection process differently