# Getting Started with Sifaka

Welcome to Sifaka! This 5-minute tutorial will get you up and running with AI text improvement in no time.

## What is Sifaka?

Sifaka transforms AI text generation from a "hope it's good" process into a **guaranteed quality** system. Instead of just generating text, Sifaka creates a feedback loop where AI systems validate, critique, and iteratively improve their outputs until they meet your standards.

**The Magic**: Research-backed techniques from papers like Reflexion, Constitutional AI, and Self-RAG, implemented as working code.

## 🚀 Quick Start (2 minutes)

### 1. Install Sifaka

```bash
pip install sifaka[openai]
```

### 2. Set Your API Key

```bash
export OPENAI_API_KEY="your-openai-key-here"
```

### 3. Your First Improvement (RECOMMENDED)

**Start with presets - they cover 90% of use cases:**

```python
import asyncio
import sifaka

async def main():
    # Just pick the preset that matches your use case
    result = await sifaka.academic_writing("Explain quantum computing")
    result = await sifaka.creative_writing("Write a short story about AI")
    result = await sifaka.technical_docs("Document the API endpoints")
    result = await sifaka.quick_draft("Brainstorm marketing ideas")

    print(result.final_text)

asyncio.run(main())
```

**That's it!** Each preset automatically:
- ✅ Uses the right model for the task
- ✅ Applies appropriate validation criteria
- ✅ Enables relevant critics for improvement
- ✅ Iterates until quality standards are met

**Available Presets:**
- `academic_writing` / `academic` - Formal tone, comprehensive coverage, 300+ words
- `creative_writing` / `creative` - Engaging narrative, emotional impact, up to 800 words
- `technical_docs` / `technical` - Clear explanations, practical examples, 200+ words
- `business_writing` / `business` - Professional tone, concise communication, up to 500 words
- `quick_draft` / `draft` - Speed over perfection, minimal processing, 2 rounds
- `high_quality` / `premium` - Maximum quality, all critics enabled, 7 rounds

### Alternative: Simple Customization

**When presets aren't enough:**

```python
import asyncio
import sifaka

async def main():
    # Simple customization with sensible defaults
    result = await sifaka.improve(
        "Write about renewable energy",
        max_rounds=5,
        model="openai:gpt-4",
        min_length=200
    )
    print(result.final_text)

asyncio.run(main())
```

## 🎯 Common Use Cases (3 minutes)

### Content Length Control

```python
# Ensure your content meets specific length requirements
result = await sifaka.improve(
    "Explain machine learning",
    min_length=200,    # At least 200 characters
    max_length=500     # No more than 500 characters
)
```

### Multiple Quality Critics

```python
# Apply multiple research-backed improvement techniques
result = await sifaka.improve(
    "Write a professional email about project delays",
    critics=["reflexion", "constitutional", "self_refine"],
    max_rounds=5
)
```

### Different AI Models

```python
# Use different models for generation
result = await sifaka.improve(
    "Create a technical explanation",
    model="anthropic:claude-3-sonnet",  # or "openai:gpt-4", "groq:llama-3.1-8b"
    min_length=300
)
```

### Content Requirements

```python
# Ensure specific topics are covered
from sifaka import SifakaEngine, SifakaConfig
from sifaka.validators import ContentValidator

config = SifakaConfig(
    model="openai:gpt-4",
    max_iterations=3,
    validators=[
        ContentValidator(required=["benefits", "challenges", "examples"])
    ]
)

engine = SifakaEngine(config=config)
result = await engine.think("Discuss renewable energy adoption")
```

## 🔍 Understanding the Results

Every Sifaka result gives you complete visibility:

```python
result = await sifaka.improve("Write about AI ethics")

# See the final improved text
print(f"Final text: {result.final_text}")

# Check how many improvement rounds it took
print(f"Iterations: {result.iteration}")

# See if all validations passed
print(f"Validation passed: {result.validation_passed()}")

# Access the complete audit trail
for generation in result.generations:
    print(f"Round {generation.iteration}: {generation.text[:100]}...")
```

## 🛠️ API Hierarchy - Progressive Disclosure

Sifaka provides **ONE clear path** with increasing complexity as needed:

### 🎯 PRIMARY: Configuration Presets (90% of users)
```python
# Direct access to presets - the recommended approach
result = await sifaka.academic_writing("Your prompt")
result = await sifaka.quick_draft("Your prompt")
result = await sifaka.high_quality("Your prompt")
```

### ⚙️ SECONDARY: Simple Customization (8% of users)
```python
# When presets aren't enough, add simple parameters
result = await sifaka.improve(
    "Your prompt",
    max_rounds=5,
    model="openai:gpt-4",
    min_length=200
)
```

### 🔧 ADVANCED: Full Control (2% of users)
```python
# For complex use cases requiring full configuration
from sifaka.advanced import SifakaEngine, SifakaConfig

config = SifakaConfig(
    model="openai:gpt-4",
    max_iterations=5,
    critics=["reflexion", "constitutional"],
    validators=[...],  # Custom validators
    enable_persistence=True,
)
engine = SifakaEngine(config=config)
result = await engine.think("Your prompt")
```

**Key Benefits of This Hierarchy:**
- ✅ **Simple Start**: 90% of users never need to leave the preset level
- ✅ **Progressive Disclosure**: Complexity only when needed
- ✅ **Clear Path**: No confusion about which API to use
- ✅ **Backwards Compatible**: Existing code still works

## 🎓 What You've Learned

In just 5 minutes, you've learned how to:
- ✅ Install and set up Sifaka
- ✅ Improve text with one line of code
- ✅ Control content length and quality
- ✅ Apply multiple improvement techniques
- ✅ Use different AI models
- ✅ Understand the results and audit trail
- ✅ Choose the right configuration level for your needs

## 🚀 Next Steps

Ready to dive deeper? Check out:

- **[Architecture Guide](ARCHITECTURE.md)** - Understand how Sifaka works under the hood
- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Examples](../examples/)** - Real-world usage examples
- **[Performance Guide](PERFORMANCE.md)** - Optimization tips and benchmarks

## 💡 Pro Tips

1. **Start Simple**: Use the one-liner API first, then add complexity as needed
2. **Experiment with Critics**: Different critics excel at different types of content
3. **Monitor Iterations**: If you're hitting max_rounds often, adjust your validation criteria
4. **Use Multiple Models**: Different models have different strengths - experiment!
5. **Check the Audit Trail**: The complete history helps you understand what's working

## 🆘 Need Help?

- **Common Issues**: Check [Troubleshooting](TROUBLESHOOTING.md)
- **API Questions**: See [API Reference](API_REFERENCE.md)
- **Examples**: Browse [examples/](../examples/) directory
- **Community**: Join our discussions on GitHub

Welcome to the future of AI text generation! 🎉
