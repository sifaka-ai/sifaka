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

### 3. Your First Improvement

```python
import asyncio
import sifaka

async def main():
    # One line to improve any text
    result = await sifaka.improve("Write about renewable energy")
    print(result.final_text)

asyncio.run(main())
```

**That's it!** Sifaka automatically:
- ✅ Generates text with GPT-4
- ✅ Validates the output quality
- ✅ Applies Reflexion critic for improvement
- ✅ Iterates until quality standards are met

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

## 🛠️ Configuration Levels

Sifaka grows with your needs:

### Level 1: One-Liner (80% of users)
```python
result = await sifaka.improve("Your prompt", max_rounds=3)
```

### Level 2: Simple Configuration (15% of users)
```python
config = SifakaConfig.simple(
    model="openai:gpt-4",
    max_iterations=5,
    min_length=200,
    critics=["reflexion", "constitutional"]
)
engine = SifakaEngine(config=config)
result = await engine.think("Your prompt")
```

### Level 3: Builder Pattern (4% of users)
```python
config = (SifakaConfig.builder()
         .model("openai:gpt-4")
         .max_iterations(5)
         .min_length(200)
         .with_reflexion()
         .with_constitutional()
         .build())
```

### Level 4: Full Control (1% of users)
```python
from sifaka.graph import SifakaDependencies
from sifaka.validators import LengthValidator
from sifaka.critics import ReflexionCritic

deps = SifakaDependencies(
    generator="openai:gpt-4",
    validators=[LengthValidator(min_length=100)],
    critics={"reflexion": ReflexionCritic()},
    validation_weight=0.7,  # 70% validation, 30% critic feedback
    critic_weight=0.3
)
engine = SifakaEngine(dependencies=deps)
```

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
