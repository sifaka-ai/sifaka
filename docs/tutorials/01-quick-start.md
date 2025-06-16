# Quick Start Tutorial - 5 Minutes to Success

**Goal**: Get up and running with Sifaka in 5 minutes or less.

## What is Sifaka?

Sifaka transforms AI text generation from "hope it's good" to "guarantee it meets your standards." Instead of just generating text, Sifaka creates a feedback loop where AI systems validate, critique, and iteratively improve their outputs until they meet your quality criteria.

**The Magic**: Research-backed improvement techniques (Reflexion, Constitutional AI, Self-RAG) implemented as working code.

## Installation (30 seconds)

```bash
# Core installation
pip install sifaka

# With model providers (recommended)
pip install sifaka[openai,anthropic,gemini]
```

## Set Your API Key (30 seconds)

```bash
# Choose your preferred provider
export OPENAI_API_KEY="your-openai-key-here"
# OR
export ANTHROPIC_API_KEY="your-anthropic-key-here"
# OR  
export GEMINI_API_KEY="your-gemini-key-here"
```

## Your First Success (2 minutes)

### Option 1: Zero Configuration (Recommended)

```python
import asyncio
import sifaka

async def main():
    # Just pick the preset that matches your use case
    result = await sifaka.academic_writing("Explain quantum computing")
    print(result.final_text)

asyncio.run(main())
```

**That's it!** Sifaka automatically:
- ✅ Chooses the right model for academic writing
- ✅ Validates the output meets academic standards
- ✅ Applies research-backed improvement techniques
- ✅ Iterates until quality criteria are met

### Option 2: Simple Customization

```python
import asyncio
import sifaka

async def main():
    # When you need specific parameters
    result = await sifaka.improve(
        "Write about renewable energy",
        max_rounds=5,
        model="openai:gpt-4",
        min_length=200
    )
    print(result.final_text)

asyncio.run(main())
```

## Available Presets (1 minute)

Choose the preset that matches your use case:

```python
import asyncio
import sifaka

async def main():
    # Academic writing (formal, comprehensive, 300+ words)
    result = await sifaka.academic_writing("Explain machine learning")
    
    # Creative writing (engaging, narrative, up to 800 words)  
    result = await sifaka.creative_writing("Write a short story about AI")
    
    # Technical documentation (clear, practical, 200+ words)
    result = await sifaka.technical_docs("Document the REST API")
    
    # Business writing (professional, concise, up to 500 words)
    result = await sifaka.business_writing("Write a project proposal")
    
    # Quick draft (speed over perfection, 2 rounds)
    result = await sifaka.quick_draft("Brainstorm marketing ideas")
    
    # High quality (maximum quality, all critics, 7 rounds)
    result = await sifaka.high_quality("Write a research summary")

asyncio.run(main())
```

## Understanding Your Results (1 minute)

Every Sifaka result gives you complete visibility:

```python
result = await sifaka.academic_writing("Explain neural networks")

# The improved final text
print(f"Final text: {result.final_text}")

# How many improvement rounds it took
print(f"Iterations: {result.iteration}")

# Whether all validations passed
print(f"All validations passed: {result.validation_passed()}")

# Complete audit trail
for generation in result.generations:
    print(f"Round {generation.iteration}: {generation.text[:100]}...")
```

## Common Use Cases (30 seconds)

### Content Length Control
```python
# Ensure specific length requirements
result = await sifaka.improve(
    "Explain machine learning",
    min_length=200,    # At least 200 characters
    max_length=500     # No more than 500 characters
)
```

### Multiple Quality Critics
```python
# Apply multiple improvement techniques
result = await sifaka.improve(
    "Write a professional email",
    critics=["reflexion", "constitutional", "self_refine"],
    max_rounds=5
)
```

### Different AI Models
```python
# Use different models
result = await sifaka.improve(
    "Create a technical explanation",
    model="anthropic:claude-3-sonnet",  # or "openai:gpt-4", "groq:llama-3.1-8b"
    min_length=300
)
```

## Troubleshooting (30 seconds)

### API Key Issues
```python
# Error: "API key error"
# Solution: Set your API key
export OPENAI_API_KEY="your-key-here"
```

### Import Errors
```python
# Error: "No module named 'sifaka'"
# Solution: Install Sifaka
pip install sifaka
```

### Model Errors
```python
# Error: "Model not found"
# Solution: Install provider dependencies
pip install sifaka[openai,anthropic,gemini]
```

## What's Next?

🎉 **Congratulations!** You've successfully used Sifaka to improve AI text generation.

**Next Steps**:
- **[Basic Usage Tutorial](02-basic-usage.md)** - Learn common patterns and workflows
- **[Advanced Features Tutorial](03-advanced-features.md)** - Unlock power user features
- **[Custom Critics Tutorial](04-custom-critics.md)** - Extend Sifaka with your own logic

## Key Takeaways

✅ **Zero Configuration**: Presets work out of the box  
✅ **Progressive Complexity**: Start simple, add features as needed  
✅ **Complete Visibility**: Full audit trail of every improvement  
✅ **Research-Backed**: Implements cutting-edge AI research papers  
✅ **Quality Guaranteed**: Iterative improvement until standards are met  

**You're now ready to transform your AI text generation! 🚀**

---

**Time to Success**: ⏱️ **5 minutes or less**  
**Next Tutorial**: 📚 **[Basic Usage Patterns](02-basic-usage.md)**
