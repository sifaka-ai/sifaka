# Advanced Features Tutorial - Power User Guide

**Goal**: Unlock Sifaka's full potential with custom validators, critics, and advanced workflows.

**Prerequisites**: Complete [Basic Usage Tutorial](02-basic-usage.md)

## Advanced API Access

For complex use cases, use the advanced API:

```python
from sifaka.advanced import SifakaEngine, SifakaConfig, SifakaDependencies
```

## Custom Validators

Validators check if generated content meets your specific criteria.

### Built-in Validators

```python
import asyncio
from sifaka.advanced import SifakaEngine, SifakaConfig
from sifaka.validators import (
    LengthValidator,
    ContentValidator, 
    SentimentValidator,
    FormatValidator
)

async def builtin_validators():
    config = SifakaConfig(
        model="openai:gpt-4",
        max_iterations=5,
        validators=[
            LengthValidator(min_length=200, max_length=800),
            ContentValidator(
                required=["examples", "benefits", "challenges"],
                prohibited=["outdated", "deprecated"]
            ),
            SentimentValidator(required_sentiment="positive"),
            FormatValidator(format_type="markdown")
        ]
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think("Write about renewable energy adoption")

asyncio.run(builtin_validators())
```

### Custom Validator Creation

```python
from sifaka.validators.base import BaseValidator, ValidationResult
import re

class EmailValidator(BaseValidator):
    """Validates that text contains valid email addresses."""
    
    def __init__(self, min_emails: int = 1):
        super().__init__()
        self.min_emails = min_emails
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    async def validate_async(self, text: str, context: dict = None) -> ValidationResult:
        emails = re.findall(self.email_pattern, text)
        
        if len(emails) >= self.min_emails:
            return ValidationResult(
                passed=True,
                score=1.0,
                feedback=f"Found {len(emails)} valid email addresses",
                validator="email_validator"
            )
        else:
            return ValidationResult(
                passed=False,
                score=len(emails) / self.min_emails,
                feedback=f"Need at least {self.min_emails} emails, found {len(emails)}",
                validator="email_validator",
                suggestions=[
                    "Include contact email addresses",
                    "Add support email information",
                    "Provide team member emails"
                ]
            )

# Usage
async def custom_validator_example():
    config = SifakaConfig(
        model="openai:gpt-4",
        validators=[EmailValidator(min_emails=2)]
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think("Write a business contact page")

asyncio.run(custom_validator_example())
```

## Advanced Critics Configuration

### Multiple Critics with Weights

```python
async def weighted_critics():
    config = SifakaConfig(
        model="openai:gpt-4",
        max_iterations=7,
        critics={
            "reflexion": "openai:gpt-4",           # Self-reflection
            "constitutional": "anthropic:claude-3-sonnet",  # Principle-based
            "self_refine": "gemini-1.5-pro",      # Iterative improvement
            "n_critics": "openai:gpt-3.5-turbo"   # Multi-perspective
        },
        validation_weight=0.7,  # 70% validation feedback
        critic_weight=0.3,      # 30% critic feedback
        always_apply_critics=True  # Apply critics even if validation passes
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think("Write a comprehensive analysis of AI ethics")

asyncio.run(weighted_critics())
```

### Custom Critic Creation

```python
from sifaka.critics.base import BaseCritic, CritiqueResult

class TechnicalAccuracyCritic(BaseCritic):
    """Critic that focuses on technical accuracy and precision."""
    
    def __init__(self, model_name: str = "openai:gpt-4"):
        super().__init__(model_name=model_name)
    
    async def critique_async(self, text: str, context: dict = None) -> CritiqueResult:
        prompt = f"""
        Analyze this technical content for accuracy and precision:
        
        {text}
        
        Focus on:
        1. Technical correctness
        2. Precision of terminology
        3. Completeness of explanations
        4. Clarity for the target audience
        
        Provide specific suggestions for improvement.
        """
        
        # Use the critic's model to generate feedback
        response = await self._generate_critique(prompt)
        
        return CritiqueResult(
            feedback=response,
            suggestions=self._extract_suggestions(response),
            confidence=0.8,
            critic_name="technical_accuracy"
        )

# Usage
async def custom_critic_example():
    config = SifakaConfig(
        model="openai:gpt-4",
        critics={"technical": TechnicalAccuracyCritic()}
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think("Explain how neural networks work")

asyncio.run(custom_critic_example())
```

## Advanced Workflows

### Multi-Stage Processing

```python
async def multi_stage_workflow():
    # Stage 1: Quick draft
    draft_config = SifakaConfig(
        model="openai:gpt-3.5-turbo",
        max_iterations=2,
        critics=["reflexion"]
    )
    
    draft_engine = SifakaEngine(config=draft_config)
    draft = await draft_engine.think("Write about sustainable technology")
    
    # Stage 2: Content enhancement
    enhance_config = SifakaConfig(
        model="openai:gpt-4",
        max_iterations=3,
        validators=[LengthValidator(min_length=500)],
        critics=["constitutional", "self_refine"]
    )
    
    enhance_engine = SifakaEngine(config=enhance_config)
    enhanced = await enhance_engine.think(
        f"Expand and improve this content: {draft.final_text}"
    )
    
    # Stage 3: Final polish
    polish_config = SifakaConfig(
        model="anthropic:claude-3-sonnet",
        max_iterations=2,
        critics=["meta_evaluation"]
    )
    
    polish_engine = SifakaEngine(config=polish_config)
    final = await polish_engine.think(
        f"Polish this content for publication: {enhanced.final_text}"
    )
    
    return final

result = await multi_stage_workflow()
```

### Conditional Processing

```python
async def conditional_processing():
    # Initial generation
    result = await sifaka.academic_writing("Explain machine learning algorithms")
    
    # Conditional improvements based on results
    if len(result.final_text) < 500:
        # Too short - expand
        result = await sifaka.improve(
            f"Expand this explanation with more detail: {result.final_text}",
            min_length=500,
            max_rounds=3
        )
    
    if not result.validation_passed():
        # Quality issues - apply stronger critics
        config = SifakaConfig(
            model="openai:gpt-4",
            max_iterations=5,
            critics={
                "reflexion": "openai:gpt-4",
                "constitutional": "anthropic:claude-3-sonnet",
                "self_refine": "gemini-1.5-pro"
            },
            always_apply_critics=True
        )
        
        engine = SifakaEngine(config=config)
        result = await engine.think(result.final_text)
    
    # Final quality check
    if result.iteration >= 5:
        print("Warning: Hit maximum iterations - consider adjusting criteria")
    
    return result

final_result = await conditional_processing()
```

## Memory and Performance Optimization

### Memory Management

```python
async def memory_optimized():
    config = SifakaConfig(
        model="openai:gpt-4",
        max_iterations=10,
        # Memory optimization settings
        auto_optimize_memory=True,
        memory_optimization_interval=3,  # Optimize every 3 iterations
        keep_last_n_iterations=2,        # Keep only last 2 iterations
        max_messages_per_iteration=5,    # Limit conversation history
        max_tool_result_size_bytes=5120  # Limit tool result size
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think("Write a comprehensive guide to Python programming")

asyncio.run(memory_optimized())
```

### Caching and Performance

```python
async def performance_optimized():
    config = SifakaConfig(
        model="openai:gpt-4",
        max_iterations=5,
        # Performance settings
        enable_caching=True,
        cache_size=1000,
        enable_timing=True,
        enable_logging=True,
        log_level="INFO"
    )
    
    engine = SifakaEngine(config=config)
    
    # First run - will be cached
    result1 = await engine.think("Explain neural networks")
    
    # Second run - will use cache if similar
    result2 = await engine.think("Explain neural networks")
    
    print(f"First run: {result1.iteration} iterations")
    print(f"Second run: {result2.iteration} iterations")

asyncio.run(performance_optimized())
```

## Integration Patterns

### With PydanticAI Tools

```python
from pydantic_ai import Agent
from pydantic_ai.tools import Tool

# Custom tool for web search
web_search_tool = Tool(
    name="web_search",
    description="Search the web for current information",
    # Tool implementation here
)

async def tool_integration():
    # Create agent with tools
    agent = Agent(
        "openai:gpt-4",
        system_prompt="You can search the web for current information.",
        tools=[web_search_tool]
    )
    
    # Use with Sifaka
    config = SifakaConfig(
        generator=agent,  # Use custom agent instead of model string
        max_iterations=3,
        critics=["reflexion"]
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think("What are the latest developments in AI research?")

asyncio.run(tool_integration())
```

### With External APIs

```python
import httpx

async def external_api_integration():
    # Custom validator that checks against external API
    class FactCheckValidator(BaseValidator):
        async def validate_async(self, text: str, context: dict = None) -> ValidationResult:
            # Call fact-checking API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.factcheck.com/verify",
                    json={"text": text}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return ValidationResult(
                        passed=data["accuracy_score"] > 0.8,
                        score=data["accuracy_score"],
                        feedback=data["feedback"],
                        validator="fact_check"
                    )
    
    config = SifakaConfig(
        model="openai:gpt-4",
        validators=[FactCheckValidator()]
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think("Write about recent scientific discoveries")

asyncio.run(external_api_integration())
```

## Debugging and Monitoring

### Detailed Logging

```python
async def detailed_monitoring():
    config = SifakaConfig(
        model="openai:gpt-4",
        max_iterations=5,
        enable_logging=True,
        log_level="DEBUG",
        log_content=True  # Include content in logs
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think("Write about quantum computing")
    
    # Access detailed information
    print(f"Total iterations: {result.iteration}")
    print(f"Validations run: {len(result.validations)}")
    print(f"Critiques applied: {len(result.critiques)}")
    
    # Detailed iteration analysis
    for i, generation in enumerate(result.generations):
        print(f"\nIteration {i+1}:")
        print(f"  Text length: {len(generation.text)}")
        print(f"  Timestamp: {generation.timestamp}")
        
    # Validation details
    for validation in result.validations:
        print(f"\nValidator: {validation.validator}")
        print(f"  Passed: {validation.passed}")
        print(f"  Score: {validation.score}")
        print(f"  Feedback: {validation.feedback}")

asyncio.run(detailed_monitoring())
```

## What's Next?

🎉 **You've mastered Sifaka's advanced features!**

**Next Steps**:
- **[Custom Critics Tutorial](04-custom-critics.md)** - Build sophisticated improvement logic
- **[API Reference](../API_REFERENCE.md)** - Complete API documentation

## Key Takeaways

✅ **Custom validators** for domain-specific requirements  
✅ **Custom critics** for specialized improvement logic  
✅ **Multi-stage workflows** for complex processing  
✅ **Performance optimization** for production use  
✅ **Integration patterns** with external systems  
✅ **Detailed monitoring** for debugging and optimization  

**You're now a Sifaka power user! 🚀**

---

**Previous**: 📚 **[Basic Usage](02-basic-usage.md)**  
**Next**: 📚 **[Custom Critics](04-custom-critics.md)**
