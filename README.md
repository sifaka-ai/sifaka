# Sifaka: Reflection and Reliability for LLMs

<div style="display: flex; align-items: start;">
<div style="flex: 1;">

**Sifaka** is an open-source framework that adds **reflection and reliability** to large language model (LLM) applications. It helps developers build safer, more reliable AI systems by:

- Catching hallucinations before they reach users
- Enforcing rules and tone consistency
- Providing transparency and auditability

Whether you're building AI-powered tools for legal research, customer support, or creative generation, Sifaka makes your outputs **safer, smarter, and more transparent.**

</div>
<div style="flex: 0 0 40%; margin-left: 20px;">
  <img src="assets/sifika.jpg" alt="Sifaka - A lemur species from Madagascar" width="100%"/>
</div>
</div>

[![PyPI version](https://badge.fury.io/py/sifaka.svg)](https://badge.fury.io/py/sifaka)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🌟 Features

- **Rule-Based Validation**: Define constraints to validate LLM outputs
- **Reflection Through Critique**: Improve responses through self-reflection
- **Modular Architecture**: Easily extend with custom rules and components
- **Provider Agnostic**: Works with OpenAI, Anthropic, and other LLM providers
- **Tracing & Auditing**: Comprehensive logging of the reflection process
- **Pydantic Integration**: Leverage Pydantic for structured data validation

## 📦 Installation

```bash
# Basic installation
pip install sifaka

# Install with sentiment analysis support
pip install sifaka[sentiment]

# With OpenAI support
pip install sifaka[openai]

# With Anthropic support
pip install sifaka[anthropic]

# With all integrations
pip install sifaka[all]

# With development tools
pip install sifaka[dev]
```

## 🔑 API Keys and Configuration

Sifaka supports multiple LLM providers and requires API keys for authentication. Here's how to set up your API keys:

### Environment Variables

The recommended way to set API keys is through environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google (if using Vertex AI)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

### Configuration File

Alternatively, you can create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
```

Then load it in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Direct Configuration

You can also pass API keys directly when initializing providers:

```python
from sifaka.models import OpenAIProvider, AnthropicProvider

# OpenAI with direct API key
openai_model = OpenAIProvider(
    model_name="gpt-4",
    api_key="your-openai-api-key"
)

# Anthropic with direct API key
anthropic_model = AnthropicProvider(
    model_name="claude-3-haiku-20240307",
    api_key="your-anthropic-api-key"
)
```

### Example Usage with API Keys

Here's a complete example showing different ways to use API keys:

```python
from sifaka import Reflector
from sifaka.models import OpenAIProvider, AnthropicProvider
from sifaka.rules import ProhibitedContentRule
from sifaka.critics import PromptCritic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Method 1: Using environment variables
# openai_model = AnthropicProvider(model_name="gpt-4")

# Method 2: Using direct API key
anthropic_model = AnthropicProvider(
    model_name="claude-3-haiku-20240307"
)

# Create rules
prohibited_terms = ProhibitedContentRule(
    name="content_filter",
    description="Checks for prohibited or inappropriate content",
    config={
        "prohibited_terms": ["controversial", "inappropriate"]
    }
)

# Create a critic for improving outputs that fail validation
critic = PromptCritic(model=anthropic_model)

# Create a reflector with rules and critic
# Note: critique functionality is automatically enabled when a critic is provided
reflector = Reflector(
    name="content_validator",
    model=anthropic_model,
    rules=[prohibited_terms],
    critic=critic
)

# Use the reflector
result = reflector.reflect(
    "Write a professional email about a search project update"
)
print(result)
```

### Security Best Practices

1. **Never commit API keys** to version control
2. Use **environment variables** in production
3. Consider using a **secrets management service** for production
4. Rotate API keys **regularly**
5. Use **least privilege** API keys with minimal permissions

## 🚀 Quick Start

```python
from sifaka import Reflector
from sifaka.models import AnthropicProvider
from sifaka.rules import LengthRule, ProhibitedContentRule, FormatRule
from sifaka.critics import PromptCritic
from sifaka.utils.logging import get_logger
import logging
from dotenv import load_dotenv

# Configure logging to show only relevant information
logging.basicConfig(level=logging.WARNING)  # Set base logging to WARNING
logger = get_logger(__name__, level=logging.INFO)  # Show info for our script
logging.getLogger("sifaka").setLevel(logging.INFO)  # Show info for Sifaka
logging.getLogger("anthropic").setLevel(logging.WARNING)  # Silence Anthropic debug logs
logging.getLogger("httpx").setLevel(logging.WARNING)  # Silence HTTP debug logs
logging.getLogger("httpcore").setLevel(logging.WARNING)  # Silence HTTP debug logs

# Load environment variables
load_dotenv()

# Initialize the provider
provider = AnthropicProvider(model_name="claude-3-haiku-20240307")

# Create rules
length_rule = LengthRule(
    name="length_check",
    description="Checks if output length is within bounds",
    config={
        "min_length": 100,
        "max_length": 500
    }
)

prohibited_terms = ProhibitedContentRule(
    name="content_filter",
    description="Checks for prohibited or inappropriate content",
    config={
        "prohibited_terms": ["controversial", "inappropriate"]
    }
)

format_rule = FormatRule(
    name="format_check",
    description="Ensures output is in markdown format",
    config={
        "required_format": "markdown"
    }
)

# Create a critic for improving outputs that fail validation
critic = PromptCritic(
    model=provider,
    system_prompt=(
        "You are an editor that makes text more concise while preserving key information. "
        "When asked to adjust text length, you MUST ensure the output is within the specified character limits. "
        "Always return your response as a plain string, not a dictionary or any other format."
    )
)

# Create a reflector with rules and critic
reflector = Reflector(
    name="content_validator",
    model=provider,
    rules=[length_rule, prohibited_terms, format_rule],
    critic=critic
)

# Use the reflector
try:
    result = reflector.reflect(
        "Write a professional email about a search project update",
        max_attempts=3  # Try up to 3 times to fix any violations
    )
    logger.info("Final result:\n%s", result)
except RuntimeError as e:
    logger.error("Failed to generate valid output: %s", e)
```

## 🧩 Architecture

Sifaka is built around a modular architecture with four main components:

1. **Reflector**: The core component that orchestrates the reflection process
2. **Rules**: Validate LLM outputs against specific criteria
3. **Critique**: Improve LLM outputs based on rule violations
4. **Model Providers**: Interface with different LLM providers

### Reflector

The `Reflector` class is the main entry point for Sifaka. It takes a list of rules and critique options, and applies them to LLM outputs.

```python
from sifaka import Reflector
from sifaka.models import AnthropicProvider

# Initialize model
model = AnthropicProvider(model_name="claude-3-haiku-20240307")

# Create reflector
reflector = Reflector(
    name="content_validator",
    model=model,
    rules=[rule1, rule2],  # List of rules to apply
    critique=True          # Enable critique
)

# Use the reflector
result = reflector.reflect("Your prompt here")
```

### Rules

Rules validate LLM outputs against specific criteria. Sifaka comes with several built-in rules:

```python
from sifaka.rules import (
    LengthRule,
    ProhibitedContentRule,
    SentimentRule,
    ToxicityRule,
    FormatRule
)
from sifaka.rules.legal import LegalCitationRule

# Create rules
length_rule = LengthRule(
    name="min_max_length_rule",
    description="Checks if output length is between 100 and 1000 characters",
    config={"min_length": 100, "max_length": 1000}
)

content_rule = ProhibitedContentRule(
    name="content_rule",
    description="Checks for prohibited terms",
    config={"prohibited_terms": ["bad", "inappropriate"]}
)

format_rule = FormatRule(
    name="format_rule",
    description="Validates output format",
    config={"required_format": "markdown"}
)

legal_rule = LegalCitationRule(
    name="legal_citations_rule",
    description="Validates legal citations in the output"
)
```

### Creating Custom Rules

You can create custom rules by inheriting from the `Rule` base class:

```python
from sifaka.rules import Rule, RuleResult
from typing import Optional, Dict, Any

class CustomRule(Rule):
    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(name=name, description=description, config=config or {}, **kwargs)
        # Initialize your rule-specific configuration here

    def validate(self, output: str) -> RuleResult:
        # Implement your validation logic here
        return RuleResult(
            valid=True,
            messages=["Validation passed"],
            metadata={"key": "value"}
        )
```

## Custom Rules

You can create custom rules by inheriting from the `Rule` base class:

```python
from sifaka.rules import Rule, RuleResult
from typing import Optional, Dict, Any

class CustomRule(Rule):
    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(name=name, description=description, config=config or {}, **kwargs)
        # Initialize your rule-specific configuration here

    def validate(self, output: str) -> RuleResult:
        # Implement your validation logic here
        return RuleResult(
            valid=True,
            messages=["Validation passed"],
            metadata={"key": "value"}
        )
```

## Framework Integrations

### LangGraph Integration

Sifaka integrates seamlessly with LangGraph for building complex LLM workflows:

```python
from langgraph.graph import Graph
from sifaka.integrations.langgraph import wrap_graph
from sifaka.rules import LengthRule, ProhibitedContentRule

# Create your LangGraph
graph = Graph()
# ... configure your graph ...

# Create rules
rules = [
    LengthRule(
        name="length_check",
        description="Checks if output length is within bounds",
        config={"min_length": 100, "max_length": 1000}
    ),
    ProhibitedContentRule(
        name="content_filter",
        description="Checks for prohibited terms",
        config={"prohibited_terms": ["bad", "worse", "worst"]}
    )
]

# Wrap the graph with Sifaka's features
sifaka_graph = wrap_graph(
    graph=graph,
    rules=rules,
    critique=True
)

# Run the graph
output = sifaka_graph({"input": "Your input here"})
```

Sifaka supports various LangGraph components:
- `wrap_graph()` - For basic graphs
- `wrap_state_graph()` - For state-based graphs
- `wrap_tool_node()` - For tool nodes
- `wrap_channel()` - For channels

### LangChain Integration

Sifaka also integrates with LangChain for enhanced LLM application development:

```python
from langchain.chains import LLMChain
from sifaka.integrations.langchain import wrap_chain
from sifaka.rules import LengthRule, ProhibitedContentRule

# Create your LangChain chain
chain = LLMChain(...)

# Create rules
rules = [
    LengthRule(
        name="length_check",
        description="Checks if output length is within bounds",
        config={"min_length": 100, "max_length": 1000}
    ),
    ProhibitedContentRule(
        name="content_filter",
        description="Checks for prohibited terms",
        config={"prohibited_terms": ["bad", "worse", "worst"]}
    )
]

# Wrap the chain with Sifaka's features
sifaka_chain = wrap_chain(
    chain=chain,
    rules=rules,
    critique=True
)

# Run the chain
output = sifaka_chain.run("Your input here")
```

Sifaka provides several LangChain integrations:
- `wrap_chain()` - For LLM chains
- `wrap_memory()` - For memory components
- `wrap_callback_handler()` - For callback handlers
- Document processing integration with rules

## Installation

```bash
pip install sifaka  # Basic installation
pip install sifaka[langgraph]  # With LangGraph support
pip install sifaka[langchain]  # With LangChain support
pip install sifaka[all]  # With all integrations
```

## Requirements

- Python 3.8+
- anthropic
- pydantic
- langgraph (optional)
- langchain (optional)

## Built-in Rules

```

## 🔄 How Reflection Works

Here's how the reflection process works:

```mermaid
graph TD
    A[Input Prompt] --> B[Generate Initial Output]
    B --> C[Validate Against All Rules]
    C --> D{Any Violations?}
    D -->|No| E[Return Output]
    D -->|Yes| F[Send to Critic]
    F --> G[Generate Improved Output]
    G --> C
    G -->|Max Attempts| H[Raise Error]
```

Let's walk through an example:

```python
from sifaka import Reflector
from sifaka.models import AnthropicProvider
from sifaka.rules import LengthRule
from sifaka.critics import PromptCritic
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__, level=logging.INFO)

# Initialize provider
provider = AnthropicProvider(model_name="claude-3-sonnet-20240229")

# Create a simple length rule
length_rule = LengthRule(
    name="length_check",
    description="Keep output between 100-200 characters",
    config={
        "min_length": 100,
        "max_length": 1200
    }
)

# Create a critic
critic = PromptCritic(
    name="content_improver",
    description="Improves content based on rule violations",
    model=provider,
    system_prompt=(
        "You are an editor that makes text more concise while preserving key information. "
        "When asked to adjust text length, you MUST ensure the output is within the specified character limits."
    ),
    user_prompt_template=(
        "Please improve this text to STRICTLY meet the length requirements:\n\n"
        "Text: {output}\n\n"
        "Issues: {feedback}\n\n"
        "IMPORTANT: Your response MUST be between {min_length}-{max_length} characters.\n"
        "Current length: {current_length} characters\n\n"
        "Provide a concise version that fits these exact requirements."
    )
)

# Create reflector with system prompt for initial generation
reflector = Reflector(
    name="length_validator",
    model=provider,
    rules=[length_rule],
    critic=critic
)

# Example walkthrough
logger.info("Step 1: Starting with input prompt...")
prompt = f"""
Explain quantum computing.

IMPORTANT: Your response must be between 100-1200 characters in length.
"""

logger.info("Step 2: Generate initial output...")
try:
    result = reflector.reflect(prompt, max_attempts=3)
    logger.info("Success! Final output:\n%s", result)
except RuntimeError as e:
    logger.error("Failed after max attempts: %s", e)

"""
The process will look like this:

1. Initial Generation:
   - Model generates first response
   - Length Rule checks character count
   - If too long/short, sends to critic

2. First Attempt (if needed):
   - Critic receives violation info
   - Generates improved version
   - Rules check again

3. Second Attempt (if needed):
   - If still violating, critic tries again
   - Rules check once more

4. Final Result:
   - Either succeeds within max attempts
   - Or raises error if can't fix
"""
```

This example shows:
1. How rules validate the output
2. How the critic tries to fix violations
3. The multiple attempt process
4. Success/failure handling

Each rule can define its own validation logic, and the critic will try to fix any violations while preserving the essential content.

## 🚀 Performance Optimization

Sifaka provides several ways to optimize performance:

### Parallel Rule Validation

Enable parallel validation when rules are independent:

```python
from sifaka import Reflector
from sifaka.models import AnthropicProvider
from sifaka.rules import LengthRule, ProhibitedContentRule

reflector = Reflector(
    name="optimized_validator",
    model=provider,
    rules=[length_rule, prohibited_terms],
    parallel_validation=True  # Enable parallel validation
)
```

### Rule Caching

Cache expensive rule validations:

```python
from sifaka.rules import ToxicityRule
from functools import lru_cache

class CachedToxicityRule(ToxicityRule):
    @lru_cache(maxsize=1000)
    def validate(self, output: str) -> RuleResult:
        return super().validate(output)

# Use the cached version
toxicity_rule = CachedToxicityRule(
    name="toxicity_check",
    description="Checks for toxic content with caching",
    config={"max_toxicity": 0.5}
)
```

### Batch Processing

Group validations for efficiency:

```python
from sifaka import BatchReflector

batch_reflector = BatchReflector(
    name="batch_validator",
    model=provider,
    rules=[length_rule, prohibited_terms],
    batch_size=10  # Process 10 items at once
)

# Process multiple items efficiently
results = batch_reflector.reflect_batch([
    "First prompt",
    "Second prompt",
    "Third prompt"
])
```

### Memory Management

Control memory usage with configuration:

```python
reflector = Reflector(
    name="memory_efficient",
    model=provider,
    rules=[length_rule, prohibited_terms],
    max_cache_size=1000,  # Limit cache size
    clear_history=True    # Clear history after each reflection
)
```

### Model Configuration

Optimize model settings for your use case:

```python
from sifaka.models import AnthropicProvider

provider = AnthropicProvider(
    model_name="claude-3-haiku-20240307",
    request_timeout=30,     # Adjust timeout
    max_retries=2,         # Limit retries
    temperature=0.1        # Lower temperature for faster responses
)
```

### Performance Monitoring

Monitor and tune performance:

```python
from sifaka.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
reflector = Reflector(
    name="monitored_validator",
    model=provider,
    rules=[length_rule, prohibited_terms],
    performance_monitor=monitor
)

# Get performance metrics
metrics = monitor.get_metrics()
print(f"Average validation time: {metrics['avg_validation_time_ms']}ms")
print(f"Rules per second: {metrics['rules_per_second']}")
```

### Best Practices

1. **Rule Ordering**: Place fast rules before slow ones
2. **Batch Size**: Adjust batch size based on available memory
3. **Cache Tuning**: Monitor cache hit rates and adjust sizes
4. **Async Operations**: Use async where possible for I/O bound operations
5. **Resource Limits**: Set appropriate timeouts and retry limits

## Rule Prioritization

Rules can be prioritized to optimize validation performance. Each rule has two attributes that control its execution order:

- `priority` (int): Higher numbers run first. Use this to run more important rules before less important ones.
- `cost` (int): Higher numbers indicate more computational expense. Among rules with the same priority, lower cost rules run first.

Example:

```python
from sifaka.rules import LengthRule, ToxicityRule
from sifaka.providers import AnthropicProvider
from sifaka import Reflector

# High priority, low cost rule runs first
length_rule = LengthRule(
    name="length",
    description="Check output length",
    config={"min_length": 100, "max_length": 200},
    priority=2,  # High priority
    cost=1       # Low cost
)

# Lower priority, high cost rule runs second
toxicity_rule = ToxicityRule(
    name="toxicity",
    description="Check for toxic content",
    config={"max_toxicity": 0.7},
    priority=1,  # Lower priority
    cost=3       # Higher cost due to API call
)

provider = AnthropicProvider(api_key="your-key")
reflector = Reflector(
    provider=provider,
    rules=[length_rule, toxicity_rule]
)

result = reflector.reflect("Tell me about machine learning")
print(result)
```

The reflector will:
1. Run the length rule first (priority=2)
2. Only if length passes, run the toxicity rule (priority=1)
3. If using parallel validation and a rule fails, cancel remaining lower priority rules

This optimization is especially useful when:
- Some rules are more critical than others
- Rules have varying computational costs
- You want to fail fast on important checks
```

## Pluggable Classifiers

Sifaka supports both LLM-based and lightweight ML classifiers for validation. This allows you to:
- Use fast, local models for simple tasks
- Leverage LLMs for complex classification
- Mix and match based on your needs

### Using Lightweight Classifiers

```python
from sifaka.classifiers import ToxicityClassifier
from sifaka.rules import ClassifierRule

# Create a lightweight toxicity classifier
classifier = ToxicityClassifier(
    name="toxicity_detector",
    model_name="original"  # Uses the Detoxify model
)

# Create a rule using the classifier
toxicity_rule = ClassifierRule(
    name="toxicity_check",
    description="Check for toxic content using Detoxify",
    classifier=classifier,
    threshold=0.7,  # Confidence threshold
    valid_labels=["clean"]  # Labels considered valid
)

# Use in a reflector
reflector = Reflector(
    name="content_validator",
    model=model,
    rules=[toxicity_rule]
)
```

### Using LLM Classifiers

```python
from sifaka.classifiers import LLMClassifier
from sifaka.rules import ClassifierRule
from sifaka.models import AnthropicProvider

# Create an LLM-based classifier
classifier = LLMClassifier(
    name="sentiment_classifier",
    description="Classifies text sentiment",
    model=AnthropicProvider(model_name="claude-3-haiku-20240307"),
    labels=["positive", "neutral", "negative"],
    system_prompt=(
        "You are a sentiment classifier that categorizes text as positive, neutral, or negative. "
        "Respond with a JSON object containing 'label' and 'confidence' fields."
    )
)

# Create a rule using the classifier
sentiment_rule = ClassifierRule(
    name="sentiment_check",
    description="Ensure positive sentiment",
    classifier=classifier,
    threshold=0.8,
    valid_labels=["positive"]
)

# Use in a reflector
reflector = Reflector(
    name="sentiment_validator",
    model=model,
    rules=[sentiment_rule]
)
```

### Creating Custom Classifiers

You can create custom classifiers by inheriting from the `Classifier` base class:

```python
from sifaka.classifiers import Classifier, ClassificationResult
from typing import List
import your_ml_library

class CustomClassifier(Classifier):
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(
            name=name,
            description=description,
            labels=["label1", "label2"],
            cost=1,  # Low cost for fast local model
            **kwargs
        )
        self.model = your_ml_library.load_model()

    def classify(self, text: str) -> ClassificationResult:
        prediction = self.model.predict(text)
        return ClassificationResult(
            label=prediction.label,
            confidence=prediction.probability,
            metadata={"extra_info": prediction.metadata}
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        predictions = self.model.predict_batch(texts)
        return [
            ClassificationResult(
                label=p.label,
                confidence=p.probability,
                metadata={"extra_info": p.metadata}
            )
            for p in predictions
        ]
```

### Performance Considerations

- Lightweight classifiers are faster but may be less accurate
- LLM classifiers are more flexible but have higher latency and cost
- Use the `cost` attribute to help the reflector optimize rule ordering
- Enable caching for frequently used classifications
- Use batch classification when possible
```

## Benchmarking

Sifaka includes comprehensive benchmarking tools to measure and visualize classifier performance. The benchmarking suite provides insights into:

1. Individual classifier performance
2. Combined classifier pipeline performance
3. Memory usage
4. Throughput and latency
5. Caching effectiveness

### Running Benchmarks

```python
from sifaka.benchmarks import ClassifierBenchmark, BenchmarkVisualizer

# Initialize and run benchmarks
benchmark = ClassifierBenchmark(num_samples=1000)
results = benchmark.run_all_benchmarks()

# Create visualizations
visualizer = BenchmarkVisualizer(results)
visualizer.create_all_visualizations()
```

The benchmarking suite will generate:
- Latency distribution plots
- Memory usage comparisons
- Throughput comparisons
- Cache effectiveness visualizations
- A detailed markdown report
- Raw results in JSON format

All visualizations and reports are saved in a `benchmark_results` directory for easy sharing and analysis.

### Available Metrics

For each classifier, the following metrics are tracked:
- Throughput (texts/second)
- Mean, median, P95, and P99 latencies
- Memory usage
- Cache size (for applicable classifiers)

The benchmarking suite automatically handles:
- Warm-up rounds before measurement
- Test data generation of varying complexity
- Memory profiling
- Cache effectiveness tracking
- Pipeline performance measurement
```

# Sifaka Framework Documentation

Sifaka is a powerful framework for validating and classifying text content. It provides a comprehensive set of rules and classifiers for various text analysis tasks.

## 📏 Rules

Sifaka provides a comprehensive set of rules to validate and control LLM outputs. Here's a detailed overview of each rule type:

### Base Rule

The foundation for all rules in Sifaka. Provides core validation functionality.

```python
from sifaka.rules import Rule

# Base class attributes
name: str  # Name of the rule
description: str  # Description of what the rule checks
config: Dict[str, Any]  # Rule configuration
```

### Content Rules

#### ProhibitedContentRule

Checks for and prevents specified prohibited content.

```python
from sifaka.rules import ProhibitedContentRule

rule = ProhibitedContentRule(
    name="content_filter",
    description="Filters prohibited content",
    config={
        "prohibited_terms": ["term1", "term2"],
        "case_sensitive": False,
        "whole_word_match": True
    }
)
```

#### FormatRule

Ensures content follows specified formatting requirements.

```python
from sifaka.rules import FormatRule

rule = FormatRule(
    name="format_check",
    description="Validates content format",
    config={
        "required_format": "markdown",  # or "json", "yaml", etc.
        "schema": {...},  # Optional: JSON schema for validation
        "strict": True  # Optional: strict format checking
    }
)
```

#### LengthRule

Controls the length of generated content.

```python
from sifaka.rules import LengthRule

rule = LengthRule(
    name="length_check",
    description="Validates content length",
    config={
        "min_length": 100,
        "max_length": 1000,
        "unit": "characters"  # or "words", "sentences"
    }
)
```

### Safety Rules

#### ToxicityRule

Prevents toxic or harmful content.

```python
from sifaka.rules import ToxicityRule

rule = ToxicityRule(
    name="toxicity_check",
    description="Checks for toxic content",
    config={
        "max_toxicity": 0.7,
        "check_categories": ["hate", "threat", "profanity"]
    }
)
```

#### SentimentRule

Controls the emotional tone of content.

```python
from sifaka.rules import SentimentRule

rule = SentimentRule(
    name="sentiment_check",
    description="Validates content sentiment",
    config={
        "min_sentiment": 0.0,  # Minimum sentiment score (-1 to 1)
        "target_sentiment": "positive"  # or "neutral", "negative"
    }
)
```

### Domain-Specific Rules

#### LegalCitationRule

Validates legal citations and references.

```python
from sifaka.rules import LegalCitationRule

rule = LegalCitationRule(
    name="citation_check",
    description="Validates legal citations",
    config={
        "citation_styles": ["bluebook", "alwd"],
        "strict_validation": True,
        "require_pinpoint": False
    }
)
```

#### CodeRule

Validates code snippets and programming content.

```python
from sifaka.rules import CodeRule

rule = CodeRule(
    name="code_check",
    description="Validates code content",
    config={
        "language": "python",
        "lint": True,
        "check_syntax": True,
        "style_guide": "pep8"
    }
)
```

### Rule Flow Diagram

```mermaid
graph TD
    A[Input Text] --> B[Rule Processor]
    B --> C{Validation Process}
    C --> D[Content Analysis]
    C --> E[Rule Application]
    C --> F[Constraint Checking]
    D --> G[Validation Result]
    E --> G
    F --> G
    G --> H[Pass/Fail]
    G --> I[Violations]
    G --> J[Suggestions]
```

### Best Practices

1. **Rule Composition**: Combine multiple rules for comprehensive validation
2. **Configuration**: Tune rule parameters based on your specific needs
3. **Error Handling**: Implement proper error handling for rule violations
4. **Performance**: Consider rule execution order for optimal performance
5. **Maintenance**: Regularly update rule configurations based on feedback

### Performance Considerations

- Rules are evaluated in order of registration
- Some rules may have dependencies on classifiers
- Consider caching rule results for frequently checked content
- Monitor rule execution time for performance optimization

## Classifiers

### Text Analysis Classifiers
- **ToxicityClassifier**: Detects toxic content using the Detoxify model
- **SentimentClassifier**: Analyzes text sentiment using VADER
- **ProfanityClassifier**: Identifies profane content
- **LanguageClassifier**: Detects and validates language
- **ReadabilityClassifier**: Assesses text readability
- **LLMClassifier**: Leverages language models for classification

## Usage

### Rule Usage Example
```python
from sifaka.rules import ToxicityRule

# Initialize the rule
rule = ToxicityRule(
    name="toxicity_check",
    description="Check for toxic content",
    threshold=0.7
)

# Validate text
result = rule.validate("Your text here")
print(f"Validation passed: {result.passed}")
print(f"Message: {result.message}")
print(f"Metadata: {result.metadata}")
```

### Classifier Usage Example
```python
from sifaka.classifiers import SentimentClassifier

# Initialize the classifier
classifier = SentimentClassifier(
    name="sentiment_analysis",
    description="Analyze text sentiment"
)

# Classify text
result = classifier.classify("Your text here")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence}")
```

## Contributing

To add new rules or classifiers:

1. Inherit from the appropriate base class (`Rule` or `Classifier`)
2. Implement required methods (`validate` for rules, `classify` for classifiers)
3. Add comprehensive tests
4. Update documentation

## License

MIT License

Copyright (c) 2024 Sifaka Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 📊 Classifiers

Sifaka provides a suite of classifiers to analyze and validate text content. Here's a detailed overview of each classifier:

### Base Classifier

The foundation for all classifiers in Sifaka. Provides core functionality and interfaces that other classifiers extend.

```python
from sifaka.classifiers import Classifier, ClassificationResult

# Base class attributes
name: str  # Name of the classifier
description: str  # Description of what the classifier does
labels: List[str]  # Possible classification labels
cost: float  # Computational cost of classification
```

### Profanity Classifier

Detects and censors profanity and inappropriate language in text.

```python
from sifaka.classifiers import ProfanityClassifier

classifier = ProfanityClassifier(
    name="profanity_check",  # Optional: custom name
    description="Detects inappropriate language",  # Optional: custom description
    custom_words={"word1", "word2"},  # Optional: additional words to check
    censor_char="*"  # Optional: character for censoring (default: *)
)

result = classifier.classify("Your text here")
# Returns ClassificationResult with:
# - label: "clean" or "profane"
# - confidence: float between 0 and 1
# - metadata: {
#     "contains_profanity": bool,
#     "censored_text": str,
#     "censored_word_count": int
# }
```

### Sentiment Classifier

Analyzes the emotional tone of text using state-of-the-art sentiment analysis.

```python
from sifaka.classifiers import SentimentClassifier

classifier = SentimentClassifier(
    name="sentiment_analyzer",
    description="Analyzes text sentiment",
    threshold=0.5  # Optional: threshold for positive/negative classification
)

result = classifier.classify("Your text here")
# Returns ClassificationResult with:
# - label: "positive", "negative", or "neutral"
# - confidence: float between 0 and 1
# - metadata: {
#     "sentiment_score": float,
#     "detailed_scores": Dict[str, float]
# }
```

### Readability Classifier

Assesses text complexity and readability using various metrics (Flesch-Kincaid, etc.).

```python
from sifaka.classifiers import ReadabilityClassifier

classifier = ReadabilityClassifier(
    name="readability_check",
    description="Analyzes text complexity",
    target_grade_level=8  # Optional: target reading grade level
)

result = classifier.classify("Your text here")
# Returns ClassificationResult with:
# - label: "easy", "moderate", or "complex"
# - confidence: float between 0 and 1
# - metadata: {
#     "grade_level": float,
#     "reading_time": float,
#     "metrics": {
#         "flesch_kincaid": float,
#         "gunning_fog": float,
#         "coleman_liau": float
#     }
# }
```

### Toxicity Classifier

Identifies toxic, harmful, or inappropriate content using advanced content moderation.

```python
from sifaka.classifiers import ToxicityClassifier

classifier = ToxicityClassifier(
    name="toxicity_check",
    description="Detects toxic content",
    threshold=0.7  # Optional: toxicity threshold
)

result = classifier.classify("Your text here")
# Returns ClassificationResult with:
# - label: "safe" or "toxic"
# - confidence: float between 0 and 1
# - metadata: {
#     "toxicity_score": float,
#     "category_scores": Dict[str, float]
# }
```

### Language Classifier

Identifies the language of text and provides language-specific metrics.

```python
from sifaka.classifiers import LanguageClassifier

classifier = LanguageClassifier(
    name="language_detector",
    description="Detects text language",
    supported_languages=["en", "es", "fr"]  # Optional: limit supported languages
)

result = classifier.classify("Your text here")
# Returns ClassificationResult with:
# - label: ISO language code (e.g., "en", "es")
# - confidence: float between 0 and 1
# - metadata: {
#     "language_name": str,
#     "language_family": str,
#     "script": str
# }
```

### LLM Classifier

Uses Large Language Models for sophisticated text classification tasks.

```python
from sifaka.classifiers import LLMClassifier

classifier = LLMClassifier(
    name="llm_classifier",
    description="Advanced text classification",
    model="gpt-4",  # Specify LLM model to use
    labels=["technical", "non-technical"],  # Custom classification labels
    prompt_template="Classify the following text as {labels}: {text}"
)

result = classifier.classify("Your text here")
# Returns ClassificationResult with:
# - label: One of the specified labels
# - confidence: float between 0 and 1
# - metadata: {
#     "model_name": str,
#     "reasoning": str,
#     "additional_insights": Dict[str, Any]
# }
```

### Data Flow Diagram

```mermaid
graph TD
    A[Input Text] --> B[Classifier]
    B --> C{Classification Process}
    C --> D[Text Analysis]
    C --> E[Feature Extraction]
    C --> F[Model Inference]
    D --> G[ClassificationResult]
    E --> G
    F --> G
    G --> H[Label]
    G --> I[Confidence]
    G --> J[Metadata]
```

### Best Practices

1. **Initialization**: Always initialize classifiers with appropriate configurations for your use case
2. **Error Handling**: Handle potential exceptions during classification
3. **Batch Processing**: Use `batch_classify()` for multiple texts to improve performance
4. **Confidence Thresholds**: Set appropriate confidence thresholds for your application
5. **Metadata Usage**: Leverage metadata for detailed insights and debugging

### Performance Considerations

- Most classifiers have a `cost` attribute indicating computational complexity
- Use batch classification when processing multiple texts
- Consider caching results for frequently classified content
- Monitor memory usage when processing large texts