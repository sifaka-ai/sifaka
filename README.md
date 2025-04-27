<p align="center">
  <img src="assets/sifika.jpg" alt="Sifaka - A lemur species from Madagascar" width="100%" max-width="800px"/>
</p>

# Sifaka: Reflection and Reliability for LLMs

[![PyPI version](https://badge.fury.io/py/sifaka.svg)](https://badge.fury.io/py/sifaka)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Sifaka** is an open-source framework that adds **reflection and reliability** to large language model (LLM) applications. It helps developers build safer, more reliable AI systems by:

- Catching hallucinations before they reach users
- Enforcing rules and tone consistency
- Providing transparency and auditability

Whether you're building AI-powered tools for legal research, customer support, or creative generation, Sifaka makes your outputs **safer, smarter, and more transparent.**

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
    model_name="claude-3-opus-20240229",
    api_key="your-anthropic-api-key"
)
```

### Example Usage with API Keys

Here's a complete example showing different ways to use API keys:

```python
from sifaka import Reflector
from sifaka.models import OpenAIProvider, AnthropicProvider
from sifaka.rules.content import ProhibitedContentRule
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Method 1: Using environment variables
openai_model = OpenAIProvider(model_name="gpt-4")

# Method 2: Using direct API key
anthropic_model = AnthropicProvider(
    model_name="claude-3-opus-20240229"
)

# Create rules
prohibited_terms = ProhibitedContentRule(
    prohibited_terms=["controversial", "inappropriate"]
)

# Create reflector
reflector = Reflector(
    rules=[prohibited_terms],
    critique=True
)

# Run with OpenAI model
result_openai = reflector.run(
    openai_model,
    "Write a professional email about a project update"
)

# Run with Anthropic model
result_anthropic = reflector.run(
    anthropic_model,
    "Write a professional email about a project update"
)

# Print results
print("OpenAI Result:", result_openai["final_output"])
print("Anthropic Result:", result_anthropic["final_output"])
```

### Security Best Practices

1. **Never commit API keys** to version control
2. Use **environment variables** in production
3. Consider using a **secrets management service** for production
4. Rotate API keys **regularly**
5. Use **least privilege** API keys with minimal permissions

## 🚀 Quick Start

```python
from sifaka import Reflector, legal_citation_check
from sifaka.models import OpenAIProvider
from sifaka.rules.content import ProhibitedContentRule

# Initialize the model provider
model = OpenAIProvider(model_name="gpt-4")

# Create a custom rule
prohibited_terms = ProhibitedContentRule(
    prohibited_terms=["controversial", "inappropriate"]
)

# Create a reflector with rules and critique
reflector = Reflector(
    rules=[legal_citation_check, prohibited_terms],
    critique=True
)

# Run the reflector
result = reflector.run(model, "Write about the Supreme Court case Brown v. Board of Education")

# Get the final output
print(result["final_output"])
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
reflector = Reflector(
    rules=[rule1, rule2],  # List of rules to apply
    critique=True,         # Enable critique
    trace=True             # Enable tracing
)

result = reflector.run(model, prompt)
```

The result contains:
- `original_output`: The original output from the LLM
- `final_output`: The final output after applying rules and critiques
- `rule_violations`: Any rule violations that were detected
- `trace`: Trace data if tracing is enabled

### Rules

Rules validate LLM outputs against specific criteria. Sifaka comes with several built-in rules:

```python
# Legal citation rule
from sifaka import legal_citation_check

# Content rules
from sifaka.rules.content import ProhibitedContentRule, ToneConsistencyRule

# Create custom rules
prohibited_terms = ProhibitedContentRule(
    prohibited_terms=["controversial", "inappropriate"]
)

formal_tone = ToneConsistencyRule(
    expected_tone="formal"
)
```

You can also create custom rules by subclassing `Rule`:

```python
from sifaka.rules import Rule, RuleResult

class MyCustomRule(Rule):
    name: str = "my_custom_rule"

    def validate(self, output: str, **kwargs) -> RuleResult:
        # Implement your validation logic here
        if some_condition:
            return RuleResult(
                passed=True,
                message="Validation passed"
            )
        else:
            return RuleResult(
                passed=False,
                message="Validation failed",
                metadata={"reason": "Some reason"}
            )
```

Or by using a function:

```python
def my_rule_function(output: str, **kwargs) -> bool:
    # Return a boolean
    return "good" in output.lower()

# Or return a tuple of (passed, message, metadata)
def my_detailed_rule(output: str, **kwargs) -> tuple:
    passed = "good" in output.lower()
    message = "Output contains 'good'" if passed else "Output does not contain 'good'"
    return passed, message, {"contains_good": passed}
```

### Critique

Critique improves LLM outputs based on rule violations. Sifaka uses the LLM itself to critique and improve outputs:

```python
from sifaka.critique import PromptCritique

# Create a custom critique
critique = PromptCritique(model)

# Or enable critique in the reflector
reflector = Reflector(
    rules=[rule1, rule2],
    critique=True  # Uses PromptCritique by default
)
```

### Model Providers

Model providers interface with different LLM providers. Sifaka supports OpenAI and Anthropic out of the box:

```python
from sifaka.models import OpenAIProvider, AnthropicProvider

# OpenAI provider
openai_model = OpenAIProvider(
    model_name="gpt-4",
    temperature=0.7
)

# Anthropic provider
anthropic_model = AnthropicProvider(
    model_name="claude-3-opus-20240229",
    temperature=0.7
)
```

## 📊 Tracing and Logging

Sifaka provides comprehensive tracing and logging to help you understand and debug the reflection process:

```python
reflector = Reflector(
    rules=[rule1, rule2],
    critique=True,
    trace=True  # Enable tracing
)

result = reflector.run(model, prompt)

# Access trace data
for event in result["trace"]:
    print(f"Stage: {event['stage']}")
```

## 🔍 Advanced Usage

### Integration with LangChain

Sifaka can be used with LangChain:

```python
from langchain.llms import OpenAI as LangChainOpenAI
from sifaka import Reflector, legal_citation_check
from sifaka.models import ModelProvider

# Create a LangChain adapter
class LangChainAdapter(ModelProvider):
    name: str = "langchain_adapter"
    model: Any

    def generate(self, prompt: str, **kwargs) -> str:
        return self.model.generate(prompt)

# Create a LangChain model
langchain_model = LangChainOpenAI()

# Create a Sifaka adapter
sifaka_model = LangChainAdapter(langchain_model)

# Create a reflector
reflector = Reflector(
    rules=[legal_citation_check],
    critique=True
)

# Run the reflector
result = reflector.run(sifaka_model, prompt)
```

### Integration with LangGraph

Sifaka provides seamless integration with LangGraph's graph-based execution system:

```python
from langgraph.graph import Graph
from sifaka.integrations.langgraph import wrap_graph
from sifaka.rules.domain import MedicalRule, LegalRule

# Create a LangGraph graph
graph = Graph()

# Add nodes and edges
graph.add_node("medical", medical_processor)
graph.add_node("legal", legal_processor)
graph.add_edge("medical", "legal")

# Wrap it with Sifaka
sifaka_graph = wrap_graph(
    graph=graph,
    rules={
        "medical": [MedicalRule()],
        "legal": [LegalRule()]
    },
    critique=True
)

# Use the graph as normal
result = sifaka_graph.run(input_data)
```

Sifaka's LangGraph integration includes:

1. **Graph Integration**:
   - Node-specific rule validation
   - State validation at each step
   - Automatic critique and improvement of outputs
   - Support for both regular and stateful graphs

2. **Tool Integration**:
   - Rule validation for tool outputs
   - Automatic improvement of tool results
   - Integration with LangGraph's tool executor

3. **Channel Integration**:
   - Message validation in channels
   - Automatic improvement of messages
   - Support for all LangGraph channel types

Example with Stateful Graph:

```python
from langgraph.graph import StateGraph
from sifaka.integrations.langgraph import wrap_state_graph
from sifaka.rules.domain import MedicalRule, LegalRule

# Create a stateful graph
graph = StateGraph()

# Add nodes and edges
graph.add_node("medical", medical_processor)
graph.add_node("legal", legal_processor)
graph.add_edge("medical", "legal")

# Wrap it with Sifaka
sifaka_graph = wrap_state_graph(
    graph=graph,
    rules={
        "medical": [MedicalRule()],
        "legal": [LegalRule()]
    },
    state_validators={
        "medical": [MedicalRule()],
        "legal": [LegalRule()]
    },
    critique=True
)

# Use the graph as normal
result = sifaka_graph.run(input_data)
```

## 📚 API Reference

For full API documentation, visit [docs.sifaka.ai](https://docs.sifaka.ai).

## 🧪 Testing

Run the tests with pytest:

```bash
# Install development dependencies
pip install sifaka[dev]

# Run tests
pytest
```

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

Sifaka is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Sifaka Rules Engine

Sifaka is a rules engine that provides content validation and critique capabilities for language model outputs. It integrates with langgraph to provide a comprehensive validation pipeline.

## Built-in Rules

Sifaka provides several built-in rules, each in its own module for easy extension:

### 1. LengthRule (`sifaka.rules.length`)
Validates the minimum length of the output.

```python
from sifaka.rules.length import LengthRule

# Create a rule that requires at least 100 characters
rule = LengthRule(min_length=100)
```

**Parameters:**
- `min_length` (int, default=50): Minimum required length in characters

**Validation:**
- Fails if output length < min_length
- Returns metadata with actual length and minimum required length

### 2. ProhibitedContentRule (`sifaka.rules.prohibited_content`)
Checks for prohibited terms in the output.

```python
from sifaka.rules.prohibited_content import ProhibitedContentRule

# Create a rule that checks for specific prohibited terms
rule = ProhibitedContentRule(prohibited_terms=["badword1", "badword2"])
```

**Parameters:**
- `prohibited_terms` (List[str]): List of terms to check for

**Validation:**
- Fails if any prohibited term is found (case-insensitive)
- Returns metadata with list of found prohibited terms

### 3. SentimentRule (`sifaka.rules.sentiment`)
Validates the sentiment of the output.

```python
from sifaka.rules.sentiment import SentimentRule

# Create a rule that requires neutral or positive sentiment
rule = SentimentRule(min_sentiment=-0.5)
```

**Parameters:**
- `min_sentiment` (float, default=-0.5): Minimum acceptable sentiment score (-1 to 1)

**Validation:**
- Fails if sentiment score < min_sentiment
- Returns metadata with actual sentiment score

### 4. ToxicityRule (`sifaka.rules.toxicity`)
Checks for toxic content in the output.

```python
from sifaka.rules.toxicity import ToxicityRule

# Create a rule that limits toxicity
rule = ToxicityRule(max_toxicity=0.5)
```

**Parameters:**
- `max_toxicity` (float, default=0.5): Maximum acceptable toxicity score (0 to 1)

**Validation:**
- Fails if toxicity score > max_toxicity
- Returns metadata with actual toxicity score

### 5. FormatRule (`sifaka.rules.format`)
Validates the formatting of the output.

```python
from sifaka.rules.format import FormatRule

# Create a rule that requires markdown formatting
rule = FormatRule(required_format="markdown")
```

**Parameters:**
- `required_format` (Literal["markdown", "plain_text", "json"]): Required format type

**Validation:**
- For markdown: Checks for basic markdown syntax
- For JSON: Validates JSON syntax
- For plain_text: Always passes
- Returns metadata with required format type

## Creating Custom Rules

### Using the Template

Sifaka provides a template for creating new rules. Copy `sifaka/rules/template.py` and modify it for your needs:

```python
from sifaka.rules.base import Rule, RuleResult

class MyCustomRule(Rule):
    def __init__(self, custom_param: str):
        super().__init__(
            name="my_custom_rule",
            description="Description of your custom rule"
        )
        self.custom_param = custom_param

    def validate(self, output: str) -> RuleResult:
        # Implement your validation logic
        if not self._check_criteria(output):
            return RuleResult(
                passed=False,
                message="Description of the failure",
                metadata={"custom_param": self.custom_param}
            )
        return RuleResult(passed=True)
```

### Rule Structure Requirements

Every rule must:

1. **Inherit from `Rule`**:
   ```python
   from sifaka.rules.base import Rule, RuleResult
   ```

2. **Initialize with name and description**:
   ```python
   super().__init__(
       name="rule_name",
       description="Description of what this rule checks"
   )
   ```

3. **Implement `validate` method**:
   ```python
   def validate(self, output: str) -> RuleResult:
       # Validation logic here
       return RuleResult(passed=True/False, message="...", metadata={...})
   ```

### Best Practices

1. **Error Handling**:
   ```python
   try:
       # Validation logic
   except Exception as e:
       return RuleResult(
           passed=False,
           message=f"Error during validation: {str(e)}",
           metadata={"error": str(e)}
       )
   ```

2. **Metadata**:
   - Include relevant parameters in metadata
   - Add any useful diagnostic information
   - Keep metadata structure consistent

3. **Documentation**:
   - Add docstrings for the class and methods
   - Document parameters and return values
   - Include usage examples

### Example: KeywordRule

Here's a complete example of a custom rule:

```python
from sifaka.rules.base import Rule, RuleResult
from typing import List

class KeywordRule(Rule):
    """Rule that checks if the output contains required keywords."""

    def __init__(self, required_keywords: List[str]):
        super().__init__(
            name="keyword_rule",
            description="Checks if the output contains all required keywords"
        )
        self.required_keywords = required_keywords

    def validate(self, output: str) -> RuleResult:
        try:
            missing_keywords = [
                keyword for keyword in self.required_keywords
                if keyword.lower() not in output.lower()
            ]

            if missing_keywords:
                return RuleResult(
                    passed=False,
                    message=f"Output is missing required keywords: {', '.join(missing_keywords)}",
                    metadata={
                        "missing_keywords": missing_keywords,
                        "required_keywords": self.required_keywords
                    }
                )
            return RuleResult(passed=True)

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error checking keywords: {str(e)}",
                metadata={"error": str(e)}
            )
```

## Using Rules with Langgraph

```python
from langgraph.graph import Graph
from sifaka.integrations.langgraph import wrap_graph
from sifaka.rules import LengthRule, ProhibitedContentRule, MyCustomRule
from sifaka.critique.base import ClaudeCritique

# Create your langgraph
graph = Graph()
# ... configure your graph ...

# Create rules
rules = [
    LengthRule(min_length=100),
    ProhibitedContentRule(["badword1", "badword2"]),
    MyCustomRule("custom_param_value")
]

# Create critique
critic = ClaudeCritique()

# Wrap the graph with Sifaka's features
sifaka_graph = wrap_graph(
    graph=graph,
    rules=rules,
    critique=True,
    critic=critic
)

# Run the graph
output = sifaka_graph({"input": "Your input here"})
```

## Installation

```bash
pip install sifaka
```

## Requirements

- Python 3.8+
- langgraph
- anthropic
- pydantic
