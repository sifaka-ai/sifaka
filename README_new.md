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

## 📚 API Reference

For full API documentation, visit [docs.sifaka.ai](https://docs.sifaka.ai).

## � Testing

Run the tests with pytest:

```bash
# Install development dependencies
pip install sifaka[dev]

# Run tests
pytest
```

## �🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

Sifaka is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
