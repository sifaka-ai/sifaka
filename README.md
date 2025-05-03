# Sifaka

Sifaka is a framework for improving large language model (LLM) outputs through validation, reflection, and refinement. It helps build more reliable AI systems by enforcing constraints and improving response quality.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

Sifaka can be installed with different sets of dependencies depending on your needs:

### Basic Installation
```bash
pip install sifaka
```

### Installation with Specific Features

```bash
# Install with OpenAI support
pip install "sifaka[openai]"

# Install with Anthropic support
pip install "sifaka[anthropic]"

# Install with all classifiers
pip install "sifaka[classifiers]"

# Install with benchmarking tools
pip install "sifaka[benchmark]"

# Install everything (except development tools)
pip install "sifaka[all]"
```

### Development Installation
```bash
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka
pip install -e ".[dev]"  # Install with development dependencies
```

## Optional Dependencies

Sifaka's functionality can be extended through optional dependencies:

### Model Providers
- `openai`: OpenAI API support
- `anthropic`: Anthropic Claude API support
- `google-generativeai`: Google Gemini API support

### Classifiers
- `toxicity`: Toxicity detection using Detoxify
- `sentiment`: Sentiment analysis using VADER
- `profanity`: Profanity detection
- `language`: Language detection
- `readability`: Text readability analysis

### Benchmarking
- `benchmark`: Tools for performance benchmarking and analysis

## Key Features

- ✅ **Validation Rules**: Enforce constraints like length limits and content restrictions
- ✅ **Response Critics**: Provide feedback to improve model outputs
- ✅ **Chain Architecture**: Create feedback loops for iterative improvement
- ✅ **Model Agnostic**: Works with Claude, OpenAI, and other LLM providers
- ✅ **Streamlined Configuration**: Unified configuration system using ClassifierConfig and RuleConfig

## Architecture

Sifaka follows a modular architecture with several key components working together:

```mermaid
graph TD
    subgraph Chain Core
        PM[Prompt Manager]
        VS[Validation Manager]
        RS[Retry Strategy]
        RF[Result Formatter]
    end

    I[Input] --> PM
    PM --> MP[Model Provider]
    MP --> VS
    VS --> R[Rules]
    VS --> C[Classifiers]

    VS --> |Validation Failed| CR[Critics]
    CR --> |Feedback| PM
    VS --> |Validation Passed| RF
    RF --> O[Output]

    style Chain Core fill:#f5f5f5,stroke:#333,stroke-width:2px
```

### Core Components

1. **Chain Core**
   - Central orchestrator containing:
     - `PromptManager`: Handles prompt formatting and feedback incorporation
     - `ValidationManager`: Coordinates rule and classifier validation
     - `RetryStrategy`: Manages retry attempts and backoff
     - `ResultFormatter`: Formats final outputs and feedback

2. **Model Providers**
   - Interface with different LLM APIs (OpenAI, Anthropic, etc.)
   - Handle API key management and request formatting
   - Support streaming and non-streaming responses

3. **Rules & Classifiers**
   - Validate responses against specific criteria
   - Rules: length, style, content restrictions
   - Classifiers: toxicity, sentiment, profanity detection
   - Can be combined and prioritized
   - Support custom implementations

4. **Critics**
   - Analyze and improve model outputs when validation fails
   - Two main types:
     - `PromptCritic`: Single-pass improvement
     - `ReflexionCritic`: Learning-based improvement with memory
   - Provide feedback for retry attempts

### Flow Description

1. Input is received by the Prompt Manager
2. Prompt Manager formats and sends to Model Provider
3. Model output is sent to Validation Manager
4. Validation Manager coordinates Rules and Classifiers
5. If validation fails:
   - Critics analyze and provide feedback
   - Feedback is sent back to Prompt Manager
   - Process repeats with retry strategy
6. If validation passes:
   - Result Formatter processes the output
   - Final result is returned

## Configuration System

Sifaka uses a streamlined configuration system with two main configuration classes:

```mermaid
classDiagram
    class ClassifierConfig {
        +List[str] labels
        +float cost
        +float min_confidence
        +Dict params
    }

    class RuleConfig {
        +RulePriority priority
        +int cache_size
        +float cost
        +Dict params
    }
```

- `ClassifierConfig`: Manages classifier parameters and thresholds
- `RuleConfig`: Controls rule behavior and execution priority

## Integration with Guardrails

Sifaka provides seamless integration with [Guardrails AI](https://www.guardrailsai.com/), allowing you to:

- Use Guardrails' validation and transformation capabilities
- Leverage Guardrails' extensive rule library
- Combine both systems' strengths for robust content validation

Example integration:
```python
from sifaka.adapters.rules.guardrails_adapter import GuardrailsAdapter
from sifaka.domain import Domain

guardrails_adapter = GuardrailsAdapter()
domain = Domain({
    "name": "text",
    "rules": {
        "guardrails": {
            "enabled": True,
            "adapter": guardrails_adapter
        }
    }
})
```

## License

Sifaka is licensed under the MIT License. See [LICENSE](LICENSE) for details.


