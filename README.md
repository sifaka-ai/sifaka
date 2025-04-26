<p align="center">
  <img src="assets/sifika.jpg" alt="Sifaka - A lemur species from Madagascar" width="100%" max-width="800px"/>
</p>

## What is Sifaka?

**Sifaka** is an open-source framework that adds **reflection and reliability** to large language model (LLM) applications. It’s built to help developers:

- Catch hallucinations before they reach users
- Enforce rules and tone consistency
- Audit and understand LLM behavior

Whether you're building AI-powered tools for legal research, customer support, or creative generation, Sifaka makes your outputs **safer, smarter, and more transparent.**

---

## Why Use Sifaka?

Modern LLMs like GPT-4 and Claude are powerful—but unreliable. Sifaka lets you:

- **Add Guardrails**: Define rules and constraints (e.g., verify citations, block certain phrases)
- **Reflect Before Responding**: Use prompt-based critiques to improve AI answers
- **Trace & Audit**: Keep logs of what was generated, what was flagged, and why

All with just a few lines of code.

---

## Key Features

- **Python SDK** (TypeScript coming soon)
- **Modular reflection pipeline**
- **Plug-and-play with OpenAI, Anthropic, or open-source models**
- **Works with LangChain, LlamaIndex, and other LLM toolkits**

```python
from sifaka import Reflector, legal_citation_check

reflector = Reflector(rules=[legal_citation_check], critique=True)
output = reflector.run(model, prompt)
