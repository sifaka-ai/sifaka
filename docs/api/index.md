# API Reference

This section provides detailed documentation for all Sifaka components.

## Core Components

- [Chain](chain.md): The main orchestrator for generation, validation, and improvement
- [Results](results.md): Result types returned by Sifaka operations

## Models

- [Model Interface](models/index.md): Base interface for all model providers
- [OpenAI](models/openai.md): OpenAI model implementation
- [Anthropic](models/anthropic.md): Anthropic model implementation
- [Gemini](models/gemini.md): Google Gemini model implementation

## Validators

- [Length](validators/length.md): Validators for text length
- [Content](validators/content.md): Validators for text content
- [Format](validators/format.md): Validators for text format

## Critics

- [Critic Base](critics/index.md): Base class for all critics
- [Clarity and Coherence](critics/clarity.md): Critics for improving text clarity
- [Factual Accuracy](critics/factual.md): Critics for validating factual accuracy

## Advanced Features

- [Caching](advanced/caching.md): Caching mechanisms for improved performance
- [Retry](advanced/retry.md): Retry mechanisms for handling failures
- [Streaming](advanced/streaming.md): Streaming support for model responses
- [Observability](advanced/observability.md): Logging and metrics collection
