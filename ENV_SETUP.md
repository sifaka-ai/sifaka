# Environment Variable Setup

Sifaka uses environment variables to manage API keys for various LLM providers. This guide explains how to set up your environment.

## Quick Start

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ANTHROPIC_API_KEY=your-anthropic-api-key-here
   GEMINI_API_KEY=your-gemini-api-key-here
   GROQ_API_KEY=your-groq-api-key-here
   ```

## Automatic Loading

Sifaka automatically loads environment variables from `.env` files using `python-dotenv`:

- When importing sifaka: `from sifaka import improve`
- In example scripts that include `load_dotenv()`
- In the LLM client module

## Provider Requirements

Not all API keys are required. You only need keys for the providers you plan to use:

- **OpenAI**: Required for GPT-3.5, GPT-4 models (default)
- **Anthropic**: Required for Claude models
- **Google Gemini**: Required for Gemini models
- **Groq**: Required for fast open-source model inference

## Security Notes

- Never commit `.env` files to version control
- The `.env` file is already in `.gitignore`
- Use `.env.example` as a template for sharing required variables
- Keep your API keys secure and rotate them regularly

## Additional Configuration

Optional environment variables:

- `GUARDRAILS_API_KEY`: For GuardrailsAI validators
- `LOGFIRE_TOKEN`: For observability with Logfire
- `REDIS_URL`: Redis connection string (default: `redis://localhost:6379`)
- `REDIS_PASSWORD`: Redis authentication password
