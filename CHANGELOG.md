# Changelog

All notable changes to Sifaka are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-07-30

### Added
- **Self-Taught Evaluator Critic**: New critic based on Wang et al. 2024 paper
  - Generates contrasting text versions for transparent evaluation
  - Provides detailed reasoning traces explaining decisions
  - Learns from evaluation history to avoid repetitive feedback
  - Tracks explored dimensions across iterations
  - No training data required - uses synthetic contrasts
- **Agent4Debate Critic**: New critic based on Chen et al. 2024 paper
  - Simulates multi-agent competitive debate dynamics
  - Three perspectives: Conservative, Transformative, and Balanced
  - Reveals trade-offs through structured argumentation
  - Judge mechanism determines winning approach
  - Particularly useful for high-stakes content decisions
- **Enhanced Dimension Tracking**: Self-Taught Evaluator tracks which aspects have been evaluated to ensure diverse feedback across iterations
- **Example Scripts**: Added comprehensive examples for both new critics demonstrating their unique capabilities

### Changed
- **Documentation**: Updated all documentation to include new critics
  - Added to Research Foundation sections
  - Created detailed usage guides with examples
  - Updated API reference with new CriticType entries
- **Type System**: Extended CriticType enum to include SELF_TAUGHT_EVALUATOR and AGENT4DEBATE

### Fixed
- **Self-Taught Evaluator**: Fixed deque slicing issue in previous context retrieval
- **Formatting**: Applied consistent quote style in f-strings for ruff compliance

## [0.1.6] - 2025-07-20

### Added
- **Logfire Integration**: Deep observability integration with Logfire/OpenTelemetry
  - Automatic tracing of LLM calls with rich metadata (model, tokens, duration)
  - Detailed critic tracking with individual critic names and performance metrics
  - Nested spans for operation hierarchy visualization
  - Service name configuration for proper identification in Logfire dashboard
  - Global monitor singleton for consistent metrics across the application
  - Performance metrics including tokens/second, critic execution times

### Fixed
- **Logfire Critic Tracking**: Fixed parallel critic execution not being tracked in Logfire
  - Critics now properly report to Logfire with individual names and timings
  - Parallel execution wrapped with monitor.track_critic_call for visibility
- **Service Naming**: Fixed "unknown_service" in Logfire by setting OTEL_SERVICE_NAME early
- **Example Configurations**: Fixed n_critics_example.py invalid critic_model parameter
  - Moved critic_model from CriticConfig to LLMConfig where it belongs

### Changed
- **Monitoring Enhancement**: Enriched Logfire spans with comprehensive LLM metadata
  - Added model name, token counts, temperature settings to spans
  - Enhanced critic spans with critic type and confidence scores
  - Improved span naming for better clarity in traces

## [0.1.5] - 2025-07-20

### Fixed
- **Ollama Critics**: Fixed critics failing with Ollama by properly setting OPENAI_BASE_URL for pydantic-ai
- **Import Error**: Made logfire import optional to avoid ModuleNotFoundError when not installed

### Changed
- **Ollama Integration**: Improved to use pydantic-ai's native OpenAI provider support for Ollama

## [0.1.4] - 2025-07-20

### Fixed
- **Ollama Support**: Fixed critical issues preventing Ollama from working correctly
  - Fixed provider not being passed to critics during orchestration
  - Fixed provider not being passed to text generator
  - Fixed pydantic_ai integration for Ollama (now uses direct completion API)
  - Ollama now correctly generates improved text
- **Documentation**: Fixed broken internal links in MkDocs documentation
- **README**: Updated Ollama usage example to include required `critic_model` setting

### Changed
- **Error Handling**: TextGenerator now gracefully handles Ollama-specific requirements
- **Tests**: Fixed test expecting specific create_critics parameters

## [0.1.3] - 2025-07-20

### Added
- **Ollama Integration Tests**: Added comprehensive integration tests for Ollama provider
  - Ollama now included in multi-provider critics integration tests
  - All critics can be tested with local Ollama models
  - Enhanced test configuration supports Ollama with `llama3.2` model

### Improved
- **Error Handling**: Enhanced error messages when no LLM provider is configured
  - Replaced generic `ValueError` with structured `ModelProviderError`
  - Added specific error code `"no_provider"` with helpful setup instructions
  - Updated documentation to reflect new error format
  - Users now get clear guidance on which API keys to set up

### Fixed
- **Test Suite**: Fixed Ollama provider tests to properly handle LLMResponse structure
  - Corrected usage data access patterns in test assertions
  - All Ollama provider tests now pass consistently

## [0.1.2] - 2025-07-20

### Fixed
- **GuardrailsAI dependency**: Temporarily disabled due to griffe version conflict with pydantic-ai
- **PyPI display**: Removed logo from README as PyPI doesn't support relative image paths

### Added
- **Ollama Support**: Added support for local LLMs via Ollama
  - New provider: `Provider.OLLAMA`
  - Supports popular models: llama3.2, mistral, qwen2.5-coder, etc.
  - Configurable base URL via `OLLAMA_BASE_URL` environment variable
  - No API key required by default

### Changed
- **Dependencies**: Removed guardrails-ai from dependencies until they release a fix
- **GuardrailsValidator**: Temporarily replaced with placeholder that raises NotImplementedError

## [0.1.1] - 2025-07-20

### Fixed
- **Integration tests**: Fixed 7 failing integration tests related to Config API changes
- **CI/CD workflow**: Removed redundant tests.yml workflow
- **Documentation deployment**: Fixed GitHub Pages permissions for docs deployment
- **Release workflow**: Fixed dependency installation to use uv with all extras
- **PyPI publishing**: Configured PYPI_TOKEN secret for automated publishing
- **Warnings**: Suppressed pkg_resources deprecation warning from guardrails

### Changed
- **Version management**: Added __version__ attribute to __init__.py
- **Documentation**: Updated installation docs to reflect PyPI availability
- **Repository links**: Fixed all documentation URLs to point to sifaka-ai/sifaka

### Added
- **Documentation badge**: Added GitHub Pages documentation badge to README

## [0.1.0] - 2025-07-19

### Changed
- **Migration to PydanticAI**: Replaced OpenAI client with PydanticAI for structured outputs
- **Improved test suite**: Fixed all critic and middleware tests to work with new architecture
- **Better error handling**: More robust error messages and exception handling
- **Enhanced type safety**: Leveraging Pydantic models for all LLM interactions

### Fixed
- **Python version compatibility**: Fixed TypedDict imports for Python < 3.12
- **Test fixtures**: Updated all test mocking to work with PydanticAI agents
- **CI/CD pipeline**: Fixed GitHub Actions failures for all Python versions
- **Bandit security issues**: Resolved MD5 hash and XML parsing security warnings
- **Pre-commit hooks**: Fixed all linting and type checking issues

### Technical Details
- Migrated from direct OpenAI client usage to PydanticAI's agent-based approach
- Updated all critic implementations to use structured response models
- Fixed confidence calculation floating point precision issues
- Improved middleware chain to properly pass text modifications through

## [0.0.7] - 2025-06-28

### Added
- **Standardized API**: Single `improve()` function for all text improvement needs
- **Hybrid model optimization**: Default gpt-3.5-turbo for critics, gpt-4o-mini for generation
- **Complete observability**: Full audit trails with processing times and token usage
- **Memory-bounded operations**: Automatic cleanup and bounded collections
- **Tool integration**: Support for web search and external tools via registry system
- **Meta-Rewarding critic**: Two-stage evaluation system for higher quality feedback
- **Self-RAG critic**: Retrieval-augmented critique with fact-checking capabilities
- **Comprehensive documentation**: Complete rewrite of all documentation for accuracy

### Changed
- **Simplified codebase**: Dramatically reduced complexity while maintaining functionality
- **Unified configuration**: Single `Config` class for all settings
- **Streamlined imports**: Clean, consistent import structure
- **Performance optimized**: 50-60% faster processing with hybrid model approach
- **Standardized result objects**: Consistent `SifakaResult` with all metadata
- **Improved error handling**: Robust retry logic and graceful degradation

### Fixed
- **Documentation accuracy**: Removed all references to non-existent functions and attributes
- **Type safety**: Complete type annotations throughout codebase
- **Import consistency**: Fixed all import paths and module references
- **API consistency**: Standardized parameter names and function signatures
- **Memory leaks**: Implemented proper cleanup and bounded collections
- **Duplicate feedback**: Fixed critic deduplication in improvement prompts

### Removed
- **Legacy APIs**: Removed inconsistent and deprecated function variants
- **Redundant code**: Eliminated duplicate implementations and unused modules
- **Inconsistent patterns**: Unified all usage patterns under single API
- **Draft documentation**: Removed working/draft README files

---

*Initial release focused on establishing a clean, consistent, and well-documented foundation for AI text improvement.*
