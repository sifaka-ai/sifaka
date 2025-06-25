# Changelog

All notable changes to Sifaka are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced NCriticsCritic with dynamic perspective generation
- Advanced confidence calculation with critic-specific strategies
- SelfRAGEnhanced critic with pluggable retrieval backends
- Composable validator system with operator overloading
- Middleware system for cross-cutting concerns
- Performance monitoring capabilities
- Comprehensive integration tests for multiple LLM providers
- Critic selection guide and comparison table

### Changed
- Refactored engine.py into modular components (generation, orchestration, validation, core)
- Improved confidence calculation to be critic-specific
- Updated all critics with "When to Use" documentation

### Fixed
- All mypy strict mode errors (0 errors with --strict)
- Type annotations throughout the codebase
- SifakaResult.confidence access (now from critiques)

## [2024-12-23]

### Added
- Comprehensive test suite for all critics
- Error handling guide documentation
- Validator examples (pattern, numeric validators)
- Retry logic for LLM API calls

### Changed
- Simplified metrics.py (190 → 88 lines)
- Consolidated API documentation
- Unified configuration system (merged CriticConfig into Config)
- Standardized error handling patterns

### Removed
- Redundant validators (URL, email, toxicity - using GuardrailsAI instead)
- Unused cache.py module
- Duplicate API_REFERENCE.md file

### Fixed
- 60 mypy type errors
- All ruff linting issues
- Import structure and circular dependencies
- Config class location and imports

## Initial Release

- Core improvement API with `improve()` function
- 8 research-based critics
- Validator system
- Storage backends (memory and file)
- Plugin architecture
- Multi-provider support (OpenAI, Anthropic, Google, etc.)
