# Changelog

All notable changes to Sifaka are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
