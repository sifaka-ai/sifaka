# Changelog

All notable changes to Sifaka will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-03-14

### Added
- New `SymmetryRule` for text symmetry validation
- New `RepetitionRule` for pattern detection
- Comprehensive pattern detection configuration options
- Migration guide in README.md
- This CHANGELOG file

### Changed
- **BREAKING**: Deprecated the `Reflector` class
- Moved pattern detection functionality to specialized rules
- Updated documentation to reflect new architecture
- Improved error messages and validation feedback
- Updated examples to use new pattern rules
- Changed development status to Production/Stable

### Deprecated
- The `Reflector` class (will be removed in 2.0.0)
- Old pattern detection methods
- Direct reflection validation through the reflector

## [0.2.0] - 2024-02-28

### Added
- Initial public release
- Basic reflection capabilities
- Integration with LangChain and LangGraph
- Support for multiple LLM providers
- Basic rule system
- Critic framework for content improvement