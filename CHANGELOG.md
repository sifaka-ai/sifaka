# Changelog

All notable changes to Sifaka will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2024-05-02

### Added
- Enhanced documentation with detailed integration guides
  - Django integration guide
  - FastAPI integration guide
  - Flask integration guide
- New use case examples
  - Content moderation examples
  - Text analysis examples
- Architecture diagrams with Mermaid syntax
- Unit tests for common patterns

### Changed
- Improved documentation structure
- Standardized example code patterns
- Updated base classes for better type support

### Fixed
- Documentation typos and outdated references
- Inconsistent API descriptions

## [0.1.0] - 2024-04-28

### Added
- Initial release
- Basic framework for validating and improving LLM outputs
- Support for multiple LLM providers (Claude, OpenAI)
- Rule system for enforcing constraints on outputs
- Critic framework for content improvement
- Chain architecture for feedback loops
- Classifier system for text analysis
- Unified configuration system with ClassifierConfig and RuleConfig

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

### Known Limitations
- Limited test coverage
- Some classifiers may have high false positive rates
- Documentation is still evolving

[unreleased]: https://github.com/username/sifaka/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/username/sifaka/releases/tag/v0.1.0