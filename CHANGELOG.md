# Changelog

All notable changes to the Sifaka project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.7] - 2024-06-22

### 🎯 Major Rewrite and Architecture Improvements

This version represents a complete rewrite of Sifaka with a focus on simplicity, production readiness, and research-backed critique methods.

### ✨ Added

#### Core Functionality
- **Complete rewrite** with clean, production-ready architecture
- **Single `improve()` function** replacing multiple confusing APIs
- **Research-backed critics** implementing 7 peer-reviewed methodologies:
  - Reflexion - Self-reflection and learning from mistakes
  - Constitutional AI - Principle-based ethical evaluation
  - Self-Refine - Iterative self-improvement
  - N-Critics - Multi-perspective ensemble critique
  - Self-RAG - Retrieval-augmented critique for factual accuracy
  - Meta-Rewarding - Two-stage judgment with meta-evaluation
  - Self-Consistency - Consensus-based evaluation through multiple assessments

#### Memory and Performance
- **Memory-bounded collections** preventing OOM crashes
- **Timeout handling** with configurable limits
- **Comprehensive error handling** with clear, actionable messages

#### Storage and Plugins
- **Plugin architecture** for extensible storage backends
- **File storage** with JSON serialization and cleanup
- **Memory storage** with LRU eviction
- **Plugin discovery** via entry points

#### Validation System
- **LengthValidator** for text length constraints
- **ContentValidator** for required/forbidden terms
- **Custom validator** interface for extensibility
- **Async validation** with detailed results

#### Developer Experience
- **Type safety** with Pydantic models throughout
- **Async/await** patterns for modern Python
- **Comprehensive test suite** with 75%+ coverage
- **Working examples** for all critics with thought logging
- **Complete API documentation** with examples
- **Architectural decision records** (ADRs)

### 🔧 Technical Improvements

#### Dependencies
- **Minimal dependencies** (4 core dependencies vs 41 in previous version)
- **Optional extras** for different model providers:
  - `sifaka[anthropic]` - Anthropic Claude support
  - `sifaka[gemini]` - Google Gemini support  
  - `sifaka[guardrails]` - GuardrailsAI integration
  - `sifaka[all]` - All optional dependencies

#### Model Support
- **OpenAI** models: GPT-4, GPT-4 Turbo, GPT-4o, GPT-4o-mini, GPT-3.5 Turbo
- **Anthropic** models: Claude-3 Opus, Sonnet, Haiku
- **Google** models: Gemini 1.5 Pro, Gemini 1.5 Flash

#### Configuration
- **Simple configuration** with sensible defaults
- **Validation** of all configuration parameters
- **Iteration bounds** and timeout controls
- **Model-specific** temperature and parameter handling

### 📁 Project Structure

```
sifaka/
├── sifaka/
│   ├── core/           # Core engine, models, interfaces
│   ├── critics/        # Research-backed critic implementations  
│   ├── validators/     # Text validation system
│   ├── storage/        # Storage backends and plugin system
│   └── plugins.py      # Plugin discovery and registration
├── examples/           # Working examples for all critics
├── thoughts/           # Critic analysis output folder
├── tests/              # Comprehensive test suite
└── docs/              # Architecture and decision documentation
```

### 🧪 Testing and Quality

#### Comprehensive Test Suite
- **Unit tests** for all components
- **Integration tests** for end-to-end workflows
- **Performance tests** and stress testing
- **Concurrency tests** for thread safety
- **Edge case tests** for robustness
- **Model provider tests** across all supported APIs

#### Code Quality
- **Type checking** with mypy (99%+ coverage)
- **Linting** with ruff (zero errors)
- **Formatting** with black
- **Pre-commit hooks** for quality gates
- **GitHub Actions** CI/CD pipeline

### 📖 Documentation

#### User Documentation
- **Quick Start Guide** (QUICKSTART.md) - 5-minute setup
- **API Reference** (API.md) - Complete API documentation
- **Examples** (examples.py) - Comprehensive usage examples
- **README** with architecture diagram and feature overview

#### Developer Documentation
- **Architecture Decisions** (docs/decisions/) - Key technical decisions
- **API Documentation** with detailed examples
- **Plugin Development Guide** for extensions

### 🔄 Migration from Previous Versions

#### Breaking Changes
- **Complete API rewrite** - previous APIs are not compatible
- **New import structure** - `from sifaka import improve`
- **Configuration changes** - simplified configuration model
- **Result format changes** - new `SifakaResult` model with audit trail

#### Migration Benefits
- **4 dependencies** instead of 41 (90% reduction)
- **Memory-bounded** operation prevents crashes
- **Clear error messages** for better debugging
- **Working tests** that actually run and pass
- **Production-ready** stability and performance

### 🎯 Research Foundation

All critics implement peer-reviewed research papers:

- **[Reflexion](https://arxiv.org/abs/2303.11366)** - Self-reflection for iterative improvement
- **[Constitutional AI](https://arxiv.org/abs/2212.08073)** - Principle-based evaluation  
- **[Self-Refine](https://arxiv.org/abs/2303.17651)** - Iterative self-improvement
- **[N-Critics](https://arxiv.org/abs/2310.18679)** - Ensemble of diverse critical perspectives
- **[Self-RAG](https://arxiv.org/abs/2310.11511)** - Retrieval-augmented self-critique
- **[Meta-Rewarding](https://arxiv.org/abs/2407.19594)** - Two-stage judgment with meta-evaluation
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)** - Multiple reasoning paths with consensus

---

## Previous Versions

### [0.1.x - 1.0.0] - Legacy Versions

Previous versions of Sifaka (0.1.x through 1.0.0) represented initial explorations of AI text improvement. Key limitations that led to the rewrite:

#### Issues with Legacy Versions
- **Complex API** with three different improvement methods
- **41 dependencies** causing installation and conflict issues
- **Memory leaks** from unbounded collections
- **Failing tests** that didn't reflect actual functionality
- **Unclear error messages** making debugging difficult
- **No production readiness** due to stability issues

#### Migration Path
Users of legacy versions should:
1. **Update imports**: `from sifaka import improve`
2. **Simplify API calls**: Use single `improve()` function
3. **Update configuration**: Use new parameter-based configuration
4. **Review documentation**: Follow new Quick Start Guide
5. **Test thoroughly**: Verify behavior with new API