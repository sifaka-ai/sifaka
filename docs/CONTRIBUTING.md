# Contributing to Sifaka

Welcome to Sifaka! We're excited that you're interested in contributing to the future of reliable AI text generation. This guide will help you get started and make meaningful contributions to the project.

## 🎯 Our Mission

Sifaka makes AI text generation reliable, observable, and trustworthy through research-backed validation, criticism, and iterative improvement. Every contribution should align with this mission.

## 🚀 Quick Start for Contributors

### 1. Development Setup

```bash
# Clone the repository
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka

# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to ensure everything works
make test
```

### 2. Development Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... code, test, document ...

# Run the full test suite
make test

# Format and lint your code
make format

# Commit your changes
git commit -m "feat: add your feature description"

# Push and create a pull request
git push origin feature/your-feature-name
```

## 🎯 Contribution Areas

### **High Priority: Coming Soon Features**

#### **🚀 Enhanced T5 Summarization**
- **Goal**: Advanced text summarization for validation and critic feedback
- **Skills**: NLP, Transformers, Text Summarization
- **Impact**: Improve feedback quality and conciseness
- **Files**: `sifaka/critics/feedback_summarizer.py`

#### **🚀 HuggingFace Integration Restoration**
- **Goal**: Restore HuggingFace model support in PydanticAI chains
- **Skills**: PydanticAI, HuggingFace Transformers, Dependency Management
- **Impact**: Expand model provider support
- **Blocker**: Waiting for PydanticAI native support

#### **🚀 Guardrails AI Integration**
- **Goal**: Restore GuardrailsValidator functionality
- **Skills**: Guardrails AI, Dependency Management, Validation
- **Impact**: Advanced validation capabilities
- **Blocker**: Griffe version compatibility

#### **🚀 MCP Storage Restoration**
- **Goal**: Fix Redis and Milvus storage backends via MCP
- **Skills**: MCP Protocol, Redis, Milvus, Distributed Systems
- **Impact**: Production-ready storage
- **Priority**: High

### **Core Framework Development**

#### **PydanticAI Integration**
- **Tool Development**: Create new PydanticAI tools for validation and criticism
- **Agent Patterns**: Develop advanced agent patterns for reliability
- **Type Safety**: Enhance Pydantic integration for structured outputs
- **Performance**: Optimize PydanticAI chain execution

#### **Critics and Validators**
- **Research Implementation**: Implement new research papers as critics
- **Domain-Specific Validators**: Create validators for specific domains (legal, medical, technical)
- **Ensemble Methods**: Combine multiple critics for better feedback
- **Adaptive Critics**: Critics that learn from validation patterns

#### **Storage and Retrieval**
- **Semantic Search**: Advanced search across thought histories
- **Distributed Storage**: Scale storage for large applications
- **Real-time Indexing**: Index thoughts and context documents
- **Caching Strategies**: Optimize performance through intelligent caching

### **Documentation and Examples**

#### **User Guides**
- **Getting Started**: Improve onboarding experience
- **Advanced Patterns**: Document complex use cases
- **Best Practices**: Share production deployment patterns
- **Troubleshooting**: Help users solve common problems

#### **Examples and Tutorials**
- **Domain-Specific Examples**: Healthcare, legal, technical writing
- **Integration Examples**: Popular AI platforms and tools
- **Performance Examples**: High-throughput production patterns
- **Research Examples**: Academic use cases and benchmarks

## 📋 Contribution Guidelines

### **Code Quality Standards**

#### **Code Style**
- **Formatting**: Use `black` for code formatting (enforced by pre-commit)
- **Imports**: Use `isort` for import organization (enforced by pre-commit)
- **Linting**: Use `ruff` for linting (enforced by pre-commit)
- **Type Hints**: All public APIs must have complete type hints

#### **Testing Requirements**
- **Unit Tests**: All new code must have unit tests (>90% coverage)
- **Integration Tests**: Complex features need integration tests
- **Example Tests**: All examples must have working tests
- **Performance Tests**: Performance-critical code needs benchmarks

#### **Documentation Requirements**
- **Docstrings**: All public APIs must have comprehensive docstrings
- **Type Documentation**: Complex types should be documented
- **Examples**: Public APIs should include usage examples
- **Changelog**: Breaking changes must be documented

### **Research-Backed Development**

#### **Academic Rigor**
- **Citations**: All research implementations must cite original papers
- **Accuracy**: Implementations must faithfully represent research
- **Parameters**: Configurable parameters should match research specifications
- **Evaluation**: Include evaluation metrics from original papers

#### **Reproducibility**
- **Deterministic**: Results should be reproducible with fixed seeds
- **Configurable**: All parameters should be configurable
- **Documented**: Implementation decisions should be documented
- **Tested**: Research implementations need comprehensive tests

### **Architecture Principles**

#### **Thought-First Design**
- **Immutable State**: Thoughts should be immutable after iteration completion
- **Complete Audit Trails**: Every operation should be traceable
- **Serializable**: All state should be serializable for storage
- **Observable**: All operations should be observable and debuggable

#### **PydanticAI Integration**
- **Native Tools**: Use PydanticAI tools for extensibility
- **Type Safety**: Leverage Pydantic for validation and serialization
- **Async Patterns**: Use async/await throughout
- **Agent Patterns**: Follow PydanticAI best practices

## 🔄 Development Process

### **Issue Workflow**

1. **Check Existing Issues**: Search for existing issues before creating new ones
2. **Use Issue Templates**: Follow the provided templates for bugs and features
3. **Label Appropriately**: Use labels to categorize issues
4. **Assign Yourself**: Assign yourself to issues you're working on

### **Pull Request Process**

1. **Create Feature Branch**: Branch from `main` for new features
2. **Small, Focused PRs**: Keep PRs small and focused on single features
3. **Comprehensive Testing**: Include tests for all new functionality
4. **Documentation Updates**: Update documentation for user-facing changes
5. **Review Process**: Address all review feedback before merging

### **Release Process**

1. **Semantic Versioning**: Follow semantic versioning (MAJOR.MINOR.PATCH)
2. **Changelog Updates**: Update CHANGELOG.md with all changes
3. **Breaking Changes**: Clearly document breaking changes
4. **Migration Guides**: Provide migration guides for breaking changes

## 🏆 Recognition

### **Contributor Recognition**
- **Contributors File**: All contributors are listed in CONTRIBUTORS.md
- **Release Notes**: Significant contributions are highlighted in releases
- **Academic Credit**: Research implementations credit original authors and implementers
- **Community Spotlight**: Outstanding contributors are featured in community updates

### **Maintainer Path**
- **Consistent Contributions**: Regular, high-quality contributions
- **Community Engagement**: Active participation in discussions and reviews
- **Technical Leadership**: Demonstrating technical expertise and judgment
- **Mentorship**: Helping other contributors and users

## 🤝 Community Guidelines

### **Code of Conduct**
- **Respectful**: Treat all community members with respect
- **Inclusive**: Welcome contributors from all backgrounds
- **Constructive**: Provide constructive feedback and criticism
- **Professional**: Maintain professional standards in all interactions

### **Communication Channels**
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Request Reviews**: For code review and technical discussion
- **Documentation**: For user guides and API documentation

## 🎓 Learning Resources

### **Getting Started with Sifaka**
- **[Architecture Document](docs/ARCHITECTURE.md)**: Understand the system design
- **[Design Decisions](docs/DESIGN_DECISIONS.md)**: Learn about key architectural choices
- **[Examples](examples/)**: Study working examples
- **[API Reference](docs/api/)**: Comprehensive API documentation

### **Research Background**
- **[Critics Documentation](docs/critics.md)**: Understand criticism techniques
- **[Validation Guide](docs/guides/custom-validators.md)**: Learn about validation patterns
- **[Research Papers](docs/research/)**: Read the papers behind our implementations

### **Technical Skills**
- **[PydanticAI Documentation](https://ai.pydantic.dev/)**: Learn PydanticAI patterns
- **[Pydantic Documentation](https://docs.pydantic.dev/)**: Understand type validation
- **[Async Python](https://docs.python.org/3/library/asyncio.html)**: Master async patterns

## 🚀 Getting Help

### **For Contributors**
- **GitHub Discussions**: Ask questions about contributing
- **Issue Comments**: Get help with specific issues
- **Code Reviews**: Learn from feedback on your PRs
- **Documentation**: Check existing documentation first

### **For Maintainers**
- **Maintainer Guidelines**: Follow maintainer-specific guidelines
- **Release Process**: Understand the release workflow
- **Community Management**: Help foster a welcoming community
- **Technical Decisions**: Participate in architectural discussions

---

**Thank you for contributing to Sifaka!** Together, we're building the future of reliable AI text generation.
