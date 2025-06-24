# Sifaka Documentation

This directory contains comprehensive documentation for the Sifaka project architecture, design decisions, and development guidelines.

## 📁 Documentation Structure

### Architecture Documentation
- **[Architecture Overview](architecture.md)** - High-level system architecture and component relationships
- **[Design Decisions](decisions/)** - Architectural Decision Records (ADRs) documenting key choices

### Development Documentation
- **[Plugin Development](../examples/plugins/README.md)** - Guide for creating Sifaka plugins with examples
- **[Contributing](../CONTRIBUTING.md)** - Guidelines for contributing to the project
- **[FAQ](FAQ.md)** - Frequently asked questions and troubleshooting

## 🎯 Key Architectural Principles

### 1. **Simplicity First**
- Single function API: `improve()`
- Minimal dependencies (5 core)
- Clear, obvious interfaces

### 2. **Production Ready**
- Memory-bounded operations
- Cost tracking and limits
- Comprehensive error handling
- Observable operations with audit trails

### 3. **Research-Backed**
- All critics implement peer-reviewed papers
- Evidence-based improvement methodologies
- Scientific foundation for critique approaches

### 4. **Extensible Design**
- Plugin architecture for storage backends
- Custom critic interface
- Validator extensibility
- Model provider abstraction

### 5. **Type Safety**
- Pydantic models throughout
- mypy type checking
- Clear contracts and interfaces

## 🏗️ Architecture Highlights

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   improve()     │    │  SifakaEngine   │    │    Critics      │
│   Function      │───▶│     Core        │───▶│   (Research)    │
│   (User API)    │    │   Orchestrator  │    │   Implementations│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │                        │
                               ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Validators    │    │    Storage      │
                       │   (Quality)     │    │   (Persistence) │
                       └─────────────────┘    └─────────────────┘
```

### Research Foundation

Each critic implements a specific research methodology:

- **Reflexion** - Self-reflection and iterative learning
- **Constitutional AI** - Principle-based evaluation
- **Self-Refine** - Quality-focused iterative improvement
- **N-Critics** - Multi-perspective ensemble analysis
- **Self-RAG** - Retrieval-augmented factual critique
- **Meta-Rewarding** - Two-stage judgment evaluation
- **Self-Consistency** - Consensus-based assessment

## 📖 Documentation Guidelines

### For Contributors
1. **Read the Architecture Overview** first to understand the system
2. **Review relevant ADRs** before making design changes
3. **Document new decisions** using the ADR template
4. **Update examples** when adding new features
5. **Maintain test coverage** for all components

### For Plugin Developers
1. **Follow the Plugin Development Guide** for creating extensions
2. **Use the provided interfaces** for consistency
3. **Include comprehensive tests** for plugin functionality
4. **Document plugin-specific configuration** options

### For Users
1. **Start with the Quick Start Guide** (../QUICKSTART.md)
2. **Reference the API Documentation** (../API.md) for details
3. **Explore examples** (../examples/) for usage patterns

## 🔗 Related Resources

- **[GitHub Repository](https://github.com/sifaka-ai/sifaka)** - Source code and issues
- **[PyPI Package](https://pypi.org/project/sifaka/)** - Installation and releases
- **[Research Papers](../README.md#research-foundation)** - Academic foundation
- **[Examples](../examples/)** - Working code examples

## 🤝 Getting Help

- **Documentation Issues**: Open an issue on GitHub
- **Usage Questions**: Check the API documentation first
- **Feature Requests**: Discuss in GitHub issues
- **Bug Reports**: Use the issue template with reproduction steps