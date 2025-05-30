# Sifaka Documentation

Welcome to the Sifaka documentation! This guide will help you get started with Sifaka and make the most of its powerful features.

## � **Current Status (v0.2.1)**

**Important**: Sifaka 0.2.1 introduces temporary compatibility changes due to dependency conflicts:
- **⚠️ HuggingFace Models**: Temporarily disabled due to PydanticAI dependency conflicts
- **⚠️ Guardrails AI**: Temporarily disabled due to griffe version incompatibility
- **⚠️ MCP Storage**: Redis and Milvus backends experiencing issues
- **✅ PydanticAI Chain**: Now the primary and recommended approach with full feature parity

## �🚀 Getting Started

New to Sifaka? Start here:

- **[Installation](getting-started/installation.md)** - Install Sifaka and set up your environment
- **[Your First Chain](getting-started/first-chain.md)** - Create and run your first Sifaka chain
- **[Basic Concepts](getting-started/basic-concepts.md)** - Understand Thoughts, Models, Validators, and Critics

## 📚 User Guides

Learn how to customize and extend Sifaka:

- **[Configuration Guide](guides/configuration.md)** - Configure Sifaka for your needs
- **[Custom Models](guides/custom-models.md)** - Create custom model integrations
- **[Custom Validators](guides/custom-validators.md)** - Build domain-specific validation
- **[Classifiers](guides/classifiers.md)** - Using built-in text classifiers for content analysis
- **[Storage Setup](guides/storage-setup.md)** - Configure Redis, Milvus, and other storage backends
- **[Performance Tuning](guides/performance-tuning.md)** - Optimize your Sifaka applications

## 🔧 Troubleshooting

Having issues? Check these guides:

- **[Common Issues](troubleshooting/common-issues.md)** - Solutions to frequently encountered problems
- **[Import Problems](troubleshooting/import-problems.md)** - Resolve import and dependency issues
- **[Configuration Errors](troubleshooting/configuration-errors.md)** - Fix configuration-related problems

## 📖 API Reference

Complete technical documentation:

- **[API Reference](api/api-reference.md)** - Complete API documentation for all components
- **[Architecture](architecture.md)** - System design and component interactions

## 🛠️ Development Guidelines

Contributing to Sifaka? Follow these standards:

- **[Contributing Guide](guidelines/contributing.md)** - How to contribute to Sifaka
- **[Docstring Standards](guidelines/docstring-standards.md)** - Documentation standards for code
- **[Import Standards](guidelines/import-standards.md)** - Import style guidelines
- **[Async/Sync Guidelines](guidelines/async-sync-guidelines.md)** - Patterns for async and sync code

## 📁 Documentation Structure

```
docs/
├── getting-started/          # New user guides
│   ├── installation.md
│   ├── first-chain.md
│   └── basic-concepts.md
├── guides/                   # User guides and tutorials
│   ├── configuration.md
│   ├── custom-models.md
│   ├── custom-validators.md
│   ├── classifiers.md
│   ├── storage-setup.md
│   └── performance-tuning.md
├── troubleshooting/          # Problem-solving guides
│   ├── common-issues.md
│   ├── import-problems.md
│   └── configuration-errors.md
├── api/                      # Technical reference
│   └── api-reference.md
├── guidelines/               # Development standards
│   ├── contributing.md
│   ├── docstring-standards.md
│   ├── import-standards.md
│   └── async-sync-guidelines.md
└── architecture.md           # System architecture
```

## 🎯 Quick Navigation

### I want to...

**Get started quickly** → [Installation](getting-started/installation.md) → [First Chain](getting-started/first-chain.md)

**Understand the concepts** → [Basic Concepts](getting-started/basic-concepts.md) → [Architecture](architecture.md)

**Customize Sifaka** → [Custom Models](guides/custom-models.md) or [Custom Validators](guides/custom-validators.md)

**Use text classifiers** → [Classifiers Guide](guides/classifiers.md)

**Set up storage** → [Storage Setup](guides/storage-setup.md)

**Optimize performance** → [Performance Tuning](guides/performance-tuning.md)

**Fix a problem** → [Common Issues](troubleshooting/common-issues.md) or specific troubleshooting guides

**Contribute code** → [Contributing Guide](guidelines/contributing.md) → [Development Guidelines](guidelines/)

**Find API details** → [API Reference](api/api-reference.md)

## 💡 Examples

Looking for working examples? Check out the [examples directory](../examples/) in the repository:

- **[OpenAI Examples](../examples/openai/)** - Using OpenAI models ✅ **Fully supported**
- **[Anthropic Examples](../examples/anthropic/)** - Using Claude models ✅ **Fully supported**
- **[Google Gemini Examples](../examples/gemini/)** - Using Gemini models ✅ **Fully supported**
- **[Ollama Examples](../examples/ollama/)** - Using local models ✅ **Fully supported**
- **[Mock Examples](../examples/mock/)** - Development and testing ✅ **Fully supported**
- **~~HuggingFace Examples~~** - ⚠️ **Temporarily disabled** due to dependency conflicts
- **~~Storage Examples~~** - ⚠️ **MCP storage currently broken** (use Memory/File storage)
- **~~Validation Examples~~** - ⚠️ **Guardrails temporarily disabled** (use built-in validators)

## 🆘 Getting Help

If you can't find what you're looking for:

1. **Check the troubleshooting guides** - Most common issues are covered
2. **Search the documentation** - Use your browser's search (Ctrl/Cmd+F)
3. **Review the examples** - See working code for your use case
4. **Check the API reference** - Complete technical details
5. **Open an issue** - Report bugs or request features on GitHub

## 📝 Documentation Standards

This documentation follows these principles:

- **User-focused**: Written for developers using Sifaka
- **Example-driven**: Every concept includes working code
- **Progressive**: Start simple, build to advanced topics
- **Searchable**: Clear headings and consistent structure
- **Accurate**: All examples are tested and working

---

**Ready to get started?** Begin with the [Installation Guide](getting-started/installation.md)!
