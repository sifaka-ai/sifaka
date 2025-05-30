# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-05-30

### 🚨 BREAKING CHANGES

#### **Temporary Removal of HuggingFace and Guardrails Support**
- **REMOVED**: HuggingFace examples and direct integration due to dependency conflicts with PydanticAI
- **DISABLED**: Guardrails AI integration due to griffe version incompatibility with PydanticAI
- **IMPACT**: Affects users relying on HuggingFace models in PydanticAI chains or GuardrailsValidator

### Added
- **ToolCall Tracking**: Enhanced Thought containers now track tool calls made during PydanticAI agent execution
  - **ToolCall Model**: Records tool name, arguments, results, timing, and success status
  - **Thought Integration**: `add_tool_call()` method for tracking tool usage
  - **Complete Observability**: Full audit trail of tool interactions within thought history
- **Enhanced Documentation**: Comprehensive PydanticAI chain migration guides and architectural documentation
- **Design Decisions Documentation**: Added `docs/DESIGN_DECISIONS.md` explaining architectural choices and trade-offs

### Changed
- **Thought Container**: Extended with `tool_calls` field for tracking PydanticAI agent tool usage
- **PydanticAI Chain Documentation**: Updated all documentation to reflect PydanticAI chain as primary approach
- **Architecture Documentation**: Enhanced with detailed flow diagrams and implementation patterns
- **Examples**: Updated all examples to use PydanticAI chains consistently where possible
- **GuardrailsValidator**: Marked as prerelease/disabled with clear documentation
- **Documentation Structure**: Reorganized and updated for better clarity and current architecture

### Removed
- **HuggingFace Examples**: Removed `examples/huggingface/` directory and all HuggingFace-specific examples
- **HuggingFace PydanticAI Integration**: Disabled due to dependency conflicts
- **Guardrails Dependency**: Removed from pyproject.toml due to griffe version conflicts
- **Deprecated Documentation**: Removed outdated AGENTS.md and examples/README.md files

### Fixed
- **Async Event Loop Issues**: Fixed "This event loop is already running" errors in PydanticAI chains
- **Dependency Conflicts**: Resolved installation issues by removing conflicting packages
- **PydanticAI Chain Stability**: Improved async handling and error recovery
- **Async Consistency**: Resolved mixed async/sync patterns in PydanticAI chain implementation

### Enhanced
- **Tool Call Observability**: Complete tracking of tool interactions with timing and success metrics
- **Thought History**: Improved iteration tracking with tool call preservation across iterations
- **Documentation Coverage**: Comprehensive guides for migration, architecture, and design decisions

### Workarounds
- **HuggingFace Models**: Use Traditional chains for HuggingFace models, or use OpenAI/Anthropic/Gemini with PydanticAI chains
- **Guardrails Validation**: Use built-in validators (LengthValidator, RegexValidator) or create custom validators

### Future Plans
- **HuggingFace Support**: Will be restored when PydanticAI adds native HuggingFace support
- **Guardrails Support**: Will be restored when dependency conflicts are resolved
- **Enhanced Integration**: Planning improved integration patterns for both systems
- **Tool Call Analytics**: Advanced analytics and visualization for tool usage patterns

---

## [0.2.0] - 2025-05-28

### 🚨 BREAKING CHANGES

This release introduces significant architectural improvements with breaking changes:

#### **PydanticAI Chain is now the primary approach**
- **NEW**: PydanticAI Chain is now the recommended chain implementation for all new projects
- **DEPRECATED**: Traditional Chain is now in maintenance mode (still available but deprecated)
- **FEATURE PARITY**: PydanticAI chains now support retrievers, making them functionally equivalent to Traditional chains
- **STRATEGIC ALIGNMENT**: Sifaka's development roadmap is now fully aligned with PydanticAI's evolution

#### **Full Retriever Support for PydanticAI Chains**
- **NEW**: Added `model_retrievers` and `critic_retrievers` parameters to PydanticAI chains
- **NEW**: Automatic retrieval orchestration (pre-generation and critic-specific retrieval)
- **NEW**: Full compatibility with Sifaka's retriever system (InMemoryRetriever, Redis, Milvus, etc.)

### Added
- **PydanticAI Chain Retrievers**: Full retriever support with same API as Traditional chains
- **Migration Guide**: Comprehensive guide for migrating from Traditional to PydanticAI chains
- **Breaking Changes Documentation**: Clear documentation of v0.2.0 changes
- **Feature Parity**: PydanticAI chains now match Traditional chains in capabilities
- **Vision Alignment**: Updated project vision to emphasize PydanticAI partnership and alignment

### Changed
- **Primary Recommendation**: PydanticAI Chain is now the recommended approach for new projects
- **Documentation Focus**: Updated all documentation to emphasize PydanticAI Chain
- **Chain Selection Guide**: Updated to reflect new capabilities and recommendations
- **Module Organization**: Moved `validation_context.py` from `sifaka.core` to `sifaka.validators` for better organization

### Deprecated
- **Traditional Chain**: Now in maintenance mode, still available but deprecated
- **Traditional Chain Examples**: Marked as legacy in documentation

### Migration Path

**Before (Traditional Chain)**:
```python
from sifaka import Chain
from sifaka.models import create_model

chain = Chain(
    model=create_model("openai:gpt-4"),
    prompt="Your prompt here",
    model_retrievers=[retriever],
    max_improvement_iterations=2,
    always_apply_critics=True
)
chain = chain.validate_with(validator).improve_with(critic)
result = chain.run()
```

**After (PydanticAI Chain)**:
```python
from sifaka.agents import create_pydantic_chain
from pydantic_ai import Agent

agent = Agent("openai:gpt-4", system_prompt="You are a helpful assistant")
chain = create_pydantic_chain(
    agent=agent,
    model_retrievers=[retriever],  # NEW: Now supported!
    validators=[validator],
    critics=[critic],
    max_improvement_iterations=2,
    always_apply_critics=True
)
result = chain.run("Your prompt here")
```

### Technical Details

#### **PydanticAI Chain Enhancements**
- Added `model_retrievers` parameter for pre-generation retrieval
- Added `critic_retrievers` parameter for critic-specific retrieval
- Added `_execute_model_retrieval()` method for pre-generation context
- Added `_execute_critic_retrieval()` method for critic context
- Added `always_apply_critics` parameter support
- Integrated retrieval into the chain execution flow

#### **API Changes**
- **PydanticAI Chain Constructor**: Added `model_retrievers` and `critic_retrievers` parameters
- **Factory Function**: `create_pydantic_chain()` now passes through retriever parameters
- **Execution Flow**: Added retrieval phases to PydanticAI chain execution

### Compatibility
- **Backward Compatible**: Traditional Chain still works (deprecated but functional)
- **Forward Compatible**: PydanticAI Chain now supports all Traditional Chain features
- **Migration Friendly**: Clear migration path with minimal code changes

---

## [0.1.0] - 2025-05-27

### Added
- Initial release of Sifaka framework
- Traditional Chain implementation
- PydanticAI Chain implementation (basic)
- Validators: Length, Regex, Content, Bias, Reading Level
- Critics: Reflexion, Self-Refine, Constitutional AI, Self-RAG, N-Critics
- Models: OpenAI, Anthropic, Google Gemini, HuggingFace, Ollama, Mock
- Storage: Memory, File, Redis (via MCP), Milvus (via MCP)
- Retrievers: InMemory, Mock
- Classifiers: Toxicity, Sentiment, Bias, Language, Profanity, Spam
- QuickStart utilities for rapid setup
- Comprehensive documentation and examples

### Features
- Thought-centric architecture with complete state tracking
- Validation-first design with iterative improvement
- Academic research integration (Reflexion, Self-Refine, etc.)
- MCP integration for external services
- Complete observability and audit trails
- Type-safe components with Pydantic models
