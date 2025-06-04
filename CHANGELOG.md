# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-05-30

### Added
- **Rich Result Capture Foundation**: Enhanced PydanticAI integration with comprehensive result capture
  - Added rich result methods to PydanticAIModel for detailed response metadata
  - Extended Thought object with PydanticAI-specific fields (usage, cost, timestamp, model info)
  - Updated AgentDataExtractor for comprehensive data extraction from PydanticAI responses
  - Integrated rich data capture in GenerationExecutor for complete observability
- **Comprehensive Codebase Review**: Added detailed REVIEW.md with analysis and recommendations
  - Code quality assessment across maintainability, extensibility, usability criteria
  - Architectural recommendations and improvement suggestions
  - Performance optimization guidelines
- **Example Verification**: Validated all 8 example files working seamlessly
  - Constitutional critic, meta rewarding, n-critics, prompt critic all functional
  - Reflexion, self-consistency, self-RAG, self-refine examples verified
  - No async/sync issues or import errors detected
  - PydanticAI integration working across all examples

### Enhanced
- **PydanticAI Integration**: Phase 1 of enhanced integration complete
  - Maintains full backward compatibility while adding rich features
  - Better observability and debugging capabilities
  - Enhanced metadata capture for cost tracking and performance analysis
- **Development Environment**: Improved uv environment compatibility
  - All examples verified working with new uv package management
  - Streamlined dependency management and installation process
- **Documentation**: Updated README and comprehensive documentation review
  - Import ordering fixes and consistency improvements
  - Enhanced troubleshooting and configuration guidance

### Technical Improvements
- **AgentDataExtractor**: Enhanced extraction capabilities for PydanticAI responses
- **Thought Object**: Extended with rich PydanticAI metadata fields
- **GenerationExecutor**: Integrated rich data capture for complete audit trails
- **Example Stability**: All critics and validators working reliably across model providers

### Quality Assurance
- **Zero Breaking Changes**: All enhancements maintain backward compatibility
- **Example Coverage**: 100% of examples verified and working
- **Integration Testing**: Comprehensive validation across all supported model providers
- **Documentation Accuracy**: All guides and examples updated and verified

---

## [0.3.0] - 2025-05-30

### 🚨 BREAKING CHANGES

#### **Complete Removal of Traditional Chain Implementation**
- **REMOVED**: Entire `sifaka.core.chain` module and traditional Chain class
- **REMOVED**: `QuickStart` utility class (was based on traditional Chain)
- **REMOVED**: All backward compatibility code and properties
- **REMOVED**: Traditional chain examples, tests, and documentation
- **IMPACT**: All existing code using `from sifaka import Chain` will break

#### **PydanticAI-Only Architecture**
- **NEW**: Sifaka is now exclusively PydanticAI-native
- **SIMPLIFIED**: Single chain implementation via `sifaka.agents.create_pydantic_chain`
- **MODERNIZED**: All workflows now use PydanticAI agents with tool calling
- **STREAMLINED**: Removed dual-API confusion and maintenance burden

### Migration Guide

**Before (v0.2.x - Traditional Chain)**:
```python
from sifaka import Chain
from sifaka.models import create_model

chain = Chain(
    model=create_model("openai:gpt-4"),
    prompt="Your prompt here"
)
chain = chain.validate_with(validator).improve_with(critic)
result = chain.run()
```

**After (v0.3.0 - PydanticAI Only)**:
```python
from pydantic_ai import Agent
from sifaka.agents import create_pydantic_chain

agent = Agent("openai:gpt-4", system_prompt="You are a helpful assistant")
chain = create_pydantic_chain(
    agent=agent,
    validators=[validator],
    critics=[critic]
)
result = chain.run("Your prompt here")
```

### Removed
- **Traditional Chain**: Complete removal of `sifaka.core.chain` module
- **QuickStart Class**: Removed `sifaka.quickstart.QuickStart`
- **Backward Compatibility**: All compatibility properties and deprecated classes
- **Legacy Tests**: Removed traditional chain integration and unit tests
- **Legacy Documentation**: Removed all traditional chain examples and guides

### Changed
- **Version**: Bumped to 0.3.0 to reflect breaking changes
- **Package Description**: Updated to "PydanticAI-native AI validation framework"
- **Main Exports**: `sifaka.__init__.py` now only exports core interfaces and Thought objects
- **Documentation**: All examples now use PydanticAI chains exclusively

### Benefits
- ✅ **Simplified Architecture**: Single, modern chain implementation
- ✅ **No Backward Compatibility Burden**: Clean, focused codebase
- ✅ **PydanticAI-Native**: Full integration with modern agent patterns
- ✅ **Reduced Confusion**: No more dual-API decision paralysis
- ✅ **Easier Maintenance**: Significantly reduced codebase complexity

---

## [0.2.1] - 2025-05-29

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
- **NEW**: Full compatibility with Sifaka's retriever system (InMemoryRetriever, Redis, etc.)

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
- Storage: Memory, File, Redis (via MCP)
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
