# PydanticAI Integration Enhancement Plan

## Overview

Transform Sifaka from a post-processing validation/criticism system to a generation-time guidance system using PydanticAI's advanced features. This plan addresses three key PydanticAI features we're currently underutilizing:

1. **Dependencies** - For real-time validator/critic guidance during generation
2. **AgentRunResult** - For rich metadata capture (usage, cost, messages, tool calls)
3. **Message History** - For conversation memory and learning from previous iterations

## Current State Analysis

### ✅ What We're Doing Well
- Basic PydanticAI agent wrapping via `PydanticAIModel`
- Hybrid composition architecture (PydanticAI + Sifaka components)
- Proper async/sync compatibility
- Basic tool support documentation

### ❌ Major Integration Gaps
- **No dependency injection** - Validators/critics are post-processing only
- **Limited result capture** - Only extracting text, discarding rich metadata
- **No conversation memory** - Each generation starts fresh, no learning
- **Placeholder tool integration** - Tool registration not implemented

## Implementation Phases

## Phase 1: Rich Result Capture & Foundation
**Timeline**: 1-2 days
**Goal**: Capture full PydanticAI AgentRunResult data and establish foundation

### 1.1 Enhance PydanticAIModel Result Handling
- [ ] Add `_extract_rich_result()` method to capture full `AgentRunResult`
- [ ] Modify `generate()` to optionally return rich data
- [ ] Add `generate_with_rich_result()` method for full metadata access
- [ ] Ensure backward compatibility for existing text-only usage

### 1.2 Enhance Thought Object
- [ ] Add fields: `usage`, `cost`, `messages`, `tool_calls`, `pydantic_metadata`
- [ ] Update `Thought.from_dict()` and `model_dump()` serialization
- [ ] Modify storage implementations to handle new fields
- [ ] Add helper methods for accessing PydanticAI data

### 1.3 Update Chain Execution
- [ ] Modify `GenerationExecutor` to capture rich result data
- [ ] Update thought logging to include PydanticAI metadata
- [ ] Enhance examples to demonstrate rich data access
- [ ] Add usage/cost tracking across iterations

**Deliverables**:
- Enhanced `PydanticAIModel` with rich result capture
- Extended `Thought` object with PydanticAI fields
- Updated examples showing usage/cost/message data
- Backward compatible API

## Phase 2: Dependency Integration
**Timeline**: 2-3 days
**Goal**: Convert validators/critics to PydanticAI dependencies for generation-time guidance

### 2.1 Create Dependency Adapters
- [ ] Create `ValidatorDependency` wrapper class
- [ ] Create `CriticDependency` wrapper class
- [ ] Implement dependency injection patterns for Sifaka components
- [ ] Add dependency lifecycle management

### 2.2 Modify PydanticAIChain
- [ ] Add dependency registration during chain initialization
- [ ] Update agent creation to include Sifaka dependencies
- [ ] Implement `enable_generation_time_validation` option
- [ ] Maintain backward compatibility with post-processing

### 2.3 Update Validation/Criticism Flow
- [ ] Add generation-time validation hooks
- [ ] Implement real-time critic feedback during generation
- [ ] Create hybrid validation (generation-time + post-processing)
- [ ] Add dependency-aware error handling

**Deliverables**:
- Validator/critic dependency wrappers
- Generation-time validation capabilities
- Hybrid validation workflows
- Enhanced chain configuration options

## Phase 3: Message History Integration
**Timeline**: 2-3 days
**Goal**: Replace custom thought storage with PydanticAI conversation history

### 3.1 Message History Adapter
- [ ] Create `ConversationHistoryAdapter` class
- [ ] Implement Sifaka thought → PydanticAI message conversion
- [ ] Design hybrid storage (PydanticAI memory + Sifaka persistence)
- [ ] Add conversation reconstruction capabilities

### 3.2 Conversation Memory Integration
- [ ] Modify chain execution to use agent conversation history
- [ ] Convert critic feedback to conversation context
- [ ] Implement thought reconstruction from message history
- [ ] Add conversation-aware context building

### 3.3 Enhanced Context Awareness
- [ ] Use conversation history for better context in subsequent generations
- [ ] Implement learning from validation failures
- [ ] Add conversation-aware critic feedback
- [ ] Create memory-based improvement strategies

**Deliverables**:
- Conversation history integration
- Memory-aware chain execution
- Learning from previous iterations
- Enhanced context building

## Phase 4: Advanced Tool Integration
**Timeline**: 3-4 days
**Goal**: Implement validators/critics as PydanticAI tools for maximum integration

### 4.1 Tool Registration System
- [ ] Implement `_setup_validator_tools()` method
- [ ] Implement `_setup_critic_tools()` method
- [ ] Create tool wrappers for Sifaka components
- [ ] Add dynamic tool registration/deregistration

### 4.2 Real-time Guidance Tools
- [ ] Implement validation tools that guide generation
- [ ] Create critic tools for real-time feedback
- [ ] Add retrieval tools for context enhancement
- [ ] Implement tool-based improvement workflows

### 4.3 Advanced Workflows
- [ ] Implement self-correcting generation loops
- [ ] Add tool-based improvement iterations
- [ ] Create sophisticated agent behaviors
- [ ] Add tool usage analytics and optimization

**Deliverables**:
- Complete tool integration system
- Self-correcting generation capabilities
- Advanced agent workflows
- Tool usage analytics

## Technical Considerations

### Backward Compatibility
- All changes must maintain existing API compatibility
- Add feature flags for new behaviors
- Provide migration guides for advanced features
- Ensure examples continue to work

### Performance Impact
- Rich result capture should be optional
- Dependency injection should not slow generation
- Message history should be memory-efficient
- Tool integration should be lightweight

### Error Handling
- Graceful degradation when PydanticAI features unavailable
- Fallback to post-processing when dependencies fail
- Clear error messages for configuration issues
- Robust handling of malformed results

## Success Metrics

### Phase 1 Success
- [ ] Rich result data captured in all examples
- [ ] Usage/cost tracking working across iterations
- [ ] No performance regression in existing workflows
- [ ] All existing tests pass

### Phase 2 Success
- [ ] Validators can guide generation in real-time
- [ ] Critics provide feedback during generation
- [ ] Hybrid validation workflows functional
- [ ] Generation quality improves with guidance

### Phase 3 Success
- [ ] Conversation history preserved across iterations
- [ ] Agent learns from previous validation failures
- [ ] Context awareness improves generation quality
- [ ] Memory usage remains reasonable

### Phase 4 Success
- [ ] Validators/critics work as PydanticAI tools
- [ ] Self-correcting generation loops functional
- [ ] Tool usage provides measurable improvements
- [ ] Advanced workflows demonstrate value

## Risk Mitigation

### Technical Risks
- **PydanticAI API changes**: Pin versions, test compatibility
- **Performance degradation**: Benchmark each phase, optimize bottlenecks
- **Memory usage**: Monitor conversation history size, implement cleanup
- **Complexity increase**: Maintain clear separation of concerns

### Integration Risks
- **Breaking changes**: Extensive testing, feature flags
- **Dependency conflicts**: Careful version management
- **Configuration complexity**: Provide sensible defaults, clear documentation

## Next Steps

1. **Review this plan** - Gather feedback on approach and priorities
2. **Start Phase 1** - Begin with rich result capture foundation
3. **Iterative development** - Complete each phase before moving to next
4. **Continuous testing** - Ensure no regressions throughout process
5. **Documentation updates** - Keep docs current with new capabilities

This plan transforms Sifaka from a post-processing framework to a generation-time guidance system, fully leveraging PydanticAI's advanced capabilities while maintaining backward compatibility and our preferred architectural patterns.
