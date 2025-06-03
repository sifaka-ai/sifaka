# Sifaka Critics Enhancement TODO List

Based on the IMPROVEMENT.md plan, here's a comprehensive todo list organized by phase and priority.

## 🚀 **IMMEDIATE ACTION PLAN** - Addressing PydanticAI Integration Gaps

### **Current Status**: 85% Complete Phase 1, Ready for Advanced PydanticAI Features

**Key Insight**: We have a solid foundation with structured output but are missing PydanticAI's most powerful features:
- ❌ **Tools** - Only basic retrievers, missing web search, analysis, fact-checking
- ❌ **Graphs** - Zero usage of multi-step workflows and state management
- ❌ **Streaming** - No real-time feedback for long-running operations
- ❌ **Advanced Message History** - Basic integration vs. PydanticAI's conversation memory

### **Recommended Execution Order** (Maximum Impact)

1. **Phase 2A: Tool-Enabled Critics** (2-3 weeks) 🎯 **START IMMEDIATELY**
   - **Week 1**: Core tool infrastructure + web search + code analysis tools
   - **Week 2**: Enhanced existing critics (Constitutional + SelfRAG) with tools
   - **Week 3**: New tool-enabled critics (Research, FactCheck, CodeReview)
   - **ROI**: Transform critics from basic text processors → sophisticated AI systems

2. **Phase 2B: Graph-Based Workflows** (3-4 weeks) 🎯 **HIGH IMPACT**
   - **Week 1**: Graph infrastructure and ReflexionCritic conversion
   - **Week 2-3**: MetaRewardingCritic + NCriticsCritic graph workflows
   - **Week 4**: Human-in-the-loop + visualization + templates
   - **ROI**: Enable complex multi-step reasoning and resumable workflows

3. **Phase 3: Streaming + Production** (2-3 weeks) 🎯 **POLISH & SCALE**
   - **Week 1**: Streaming implementation for real-time feedback
   - **Week 2**: Performance optimization and caching
   - **Week 3**: Production features and analytics
   - **ROI**: Production-ready performance and user experience

### **🔥 IMMEDIATE MIGRATION: Retrievers → Tools**

**Current Problem**: Legacy retriever system with limited functionality
- Only basic keyword search (InMemoryRetriever)
- Static retrieval phases in chain execution
- No intelligent decision-making about when/what to retrieve
- Complex integration with separate RetrievalExecutor

**Solution**: Replace with PydanticAI tools for intelligent, dynamic retrieval
- LLM decides when and what to retrieve
- Rich tool ecosystem (web search, APIs, databases)
- Native PydanticAI integration with proper error handling
- Composable tools that can chain operations

**Migration Steps** (NO BACKWARD COMPATIBILITY):
1. **Day 1: DELETE** `sifaka/retrievers/` + `RetrievalExecutor` + retrieval phases
2. **Day 2: CONVERT** SelfRAGCritic to use `@agent.tool` + update examples
3. **Day 3: CLEAN** documentation + tests + any remaining references
**Total: 2-3 days for complete migration**

## Phase 1: Structured Output Foundation (1-2 weeks) 🎯 **Priority 1** ✅ **PARTIALLY COMPLETE**

### Core Pydantic Models ✅ **COMPLETE**
- [x] Create `CriticResult` Pydantic model with validation
- [x] Create `CritiqueFeedback` Pydantic model to replace string parsing
- [x] Create `ImprovementSuggestion` Pydantic model with metadata
- [x] Create `ViolationReport` Pydantic model with severity levels
- [x] Create `ConfidenceScore` Pydantic model with detailed breakdowns

### New PydanticAI Architecture ✅ **COMPLETE**
- [x] Create `PydanticAICritic` base class with structured output
- [x] Implement async-first design with PydanticAI agents
- [x] Add validation context awareness and retrieval integration
- [x] Create tool integration support for real-time feedback
- [x] Implement proper error handling with `CriticResult` objects

### Update Existing Critics ✅ **COMPLETE**
- [x] ✅ **COMPLETE** - Update `ConstitutionalCritic` to use PydanticAI + structured output
- [x] ✅ **COMPLETE** - Create new `ReflexionCritic` with PydanticAI + structured output
- [x] ✅ **COMPLETE** - Update `SelfRefineCritic` to use structured output
- [x] ✅ **COMPLETE** - Update `SelfRAGCritic` to use structured output
- [x] ✅ **COMPLETE** - Update `MetaRewardingCritic` to use structured output
- [x] ✅ **COMPLETE** - Update `SelfConsistencyCritic` to use structured output
- [x] ✅ **COMPLETE** - Update `NCriticsCritic` to use structured output
- [x] ✅ **COMPLETE** - Update `PromptCritic` to use structured output

### Remove String Parsing ✅ **COMPLETE**
- [x] ✅ **COMPLETE** - Remove string parsing from new PydanticAI critics
- [x] ✅ **COMPLETE** - Remove `_parse_critique()` methods from remaining critics
- [x] ✅ **COMPLETE** - Remove `_extract_violations()` string parsing logic
- [x] ✅ **COMPLETE** - Update error handling to work with structured models
- [x] ✅ **COMPLETE** - Update validation logic for new models

### Testing and Documentation ✅ **COMPLETE**
- [x] ✅ **COMPLETE** - Create working test examples for new critics
- [x] ✅ **COMPLETE** - Write comprehensive tests for new Pydantic models
- [x] ✅ **COMPLETE** - Update existing critic tests for structured output
- [x] ✅ **COMPLETE** - Update documentation with new model schemas
- [x] ✅ **COMPLETE** - Create migration guide for existing users

### Key Learnings from Implementation:
- **PydanticAI Integration**: Successfully integrated PydanticAI agents with `output_type` for structured output
- **Async Patterns**: Pure async implementation works well with `await agent.run()` pattern
- **Validation Context**: Existing validation awareness mixins integrate seamlessly
- **Error Handling**: `CriticResult` objects provide much richer error information than dictionaries
- **Performance**: Initial testing shows good performance with structured output

## Phase 2A: Tool-Enabled Critics (2-3 weeks) 🎯 **Priority 1**

**IMMEDIATE FOCUS**: Transform critics into sophisticated AI-powered evaluation systems with external tool access.

### Core Tool Infrastructure ⚡ **Days 1-3** 🔥 **IMMEDIATE PRIORITY**
- [x] **Day 1: AGGRESSIVE RETRIEVER REMOVAL** 🔥 **IN PROGRESS**
  - [ ] DELETE `sifaka/retrievers/` directory completely
  - [ ] DELETE `sifaka/agents/execution/retrieval.py`
  - [ ] REMOVE `model_retrievers` and `critic_retrievers` from `ChainConfig`
  - [ ] REMOVE retrieval phases from chain execution workflow
  - [ ] REMOVE retrieval parameters from `create_pydantic_chain()`
- [ ] **Day 2: CONVERT TO TOOLS**
  - [ ] Study PydanticAI's `@agent.tool` and `@agent.tool_plain` decorators
  - [ ] Convert InMemoryRetriever functionality to simple `@agent.tool`
  - [ ] Rewrite SelfRAGCritic to use tools instead of retriever parameter
  - [ ] Update all examples to use tools instead of retrievers
- [ ] **Day 3: CLEAN & FRAMEWORK**
  - [ ] Create `CriticToolMixin` for easy tool integration in critics
  - [ ] Implement tool error handling with `ModelRetry` exceptions
  - [ ] Update documentation and remove all retriever references

### High-Impact Tool Categories ⚡ **Week 1-2**

#### **Information Retrieval Tools** (Immediate ROI)
- [ ] **Web Search Tool** - Google Custom Search API integration
  - [ ] Implement `@agent.tool` decorated web search function
  - [ ] Add query optimization and result filtering
  - [ ] Include source credibility scoring
- [ ] **Academic Research Tool** - arXiv/PubMed integration
  - [ ] Paper search and citation extraction
  - [ ] Abstract summarization and relevance scoring
- [ ] **Knowledge Base Tool** - Wikipedia/Wikidata integration
  - [ ] Fact verification and cross-referencing
  - [ ] Entity disambiguation and linking

#### **Analysis Tools** (High Value)
- [ ] **Code Analysis Tool** - AST parsing and static analysis
  - [ ] Security vulnerability detection
  - [ ] Code quality and style checking
  - [ ] Performance optimization suggestions
- [ ] **Text Analysis Tool** - Advanced NLP capabilities
  - [ ] Sentiment analysis and bias detection
  - [ ] Readability and clarity scoring
  - [ ] Fact-checking and source validation

### Enhanced Existing Critics ⚡ **Week 2**
- [ ] **ConstitutionalCritic + Tools**
  - [ ] Add `@agent.tool` for principle lookup and legal database queries
  - [ ] Implement ethical framework validation tools
  - [ ] Add precedent search and compliance checking
- [ ] **SelfRAGCritic + Advanced Tools**
  - [ ] Enhance with multi-source retrieval tools
  - [ ] Add real-time fact-checking capabilities
  - [ ] Implement knowledge base updating tools

### New Tool-Enabled Critics ⚡ **Week 3**
- [ ] **ResearchCritic** - Web search + academic validation
- [ ] **FactCheckCritic** - Multi-source verification + credibility scoring
- [ ] **CodeReviewCritic** - Static analysis + security scanning
- [ ] **ComplianceCritic** - Regulatory checking + policy validation

### Tool Integration Examples & Documentation ⚡ **Week 3**
- [ ] Create comprehensive tool usage examples
- [ ] Document PydanticAI tool integration patterns
- [ ] Write tool development and testing guides
- [ ] Performance benchmarking vs. non-tool critics

## Phase 2B: Graph-Based Workflows (3-4 weeks) 🎯 **Priority 2**

**FOCUS**: Leverage PydanticAI's powerful graph capabilities for complex multi-step critique workflows.

### Graph Infrastructure Foundation ⚡ **Week 1**
- [ ] **Study PydanticAI Graph Architecture**
  - [ ] Analyze `pydantic_graph` library and `BaseNode` patterns
  - [ ] Understand `GraphRunContext`, `End`, and state management
  - [ ] Study graph persistence and resumption capabilities
- [ ] **Create Graph-Enabled Critic Base Classes**
  - [ ] Implement `GraphCritic` base class extending `PydanticAICritic`
  - [ ] Add graph state management for critic workflows
  - [ ] Integrate with existing validation and context systems

### High-Impact Graph Conversions ⚡ **Week 2-3**

#### **ReflexionCritic → Graph Workflow** (Perfect fit for graphs)
- [ ] **Multi-Step Reflection Nodes**
  - [ ] `InitialCritique` node - Generate first critique
  - [ ] `ReflectOnCritique` node - Self-reflection on critique quality
  - [ ] `RefineOrEnd` node - Decision to refine or complete
  - [ ] `GenerateImprovement` node - Create improved text
- [ ] **Episodic Memory Management**
  - [ ] State persistence for trial history
  - [ ] Success/failure pattern analysis
  - [ ] Learning from previous attempts

#### **MetaRewardingCritic → Graph Workflow** (Complex evaluation pipeline)
- [ ] **Reward Model Evaluation Nodes**
  - [ ] `GenerateRewards` node - Multiple reward model evaluation
  - [ ] `ConsensusBuilding` node - Aggregate and reconcile rewards
  - [ ] `IterativeRefinement` node - Improve based on reward feedback
  - [ ] `FinalEvaluation` node - Meta-evaluation of the process

#### **NCriticsCritic → Graph Workflow** (Parallel specialist coordination)
- [ ] **Parallel Critique Generation**
  - [ ] Multiple specialist critic nodes running in parallel
  - [ ] Consensus building and conflict resolution
  - [ ] Weighted aggregation based on expertise areas

### Advanced Graph Features ⚡ **Week 3-4**

#### **Human-in-the-Loop Integration**
- [ ] **Interactive Decision Nodes**
  - [ ] Human approval nodes for critical decisions
  - [ ] User feedback collection mechanisms
  - [ ] Manual override and intervention capabilities
- [ ] **Workflow Pause/Resume**
  - [ ] State persistence for long-running critiques
  - [ ] Resume from any point in the workflow
  - [ ] External input integration (web requests, user input)

#### **Graph Visualization and Monitoring**
- [ ] **Mermaid Diagram Generation**
  - [ ] Auto-generate workflow diagrams for critic graphs
  - [ ] Real-time execution state visualization
  - [ ] Performance metrics and bottleneck identification
- [ ] **Workflow Execution Monitoring**
  - [ ] Node execution timing and performance
  - [ ] State transition tracking and debugging
  - [ ] Error handling and recovery mechanisms

### Graph Optimization and Templates ⚡ **Week 4**
- [ ] **Workflow Templates**
  - [ ] Common critique patterns as reusable templates
  - [ ] Configurable graph structures for different critic types
  - [ ] Template library for rapid critic development
- [ ] **Performance Optimization**
  - [ ] Parallel node execution where possible
  - [ ] Intelligent caching and memoization
  - [ ] Conditional branching optimization

## Phase 3: Streaming and Advanced Features (2-3 weeks) 🎯 **Priority 3**

**FOCUS**: Add real-time feedback, advanced message history, and production-ready optimizations.

### Streaming Implementation ⚡ **Week 1**
- [ ] **PydanticAI Streaming Integration**
  - [ ] Study `agent.run_stream()` and `StreamedRunResult` patterns
  - [ ] Implement streaming support for long-running critics
  - [ ] Add `stream_text()` and `stream()` for structured output
- [ ] **Progressive Feedback Mechanisms**
  - [ ] Real-time confidence updates during critique generation
  - [ ] Incremental suggestion building and partial results
  - [ ] Streaming cancellation and interruption support
- [ ] **Streaming-Enabled Critics**
  - [ ] Convert `ResearchCritic` to use streaming for web search results
  - [ ] Add streaming to `ReflexionCritic` for multi-step reflection
  - [ ] Implement streaming progress for `MetaRewardingCritic`

### Enhanced Message History Integration ⚡ **Week 1-2**
- [ ] **PydanticAI Conversation Memory**
  - [ ] Study PydanticAI's message history and conversation flow
  - [ ] Integrate with Sifaka's superior Thought infrastructure
  - [ ] Create hybrid storage strategy (thoughts + conversations)
- [ ] **Advanced Memory Management**
  - [ ] Optimize memory usage for long conversations
  - [ ] Add conversation analytics and insights
  - [ ] Implement cross-iteration memory with thought references
- [ ] **Message History Optimization**
  - [ ] Preserve Sifaka's analytics while leveraging PydanticAI's flow
  - [ ] Add conversation summarization for long contexts
  - [ ] Implement intelligent message pruning and compression

### Performance Optimizations ⚡ **Week 2**
- [ ] **Caching and Parallelization**
  - [ ] Implement critic result caching with intelligent invalidation
  - [ ] Add parallel critic execution for independent critics
  - [ ] Optimize tool usage and API call batching
- [ ] **Performance Monitoring**
  - [ ] Create comprehensive performance benchmarking suite
  - [ ] Add detailed timing and resource usage metrics
  - [ ] Implement performance regression detection

### Advanced Analytics and Production Features ⚡ **Week 3**
- [ ] **Analytics and Reporting**
  - [ ] Create critic performance dashboards
  - [ ] Implement A/B testing framework for critics
  - [ ] Add detailed usage analytics and cost monitoring
- [ ] **Production Readiness**
  - [ ] Add rate limiting and throttling for external APIs
  - [ ] Implement circuit breakers for external tools
  - [ ] Create health checks and monitoring endpoints
  - [ ] Add configuration management for production deployments

## Cross-Phase Tasks

### Documentation
- [ ] Update README with new capabilities
- [ ] Create comprehensive API documentation
- [ ] Write user guides for each critic type
- [ ] Create troubleshooting guides
- [ ] Document best practices and patterns

### Testing
- [ ] Create integration test suite
- [ ] Add performance regression tests
- [ ] Implement end-to-end workflow tests
- [ ] Create mock tools for testing
- [ ] Add chaos engineering tests

### Examples and Demos
- [ ] Create example applications showcasing new features
- [ ] Build demo notebooks for each critic type
- [ ] Create video tutorials and walkthroughs
- [ ] Build interactive web demos

### Infrastructure
- [ ] Set up CI/CD pipelines for new features
- [ ] Create deployment scripts and configurations
- [ ] Set up monitoring and alerting
- [ ] Create backup and disaster recovery procedures

## Quick Wins (Can be done in parallel)

- [ ] Add type hints to all critic methods
- [ ] Improve error messages and logging
- [ ] Create critic configuration validation
- [ ] Add progress bars for long-running operations
- [ ] Implement basic caching for repeated operations
- [ ] Create utility functions for common critic tasks
- [ ] Add debug mode with detailed execution traces
- [ ] Create critic comparison and benchmarking tools

## Dependencies and Prerequisites

- [ ] Ensure PydanticAI version supports all required features
- [ ] Set up API keys and credentials for external tools
- [ ] Configure development environment with all dependencies
- [ ] Set up testing infrastructure and databases
- [ ] Create staging environment for integration testing

## Success Criteria

### Phase 1 Complete When:
- [x] ✅ **COMPLETE** - Core Pydantic models are implemented and working
- [x] ✅ **COMPLETE** - New PydanticAI architecture is established
- [x] ✅ **COMPLETE** - At least 2 critics (Constitutional, Reflexion) use structured output
- [x] ✅ **COMPLETE** - All remaining critics use Pydantic models instead of string parsing
- [x] ✅ **COMPLETE** - New critics maintain backward compatibility through legacy imports
- [ ] 🔄 **TODO** - All tests pass with new structured output
- [ ] 🔄 **TODO** - Documentation is updated and complete

### Current Status: **85% Complete** - All critics converted, testing and documentation remain

### Phase 2 Complete When:
- [ ] At least 3 new tool-enabled critics are working
- [ ] Tool integration framework is stable and documented
- [ ] Enhanced critics show measurable improvement
- [ ] Tool usage is properly monitored and optimized

### Phase 3 Complete When:
- [ ] Graph-based ReflexionCritic and MetaRewardingCritic are working
- [ ] Human-in-the-loop workflows are functional
- [ ] Graph visualization and monitoring are available
- [ ] Performance is equal or better than original critics

### Phase 4 Complete When:
- [ ] Streaming works for all applicable critics
- [ ] Performance optimizations show measurable improvements
- [ ] Production deployment is successful
- [ ] Analytics and monitoring are fully operational
