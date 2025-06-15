# Sifaka Architecture Guide

This guide explains how Sifaka works under the hood, from high-level concepts to implementation details.

## 🎯 Core Philosophy

Sifaka transforms AI text generation from a **black box** into a **transparent, iterative improvement system**:

- **Traditional AI**: Prompt → Model → Text → Hope it's good ❌
- **Sifaka**: Prompt → Generate → Validate → Critique → Improve → Repeat → Guaranteed Quality ✅

## 🏗️ High-Level Architecture

```mermaid
graph TB
    subgraph "User Interface"
        API[Simple API]
        CONFIG[Configuration]
        ENGINE[SifakaEngine]
    end
    
    subgraph "Core Workflow"
        THOUGHT[SifakaThought<br/>State Container]
        GRAPH[PydanticAI Graph]
        
        subgraph "Graph Nodes"
            GEN[Generate Node]
            VAL[Validate Node] 
            CRIT[Critique Node]
        end
    end
    
    subgraph "Components"
        MODELS[AI Models<br/>OpenAI, Anthropic, etc.]
        VALIDATORS[Validators<br/>Length, Content, etc.]
        CRITICS[Critics<br/>Reflexion, Constitutional]
    end
    
    subgraph "Storage"
        MEMORY[Memory]
        FILE[File System]
        REDIS[Redis]
        POSTGRES[PostgreSQL]
    end
    
    API --> ENGINE
    CONFIG --> ENGINE
    ENGINE --> THOUGHT
    THOUGHT --> GRAPH
    GRAPH --> GEN
    GRAPH --> VAL
    GRAPH --> CRIT
    
    GEN --> MODELS
    VAL --> VALIDATORS
    CRIT --> CRITICS
    
    THOUGHT --> STORAGE
    
    classDef userLayer fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef coreLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef componentLayer fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef storageLayer fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class API,CONFIG,ENGINE userLayer
    class THOUGHT,GRAPH,GEN,VAL,CRIT coreLayer
    class MODELS,VALIDATORS,CRITICS componentLayer
    class MEMORY,FILE,REDIS,POSTGRES storageLayer
```

## 🔄 Core Workflow

The heart of Sifaka is an iterative improvement loop:

```mermaid
graph TD
    START[User Prompt] --> THOUGHT[Create SifakaThought]
    THOUGHT --> GEN[🤖 Generate Text]
    GEN --> VAL[✅ Validate Output]
    VAL --> CRIT[🔍 Apply Critics]
    CRIT --> DECISION{Continue?}
    
    DECISION -->|✅ Quality Met<br/>OR Max Iterations| COMPLETE[🎯 Return Result]
    DECISION -->|❌ Needs Improvement| FEEDBACK[📝 Combine Feedback]
    FEEDBACK --> GEN
    
    classDef startEnd fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class START,COMPLETE startEnd
    class THOUGHT,GEN,VAL,CRIT,FEEDBACK process
    class DECISION decision
```

### Decision Logic

The system continues iterating when:
- ✅ **Validation fails** (requirements not met)
- ✅ **Critics suggest improvements** (quality can be better)
- ✅ **Under max iterations** (haven't hit the limit)

The system stops when:
- 🎯 **All validations pass AND no critic suggestions** (perfect quality)
- 🔄 **Max iterations reached** (time to finalize)

## 🧠 State Management

### SifakaThought: The Central State Container

Every improvement process is tracked in a `SifakaThought` object:

```mermaid
graph LR
    subgraph "SifakaThought State"
        PROMPT[User Prompt]
        GENS[Generations<br/>History]
        VALS[Validation<br/>Results]
        CRITS[Critique<br/>Feedback]
        CONVS[Conversation<br/>History]
        META[Metadata<br/>& Timing]
    end
    
    PROMPT --> GENS
    GENS --> VALS
    VALS --> CRITS
    CRITS --> CONVS
    CONVS --> META
    
    classDef stateBox fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    class PROMPT,GENS,VALS,CRITS,CONVS,META stateBox
```

**Complete Audit Trail**: Every generation, validation result, critique, and model conversation is preserved.

## 🎭 Component Architecture

### 1. Generators (AI Models)

```mermaid
graph LR
    subgraph "Model Providers"
        OPENAI[OpenAI<br/>GPT-4, GPT-3.5]
        ANTHROPIC[Anthropic<br/>Claude-3]
        GOOGLE[Google<br/>Gemini]
        GROQ[Groq<br/>Llama-3.1]
    end
    
    subgraph "PydanticAI Integration"
        AGENT[PydanticAI Agent]
        TOOLS[Tool Calling]
        STRUCT[Structured Output]
    end
    
    OPENAI --> AGENT
    ANTHROPIC --> AGENT
    GOOGLE --> AGENT
    GROQ --> AGENT
    
    AGENT --> TOOLS
    AGENT --> STRUCT
    
    classDef provider fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef integration fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class OPENAI,ANTHROPIC,GOOGLE,GROQ provider
    class AGENT,TOOLS,STRUCT integration
```

### 2. Validators (Quality Checks)

```mermaid
graph TD
    subgraph "Built-in Validators"
        LENGTH[Length Validator<br/>Min/Max words/chars]
        CONTENT[Content Validator<br/>Required/Prohibited terms]
        FORMAT[Format Validator<br/>JSON, Markdown, etc.]
    end
    
    subgraph "AI-Powered Validators"
        SENTIMENT[Sentiment Validator<br/>Positive/Negative/Neutral]
        TOXICITY[Toxicity Validator<br/>Harmful content detection]
        READABILITY[Readability Validator<br/>Grade level assessment]
    end
    
    subgraph "Custom Validators"
        CUSTOM[Your Custom Logic]
    end
    
    classDef builtin fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef ai fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef custom fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class LENGTH,CONTENT,FORMAT builtin
    class SENTIMENT,TOXICITY,READABILITY ai
    class CUSTOM custom
```

### 3. Critics (Improvement Agents)

```mermaid
graph TD
    subgraph "Research-Backed Critics"
        REFLEXION[Reflexion Critic<br/>Self-reflection & improvement<br/>📄 Shinn et al. 2023]
        CONSTITUTIONAL[Constitutional Critic<br/>Principle-based evaluation<br/>📄 Anthropic 2022]
        SELFREFINE[Self-Refine Critic<br/>Iterative self-improvement<br/>📄 Madaan et al. 2023]
    end
    
    subgraph "Advanced Critics"
        SELFRAG[Self-RAG Critic<br/>Retrieval-augmented critique<br/>📄 Asai et al. 2023]
        METAREWARD[Meta-Rewarding Critic<br/>Two-stage judgment<br/>📄 Wu et al. 2024]
    end
    
    classDef research fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef advanced fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    
    class REFLEXION,CONSTITUTIONAL,SELFREFINE research
    class SELFRAG,METAREWARD advanced
```

## 🔧 Configuration System

Sifaka provides multiple configuration levels for different user needs:

```mermaid
graph TD
    subgraph "Configuration Levels"
        SIMPLE[Simple API<br/>sifaka.improve()]
        CONFIG[SifakaConfig<br/>Builder Pattern]
        DEPS[SifakaDependencies<br/>Full Control]
    end
    
    subgraph "Configuration Options"
        MODELS[Model Selection]
        VALIDATORS[Validator Setup]
        CRITICS[Critic Configuration]
        WEIGHTS[Feedback Weighting]
        STORAGE[Storage Backend]
    end
    
    SIMPLE --> MODELS
    CONFIG --> MODELS
    DEPS --> MODELS
    
    CONFIG --> VALIDATORS
    DEPS --> VALIDATORS
    
    CONFIG --> CRITICS
    DEPS --> CRITICS
    
    DEPS --> WEIGHTS
    DEPS --> STORAGE
    
    classDef level fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef option fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class SIMPLE,CONFIG,DEPS level
    class MODELS,VALIDATORS,CRITICS,WEIGHTS,STORAGE option
```

## 📊 Feedback Weighting System

Sifaka combines validation and critic feedback using configurable weights:

```mermaid
graph LR
    subgraph "Feedback Sources"
        VAL_FEEDBACK[Validation Feedback<br/>60% weight default]
        CRIT_FEEDBACK[Critic Feedback<br/>40% weight default]
    end
    
    subgraph "Combination"
        WEIGHTED[Weighted Combination]
        NEXT_PROMPT[Next Iteration Prompt]
    end
    
    VAL_FEEDBACK --> WEIGHTED
    CRIT_FEEDBACK --> WEIGHTED
    WEIGHTED --> NEXT_PROMPT
    
    classDef feedback fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class VAL_FEEDBACK,CRIT_FEEDBACK feedback
    class WEIGHTED,NEXT_PROMPT process
```

**Why This Works**: Validation feedback ensures requirements are met, while critic feedback drives quality improvements.

## 💾 Storage Architecture

Sifaka supports multiple storage backends with automatic failover:

```mermaid
graph TD
    subgraph "Storage Backends"
        MEMORY[Memory Storage<br/>Fast, No Persistence]
        FILE[File Storage<br/>Local Persistence]
        REDIS[Redis Storage<br/>Distributed Cache]
        POSTGRES[PostgreSQL<br/>Full Database]
    end
    
    subgraph "Hybrid Configuration"
        CACHE[Cache Layer<br/>Memory]
        PRIMARY[Primary Storage<br/>Redis]
        BACKUP[Backup Storage<br/>File]
        SEARCH[Search Storage<br/>PostgreSQL]
    end
    
    MEMORY --> CACHE
    REDIS --> PRIMARY
    FILE --> BACKUP
    POSTGRES --> SEARCH
    
    classDef storage fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef hybrid fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class MEMORY,FILE,REDIS,POSTGRES storage
    class CACHE,PRIMARY,BACKUP,SEARCH hybrid
```

## 🚀 Performance Characteristics

### Parallel Processing

```mermaid
graph TD
    START[Text Generated] --> PARALLEL{Parallel Processing}
    
    PARALLEL --> VAL1[Validator 1]
    PARALLEL --> VAL2[Validator 2]
    PARALLEL --> VAL3[Validator 3]
    
    PARALLEL --> CRIT1[Critic 1]
    PARALLEL --> CRIT2[Critic 2]
    
    VAL1 --> COMBINE[Combine Results]
    VAL2 --> COMBINE
    VAL3 --> COMBINE
    CRIT1 --> COMBINE
    CRIT2 --> COMBINE
    
    COMBINE --> DECISION[Continue?]
    
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef parallel fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class START,COMBINE,DECISION process
    class VAL1,VAL2,VAL3,CRIT1,CRIT2 parallel
```

**Key Benefits**:
- ⚡ **Parallel Validation**: All validators run simultaneously
- ⚡ **Parallel Critique**: All critics run simultaneously  
- ⚡ **Async Throughout**: Full async/await support
- ⚡ **Caching**: Model responses and validation results cached

## 🔍 Observability

Every aspect of the improvement process is observable:

```mermaid
graph LR
    subgraph "Audit Trail"
        ITERATIONS[Iteration History]
        CONVERSATIONS[Model Conversations]
        VALIDATIONS[Validation Results]
        CRITIQUES[Critique Feedback]
        TIMINGS[Performance Metrics]
    end
    
    subgraph "Analysis Tools"
        INSPECTOR[Thought Inspector]
        METRICS[Performance Metrics]
        DEBUGGING[Debug Information]
    end
    
    ITERATIONS --> INSPECTOR
    CONVERSATIONS --> INSPECTOR
    VALIDATIONS --> INSPECTOR
    CRITIQUES --> INSPECTOR
    TIMINGS --> METRICS
    
    classDef audit fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef tools fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    
    class ITERATIONS,CONVERSATIONS,VALIDATIONS,CRITIQUES,TIMINGS audit
    class INSPECTOR,METRICS,DEBUGGING tools
```

## 🎯 Design Principles

1. **Transparency**: Every decision is logged and auditable
2. **Modularity**: Components can be mixed and matched
3. **Extensibility**: Easy to add custom validators and critics
4. **Performance**: Parallel processing and caching throughout
5. **Type Safety**: Full Pydantic integration for reliability
6. **Research-Backed**: Implementations of proven academic techniques

## 🔮 Future Architecture

Planned enhancements:
- **Multi-Agent Workflows**: Specialized agents for different tasks
- **Retrieval Integration**: RAG-based critics and validators
- **Streaming Support**: Real-time improvement feedback
- **Distributed Processing**: Scale across multiple machines
- **Advanced Caching**: Semantic similarity-based cache hits

This architecture enables Sifaka to provide guaranteed quality improvements while maintaining full transparency and extensibility.
