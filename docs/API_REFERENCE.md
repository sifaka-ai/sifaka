# Sifaka API Reference

This document provides comprehensive API documentation for Sifaka's core components, models, and utilities.

## Architecture Overview

Sifaka is built on PydanticAI with a simple graph-based workflow:

- **SifakaEngine**: Main orchestration engine using PydanticAI graphs
- **SifakaThought**: Core state container with complete audit trails
- **SifakaDependencies**: Dependency injection system for agents, validators, and critics
- **Graph Nodes**: Generate, Validate, and Critique operations
- **Storage**: Production-ready backends (memory, file, Redis MCP, PostgreSQL, hybrid)

### Key Features

- **Complete Observability**: Full audit trails with conversation history
- **Configurable Weighting**: Default 60/40 split between validation and critic feedback
- **Research-Backed**: Direct implementations of academic papers
- **Type Safety**: Full Pydantic integration throughout
- **Active Development**: Built on PydanticAI (both projects are in active development)

## Core API Components

### SifakaEngine

Main orchestration engine for processing thoughts through the complete workflow.

```python
from sifaka import SifakaEngine
from sifaka.graph import SifakaDependencies

# Create with default configuration
engine = SifakaEngine()

# Create with custom dependencies
dependencies = SifakaDependencies(
    generator="openai:gpt-4",
    validation_weight=0.6,  # 60% weight for validation feedback
    critic_weight=0.4,      # 40% weight for critic feedback
)
engine = SifakaEngine(dependencies=dependencies)

# Process a thought
thought = await engine.think("Your prompt here", max_iterations=3)
```

### SifakaThought

Core state container that tracks the complete audit trail of a thought's evolution.

```python
from sifaka.core.thought import SifakaThought

# Access thought properties
print(f"Final text: {thought.final_text}")
print(f"Iterations: {thought.iteration}")
print(f"Validation passed: {thought.validation_passed()}")

# Access audit trail
for generation in thought.generations:
    print(f"Iteration {generation.iteration}: {generation.text}")
    print(f"Conversation history: {len(generation.conversation_history)} messages")

# Access conversation messages
latest_messages = thought.get_latest_conversation_messages()
iteration_messages = thought.get_conversation_messages_for_iteration(1)
```

### SifakaDependencies

Dependency injection system for configuring agents, validators, and critics.

```python
from sifaka.graph import SifakaDependencies
from sifaka.validators import LengthValidator
from sifaka.critics import ReflexionCritic

# Create custom configuration
dependencies = SifakaDependencies(
    generator="openai:gpt-4",
    validators=[LengthValidator(min_length=50, max_length=500)],
    critics={
        "reflexion": "openai:gpt-3.5-turbo",
        "constitutional": "anthropic:claude-3-haiku"
    },
    # Feedback weighting configuration
    validation_weight=0.6,  # Default: 60% weight for validation
    critic_weight=0.4,      # Default: 40% weight for critics
    # Critic behavior
    always_apply_critics=False,  # Only apply when validation fails
    never_apply_critics=False,   # Allow critics to run
    always_include_validation_results=True  # Include validation in context
)
```

## Storage and Persistence

Sifaka provides multiple storage backends for production use with automatic failover and hybrid configurations.

### Storage Backends

```python
from sifaka.storage import (
    MemoryPersistence,
    SifakaFilePersistence,
    RedisPersistence,
    PostgreSQLPersistence,
    FlexibleHybridPersistence,
    BackendConfig,
    BackendRole
)

# Memory storage (default, no persistence)
memory_storage = MemoryPersistence(key_prefix="sifaka")

# File storage with indexing and search
file_storage = SifakaFilePersistence(
    storage_dir="./thoughts",
    auto_backup=True,
    max_backup_count=10
)

# Redis storage via MCP (production-ready)
from pydantic_ai.mcp import MCPServerStdio
redis_mcp = MCPServerStdio("redis-mcp-server")
redis_storage = RedisPersistence(
    mcp_server=redis_mcp,
    key_prefix="sifaka",
    ttl_seconds=3600  # Optional TTL
)

# PostgreSQL storage (enterprise-grade)
postgres_storage = PostgreSQLPersistence(
    connection_string="postgresql://user:pass@localhost/sifaka",
    key_prefix="sifaka"
)
```

### Hybrid Storage

Combine multiple backends with automatic failover:

```python
# Create hybrid storage with cache → primary → backup → search
hybrid_storage = FlexibleHybridPersistence([
    BackendConfig(
        backend=MemoryPersistence(),
        role=BackendRole.CACHE,
        priority=0,  # Highest priority
        read_enabled=True,
        write_enabled=True
    ),
    BackendConfig(
        backend=redis_storage,
        role=BackendRole.PRIMARY,
        priority=1
    ),
    BackendConfig(
        backend=file_storage,
        role=BackendRole.BACKUP,
        priority=2
    ),
    BackendConfig(
        backend=postgres_storage,
        role=BackendRole.SEARCH,
        priority=3,
        read_enabled=True,
        write_enabled=False  # Read-only for search
    )
])

# Use with engine
engine = SifakaEngine(persistence=hybrid_storage)
```

### Storage Operations

```python
# Store and retrieve thoughts
thought = await engine.think("Your prompt")
await storage.store_thought(thought)

# Retrieve by ID
retrieved = await storage.retrieve_thought(thought.id)

# List thoughts with filtering
thoughts = await storage.list_thoughts(
    conversation_id="conv_123",
    limit=10
)

# Search thoughts (PostgreSQL backend)
if hasattr(storage, 'search_thoughts_by_text'):
    results = await storage.search_thoughts_by_text("renewable energy")
```

## Utilities

### Thought Inspector

Utilities for analyzing and debugging thought processes with conversation history.

```python
from sifaka.utils.thought_inspector import (
    print_iteration_details,
    print_all_iterations,
    get_latest_conversation_messages,
    get_conversation_messages_for_iteration,
    print_conversation_messages,
    print_critic_summary,
    print_validation_summary,
    get_thought_overview,
)

# Print detailed information about the latest iteration
print_iteration_details(thought)

# Print all iterations with full audit trail
print_all_iterations(thought)

# Get conversation messages (requests and responses)
latest_messages = get_latest_conversation_messages(thought)
iteration_messages = get_conversation_messages_for_iteration(thought, iteration=1)

# Print conversation history
print_conversation_messages(thought, full_messages=True)

# Print summaries
print_critic_summary(thought)
print_validation_summary(thought)

# Get overview statistics
overview = get_thought_overview(thought)
print(f"Total iterations: {overview['total_iterations']}")
print(f"Validation success rate: {overview['validation_success_rate']}")
```

### Configuration

Global configuration management for Sifaka.

```python
from sifaka.utils.config import SifakaConfig

# Create configuration
config = SifakaConfig(
    default_model="openai:gpt-4",
    max_iterations=3,
    enable_critics=True,
    timeout_seconds=30.0,
    storage_backend="memory"
)

# Access configuration values
print(f"Default model: {config.default_model}")
print(f"Max iterations: {config.max_iterations}")
```

## Pydantic Models

### SeverityLevel

An enumeration defining severity levels for violations and suggestions.

```python
from sifaka.models.critic_results import SeverityLevel

# Available levels
SeverityLevel.CRITICAL  # "critical"
SeverityLevel.HIGH      # "high"
SeverityLevel.MEDIUM    # "medium"
SeverityLevel.LOW       # "low"
SeverityLevel.INFO      # "info"
```

### ConfidenceScore

Detailed confidence information with breakdowns by category.

```python
from sifaka.models.critic_results import ConfidenceScore

confidence = ConfidenceScore(
    overall=0.85,                    # Required: Overall confidence (0.0-1.0)
    content_quality=0.9,             # Optional: Content quality confidence
    grammar_accuracy=0.8,            # Optional: Grammar accuracy confidence
    factual_accuracy=0.7,            # Optional: Factual accuracy confidence
    coherence=0.95,                  # Optional: Coherence confidence
    calculation_method="weighted",    # Optional: How confidence was calculated
    factors_considered=["grammar", "content"],  # Factors in calculation
    uncertainty_sources=["ambiguous_context"],  # Sources of uncertainty
    metadata={"model_version": "v1.0"}  # Additional metadata
)
```

**Fields:**
- `overall` (float, required): Overall confidence score between 0.0 and 1.0
- `content_quality` (float, optional): Confidence in content quality assessment
- `grammar_accuracy` (float, optional): Confidence in grammar assessment
- `factual_accuracy` (float, optional): Confidence in factual accuracy
- `coherence` (float, optional): Confidence in coherence assessment
- `calculation_method` (str, optional): Method used to calculate confidence
- `factors_considered` (List[str]): Factors considered in calculation
- `uncertainty_sources` (List[str]): Sources that lower confidence
- `metadata` (Dict[str, Any]): Additional confidence-related metadata

### ViolationReport

Structured report of a specific violation or issue found in the text.

```python
from sifaka.models.critic_results import ViolationReport, SeverityLevel

violation = ViolationReport(
    violation_type="grammar_error",           # Required: Type of violation
    description="Subject-verb disagreement", # Required: Description
    severity=SeverityLevel.HIGH,             # Required: Severity level
    location="paragraph 2, sentence 3",     # Optional: Location in text
    start_position=45,                       # Optional: Start character position
    end_position=52,                         # Optional: End character position
    rule_violated="subject_verb_agreement",  # Optional: Specific rule
    evidence="The cats was sleeping",        # Optional: Evidence text
    suggested_fix="Change 'was' to 'were'", # Optional: Suggested fix
    confidence=0.95,                         # Confidence in this violation
    metadata={"rule_id": "SVA001"}           # Additional metadata
)
```

**Fields:**
- `violation_type` (str, required): Type or category of the violation
- `description` (str, required): Detailed description of the violation
- `severity` (SeverityLevel, required): Severity level of the violation
- `location` (str, optional): Location in text where violation occurs
- `start_position` (int, optional): Character position where violation starts
- `end_position` (int, optional): Character position where violation ends
- `rule_violated` (str, optional): Specific rule or principle violated
- `evidence` (str, optional): Text evidence supporting the violation
- `suggested_fix` (str, optional): Suggested fix for this violation
- `confidence` (float): Confidence in this violation report (0.0-1.0)
- `metadata` (Dict[str, Any]): Additional violation-specific metadata

### ImprovementSuggestion

Structured improvement suggestion with implementation details.

```python
from sifaka.models.critic_results import ImprovementSuggestion, SeverityLevel

suggestion = ImprovementSuggestion(
    suggestion="Add more specific examples",     # Required: The suggestion
    category="clarity",                          # Required: Category
    priority=SeverityLevel.HIGH,                # Priority level
    rationale="Examples help understanding",     # Why this is suggested
    implementation="Insert 2-3 examples",       # How to implement
    example="For instance, when discussing...", # Example of improvement
    applies_to="paragraph 3",                   # Where it applies
    start_position=120,                         # Start position
    end_position=180,                           # End position
    expected_impact="Better comprehension",      # Expected impact
    confidence=0.85,                            # Confidence in suggestion
    depends_on=["fix_grammar_first"],           # Dependencies
    conflicts_with=["remove_examples"],         # Conflicts
    metadata={"suggestion_id": "CLAR001"}       # Additional metadata
)
```

**Fields:**
- `suggestion` (str, required): The improvement suggestion text
- `category` (str, required): Category of improvement (e.g., 'grammar', 'clarity')
- `priority` (SeverityLevel): Priority level for this suggestion
- `rationale` (str, optional): Explanation of why this improvement is suggested
- `implementation` (str, optional): Specific steps to implement this suggestion
- `example` (str, optional): Example of how the improvement would look
- `applies_to` (str, optional): Part of text this suggestion applies to
- `start_position` (int, optional): Character position where suggestion applies
- `end_position` (int, optional): Character position where suggestion applies
- `expected_impact` (str, optional): Expected impact of implementing this suggestion
- `confidence` (float): Confidence in this suggestion (0.0-1.0)
- `depends_on` (List[str]): Other suggestions this one depends on
- `conflicts_with` (List[str]): Other suggestions this one conflicts with
- `metadata` (Dict[str, Any]): Additional suggestion-specific metadata

### CritiqueFeedback

Main structured critique feedback model containing all feedback components.

```python
from sifaka.models.critic_results import (
    CritiqueFeedback, ConfidenceScore, ViolationReport, ImprovementSuggestion
)

feedback = CritiqueFeedback(
    message="The text has several issues",      # Required: Main message
    needs_improvement=True,                     # Required: Whether improvement needed
    violations=[violation1, violation2],       # List of violations found
    suggestions=[suggestion1, suggestion2],    # List of improvement suggestions
    confidence=confidence_score,                # Required: Confidence information
    critic_name="ConstitutionalCritic",        # Required: Name of critic
    critic_version="v1.0",                     # Optional: Critic version
    processing_time_ms=150.5,                  # Optional: Processing time
    metadata={"rules_applied": ["grammar"]}    # Additional metadata
)
```

**Fields:**
- `message` (str, required): Main critique message or summary
- `needs_improvement` (bool, required): Whether the text needs improvement
- `violations` (List[ViolationReport]): Specific violations or issues found
- `suggestions` (List[ImprovementSuggestion]): Specific improvement suggestions
- `confidence` (ConfidenceScore, required): Detailed confidence information
- `critic_name` (str, required): Name of the critic that generated this feedback
- `critic_version` (str, optional): Version of the critic
- `processing_time_ms` (float, optional): Time taken to generate feedback
- `timestamp` (datetime): When this feedback was generated (auto-set)
- `metadata` (Dict[str, Any]): Additional critic-specific metadata

### CriticResult

Main result container for critic operations with validation and metadata.

```python
from sifaka.models.critic_results import CriticResult, CritiqueFeedback

result = CriticResult(
    feedback=critique_feedback,                 # Required: Structured feedback
    operation_type="critique",                  # Required: Type of operation
    success=True,                              # Whether operation succeeded
    total_processing_time_ms=245.7,           # Required: Total processing time
    input_text_length=150,                    # Required: Input text length
    model_calls=2,                            # Number of model API calls
    tokens_used=1500,                         # Total tokens used
    error_message=None,                       # Error message if failed
    error_type=None,                          # Type of error
    input_hash="abc123",                      # Hash of input text
    validation_context={"min_length": 50},    # Validation context
    metadata={"version": "v1.0"}              # Additional metadata
)
```

**Fields:**
- `feedback` (CritiqueFeedback, required): Structured critique feedback
- `operation_type` (str, required): Type of operation (e.g., 'critique', 'improve')
- `success` (bool): Whether the operation completed successfully
- `total_processing_time_ms` (float, required): Total time for operation in milliseconds
- `input_text_length` (int, required): Length of input text in characters
- `model_calls` (int): Number of model API calls made
- `tokens_used` (int, optional): Total tokens used (if available)
- `error_message` (str, optional): Error message if operation failed
- `error_type` (str, optional): Type of error that occurred
- `input_hash` (str, optional): Hash of input text for caching/deduplication
- `validation_context` (Dict[str, Any], optional): Validation context that influenced critique
- `metadata` (Dict[str, Any]): Additional result metadata
- `created_at` (datetime): When this result was created (auto-set)

## Model Validation

All models include comprehensive validation:

### Consistency Validation

The `CriticResult` model includes a validator that ensures internal consistency:

```python
# This will raise a ValidationError
CriticResult(
    feedback=CritiqueFeedback(
        message="All good",
        needs_improvement=False,  # Inconsistent with success=False
        confidence=confidence,
        critic_name="TestCritic"
    ),
    operation_type="critique",
    success=False,  # Failed operations should indicate improvement needed
    total_processing_time_ms=100.0,
    input_text_length=50
)
```

### Range Validation

Confidence scores and positions are validated to be within valid ranges:

```python
# This will raise a ValidationError
ConfidenceScore(overall=1.5)  # Must be between 0.0 and 1.0

# This will raise a ValidationError
ViolationReport(
    violation_type="test",
    description="test",
    severity=SeverityLevel.LOW,
    start_position=-1  # Must be >= 0
)
```

## Usage Examples

### Basic Critic Usage

```python
import asyncio
from sifaka.critics import ConstitutionalCritic
from sifaka.core.thought import Thought
from datetime import datetime

async def main():
    critic = ConstitutionalCritic(model_name="openai:gpt-4o-mini")

    thought = Thought(
        prompt="Write about AI safety",
        text="AI is completely safe and needs no oversight.",
        timestamp=datetime.now(),
        id="test_001"
    )

    result = await critic.critique(thought)

    print(f"Success: {result.success}")
    print(f"Message: {result.feedback.message}")
    print(f"Needs improvement: {result.feedback.needs_improvement}")
    print(f"Violations: {len(result.feedback.violations)}")
    print(f"Suggestions: {len(result.feedback.suggestions)}")

asyncio.run(main())
```

### Analyzing Results

```python
def analyze_critic_result(result: CriticResult):
    """Analyze a critic result and print detailed information."""
    print(f"=== Critic Result Analysis ===")
    print(f"Operation: {result.operation_type}")
    print(f"Success: {result.success}")
    print(f"Processing time: {result.total_processing_time_ms}ms")
    print(f"Input length: {result.input_text_length} characters")

    feedback = result.feedback
    print(f"\n=== Feedback ===")
    print(f"Critic: {feedback.critic_name}")
    print(f"Message: {feedback.message}")
    print(f"Needs improvement: {feedback.needs_improvement}")
    print(f"Overall confidence: {feedback.confidence.overall}")

    if feedback.violations:
        print(f"\n=== Violations ({len(feedback.violations)}) ===")
        for i, violation in enumerate(feedback.violations):
            print(f"{i+1}. {violation.violation_type}: {violation.description}")
            print(f"   Severity: {violation.severity}")
            if violation.suggested_fix:
                print(f"   Fix: {violation.suggested_fix}")

    if feedback.suggestions:
        print(f"\n=== Suggestions ({len(feedback.suggestions)}) ===")
        for i, suggestion in enumerate(feedback.suggestions):
            print(f"{i+1}. {suggestion.category}: {suggestion.suggestion}")
            print(f"   Priority: {suggestion.priority}")
            if suggestion.rationale:
                print(f"   Rationale: {suggestion.rationale}")
```

### Serialization

```python
# Serialize to JSON
result_dict = result.model_dump()
import json
json_str = json.dumps(result_dict, default=str)

# Deserialize from JSON
result_data = json.loads(json_str)
restored_result = CriticResult.model_validate(result_data)
```

## Key Changes in v0.4.0

Sifaka v0.4.0 represents a complete rewrite built on PydanticAI:

### Architecture Changes
- **Complete PydanticAI Integration**: Built on PydanticAI graphs with full type safety
- **Simplified Workflow**: Streamlined Generate → Validate → Critique process
- **Enhanced Observability**: Complete audit trails with conversation history
- **Configurable Weighting**: Default 60/40 split between validation and critic feedback

### API Changes
```python
# v0.4.0 - PydanticAI-native approach
from sifaka import SifakaEngine
from sifaka.graph import SifakaDependencies

dependencies = SifakaDependencies(
    generator="openai:gpt-4",
    validation_weight=0.6,  # New: configurable weighting
    critic_weight=0.4
)
engine = SifakaEngine(dependencies=dependencies)
thought = await engine.think("Your prompt")

# Access conversation history (new feature)
messages = thought.get_latest_conversation_messages()
```

### Conversation History
- **Complete Message Tracking**: All PydanticAI conversation messages preserved
- **Iteration-Specific Access**: Get messages for specific iterations
- **Rich Debugging**: Full request/response history for analysis

See the [Migration Guide](MIGRATION.md) for complete migration instructions.
