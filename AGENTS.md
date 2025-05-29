# PydanticAI Integration for Sifaka

## Overview

This document outlines the implementation of PydanticAI integration into Sifaka using a **hybrid Chain-Agent architecture** with composition over inheritance. This approach allows Sifaka's critic and validation system to provide feedback to PydanticAI agents while maintaining clean separation of concerns.

## Architecture

### Core Components

1. **PydanticAIChain** - Main orchestrator that composes Sifaka components with PydanticAI agents
2. **AgentModel** - Adapter that wraps PydanticAI agents to work with Sifaka's model interface
3. **CriticTool** - Converts Sifaka critics into PydanticAI tools for internal feedback
4. **ValidationTool** - Converts Sifaka validators into PydanticAI tools
5. **FeedbackOrchestrator** - Manages the flow of feedback between Sifaka and PydanticAI

### Design Principles

- **Composition over Inheritance**: PydanticAIChain composes Sifaka components rather than inheriting from Chain
- **Dual Feedback Paths**: Critics work both as traditional Sifaka critics AND as PydanticAI tools
- **Clean Separation**: Sifaka manages workflow orchestration, PydanticAI handles generation + tools
- **Backward Compatibility**: Existing Sifaka components work unchanged

## Implementation Plan

### Phase 1: Core Infrastructure

#### 1.1 PydanticAI Model Adapter

```python
# sifaka/models/pydantic_ai.py
from typing import Any, Optional
from pydantic_ai import Agent
from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought

class PydanticAIModel(Model):
    """Adapter to use PydanticAI agents as Sifaka models."""

    def __init__(self, agent: Agent, model_name: Optional[str] = None):
        self.agent = agent
        self.model_name = model_name or f"pydantic-ai-{agent.model}"

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text using PydanticAI agent."""
        result = self.agent.run_sync(prompt, **options)
        return result.output

    def generate_with_thought(self, thought: Thought, **options: Any) -> tuple[str, str]:
        """Generate text using Thought context."""
        # Convert thought context to PydanticAI format
        enhanced_prompt = self._build_prompt_from_thought(thought)
        result = self.agent.run_sync(enhanced_prompt, **options)
        return result.output, enhanced_prompt

    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Use PydanticAI's token counting if available
        return len(text.split()) * 1.3  # Rough approximation
```

#### 1.2 PydanticAI Chain

```python
# sifaka/agents/chain.py
from typing import List, Optional, Any
from pydantic_ai import Agent
from sifaka.core.thought import Thought
from sifaka.core.interfaces import Validator, Critic
from sifaka.storage.base import Storage
from sifaka.storage.memory import MemoryStorage

class PydanticAIChain:
    """Hybrid chain that orchestrates PydanticAI agents with Sifaka components."""

    def __init__(
        self,
        agent: Agent,
        storage: Optional[Storage] = None,
        validators: Optional[List[Validator]] = None,
        critics: Optional[List[Critic]] = None,
        max_improvement_iterations: int = 2,
        enable_critic_tools: bool = True,
        enable_validator_tools: bool = True,
    ):
        self.agent = agent
        self.storage = storage or MemoryStorage()
        self.validators = validators or []
        self.critics = critics or []
        self.max_improvement_iterations = max_improvement_iterations

        # Setup agent tools if enabled
        if enable_critic_tools:
            self._setup_critic_tools()
        if enable_validator_tools:
            self._setup_validator_tools()

    def run(self, prompt: str, **kwargs) -> Thought:
        """Execute the hybrid chain."""
        # Create initial thought
        thought = Thought(prompt=prompt)

        # Phase 1: Initial generation with PydanticAI
        thought = self._execute_agent_generation(thought, **kwargs)

        # Phase 2: Sifaka validation
        thought = self._execute_validation(thought)

        # Phase 3: Improvement loop if validation fails
        if not self._validation_passed(thought):
            thought = self._execute_improvement_loop(thought, **kwargs)

        # Save final result
        self.storage.save(thought)
        return thought
```

### Phase 2: Tool Integration

#### 2.1 Critic Tools

```python
# sifaka/agents/tools.py
from typing import Dict, Any
from pydantic_ai.tools import RunContext
from sifaka.core.interfaces import Critic
from sifaka.core.thought import Thought

class CriticTool:
    """Converts Sifaka critics into PydanticAI tools."""

    def __init__(self, critic: Critic):
        self.critic = critic

    def create_tool_function(self):
        """Create a PydanticAI tool function from this critic."""

        async def critique_text(ctx: RunContext, text: str) -> Dict[str, Any]:
            """Critique the provided text and return feedback."""
            # Create temporary thought for critic
            temp_thought = Thought(prompt="", text=text)

            # Run critic
            feedback = self.critic.critique(temp_thought)

            # Format for PydanticAI
            return {
                "feedback": feedback.get("feedback", ""),
                "issues": feedback.get("issues", []),
                "suggestions": feedback.get("suggestions", []),
                "confidence": feedback.get("confidence", 0.0),
                "should_improve": feedback.get("confidence", 0.0) < 0.7
            }

        # Set function metadata for PydanticAI
        critique_text.__name__ = f"critique_with_{self.critic.__class__.__name__.lower()}"
        critique_text.__doc__ = f"Get feedback from {self.critic.__class__.__name__}"

        return critique_text
```

#### 2.2 Validation Tools

```python
class ValidationTool:
    """Converts Sifaka validators into PydanticAI tools."""

    def __init__(self, validator: Validator):
        self.validator = validator

    def create_tool_function(self):
        """Create a PydanticAI tool function from this validator."""

        async def validate_text(ctx: RunContext, text: str) -> Dict[str, Any]:
            """Validate the provided text."""
            # Create temporary thought for validator
            temp_thought = Thought(prompt="", text=text)

            # Run validator
            result = self.validator.validate(temp_thought)

            return {
                "passed": result.passed,
                "score": result.score,
                "feedback": result.feedback,
                "suggestions": result.suggestions,
                "errors": result.errors
            }

        validate_text.__name__ = f"validate_with_{self.validator.__class__.__name__.lower()}"
        validate_text.__doc__ = f"Validate text using {self.validator.__class__.__name__}"

        return validate_text
```

### Phase 3: Feedback Integration

#### 3.1 Feedback Orchestrator

```python
# sifaka/agents/feedback.py
class FeedbackOrchestrator:
    """Manages feedback flow between Sifaka and PydanticAI."""

    def __init__(self, chain: PydanticAIChain):
        self.chain = chain

    def create_improvement_prompt(self, thought: Thought) -> str:
        """Create an improvement prompt based on Sifaka feedback."""
        base_prompt = f"Please improve the following text:\n\n{thought.text}\n\n"

        # Add validation feedback
        if thought.validation_results:
            base_prompt += "Validation Issues:\n"
            for name, result in thought.validation_results.items():
                if not result.passed:
                    base_prompt += f"- {name}: {result.feedback}\n"
                    if result.suggestions:
                        base_prompt += f"  Suggestions: {', '.join(result.suggestions)}\n"

        # Add critic feedback
        if thought.critic_feedback:
            base_prompt += "\nCritic Feedback:\n"
            for feedback in thought.critic_feedback:
                if feedback.confidence < 0.7:  # Only include low-confidence feedback
                    base_prompt += f"- {feedback.critic_name}: {feedback.feedback}\n"
                    if feedback.suggestions:
                        base_prompt += f"  Suggestions: {', '.join(feedback.suggestions)}\n"

        base_prompt += "\nPlease provide an improved version that addresses these issues."
        return base_prompt
```

### Phase 4: Integration Points

#### 4.1 Factory Functions

```python
# sifaka/agents/__init__.py
from typing import List, Optional
from pydantic_ai import Agent
from sifaka.core.interfaces import Validator, Critic

def create_pydantic_chain(
    agent: Agent,
    validators: Optional[List[Validator]] = None,
    critics: Optional[List[Critic]] = None,
    **kwargs
) -> PydanticAIChain:
    """Factory function to create a PydanticAI chain with Sifaka components."""
    return PydanticAIChain(
        agent=agent,
        validators=validators,
        critics=critics,
        **kwargs
    )

def create_agent_model(agent: Agent, **kwargs) -> PydanticAIModel:
    """Factory function to create a PydanticAI model adapter."""
    return PydanticAIModel(agent=agent, **kwargs)
```

## Usage Examples

### Basic Usage

```python
from pydantic_ai import Agent
from sifaka.agents import create_pydantic_chain
from sifaka.validators import LengthValidator
from sifaka.critics import ReflexionCritic
from sifaka.models import create_model

# Create PydanticAI agent
agent = Agent('openai:gpt-4o', system_prompt='You are a helpful writing assistant.')

# Create Sifaka components
validator = LengthValidator(min_length=100, max_length=500)
critic_model = create_model("openai:gpt-3.5-turbo")
critic = ReflexionCritic(model=critic_model)

# Create hybrid chain
chain = create_pydantic_chain(
    agent=agent,
    validators=[validator],
    critics=[critic]
)

# Run the chain
result = chain.run("Write a story about AI")
print(f"Generated: {result.text}")
print(f"Validation passed: {result.validation_passed}")
```

### Advanced Usage with Tools

```python
from pydantic_ai import Agent
from sifaka.agents import create_pydantic_chain
from sifaka.validators import LengthValidator, RegexValidator
from sifaka.critics import SelfRefineCritic, ConstitutionalCritic

# Create agent with tools
agent = Agent('anthropic:claude-3-sonnet')

@agent.tool
async def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation here
    return f"Search results for: {query}"

# Create Sifaka components
validators = [
    LengthValidator(min_length=200, max_length=1000),
    RegexValidator(pattern=r'\b\d{4}\b', description="Must contain a year")
]

critics = [
    SelfRefineCritic(model=create_model("openai:gpt-4")),
    ConstitutionalCritic(
        model=create_model("anthropic:claude-3-haiku"),
        principles=["Be factual", "Be helpful", "Be concise"]
    )
]

# Create chain with both PydanticAI tools and Sifaka feedback
chain = create_pydantic_chain(
    agent=agent,
    validators=validators,
    critics=critics,
    enable_critic_tools=True,  # Critics available as agent tools
    enable_validator_tools=True  # Validators available as agent tools
)

result = chain.run("Research and write about the history of AI, including key milestones")
```

## Benefits

1. **Best of Both Worlds**: Combines PydanticAI's agent capabilities with Sifaka's evaluation framework
2. **Flexible Feedback**: Critics and validators work both externally (Sifaka) and internally (PydanticAI tools)
3. **Clean Architecture**: Composition-based design is maintainable and extensible
4. **Backward Compatibility**: Existing Sifaka components work unchanged
5. **Type Safety**: Leverages both frameworks' type systems

## Implementation Timeline

- **Week 1**: Core infrastructure (PydanticAIModel, PydanticAIChain)
- **Week 2**: Tool integration (CriticTool, ValidationTool)
- **Week 3**: Feedback orchestration and improvement loops
- **Week 4**: Testing, documentation, and examples

## Future Enhancements

- Stream support for real-time feedback
- Multi-agent workflows
- Advanced retrieval integration
- Performance optimizations
- Additional tool types (classifiers, retrievers)
```
