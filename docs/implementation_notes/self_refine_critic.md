# Implementation Notes: Self-Refine Critic

This document provides implementation details and notes for the Self-Refine Critic in the Sifaka project.

## Overview

The Self-Refine Critic implements the Self-Refine approach from the paper [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651). It enables language models to iteratively critique and revise their own outputs without requiring external feedback or multiple models.

## Implementation Details

### State Management

The Self-Refine Critic follows the standard state management pattern used in other Sifaka critics:

- Uses a `CriticState` object to store all mutable state
- Stores configuration values in the state's cache dictionary
- Accesses state through direct state access

```python
# Initialize state
self._state = CriticState()

# Store components in state
self._state.model = llm_provider
self._state.cache = {
    "max_iterations": config.max_iterations,
    "critique_prompt_template": config.critique_prompt_template,
    "revision_prompt_template": config.revision_prompt_template,
    "system_prompt": config.system_prompt,
    "temperature": config.temperature,
    "max_tokens": config.max_tokens,
}
self._state.initialized = True
```

### Configuration

The Self-Refine Critic uses a dedicated configuration class that extends `PromptCriticConfig`:

```python
class SelfRefineCriticConfig(PromptCriticConfig):
    max_iterations: int = Field(
        default=3, description="Maximum number of refinement iterations", gt=0
    )
    critique_prompt_template: str = Field(
        default=(
            "Critique the following response and suggest improvements:\n\n"
            "Task:\n{task}\n\n"
            "Response:\n{response}\n\n"
            "Critique:"
        ),
        description="Template for critique prompts",
    )
    revision_prompt_template: str = Field(
        default=(
            "Revise the original response using the critique:\n\n"
            "Task:\n{task}\n\n"
            "Original Response:\n{response}\n\n"
            "Critique:\n{critique}\n\n"
            "Revised Response:"
        ),
        description="Template for revision prompts",
    )
```

### Core Algorithm

The core algorithm for the Self-Refine Critic is implemented in the `improve` method:

1. Generate a critique of the current text
2. If the critique indicates no issues, return the current text
3. Generate a revised version of the text based on the critique
4. If the revised text is the same as the current text, return the current text
5. Update the current text to the revised text
6. Repeat steps 1-5 for a specified number of iterations or until no further improvements are needed

```python
# Perform iterative refinement
for _ in range(max_iterations):
    # Step 1: Critique the current output
    critique = self._state.model.generate(critique_prompt)
    
    # Check if critique indicates no issues
    if any(phrase in critique.lower() for phrase in no_issues_phrases):
        return current_output
    
    # Step 2: Revise using the critique
    revised_output = self._state.model.generate(revision_prompt)
    
    # Check if there's no improvement
    if revised_output == current_output:
        return current_output
    
    # Update current output
    current_output = revised_output

return current_output
```

### Factory Function

The Self-Refine Critic provides a factory function for easy creation:

```python
def create_self_refine_critic(
    llm_provider: Any,
    name: str = "self_refine_critic",
    description: str = "Improves text through iterative self-critique and revision",
    max_iterations: int = 3,
    system_prompt: str = "You are an expert at critiquing and revising content.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    critique_prompt_template: Optional[str] = None,
    revision_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], SelfRefineCriticConfig]] = None,
    **kwargs: Any,
) -> SelfRefineCritic:
    # Implementation details...
```

## Integration with Sifaka

The Self-Refine Critic is integrated with the Sifaka project in the following ways:

1. Added to the `critics` module with proper imports and exports
2. Added to the `__all__` list in `critics/__init__.py`
3. Added a default configuration `DEFAULT_SELF_REFINE_CONFIG`
4. Provided comprehensive tests in `tests/critics/test_self_refine.py`
5. Provided an example in `examples/self_refine_critic_example.py`

## Testing

The Self-Refine Critic includes comprehensive tests that verify:

1. Initialization with different configurations
2. Validation of text
3. Critique generation
4. Text improvement through iterative refinement
5. Factory function behavior

## Future Improvements

Potential future improvements for the Self-Refine Critic include:

1. Adding support for more sophisticated stopping criteria
2. Implementing a more robust parsing of critiques
3. Adding support for multi-step refinement with different prompts at each step
4. Implementing a more sophisticated scoring mechanism for critiques
5. Adding support for tracking the history of refinements

## References

- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- [Sifaka Critics Documentation](../components/critics.md)
