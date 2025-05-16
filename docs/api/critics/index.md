# Critics

This page documents the critics framework in Sifaka.

## Overview

Critics are LLM-based components that can both validate and improve text. They use language models to evaluate text quality and make improvements based on specific criteria.

The critics framework includes:

- A base `Critic` class that serves as a foundation for all critics
- Specific critic implementations for different aspects of text quality
- Convenience functions for creating critics

## Basic Usage

```python
from sifaka.validators import clarity, factual_accuracy

# Create a clarity critic
clarity_critic = clarity(model="openai:gpt-4")

# Validate text
validation_result = clarity_critic.validate("This is a text that could be clearer.")
print(f"Validation passed: {validation_result.passed}")
print(f"Message: {validation_result.message}")

# Improve text
improved_text, improvement_result = clarity_critic.improve("This is a text that could be clearer.")
print(f"Improved text: {improved_text}")
print(f"Changes made: {improvement_result.changes_made}")
print(f"Message: {improvement_result.message}")

# Create a factual accuracy critic
factual_critic = factual_accuracy(model="openai:gpt-4")

# Validate text
validation_result = factual_critic.validate("The Earth is flat and has four corners.")
print(f"Validation passed: {validation_result.passed}")
print(f"Message: {validation_result.message}")

# Improve text
improved_text, improvement_result = factual_critic.improve("The Earth is flat and has four corners.")
print(f"Improved text: {improved_text}")
print(f"Changes made: {improvement_result.changes_made}")
print(f"Message: {improvement_result.message}")
```

## API Reference

### Base Critic Class

```python
Critic(
    model: Union[str, Model],
    name: Optional[str] = None,
    **model_options: Any
)
```

Base class for LLM-based critics.

**Parameters:**
- `model`: Model to use for validation and improvement. Can be a model instance or a string in the format "provider:model_name".
- `name`: Name of the critic. If not provided, will be derived from the class name.
- `**model_options`: Additional options to pass to the model.

#### Methods

##### `validate`

```python
validate(self, text: str) -> ValidationResult
```

Validate text using the critic.

**Parameters:**
- `text`: Text to validate.

**Returns:**
- ValidationResult indicating whether the text passed validation.

##### `improve`

```python
improve(self, text: str) -> Tuple[str, ImprovementResult]
```

Improve text using the critic.

**Parameters:**
- `text`: Text to improve.

**Returns:**
- Tuple of (improved_text, ImprovementResult).

### Specific Critics

#### ClarityAndCoherenceCritic

```python
ClarityAndCoherenceCritic(
    model: Union[str, Model],
    name: Optional[str] = None,
    **model_options: Any
)
```

Critic that evaluates and improves text clarity and coherence.

**Parameters:**
- `model`: Model to use for validation and improvement.
- `name`: Name of the critic.
- `**model_options`: Additional options to pass to the model.

#### FactualAccuracyCritic

```python
FactualAccuracyCritic(
    model: Union[str, Model],
    name: Optional[str] = None,
    **model_options: Any
)
```

Critic that evaluates and improves factual accuracy of text.

**Parameters:**
- `model`: Model to use for validation and improvement.
- `name`: Name of the critic.
- `**model_options`: Additional options to pass to the model.

### Convenience Functions

#### `clarity`

```python
clarity(model: str = "openai:gpt-3.5-turbo", **options)
```

Create a clarity and coherence critic.

**Parameters:**
- `model`: Model to use for validation and improvement.
- `**options`: Additional options to pass to the model.

**Returns:**
- A ClarityAndCoherenceCritic instance.

#### `factual_accuracy`

```python
factual_accuracy(model: str = "openai:gpt-3.5-turbo", **options)
```

Create a factual accuracy critic.

**Parameters:**
- `model`: Model to use for validation and improvement.
- `**options`: Additional options to pass to the model.

**Returns:**
- A FactualAccuracyCritic instance.

## Examples

### Using Critics with Chain

Critics can be used both as validators and improvers in a Chain:

```python
from sifaka import Chain
from sifaka.validators import clarity, factual_accuracy, length

result = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short explanation of quantum computing.")
    .validate_with(length(min_words=50, max_words=200))
    .validate_with(factual_accuracy())  # Used as a validator
    .improve_with(clarity())  # Used as an improver
    .run())

print(f"Result passed validation: {result.passed}")
print(result.text)

# Print validation results
for i, validation_result in enumerate(result.validation_results):
    print(f"Validation {i+1}: {validation_result.message}")

# Print improvement results
for i, improvement_result in enumerate(result.improvement_results):
    print(f"Improvement {i+1}: {improvement_result.message}")
```

### Creating a Custom Critic

You can create custom critics by extending the Critic base class:

```python
from typing import Tuple
import re
from sifaka.validators import Critic
from sifaka.results import ValidationResult, ImprovementResult

class SimplificationCritic(Critic):
    """Critic that simplifies complex text."""
    
    def validate(self, text: str) -> ValidationResult:
        """Validate text simplicity."""
        prompt = f"""
        Evaluate the simplicity of the following text.
        Consider factors such as:
        - Use of simple language
        - Short sentences
        - Clear explanations without jargon
        
        Text to evaluate:
        ---
        {text}
        ---
        
        First, provide a score from 1-10 where:
        1-3: Very complex
        4-6: Moderately complex
        7-10: Very simple
        
        Then, explain your reasoning in detail.
        
        Format your response as:
        SCORE: [your score]
        REASONING: [your detailed explanation]
        PASSED: [YES if score >= 7, NO if score < 7]
        """
        
        response = self.model.generate(prompt, **self.model_options)
        
        # Extract score and passed status
        score_match = re.search(r"SCORE:\s*(\d+)", response)
        passed_match = re.search(r"PASSED:\s*(YES|NO)", response)
        reasoning_match = re.search(r"REASONING:\s*(.*?)(?=PASSED:|$)", response, re.DOTALL)
        
        if not score_match or not passed_match:
            return ValidationResult(
                passed=False,
                message="Could not determine validation result from critic response",
                details={"response": response}
            )
        
        score = int(score_match.group(1))
        passed = passed_match.group(1) == "YES"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        return ValidationResult(
            passed=passed,
            message=f"Simplicity score: {score}/10. {reasoning}",
            details={
                "score": score,
                "reasoning": reasoning,
                "response": response
            }
        )
    
    def improve(self, text: str) -> Tuple[str, ImprovementResult]:
        """Simplify text."""
        prompt = f"""
        Simplify the following text to make it more accessible.
        Focus on:
        - Using simpler language
        - Shortening sentences
        - Removing jargon and technical terms
        - Explaining complex concepts clearly
        
        Text to simplify:
        ---
        {text}
        ---
        
        First, rewrite the text to make it simpler.
        Then, explain the changes you made.
        
        Format your response as:
        SIMPLIFIED TEXT:
        [your simplified version of the text]
        
        EXPLANATION:
        [explanation of changes made]
        """
        
        response = self.model.generate(prompt, **self.model_options)
        
        # Extract simplified text and explanation
        text_match = re.search(r"SIMPLIFIED TEXT:\s*(.*?)(?=EXPLANATION:|$)", response, re.DOTALL)
        explanation_match = re.search(r"EXPLANATION:\s*(.*?)$", response, re.DOTALL)
        
        if not text_match:
            return text, ImprovementResult(
                original_text=text,
                improved_text=text,
                changes_made=False,
                message="Could not extract simplified text from critic response",
                details={"response": response}
            )
        
        simplified_text = text_match.group(1).strip()
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        # Check if the text was actually changed
        changes_made = simplified_text != text
        
        return simplified_text, ImprovementResult(
            original_text=text,
            improved_text=simplified_text,
            changes_made=changes_made,
            message=f"Simplification improvements: {explanation}",
            details={
                "explanation": explanation,
                "response": response
            }
        )

# Create a convenience function
def simplify(model: str = "openai:gpt-3.5-turbo", **options):
    """Create a simplification critic."""
    return SimplificationCritic(model, **options)
```
