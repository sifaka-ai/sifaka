# Using Critics in Sifaka

This tutorial will guide you through using critics in Sifaka to validate and improve text.

## What are Critics?

Critics are LLM-based components that can both validate and improve text. They use language models to evaluate text quality and make improvements based on specific criteria.

Unlike simple rule-based validators, critics can perform complex evaluations that require understanding the content and context of the text.

## Available Critics

Sifaka includes several built-in critics:

- **Clarity and Coherence**: Evaluates and improves the clarity and coherence of text
- **Factual Accuracy**: Evaluates and improves the factual accuracy of text

## Basic Usage

Let's start with a simple example of using a critic to validate text:

```python
import sifaka
from sifaka.validators import clarity

# Create a clarity critic
critic = clarity(model="openai:gpt-4")

# Validate text
text = "This text is somewhat unclear and could benefit from better organization and more precise language."
result = critic.validate(text)

print(f"Validation passed: {result.passed}")
print(f"Message: {result.message}")
print(f"Score: {result.details.get('score')}")
```

This example:
1. Creates a clarity critic using the OpenAI GPT-4 model
2. Validates a piece of text
3. Prints the validation result, including whether it passed, the message, and the clarity score

## Improving Text

Now, let's use a critic to improve text:

```python
import sifaka
from sifaka.validators import clarity

# Create a clarity critic
critic = clarity(model="openai:gpt-4")

# Improve text
text = "This text is somewhat unclear and could benefit from better organization and more precise language."
improved_text, result = critic.improve(text)

print("Original text:")
print(text)
print("\nImproved text:")
print(improved_text)
print(f"\nChanges made: {result.changes_made}")
print(f"Message: {result.message}")
```

This example:
1. Creates a clarity critic
2. Improves a piece of text
3. Prints the original text, improved text, whether changes were made, and the improvement message

## Using Critics in Chains

Critics can be used both as validators and improvers in a Chain:

```python
import sifaka
from sifaka.validators import clarity, factual_accuracy, length

# Create a chain with critics
result = (sifaka.Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short explanation of quantum computing.")
    .validate_with(length(min_words=50, max_words=200))
    .validate_with(factual_accuracy())  # Used as a validator
    .improve_with(clarity())  # Used as an improver
    .run())

print(f"Result passed validation: {result.passed}")
print(result.text)

# Print validation results
print("\nValidation Results:")
for i, validation_result in enumerate(result.validation_results):
    print(f"  {i+1}. Passed: {validation_result.passed}")
    print(f"     Message: {validation_result.message}")

# Print improvement results
print("\nImprovement Results:")
for i, improvement_result in enumerate(result.improvement_results):
    print(f"  {i+1}. Changes Made: {improvement_result.changes_made}")
    print(f"     Message: {improvement_result.message}")
```

This example:
1. Creates a chain with the OpenAI GPT-4 model
2. Sets the prompt to "Write a short explanation of quantum computing."
3. Adds a length validator
4. Adds a factual accuracy critic as a validator
5. Adds a clarity critic as an improver
6. Runs the chain
7. Prints the result, including validation and improvement details

## Configuring Critics

You can configure critics with additional options:

```python
import sifaka
from sifaka.validators import clarity

# Create a clarity critic with options
critic = clarity(
    model="openai:gpt-4",
    temperature=0.3,  # Lower temperature for more consistent results
    max_tokens=1000,  # Allow more tokens for detailed analysis
    system_message="You are an expert editor focused on clarity and coherence."
)

# Validate text
text = "This text is somewhat unclear and could benefit from better organization and more precise language."
result = critic.validate(text)

print(f"Validation passed: {result.passed}")
print(f"Message: {result.message}")
```

This example:
1. Creates a clarity critic with additional options
2. Validates a piece of text
3. Prints the validation result

## Creating a Custom Critic

You can create custom critics by extending the Critic base class:

```python
import re
from typing import Tuple
from sifaka.validators import Critic
from sifaka.results import ValidationResult, ImprovementResult

class TechnicalAccuracyCritic(Critic):
    """Critic that evaluates and improves technical accuracy of text."""
    
    def validate(self, text: str) -> ValidationResult:
        """Validate technical accuracy of text."""
        prompt = f"""
        Evaluate the technical accuracy of the following text.
        Consider factors such as:
        - Correct use of technical terms
        - Accurate descriptions of technical concepts
        - Proper explanation of technical processes
        - Consistency with established technical knowledge
        
        Text to evaluate:
        ---
        {text}
        ---
        
        First, provide a score from 1-10 where:
        1-3: Poor technical accuracy
        4-6: Moderate technical accuracy
        7-10: Excellent technical accuracy
        
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
            message=f"Technical accuracy score: {score}/10. {reasoning}",
            details={
                "score": score,
                "reasoning": reasoning,
                "response": response
            }
        )
    
    def improve(self, text: str) -> Tuple[str, ImprovementResult]:
        """Improve technical accuracy of text."""
        prompt = f"""
        Improve the technical accuracy of the following text.
        Focus on:
        - Correcting technical terms
        - Clarifying technical concepts
        - Improving explanations of technical processes
        - Ensuring consistency with established technical knowledge
        
        Text to improve:
        ---
        {text}
        ---
        
        First, rewrite the text to improve technical accuracy.
        Then, explain the changes you made.
        
        Format your response as:
        IMPROVED TEXT:
        [your improved version of the text]
        
        EXPLANATION:
        [explanation of changes made]
        """
        
        response = self.model.generate(prompt, **self.model_options)
        
        # Extract improved text and explanation
        text_match = re.search(r"IMPROVED TEXT:\s*(.*?)(?=EXPLANATION:|$)", response, re.DOTALL)
        explanation_match = re.search(r"EXPLANATION:\s*(.*?)$", response, re.DOTALL)
        
        if not text_match:
            return text, ImprovementResult(
                original_text=text,
                improved_text=text,
                changes_made=False,
                message="Could not extract improved text from critic response",
                details={"response": response}
            )
        
        improved_text = text_match.group(1).strip()
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        # Check if the text was actually changed
        changes_made = improved_text != text
        
        return improved_text, ImprovementResult(
            original_text=text,
            improved_text=improved_text,
            changes_made=changes_made,
            message=f"Technical accuracy improvements: {explanation}",
            details={
                "explanation": explanation,
                "response": response
            }
        )

# Create a convenience function
def technical_accuracy(model: str = "openai:gpt-3.5-turbo", **options):
    """Create a technical accuracy critic."""
    return TechnicalAccuracyCritic(model, **options)

# Use the custom critic
critic = technical_accuracy(model="openai:gpt-4")
result = critic.validate("The CPU is the main memory of a computer where all programs are stored.")
print(f"Validation passed: {result.passed}")
print(f"Message: {result.message}")
```

This example:
1. Creates a custom critic class that evaluates and improves technical accuracy
2. Implements the validate and improve methods
3. Creates a convenience function for creating instances of the critic
4. Uses the custom critic to validate a piece of text

## Conclusion

Critics are a powerful feature of Sifaka that allow you to leverage LLMs to validate and improve text. They can be used both as validators and improvers in a Chain, and you can create custom critics for specific requirements.

For more information, see the [Critics API Reference](../api/critics/index.md).
