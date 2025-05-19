# Critics Documentation

Critics (also called Improvers) are components in the Sifaka framework that enhance the quality of text by applying various improvement strategies. They analyze text, identify areas for improvement, and generate improved versions.

## Overview

Critics are used by the Chain class in two ways:

1. **Feedback Loop**: When validation fails and the feedback loop is enabled, critics provide feedback on how to improve the text. This feedback is combined with validation feedback and sent to the model to generate improved text.

2. **Post-Validation Improvement**: After validation passes, critics improve the quality of the generated text.

Critics implement the Improver protocol, which defines a consistent interface for all critics. Critics can be chained together to apply multiple improvement strategies in sequence.

## Built-in Critics

Sifaka includes several built-in critics that you can use out of the box:

### Reflexion Critic

The Reflexion critic implements the approach described in the paper ["Reflexion: Language Agents with Verbal Reinforcement Learning"](https://arxiv.org/abs/2303.11366). It improves text by having the model reflect on its own output and make improvements based on that reflection.

```python
from sifaka.critics.reflexion import create_reflexion_critic
from sifaka.models.openai import OpenAIModel

# Create a model
model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

# Create a reflexion critic
critic = create_reflexion_critic(model=model)

# Improve text
improved_text, result = critic.improve("This is a text that needs improvement.")
print(f"Original: {result.original_text}")
print(f"Improved: {result.improved_text}")
print(f"Changes made: {result.changes_made}")
```

### N-Critics Critic

The N-Critics critic implements the approach described in the paper ["Language Models Can Solve Computer Tasks"](https://arxiv.org/abs/2303.17491). It improves text by generating multiple independent critiques and then synthesizing them into a final improved version.

```python
from sifaka.critics.n_critics import create_n_critics_critic
from sifaka.models.openai import OpenAIModel

# Create a model
model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

# Create an n-critics critic with 3 critics
critic = create_n_critics_critic(model=model, num_critics=3)

# Improve text
improved_text, result = critic.improve("This is a text that needs improvement.")
print(f"Original: {result.original_text}")
print(f"Improved: {result.improved_text}")
print(f"Changes made: {result.changes_made}")
```

### Self-RAG Critic

The Self-RAG critic implements the approach described in the paper ["Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"](https://arxiv.org/abs/2310.11511). It improves text by retrieving relevant information and incorporating it into the text.

```python
from sifaka.critics.self_rag import create_self_rag_critic
from sifaka.models.openai import OpenAIModel
from sifaka.retrievers import create_retriever

# Create a model
model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

# Create a retriever
retriever = create_retriever("elasticsearch", index="my_index", host="localhost", port=9200)

# Create a self-rag critic
critic = create_self_rag_critic(model=model, retriever=retriever)

# Improve text
improved_text, result = critic.improve("This is a text that needs improvement.")
print(f"Original: {result.original_text}")
print(f"Improved: {result.improved_text}")
print(f"Changes made: {result.changes_made}")
```

### Constitutional Critic

The Constitutional critic improves text by ensuring it adheres to a set of constitutional principles or guidelines.

```python
from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.models.openai import OpenAIModel

# Create a model
model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

# Define constitutional principles
principles = [
    "All content must be family-friendly and appropriate for all ages.",
    "Content should be factually accurate and not misleading.",
    "Content should be respectful and not contain offensive language or stereotypes."
]

# Create a constitutional critic
critic = create_constitutional_critic(model=model, principles=principles)

# Improve text
improved_text, result = critic.improve("This is a text that needs improvement.")
print(f"Original: {result.original_text}")
print(f"Improved: {result.improved_text}")
print(f"Changes made: {result.changes_made}")
```

## Using Critics with Chain

Critics are typically used with the Chain class to improve generated text after validation:

```python
from sifaka import Chain
from sifaka.validators import length, prohibited_content
from sifaka.critics.reflexion import create_reflexion_critic

# Create a chain with validators and critics
chain = (Chain(max_improvement_iterations=3)
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .validate_with(length(min_words=100, max_words=500))
    .validate_with(prohibited_content(prohibited=["violent", "harmful"]))
    .improve_with(create_reflexion_critic(model="openai:gpt-4"))
    .with_options(apply_improvers_on_validation_failure=True)  # Enable feedback loop
)

# Run the chain
result = chain.run()

# Check if validation passed and text was improved
if result.passed:
    print("Chain execution succeeded!")
    print(result.text)

    # Access improvement details
    for i, improvement in enumerate(result.improvement_results):
        print(f"Improvement {i+1}:")
        print(f"- Original length: {len(improvement.original_text)}")
        print(f"- Improved length: {len(improvement.improved_text)}")
        print(f"- Changes made: {improvement.changes_made}")
```

## Creating Custom Critics

You can create custom critics by inheriting from the Critic base class and implementing the _critique and _improve methods:

```python
from sifaka.critics.base import Critic
from sifaka.models.base import Model
from typing import Dict, Any

class SimplificationCritic(Critic):
    def __init__(self, model: Model, name: str = "SimplificationCritic"):
        super().__init__(model=model, name=name)

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text for complexity and identify areas to simplify."""
        # Use the model to analyze the text complexity
        prompt = f"""
        Analyze the following text for complexity and identify areas that could be simplified:

        TEXT:
        {text}

        Provide your analysis in the following format:
        - needs_improvement: true/false
        - message: A summary of your analysis
        - issues: A list of specific issues related to complexity
        - suggestions: A list of suggestions for simplification
        """

        response = self._model.generate(prompt)

        # Parse the response (simplified example)
        lines = response.strip().split('\n')
        needs_improvement = any("needs_improvement: true" in line.lower() for line in lines)

        issues = []
        suggestions = []
        message = "Text complexity analysis"

        for line in lines:
            if line.startswith("- message:"):
                message = line[len("- message:"):].strip()
            elif line.startswith("- issues:") or line.startswith("  -"):
                issue = line.split(":", 1)[1].strip() if ":" in line else line.strip("- ")
                if issue and not issue.startswith("A list"):
                    issues.append(issue)
            elif line.startswith("- suggestions:") or line.startswith("  -"):
                suggestion = line.split(":", 1)[1].strip() if ":" in line else line.strip("- ")
                if suggestion and not suggestion.startswith("A list"):
                    suggestions.append(suggestion)

        return {
            "needs_improvement": needs_improvement,
            "message": message,
            "issues": issues,
            "suggestions": suggestions
        }

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text by simplifying it based on the critique."""
        if not critique["needs_improvement"]:
            return text

        # Use the model to simplify the text based on the critique
        issues_str = "\n".join(f"- {issue}" for issue in critique["issues"])
        suggestions_str = "\n".join(f"- {suggestion}" for suggestion in critique["suggestions"])

        prompt = f"""
        Simplify the following text to make it more accessible and easier to understand.

        ORIGINAL TEXT:
        {text}

        ISSUES IDENTIFIED:
        {issues_str}

        SUGGESTIONS FOR IMPROVEMENT:
        {suggestions_str}

        Please rewrite the text to address these issues and make it simpler while preserving the original meaning.
        """

        improved_text = self._model.generate(prompt)
        return improved_text
```

## Critic Base Class

All critics should inherit from the Critic base class, which provides common functionality and implements the Improver protocol:

```python
class Critic(Improver):
    def __init__(self, model: Model, name: Optional[str] = None, **options: Any):
        """Initialize the critic.

        Args:
            model: The model to use for generating improvements
            name: Optional name for the critic
            **options: Additional options for the critic
        """

    def improve(self, text: str) -> tuple[str, SifakaImprovementResult]:
        """Improve text using this critic.

        This method analyzes the provided text and generates an improved version
        based on the critic's improvement strategy. It follows a two-step process:
        1. Critique the text to identify areas for improvement
        2. Improve the text based on the critique

        Args:
            text: The text to improve

        Returns:
            A tuple of (improved_text, improvement_result)
        """

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text based on specific criteria.

        This method should be overridden by subclasses to implement specific
        critique logic.

        Args:
            text: The text to critique

        Returns:
            A dictionary with critique information
        """

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text based on critique.

        This method should be overridden by subclasses to implement specific
        improvement logic.

        Args:
            text: The text to improve
            critique: The critique information

        Returns:
            The improved text
        """
```

## Improvement Results

Critics return ImprovementResult objects that contain information about the improvement:

```python
@dataclass
class ImprovementResult:
    _original_text: str  # The original text before improvement
    _improved_text: str  # The improved text after improvement
    _changes_made: bool  # Whether any changes were made
    message: str  # Human-readable message describing the improvement
    _details: Dict[str, Any] = field(default_factory=dict)  # Additional details
    processing_time_ms: Optional[float] = None  # Processing time in milliseconds
```

You can access these properties to get information about the improvement:

```python
improved_text, result = critic.improve(text)

print(f"Original: {result.original_text}")
print(f"Improved: {result.improved_text}")

if result.changes_made:
    print(f"Changes made: {result.message}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
else:
    print("No changes were made to the text")
```

## Using Critics in the Feedback Loop

When validation fails, critics can provide feedback on how to improve the text. This feedback is combined with validation feedback and sent to the model to generate improved text. To enable this feature:

```python
from sifaka import Chain
from sifaka.validators import length
from sifaka.critics.reflexion import create_reflexion_critic

# Create a chain with feedback loop enabled
chain = (Chain(max_improvement_iterations=3)
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .validate_with(length(min_words=100, max_words=500))
    .improve_with(create_reflexion_critic(model="openai:gpt-4"))
    .with_options(apply_improvers_on_validation_failure=True)  # Enable feedback loop
)

# Run the chain
result = chain.run()
```

The feedback loop process:

1. If validation fails, the Chain collects feedback from validators on why validation failed
2. The Chain gets feedback from critics on how to improve the text
3. The combined feedback is sent to the model along with the original text to generate improved text
4. The improved text is re-validated
5. This process repeats until validation passes or the maximum number of iterations is reached

## Best Practices

1. **Use built-in critics** when possible, as they implement well-researched improvement strategies
2. **Combine multiple critics** to apply different improvement strategies in sequence
3. **Provide clear critique information** in custom critics to guide the improvement process
4. **Handle edge cases** like empty text or very short text
5. **Use appropriate models** for each critic, as some improvement strategies may require more advanced models
6. **Consider performance** when chaining multiple critics, as each critic adds to the overall processing time
7. **Enable the feedback loop** when you want critics to help improve text that fails validation
8. **Set an appropriate max_improvement_iterations** to balance between quality and performance
