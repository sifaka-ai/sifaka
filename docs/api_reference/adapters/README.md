# Adapters API Reference

Adapters in Sifaka are components that integrate external frameworks and systems with the Sifaka framework. They allow you to use components from other libraries as Sifaka components, and vice versa.

## Core Classes and Protocols

### BaseAdapter

`BaseAdapter` is the abstract base class for all adapters in Sifaka.

```python
from sifaka.adapters.rules.base import BaseAdapter, Adaptable
from sifaka.rules.base import RuleResult

class MyAdapter(BaseAdapter[str]):
    """Custom adapter implementation."""
    
    def __init__(self, adaptee: Adaptable):
        super().__init__(adaptee)
    
    def validate(self, text: str) -> RuleResult:
        """Validate the text using the adaptee."""
        # Implementation details
        result = self.adaptee.some_method(text)
        return RuleResult(
            passed=result,
            rule_name=self.name,
            message="Validation result",
            metadata={"adaptee_result": result}
        )
```

### Adaptable

`Adaptable` is a protocol for components that can be adapted.

```python
from sifaka.adapters.rules.base import Adaptable
from typing import Any

class MyAdaptable(Adaptable):
    """Custom adaptable implementation."""
    
    def some_method(self, input: Any) -> bool:
        """Some method that can be adapted."""
        return True
```

## Adapter Types

Sifaka provides several types of adapters:

### ClassifierAdapter

`ClassifierAdapter` adapts classifiers to work as rules.

```python
from sifaka.adapters.rules.classifier import create_classifier_rule
from sifaka.classifiers.toxicity import create_toxicity_classifier

# Create a classifier
classifier = create_toxicity_classifier()

# Create a rule from the classifier
rule = create_classifier_rule(
    classifier=classifier,
    valid_labels=["safe"],
    name="toxicity_rule",
    description="Ensures text is not toxic"
)
```

### GuardrailsAdapter

`GuardrailsAdapter` adapts Guardrails AI validators to work as rules.

```python
from sifaka.adapters.rules.guardrails_adapter import create_guardrails_rule
from guardrails.validators import ProfanityValidator

# Create a Guardrails validator
validator = ProfanityValidator()

# Create a rule from the Guardrails validator
rule = create_guardrails_rule(
    validator=validator,
    name="profanity_rule",
    description="Ensures text does not contain profanity"
)
```

## Usage Examples

### Using ClassifierAdapter

```python
from sifaka.adapters.rules.classifier import create_classifier_rule
from sifaka.classifiers.toxicity import create_toxicity_classifier
from sifaka.chain import create_simple_chain
from sifaka.models.openai import create_openai_chat_provider
from sifaka.critics.prompt import create_prompt_critic

# Create a classifier
classifier = create_toxicity_classifier()

# Create a rule from the classifier
rule = create_classifier_rule(
    classifier=classifier,
    valid_labels=["safe"],
    name="toxicity_rule",
    description="Ensures text is not toxic"
)

# Create other components
model = create_openai_chat_provider(model_name="gpt-4")
critic = create_prompt_critic(
    system_prompt="You are an expert editor. Improve the text to remove any toxic content."
)

# Create a chain with the adapted rule
chain = create_simple_chain(
    model=model,
    rules=[rule],
    critic=critic
)

# Run the chain
result = chain.run("Write a short story about a conflict.")
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
```

### Using GuardrailsAdapter

```python
from sifaka.adapters.rules.guardrails_adapter import create_guardrails_rule
from guardrails.validators import ProfanityValidator
from sifaka.chain import create_simple_chain
from sifaka.models.openai import create_openai_chat_provider
from sifaka.critics.prompt import create_prompt_critic

# Create a Guardrails validator
validator = ProfanityValidator()

# Create a rule from the Guardrails validator
rule = create_guardrails_rule(
    validator=validator,
    name="profanity_rule",
    description="Ensures text does not contain profanity"
)

# Create other components
model = create_openai_chat_provider(model_name="gpt-4")
critic = create_prompt_critic(
    system_prompt="You are an expert editor. Improve the text to remove any profanity."
)

# Create a chain with the adapted rule
chain = create_simple_chain(
    model=model,
    rules=[rule],
    critic=critic
)

# Run the chain
result = chain.run("Write a short story about a sailor.")
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
```

### Creating a Custom Adapter

```python
from sifaka.adapters.rules.base import BaseAdapter, Adaptable
from sifaka.rules.base import RuleResult, RuleConfig, Rule
from typing import Any, Dict, Optional

# Define an adaptable class (from an external library)
class ExternalValidator:
    """An external validator class."""
    
    def check_text(self, text: str) -> Dict[str, Any]:
        """Check the text for issues."""
        issues = []
        if len(text) < 10:
            issues.append("Text is too short")
        if "bad word" in text.lower():
            issues.append("Text contains bad words")
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

# Make it implement the Adaptable protocol
class AdaptableExternalValidator(ExternalValidator, Adaptable):
    """An adaptable version of the external validator."""
    pass

# Create a custom adapter
class ExternalValidatorAdapter(BaseAdapter[str]):
    """Adapter for ExternalValidator."""
    
    def __init__(
        self,
        adaptee: AdaptableExternalValidator,
        name: str,
        description: str,
        config: Optional[RuleConfig] = None
    ):
        super().__init__(adaptee)
        self._name = name
        self._description = description
        self._config = config or RuleConfig()
    
    @property
    def name(self) -> str:
        """Get the name of the rule."""
        return self._name
    
    @property
    def description(self) -> str:
        """Get the description of the rule."""
        return self._description
    
    @property
    def config(self) -> RuleConfig:
        """Get the configuration of the rule."""
        return self._config
    
    def validate(self, text: str) -> RuleResult:
        """Validate the text using the adaptee."""
        result = self.adaptee.check_text(text)
        return RuleResult(
            passed=result["valid"],
            rule_name=self.name,
            message="Validation " + ("passed" if result["valid"] else "failed"),
            metadata={"issues": result["issues"]}
        )

# Create a factory function
def create_external_validator_rule(
    validator: ExternalValidator,
    name: str,
    description: str,
    config: Optional[RuleConfig] = None
) -> Rule:
    """Create a rule from an external validator."""
    adaptable_validator = AdaptableExternalValidator()
    adaptable_validator.check_text = validator.check_text
    return ExternalValidatorAdapter(
        adaptee=adaptable_validator,
        name=name,
        description=description,
        config=config
    )

# Use the custom adapter
validator = ExternalValidator()
rule = create_external_validator_rule(
    validator=validator,
    name="external_rule",
    description="Rule using an external validator"
)

# Validate text
result = rule.validate("This is a test")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
