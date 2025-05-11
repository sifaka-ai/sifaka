# Dependency Analysis Report

Total modules: 122
Total dependencies: 453
Circular dependencies: 80

## Circular Dependencies

### Cycle 1
```
sifaka.models.providers.anthropic -> sifaka.models.providers.anthropic
```

### Cycle 2
```
sifaka.utils.errors -> sifaka.utils.errors
```

### Cycle 3
```
sifaka.models.base -> sifaka.models.providers.anthropic -> sifaka.models.core -> sifaka.models.managers.client -> sifaka.models.base
```

### Cycle 4
```
sifaka.models.base -> sifaka.models.providers.anthropic -> sifaka.models.core -> sifaka.models.services.generation -> sifaka.models.managers.token_counter -> sifaka.models.base
```

### Cycle 5
```
sifaka.models.base -> sifaka.models.providers.anthropic -> sifaka.models.core -> sifaka.models.services.generation -> sifaka.models.base
```

### Cycle 6
```
sifaka.models.base -> sifaka.models.providers.anthropic -> sifaka.models.core -> sifaka.models.services.generation -> sifaka.models.managers.tracing -> sifaka.models.base
```

### Cycle 7
```
sifaka.models.base -> sifaka.models.providers.anthropic -> sifaka.models.core -> sifaka.models.base
```

### Cycle 8
```
sifaka.utils.state -> sifaka.utils.state
```

### Cycle 9
```
sifaka.models.base -> sifaka.models.providers.anthropic -> sifaka.models.base
```

### Cycle 10
```
sifaka.utils.patterns -> sifaka.utils.patterns
```

### Cycle 11
```
sifaka.models.config -> sifaka.models.config
```

### Cycle 12
```
sifaka.utils.config -> sifaka.models.config -> sifaka.utils.config
```

### Cycle 13
```
sifaka.utils.config -> sifaka.utils.config
```

### Cycle 14
```
sifaka.chain.config -> sifaka.chain.config
```

### Cycle 15
```
sifaka.utils.config -> sifaka.chain.config -> sifaka.utils.config
```

### Cycle 16
```
sifaka.utils.config -> sifaka.classifiers.config -> sifaka.utils.config
```

### Cycle 17
```
sifaka.utils.text -> sifaka.utils.text
```

### Cycle 18
```
sifaka.utils.text -> sifaka.rules.base -> sifaka.utils.text
```

### Cycle 19
```
sifaka.rules.base -> sifaka.rules.base
```

### Cycle 20
```
sifaka.rules.base -> sifaka.rules.formatting.length -> sifaka.rules.base
```

### Cycle 21
```
sifaka.rules.formatting.length -> sifaka.rules.formatting.length
```

### Cycle 22
```
sifaka.utils.config -> sifaka.critics.models -> sifaka.utils.config
```

### Cycle 23
```
sifaka.critics.models -> sifaka.critics.models
```

### Cycle 24
```
sifaka.models.base -> sifaka.models.base
```

### Cycle 25
```
sifaka.models.base -> sifaka.models.providers.openai -> sifaka.models.base
```

### Cycle 26
```
sifaka.models.providers.openai -> sifaka.models.providers.openai
```

### Cycle 27
```
sifaka.core.protocol -> sifaka.core.protocol
```

### Cycle 28
```
sifaka.rules.factories -> sifaka.rules.factories
```

### Cycle 29
```
sifaka.rules.formatting.format -> sifaka.rules.formatting.format
```

### Cycle 30
```
sifaka.interfaces.factories -> sifaka.interfaces.factories
```

### Cycle 31
```
sifaka.classifiers.implementations.content.toxicity -> sifaka.classifiers.implementations.content.toxicity
```

### Cycle 32
```
sifaka.chain.factories -> sifaka.chain.factories
```

### Cycle 33
```
sifaka.rules.content.sentiment -> sifaka.rules.content.sentiment
```

### Cycle 34
```
sifaka.utils.results -> sifaka.utils.results
```

### Cycle 35
```
sifaka.rules.content.prohibited -> sifaka.rules.content.prohibited
```

### Cycle 36
```
sifaka.classifiers.implementations.content.profanity -> sifaka.classifiers.implementations.content.profanity
```

### Cycle 37
```
sifaka.interfaces.adapter -> sifaka.interfaces.adapter
```

### Cycle 38
```
sifaka.rules.formatting.structure -> sifaka.rules.formatting.structure
```

### Cycle 39
```
sifaka.core.initialization -> sifaka.core.initialization
```

### Cycle 40
```
sifaka.core.dependency -> sifaka.core.dependency
```

### Cycle 41
```
sifaka.core.managers.response -> sifaka.core.managers.response
```

### Cycle 42
```
sifaka.core.managers.prompt_factories -> sifaka.core.managers.prompt_factories
```

### Cycle 43
```
sifaka.core.managers.prompt -> sifaka.core.managers.prompt
```

### Cycle 44
```
sifaka.critics.core -> sifaka.critics.core
```

### Cycle 45
```
sifaka.critics.utils -> sifaka.critics.utils
```

### Cycle 46
```
sifaka.critics.base -> sifaka.critics.base
```

### Cycle 47
```
sifaka.critics.implementations.self_refine -> sifaka.critics.implementations.self_refine
```

### Cycle 48
```
sifaka.critics.implementations.self_rag -> sifaka.critics.implementations.self_rag
```

### Cycle 49
```
sifaka.critics.implementations.constitutional -> sifaka.critics.implementations.constitutional
```

### Cycle 50
```
sifaka.critics.implementations.lac -> sifaka.critics.implementations.lac
```

### Cycle 51
```
sifaka.critics.implementations.reflexion -> sifaka.critics.implementations.reflexion
```

### Cycle 52
```
sifaka.chain.interfaces -> sifaka.chain.interfaces
```

### Cycle 53
```
sifaka.chain.result -> sifaka.chain.result
```

### Cycle 54
```
sifaka.chain.engine -> sifaka.chain.engine
```

### Cycle 55
```
sifaka.chain.adapters -> sifaka.chain.adapters
```

### Cycle 56
```
sifaka.chain.managers.cache -> sifaka.chain.managers.cache
```

### Cycle 57
```
sifaka.chain.managers.retry -> sifaka.chain.managers.retry
```

### Cycle 58
```
sifaka.models.factories -> sifaka.models.factories
```

### Cycle 59
```
sifaka.models.providers.mock -> sifaka.models.providers.mock
```

### Cycle 60
```
sifaka.models.providers.gemini -> sifaka.models.providers.gemini
```

### Cycle 61
```
sifaka.models.result -> sifaka.models.result
```

### Cycle 62
```
sifaka.models.utils -> sifaka.models.utils
```

### Cycle 63
```
sifaka.retrieval.result -> sifaka.retrieval.result
```

### Cycle 64
```
sifaka.retrieval.strategies.ranking -> sifaka.retrieval.strategies.ranking
```

### Cycle 65
```
sifaka.rules.result -> sifaka.rules.result
```

### Cycle 66
```
sifaka.rules.utils -> sifaka.rules.utils
```

### Cycle 67
```
sifaka.rules.managers.validation -> sifaka.rules.managers.validation
```

### Cycle 68
```
sifaka.rules.content.tone -> sifaka.rules.content.tone
```

### Cycle 69
```
sifaka.rules.content.language -> sifaka.rules.content.language
```

### Cycle 70
```
sifaka.rules.content.base -> sifaka.rules.content.base
```

### Cycle 71
```
sifaka.rules.content.safety -> sifaka.rules.content.safety
```

### Cycle 72
```
sifaka.rules.formatting.style -> sifaka.rules.formatting.style
```

### Cycle 73
```
sifaka.rules.formatting.whitespace -> sifaka.rules.formatting.whitespace
```

### Cycle 74
```
sifaka.classifiers.adapters -> sifaka.classifiers.adapters
```

### Cycle 75
```
sifaka.classifiers.implementations.factories -> sifaka.classifiers.implementations.factories
```

### Cycle 76
```
sifaka.classifiers.implementations.adapters -> sifaka.classifiers.implementations.adapters
```

### Cycle 77
```
sifaka.classifiers.implementations.content.sentiment -> sifaka.classifiers.implementations.content.sentiment
```

### Cycle 78
```
sifaka.classifiers.implementations.properties.language -> sifaka.classifiers.implementations.properties.language
```

### Cycle 79
```
sifaka.interfaces.classifier -> sifaka.interfaces.classifier
```

### Cycle 80
```
sifaka.interfaces.critic -> sifaka.interfaces.critic
```

## Modules with Most Dependencies

- **sifaka.models.core**: 11 dependencies
  - sifaka.models.base
  - sifaka.models.managers.client
  - sifaka.models.managers.token_counter
  - sifaka.models.managers.tracing
  - sifaka.models.services.generation
  - sifaka.utils.common
  - sifaka.utils.error_patterns
  - sifaka.utils.errors
  - sifaka.utils.logging
  - sifaka.utils.state
  - ... and 1 more

- **sifaka.classifiers.implementations.content.toxicity**: 10 dependencies
  - sifaka.classifiers.classifier
  - sifaka.classifiers.config
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.classifiers.implementations.content.toxicity_model
  - sifaka.classifiers.result
  - sifaka.utils.common
  - sifaka.utils.config
  - sifaka.utils.logging
  - sifaka.utils.state
  - sifaka.utils.text

- **sifaka.core.factories**: 9 dependencies
  - sifaka.adapters.base
  - sifaka.chain.factories
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.retrieval.factories
  - sifaka.rules.content.prohibited
  - sifaka.rules.content.sentiment
  - sifaka.rules.factories
  - sifaka.rules.formatting.length
  - sifaka.rules.formatting.structure

- **sifaka.__init__**: 8 dependencies
  - sifaka.chain
  - sifaka.core.generation
  - sifaka.core.improvement
  - sifaka.core.validation
  - sifaka.critics
  - sifaka.models
  - sifaka.rules.base
  - sifaka.rules.formatting.length

- **sifaka.models.base**: 8 dependencies
  - sifaka.interfaces
  - sifaka.models
  - sifaka.models.base
  - sifaka.models.providers.anthropic
  - sifaka.models.providers.openai
  - sifaka.utils.config
  - sifaka.utils.logging
  - sifaka.utils.tracing

- **sifaka.models.providers.openai**: 8 dependencies
  - sifaka.models.base
  - sifaka.models.core
  - sifaka.models.providers.openai
  - sifaka.utils.common
  - sifaka.utils.config
  - sifaka.utils.error_patterns
  - sifaka.utils.errors
  - sifaka.utils.logging

- **sifaka.adaptersdantic_ai.factory**: 8 dependencies
  - sifaka.adapters.base
  - sifaka.critics.base
  - sifaka.models.base
  - sifaka.models.factories
  - sifaka.rules.base
  - sifaka.rules.formatting.length
  - sifaka.utils.errors
  - sifaka.utils.logging

- **sifaka.rules.base**: 8 dependencies
  - sifaka.rules.base
  - sifaka.rules.formatting.length
  - sifaka.utils.common
  - sifaka.utils.error_patterns
  - sifaka.utils.errors
  - sifaka.utils.logging
  - sifaka.utils.state
  - sifaka.utils.text

- **sifaka.rules.managers.validation**: 8 dependencies
  - sifaka.core.base
  - sifaka.interfaces
  - sifaka.rules.base
  - sifaka.rules.formatting.format
  - sifaka.rules.formatting.length
  - sifaka.rules.managers.validation
  - sifaka.utils.logging
  - sifaka.utils.state

- **sifaka.classifiers.implementations.content.sentiment**: 8 dependencies
  - sifaka.classifiers.classifier
  - sifaka.classifiers.config
  - sifaka.classifiers.implementations.content.sentiment
  - sifaka.classifiers.result
  - sifaka.utils.config
  - sifaka.utils.logging
  - sifaka.utils.state
  - sifaka.utils.text

- **sifaka.critics.core**: 7 dependencies
  - sifaka.core.managers.memory
  - sifaka.core.managers.prompt
  - sifaka.core.managers.response
  - sifaka.critics.core
  - sifaka.critics.managers
  - sifaka.models.providers
  - sifaka.utils.text

- **sifaka.models.factories**: 7 dependencies
  - sifaka.models.core
  - sifaka.models.factories
  - sifaka.models.providers.anthropic
  - sifaka.models.providers.gemini
  - sifaka.models.providers.mock
  - sifaka.models.providers.openai
  - sifaka.utils.config

- **sifaka.models.providers.anthropic**: 7 dependencies
  - sifaka.models.base
  - sifaka.models.core
  - sifaka.models.providers.anthropic
  - sifaka.utils.errors
  - sifaka.utils.logging
  - sifaka.utils.patterns
  - sifaka.utils.tracing

- **sifaka.adaptersdantic_ai.adapter**: 7 dependencies
  - sifaka.adapters.base
  - sifaka.critics.base
  - sifaka.rules.base
  - sifaka.rules.formatting.length
  - sifaka.utils.errors
  - sifaka.utils.logging
  - sifaka.utils.state

- **sifaka.adapters.classifier.adapter**: 7 dependencies
  - sifaka.adapters.base
  - sifaka.adapters.classifier
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.utils
  - sifaka.utils.errors
  - sifaka.utils.logging
  - sifaka.utils.text

- **sifaka.retrieval.core**: 7 dependencies
  - sifaka.core.base
  - sifaka.interfaces.retrieval
  - sifaka.utils.common
  - sifaka.utils.error_patterns
  - sifaka.utils.errors
  - sifaka.utils.logging
  - sifaka.utils.text

- **sifaka.rules.content.safety**: 7 dependencies
  - sifaka.adapters.classifier
  - sifaka.classifiers.implementations.content.bias
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.rules.base
  - sifaka.rules.content.base
  - sifaka.rules.content.safety
  - sifaka.utils.logging

- **sifaka.classifiers.implementations.content.profanity**: 7 dependencies
  - sifaka.classifiers.classifier
  - sifaka.classifiers.config
  - sifaka.classifiers.implementations.content.profanity
  - sifaka.classifiers.result
  - sifaka.utils.logging
  - sifaka.utils.state
  - sifaka.utils.text

- **sifaka.utils.config**: 6 dependencies
  - sifaka.chain.config
  - sifaka.classifiers.config
  - sifaka.critics.models
  - sifaka.models.config
  - sifaka.retrieval.config
  - sifaka.utils.config

- **sifaka.models.providers.gemini**: 6 dependencies
  - sifaka.models.base
  - sifaka.models.core
  - sifaka.models.providers.gemini
  - sifaka.utils.config
  - sifaka.utils.errors
  - sifaka.utils.logging

## Most Depended-Upon Modules

- **sifaka.utils.logging**: 59 dependents
- **sifaka.utils.state**: 31 dependents
- **sifaka.utils.errors**: 28 dependents
- **sifaka.rules.base**: 18 dependents
- **sifaka.utils.config**: 17 dependents
- **sifaka.utils.text**: 13 dependents
- **sifaka.models.base**: 12 dependents
- **sifaka.core.base**: 11 dependents
- **sifaka.classifiers.config**: 11 dependents
- **sifaka.classifiers.classifier**: 10 dependents
- **sifaka.classifiers.result**: 10 dependents
- **sifaka.rules.formatting.length**: 9 dependents
- **sifaka.utils.common**: 9 dependents
- **sifaka.core.managers.memory**: 8 dependents
- **sifaka.models**: 7 dependents
- **sifaka.classifiers.implementations.content.toxicity**: 7 dependents
- **sifaka.models.core**: 6 dependents
- **sifaka.adapters.base**: 6 dependents
- **sifaka.critics.models**: 6 dependents
- **sifaka.models.providers**: 6 dependents
