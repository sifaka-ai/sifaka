# Dependency Analysis Report

Total modules: 143
Total dependencies: 581
Circular dependencies: 96

## Circular Dependencies

### Cycle 1
```
sifaka.utils.config -> sifaka.utils.config
```

### Cycle 2
```
sifaka.core.base -> sifaka.core.base
```

### Cycle 3
```
sifaka.utils.state -> sifaka.utils.state
```

### Cycle 4
```
sifaka.utils.errors -> sifaka.utils.errors
```

### Cycle 5
```
sifaka.utils.text -> sifaka.utils.text
```

### Cycle 6
```
sifaka.core.base -> sifaka.utils.text -> sifaka.core.base
```

### Cycle 7
```
sifaka.utils.common -> sifaka.utils.common
```

### Cycle 8
```
sifaka.utils.logging -> sifaka.utils.logging
```

### Cycle 9
```
sifaka.rules.base -> sifaka.rules.base
```

### Cycle 10
```
sifaka.rules.result -> sifaka.rules.result
```

### Cycle 11
```
sifaka.rules.validators -> sifaka.rules.validators
```

### Cycle 12
```
sifaka.models.managers.token_counter -> sifaka.models.managers.token_counter
```

### Cycle 13
```
sifaka.models.managers.client -> sifaka.models.managers.client
```

### Cycle 14
```
sifaka.rules.formatting.length -> sifaka.rules.formatting.length
```

### Cycle 15
```
sifaka.utils.patterns -> sifaka.utils.patterns
```

### Cycle 16
```
sifaka.classifiers.implementations.content.profanity -> sifaka.classifiers.implementations.content.profanity
```

### Cycle 17
```
sifaka.classifiers.result -> sifaka.classifiers.result
```

### Cycle 18
```
sifaka.rules.content.prohibited -> sifaka.rules.content.prohibited
```

### Cycle 19
```
sifaka.core.validation -> sifaka.core.validation
```

### Cycle 20
```
sifaka.core.plugins -> sifaka.core.plugins
```

### Cycle 21
```
sifaka.core.protocol -> sifaka.core.protocol
```

### Cycle 22
```
sifaka.core.dependency -> sifaka.core.dependency
```

### Cycle 23
```
sifaka.chain.factories -> sifaka.chain.factories
```

### Cycle 24
```
sifaka.rules.factories -> sifaka.rules.factories
```

### Cycle 25
```
sifaka.classifiers.implementations.content.toxicity -> sifaka.classifiers.implementations.content.toxicity
```

### Cycle 26
```
sifaka.rules.content.base -> sifaka.rules.content.base
```

### Cycle 27
```
sifaka.classifiers.implementations.content.bias -> sifaka.classifiers.implementations.content.bias
```

### Cycle 28
```
sifaka.rules.content.safety -> sifaka.rules.content.safety
```

### Cycle 29
```
sifaka.models.providers.openai -> sifaka.models.providers.openai
```

### Cycle 30
```
sifaka.retrieval.core -> sifaka.retrieval.core
```

### Cycle 31
```
sifaka.interfaces.factories -> sifaka.interfaces.factories
```

### Cycle 32
```
sifaka.utils.results -> sifaka.utils.results
```

### Cycle 33
```
sifaka.critics.models -> sifaka.critics.models
```

### Cycle 34
```
sifaka.rules.content.sentiment -> sifaka.rules.content.sentiment
```

### Cycle 35
```
sifaka.rules.formatting.structure -> sifaka.rules.formatting.structure
```

### Cycle 36
```
sifaka.rules.formatting.format -> sifaka.rules.formatting.format
```

### Cycle 37
```
sifaka.interfaces.adapter -> sifaka.interfaces.adapter
```

### Cycle 38
```
sifaka.core.initialization -> sifaka.core.initialization
```

### Cycle 39
```
sifaka.utils.resources -> sifaka.utils.resources
```

### Cycle 40
```
sifaka.core.managers.memory -> sifaka.core.managers.memory
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
sifaka.critics.plugins -> sifaka.critics.plugins
```

### Cycle 45
```
sifaka.critics.core -> sifaka.critics.core
```

### Cycle 46
```
sifaka.critics.utils -> sifaka.critics.utils
```

### Cycle 47
```
sifaka.critics.base -> sifaka.critics.base
```

### Cycle 48
```
sifaka.critics.implementations.self_refine -> sifaka.critics.implementations.self_refine
```

### Cycle 49
```
sifaka.critics.implementations.self_rag -> sifaka.critics.implementations.self_rag
```

### Cycle 50
```
sifaka.critics.implementations.constitutional -> sifaka.critics.implementations.constitutional
```

### Cycle 51
```
sifaka.critics.implementations.lac -> sifaka.critics.implementations.lac
```

### Cycle 52
```
sifaka.critics.implementations.reflexion -> sifaka.critics.implementations.reflexion
```

### Cycle 53
```
sifaka.critics.implementations.prompt -> sifaka.critics.implementations.prompt
```

### Cycle 54
```
sifaka.chain.interfaces -> sifaka.chain.interfaces
```

### Cycle 55
```
sifaka.chain.plugins -> sifaka.chain.plugins
```

### Cycle 56
```
sifaka.chain.result -> sifaka.chain.result
```

### Cycle 57
```
sifaka.chain.engine -> sifaka.chain.engine
```

### Cycle 58
```
sifaka.chain.state -> sifaka.chain.state
```

### Cycle 59
```
sifaka.chain.adapters -> sifaka.chain.adapters
```

### Cycle 60
```
sifaka.chain.managers.memory -> sifaka.chain.managers.memory
```

### Cycle 61
```
sifaka.chain.managers.cache -> sifaka.chain.managers.cache
```

### Cycle 62
```
sifaka.chain.managers.retry -> sifaka.chain.managers.retry
```

### Cycle 63
```
sifaka.models.plugins -> sifaka.models.plugins
```

### Cycle 64
```
sifaka.models.providers.anthropic -> sifaka.models.providers.anthropic
```

### Cycle 65
```
sifaka.models.providers.gemini -> sifaka.models.providers.gemini
```

### Cycle 66
```
sifaka.models.factories -> sifaka.models.factories
```

### Cycle 67
```
sifaka.models.providers.mock -> sifaka.models.providers.mock
```

### Cycle 68
```
sifaka.models.result -> sifaka.models.result
```

### Cycle 69
```
sifaka.models.utils -> sifaka.models.utils
```

### Cycle 70
```
sifaka.adapters.plugins -> sifaka.adapters.plugins
```

### Cycle 71
```
sifaka.retrieval.plugins -> sifaka.retrieval.plugins
```

### Cycle 72
```
sifaka.retrieval.result -> sifaka.retrieval.result
```

### Cycle 73
```
sifaka.retrieval.strategies.ranking -> sifaka.retrieval.strategies.ranking
```

### Cycle 74
```
sifaka.rules.plugins -> sifaka.rules.plugins
```

### Cycle 75
```
sifaka.rules.utils -> sifaka.rules.utils
```

### Cycle 76
```
sifaka.rules.managers.validation -> sifaka.rules.managers.validation
```

### Cycle 77
```
sifaka.rules.content.tone -> sifaka.rules.content.tone
```

### Cycle 78
```
sifaka.rules.content.language -> sifaka.rules.content.language
```

### Cycle 79
```
sifaka.rules.formatting.style -> sifaka.rules.formatting.style
```

### Cycle 80
```
sifaka.rules.formatting.whitespace -> sifaka.rules.formatting.whitespace
```

### Cycle 81
```
sifaka.classifiers.interfaces -> sifaka.classifiers.interfaces
```

### Cycle 82
```
sifaka.classifiers.plugins -> sifaka.classifiers.plugins
```

### Cycle 83
```
sifaka.classifiers.factories -> sifaka.classifiers.factories
```

### Cycle 84
```
sifaka.classifiers.engine -> sifaka.classifiers.engine
```

### Cycle 85
```
sifaka.classifiers.errors -> sifaka.classifiers.errors
```

### Cycle 86
```
sifaka.classifiers.base -> sifaka.classifiers.base
```

### Cycle 87
```
sifaka.classifiers.adapters -> sifaka.classifiers.adapters
```

### Cycle 88
```
sifaka.classifiers.implementations.factories -> sifaka.classifiers.implementations.factories
```

### Cycle 89
```
sifaka.classifiers.implementations.adapters -> sifaka.classifiers.implementations.adapters
```

### Cycle 90
```
sifaka.classifiers.implementations.content.spam -> sifaka.classifiers.implementations.content.spam
```

### Cycle 91
```
sifaka.classifiers.implementations.content.sentiment -> sifaka.classifiers.implementations.content.sentiment
```

### Cycle 92
```
sifaka.classifiers.implementations.properties.language -> sifaka.classifiers.implementations.properties.language
```

### Cycle 93
```
sifaka.classifiers.implementations.properties.topic -> sifaka.classifiers.implementations.properties.topic
```

### Cycle 94
```
sifaka.classifiers.implementations.entities.ner -> sifaka.classifiers.implementations.entities.ner
```

### Cycle 95
```
sifaka.interfaces.classifier -> sifaka.interfaces.classifier
```

### Cycle 96
```
sifaka.interfaces.critic -> sifaka.interfaces.critic
```

## Modules with Most Dependencies

- **sifaka.models.core**: 15 dependencies
  - sifaka.interfaces.client
  - sifaka.interfaces.counter
  - sifaka.interfaces.model
  - sifaka.models.base
  - sifaka.models.managers.client
  - sifaka.models.managers.token_counter
  - sifaka.models.managers.tracing
  - sifaka.models.services.generation
  - sifaka.utils.common
  - sifaka.utils.config
  - ... and 5 more

- **sifaka.core.factories**: 14 dependencies
  - sifaka.adapters.base
  - sifaka.chain.factories
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.core.dependency
  - sifaka.interfaces.model
  - sifaka.models.providers.openai
  - sifaka.retrieval.factories
  - sifaka.rules.content.prohibited
  - sifaka.rules.content.safety
  - sifaka.rules.content.sentiment
  - ... and 4 more

- **sifaka.models.providers.openai**: 11 dependencies
  - sifaka.interfaces.client
  - sifaka.interfaces.counter
  - sifaka.interfaces.model
  - sifaka.models.managers.openai_client
  - sifaka.models.managers.openai_token_counter
  - sifaka.models.providers.openai
  - sifaka.utils.common
  - sifaka.utils.config
  - sifaka.utils.errors
  - sifaka.utils.logging
  - ... and 1 more

- **sifaka.models.providers.anthropic**: 11 dependencies
  - sifaka.interfaces.client
  - sifaka.interfaces.counter
  - sifaka.interfaces.model
  - sifaka.models.managers.anthropic_client
  - sifaka.models.managers.anthropic_token_counter
  - sifaka.models.providers.anthropic
  - sifaka.utils.common
  - sifaka.utils.config
  - sifaka.utils.errors
  - sifaka.utils.logging
  - ... and 1 more

- **sifaka.rules.validators**: 9 dependencies
  - sifaka.core.base
  - sifaka.rules.result
  - sifaka.rules.validators
  - sifaka.utils.common
  - sifaka.utils.error_patterns
  - sifaka.utils.errors
  - sifaka.utils.logging
  - sifaka.utils.state
  - sifaka.utils.text

- **sifaka.classifiers.implementations.content.toxicity**: 9 dependencies
  - sifaka.classifiers.classifier
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.classifiers.implementations.content.toxicity_model
  - sifaka.classifiers.result
  - sifaka.utils.common
  - sifaka.utils.config
  - sifaka.utils.logging
  - sifaka.utils.state
  - sifaka.utils.text

- **sifaka.__init__**: 8 dependencies
  - sifaka.chain
  - sifaka.core.generation
  - sifaka.core.improvement
  - sifaka.core.validation
  - sifaka.critics
  - sifaka.models
  - sifaka.rules.base
  - sifaka.rules.formatting.length

- **sifaka.critics.core**: 8 dependencies
  - sifaka.core.managers.memory
  - sifaka.core.managers.prompt
  - sifaka.core.managers.response
  - sifaka.critics.core
  - sifaka.critics.managers
  - sifaka.models.providers
  - sifaka.utils.config
  - sifaka.utils.text

- **sifaka.models.factories**: 8 dependencies
  - sifaka.interfaces.model
  - sifaka.models.factories
  - sifaka.models.providers.anthropic
  - sifaka.models.providers.gemini
  - sifaka.models.providers.mock
  - sifaka.models.providers.openai
  - sifaka.utils.config
  - sifaka.utils.logging

- **sifaka.adaptersdantic_ai.adapter**: 8 dependencies
  - sifaka.adapters.base
  - sifaka.core.validation
  - sifaka.critics.base
  - sifaka.rules.base
  - sifaka.rules.formatting.length
  - sifaka.utils.errors
  - sifaka.utils.logging
  - sifaka.utils.state

- **sifaka.adaptersdantic_ai.factory**: 8 dependencies
  - sifaka.adapters.base
  - sifaka.critics.base
  - sifaka.models.base
  - sifaka.models.factories
  - sifaka.rules.base
  - sifaka.rules.formatting.length
  - sifaka.utils.errors
  - sifaka.utils.logging

- **sifaka.adapters.classifier.adapter**: 8 dependencies
  - sifaka.adapters.base
  - sifaka.adapters.classifier
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.classifiers.result
  - sifaka.utils
  - sifaka.utils.errors
  - sifaka.utils.logging
  - sifaka.utils.text

- **sifaka.retrieval.core**: 8 dependencies
  - sifaka.core.base
  - sifaka.interfaces.retrieval
  - sifaka.retrieval.core
  - sifaka.utils.common
  - sifaka.utils.config
  - sifaka.utils.errors
  - sifaka.utils.logging
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

- **sifaka.critics.implementations.prompt**: 7 dependencies
  - sifaka.core.dependency
  - sifaka.core.managers.memory
  - sifaka.core.managers.prompt_factories
  - sifaka.critics.implementations.prompt
  - sifaka.interfaces.model
  - sifaka.models.providers
  - sifaka.utils.config

- **sifaka.chain.factories**: 7 dependencies
  - sifaka.chain.factories
  - sifaka.core.dependency
  - sifaka.critics
  - sifaka.interfaces.model
  - sifaka.models
  - sifaka.rules
  - sifaka.utils.config

- **sifaka.models.base**: 7 dependencies
  - sifaka.interfaces.client
  - sifaka.interfaces.counter
  - sifaka.interfaces.model
  - sifaka.models
  - sifaka.utils.config
  - sifaka.utils.logging
  - sifaka.utils.tracing

- **sifaka.rules.content.safety**: 7 dependencies
  - sifaka.adapters.classifier
  - sifaka.classifiers.implementations.content.bias
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.rules.base
  - sifaka.rules.content.base
  - sifaka.rules.content.safety
  - sifaka.utils.logging

- **sifaka.classifiers.base**: 7 dependencies
  - sifaka.classifiers.base
  - sifaka.classifiers.result
  - sifaka.core.base
  - sifaka.utils.common
  - sifaka.utils.errors
  - sifaka.utils.logging
  - sifaka.utils.state

- **sifaka.classifiers.implementations.content.sentiment**: 7 dependencies
  - sifaka.classifiers.classifier
  - sifaka.classifiers.implementations.content.sentiment
  - sifaka.classifiers.result
  - sifaka.utils.config
  - sifaka.utils.logging
  - sifaka.utils.state
  - sifaka.utils.text

## Most Depended-Upon Modules

- **sifaka.utils.logging**: 70 dependents
- **sifaka.utils.config**: 48 dependents
- **sifaka.utils.state**: 37 dependents
- **sifaka.utils.errors**: 34 dependents
- **sifaka.rules.base**: 17 dependents
- **sifaka.classifiers.result**: 17 dependents
- **sifaka.core.base**: 16 dependents
- **sifaka.core.interfaces**: 15 dependents
- **sifaka.utils.text**: 13 dependents
- **sifaka.utils.common**: 12 dependents
- **sifaka.classifiers.classifier**: 10 dependents
- **sifaka.core.managers.memory**: 9 dependents
- **sifaka.models**: 8 dependents
- **sifaka.rules.formatting.length**: 8 dependents
- **sifaka.core.plugins**: 8 dependents
- **sifaka.interfaces.model**: 8 dependents
- **sifaka.classifiers.implementations.content.toxicity**: 7 dependents
- **sifaka.models.providers**: 7 dependents
- **sifaka.interfaces.counter**: 7 dependents
- **sifaka.interfaces.client**: 7 dependents
