# Dependency Analysis Report

Total modules: 183
Total dependencies: 752
Circular dependencies: 126

## Circular Dependencies

### Cycle 1
```
sifaka.utils.logging -> sifaka.utils.logging
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
sifaka.utils.common -> sifaka.utils.common
```

### Cycle 5
```
sifaka.utils.errors.safe_execution -> sifaka.utils.errors.safe_execution
```

### Cycle 6
```
sifaka.utils.result_types -> sifaka.utils.result_types
```

### Cycle 7
```
sifaka.utils.text -> sifaka.utils.text
```

### Cycle 8
```
sifaka.utils.config.models -> sifaka.utils.config.models
```

### Cycle 9
```
sifaka.core.validation -> sifaka.core.validation
```

### Cycle 10
```
sifaka.rules.formatting.length -> sifaka.rules.formatting.length
```

### Cycle 11
```
sifaka.rules.validators -> sifaka.rules.validators
```

### Cycle 12
```
sifaka.rules.base -> sifaka.rules.base
```

### Cycle 13
```
sifaka.utils.config.rules -> sifaka.utils.config.rules
```

### Cycle 14
```
sifaka.rules.formatting.length -> sifaka.rules.base -> sifaka.utils.config.rules -> sifaka.rules.formatting.length
```

### Cycle 15
```
sifaka.utils.config.classifiers -> sifaka.utils.config.classifiers
```

### Cycle 16
```
sifaka.classifiers.implementations.content.profanity -> sifaka.classifiers.implementations.content.profanity
```

### Cycle 17
```
sifaka.rules.content.prohibited -> sifaka.rules.content.prohibited
```

### Cycle 18
```
sifaka.core.plugins -> sifaka.core.plugins
```

### Cycle 19
```
sifaka.core.protocol -> sifaka.core.protocol
```

### Cycle 20
```
sifaka.rules.formatting.format.markdown -> sifaka.rules.formatting.format.markdown
```

### Cycle 21
```
sifaka.utils.config.retrieval -> sifaka.utils.config.retrieval
```

### Cycle 22
```
sifaka.utils.errors.handling -> sifaka.utils.errors.handling
```

### Cycle 23
```
sifaka.interfaces.factories -> sifaka.interfaces.factories
```

### Cycle 24
```
sifaka.core.dependency.provider -> sifaka.core.dependency.provider
```

### Cycle 25
```
sifaka.core.dependency.scopes -> sifaka.core.dependency.provider -> sifaka.core.dependency.scopes
```

### Cycle 26
```
sifaka.core.dependency.scopes -> sifaka.core.dependency.scopes
```

### Cycle 27
```
sifaka.core.dependency.utils -> sifaka.core.dependency.utils
```

### Cycle 28
```
sifaka.interfaces.adapter -> sifaka.interfaces.adapter
```

### Cycle 29
```
sifaka.classifiers.implementations.properties.language -> sifaka.classifiers.implementations.properties.language
```

### Cycle 30
```
sifaka.classifiers.implementations.content.toxicity -> sifaka.classifiers.implementations.content.toxicity
```

### Cycle 31
```
sifaka.models.managers.client -> sifaka.models.managers.client
```

### Cycle 32
```
sifaka.models.managers.token_counter -> sifaka.models.managers.token_counter
```

### Cycle 33
```
sifaka.models.providers.openai -> sifaka.models.providers.openai
```

### Cycle 34
```
sifaka.rules.formatting.format.plain_text -> sifaka.rules.formatting.format.plain_text
```

### Cycle 35
```
sifaka.utils.results -> sifaka.utils.results
```

### Cycle 36
```
sifaka.utils.base_results -> sifaka.utils.base_results
```

### Cycle 37
```
sifaka.rules.content.sentiment -> sifaka.rules.content.sentiment
```

### Cycle 38
```
sifaka.rules.content.base -> sifaka.rules.content.base
```

### Cycle 39
```
sifaka.rules.content.safety -> sifaka.rules.content.safety
```

### Cycle 40
```
sifaka.classifiers.implementations.content.bias -> sifaka.classifiers.implementations.content.bias
```

### Cycle 41
```
sifaka.rules.formatting.format.json -> sifaka.rules.formatting.format.json
```

### Cycle 42
```
sifaka.rules.factories -> sifaka.rules.factories
```

### Cycle 43
```
sifaka.chain.factories -> sifaka.chain.factories
```

### Cycle 44
```
sifaka.utils.config.chain -> sifaka.utils.config.chain
```

### Cycle 45
```
sifaka.rules.formatting.structure -> sifaka.rules.formatting.structure
```

### Cycle 46
```
sifaka.utils.resources -> sifaka.utils.resources
```

### Cycle 47
```
sifaka.core.initialization -> sifaka.core.initialization
```

### Cycle 48
```
sifaka.core.managers.memory -> sifaka.core.managers.memory
```

### Cycle 49
```
sifaka.core.managers.response -> sifaka.core.managers.response
```

### Cycle 50
```
sifaka.utils.config.critics -> sifaka.utils.config.critics
```

### Cycle 51
```
sifaka.core.managers.prompt_factories -> sifaka.core.managers.prompt_factories
```

### Cycle 52
```
sifaka.critics.models -> sifaka.critics.models
```

### Cycle 53
```
sifaka.core.managers.prompt -> sifaka.core.managers.prompt
```

### Cycle 54
```
sifaka.core.dependency.injector -> sifaka.core.dependency.injector
```

### Cycle 55
```
sifaka.critics.plugins -> sifaka.critics.plugins
```

### Cycle 56
```
sifaka.critics.core -> sifaka.critics.core
```

### Cycle 57
```
sifaka.critics.utils -> sifaka.critics.utils
```

### Cycle 58
```
sifaka.utils.patterns -> sifaka.utils.patterns
```

### Cycle 59
```
sifaka.critics.implementations.self_refine -> sifaka.critics.implementations.self_refine
```

### Cycle 60
```
sifaka.interfaces.critic -> sifaka.interfaces.critic
```

### Cycle 61
```
sifaka.critics.implementations.self_rag -> sifaka.critics.implementations.self_rag
```

### Cycle 62
```
sifaka.critics.implementations.constitutional -> sifaka.critics.implementations.constitutional
```

### Cycle 63
```
sifaka.critics.implementations.lac -> sifaka.critics.implementations.lac
```

### Cycle 64
```
sifaka.critics.implementations.reflexion -> sifaka.critics.implementations.reflexion
```

### Cycle 65
```
sifaka.critics.implementations.prompt -> sifaka.critics.implementations.prompt
```

### Cycle 66
```
sifaka.critics.base.abstract -> sifaka.critics.base.abstract
```

### Cycle 67
```
sifaka.critics.base.metadata -> sifaka.critics.base.metadata
```

### Cycle 68
```
sifaka.critics.base.protocols -> sifaka.critics.base.protocols
```

### Cycle 69
```
sifaka.critics.base.factories -> sifaka.critics.base.factories
```

### Cycle 70
```
sifaka.critics.base.implementation -> sifaka.critics.base.implementation
```

### Cycle 71
```
sifaka.utils.config.base -> sifaka.utils.config.base
```

### Cycle 72
```
sifaka.utils.errors.logging -> sifaka.utils.errors.logging
```

### Cycle 73
```
sifaka.chain.interfaces -> sifaka.chain.interfaces
```

### Cycle 74
```
sifaka.chain.plugins -> sifaka.chain.plugins
```

### Cycle 75
```
sifaka.chain.engine -> sifaka.chain.engine
```

### Cycle 76
```
sifaka.models.result -> sifaka.models.result
```

### Cycle 77
```
sifaka.chain.state -> sifaka.chain.state
```

### Cycle 78
```
sifaka.chain.managers.cache -> sifaka.chain.managers.cache
```

### Cycle 79
```
sifaka.chain.managers.retry -> sifaka.chain.managers.retry
```

### Cycle 80
```
sifaka.models.plugins -> sifaka.models.plugins
```

### Cycle 81
```
sifaka.models.providers.anthropic -> sifaka.models.providers.anthropic
```

### Cycle 82
```
sifaka.models.providers.mock -> sifaka.models.providers.mock
```

### Cycle 83
```
sifaka.models.factories -> sifaka.models.factories
```

### Cycle 84
```
sifaka.models.providers.gemini -> sifaka.models.providers.gemini
```

### Cycle 85
```
sifaka.models.utils -> sifaka.models.utils
```

### Cycle 86
```
sifaka.models.base.provider -> sifaka.models.base.provider
```

### Cycle 87
```
sifaka.models.base.types -> sifaka.models.base.types
```

### Cycle 88
```
sifaka.models.base.factory -> sifaka.models.base.factory
```

### Cycle 89
```
sifaka.adapters.plugins -> sifaka.adapters.plugins
```

### Cycle 90
```
sifaka.retrieval.plugins -> sifaka.retrieval.plugins
```

### Cycle 91
```
sifaka.retrieval.result -> sifaka.retrieval.result
```

### Cycle 92
```
sifaka.retrieval.strategies.ranking -> sifaka.retrieval.strategies.ranking
```

### Cycle 93
```
sifaka.rules.plugins -> sifaka.rules.plugins
```

### Cycle 94
```
sifaka.rules.utils -> sifaka.rules.utils
```

### Cycle 95
```
sifaka.rules.managers.validation -> sifaka.rules.managers.validation
```

### Cycle 96
```
sifaka.rules.content.tone -> sifaka.rules.content.tone
```

### Cycle 97
```
sifaka.rules.content.language -> sifaka.rules.content.language
```

### Cycle 98
```
sifaka.rules.formatting.whitespace -> sifaka.rules.formatting.whitespace
```

### Cycle 99
```
sifaka.rules.formatting.style.enums -> sifaka.rules.formatting.style.enums
```

### Cycle 100
```
sifaka.rules.formatting.style.enums -> sifaka.rules.formatting.style.factories -> sifaka.rules.formatting.style.enums
```

### Cycle 101
```
sifaka.rules.formatting.style.enums -> sifaka.rules.formatting.style.factories -> sifaka.rules.formatting.style.validators -> sifaka.rules.formatting.style.enums
```

### Cycle 102
```
sifaka.rules.formatting.style.validators -> sifaka.rules.formatting.style.validators
```

### Cycle 103
```
sifaka.rules.formatting.style.enums -> sifaka.rules.formatting.style.factories -> sifaka.rules.formatting.style.validators -> sifaka.rules.formatting.style.config -> sifaka.rules.formatting.style.enums
```

### Cycle 104
```
sifaka.rules.formatting.style.config -> sifaka.rules.formatting.style.config
```

### Cycle 105
```
sifaka.rules.formatting.style.enums -> sifaka.rules.formatting.style.factories -> sifaka.rules.formatting.style.rules -> sifaka.rules.formatting.style.enums
```

### Cycle 106
```
sifaka.rules.formatting.style.rules -> sifaka.rules.formatting.style.rules
```

### Cycle 107
```
sifaka.rules.formatting.style.enums -> sifaka.rules.formatting.style.factories -> sifaka.rules.formatting.style.rules -> sifaka.rules.formatting.style.implementations -> sifaka.rules.formatting.style.enums
```

### Cycle 108
```
sifaka.rules.formatting.style.implementations -> sifaka.rules.formatting.style.implementations
```

### Cycle 109
```
sifaka.rules.formatting.style.factories -> sifaka.rules.formatting.style.factories
```

### Cycle 110
```
sifaka.rules.formatting.style.analyzers -> sifaka.rules.formatting.style.analyzers
```

### Cycle 111
```
sifaka.rules.formatting.format.utils -> sifaka.rules.formatting.format.utils
```

### Cycle 112
```
sifaka.rules.formatting.format.base -> sifaka.rules.formatting.format.base
```

### Cycle 113
```
sifaka.classifiers.interfaces -> sifaka.classifiers.interfaces
```

### Cycle 114
```
sifaka.classifiers.plugins -> sifaka.classifiers.plugins
```

### Cycle 115
```
sifaka.classifiers.factories -> sifaka.classifiers.factories
```

### Cycle 116
```
sifaka.classifiers.engine -> sifaka.classifiers.engine
```

### Cycle 117
```
sifaka.classifiers.errors -> sifaka.classifiers.errors
```

### Cycle 118
```
sifaka.classifiers.base -> sifaka.classifiers.base
```

### Cycle 119
```
sifaka.classifiers.adapters -> sifaka.classifiers.adapters
```

### Cycle 120
```
sifaka.classifiers.implementations.factories -> sifaka.classifiers.implementations.factories
```

### Cycle 121
```
sifaka.classifiers.implementations.adapters -> sifaka.classifiers.implementations.adapters
```

### Cycle 122
```
sifaka.classifiers.implementations.content.spam -> sifaka.classifiers.implementations.content.spam
```

### Cycle 123
```
sifaka.classifiers.implementations.content.sentiment -> sifaka.classifiers.implementations.content.sentiment
```

### Cycle 124
```
sifaka.classifiers.implementations.properties.topic -> sifaka.classifiers.implementations.properties.topic
```

### Cycle 125
```
sifaka.classifiers.implementations.entities.ner -> sifaka.classifiers.implementations.entities.ner
```

### Cycle 126
```
sifaka.interfaces.classifier -> sifaka.interfaces.classifier
```

## Modules with Most Dependencies

- **sifaka.core.factories**: 22 dependencies
  - sifaka.adapters.base
  - sifaka.adapters.classifier
  - sifaka.adapters.guardrails
  - sifaka.chain.factories
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.classifiers.implementations.properties.language
  - sifaka.core.dependency.provider
  - sifaka.core.dependency.utils
  - sifaka.models.providers.openai
  - sifaka.retrieval.factories
  - ... and 12 more

- **sifaka.critics.core**: 11 dependencies
  - sifaka.core.managers.memory
  - sifaka.core.managers.prompt
  - sifaka.core.managers.response
  - sifaka.critics.core
  - sifaka.critics.managers
  - sifaka.models.core.provider
  - sifaka.models.providers
  - sifaka.utils.config.critics
  - sifaka.utils.logging
  - sifaka.utils.state
  - ... and 1 more

- **sifaka.core.base**: 10 dependencies
  - sifaka.core.base
  - sifaka.core.results
  - sifaka.utils.common
  - sifaka.utils.errors.base
  - sifaka.utils.errors.results
  - sifaka.utils.errors.safe_execution
  - sifaka.utils.logging
  - sifaka.utils.result_types
  - sifaka.utils.state
  - sifaka.utils.text

- **sifaka.models.core.provider**: 10 dependencies
  - sifaka.interfaces.client
  - sifaka.interfaces.counter
  - sifaka.interfaces.model
  - sifaka.utils.config.models
  - sifaka.utils.errors.base
  - sifaka.utils.errors.component
  - sifaka.utils.errors.safe_execution
  - sifaka.utils.logging
  - sifaka.utils.state
  - sifaka.utils.tracing

- **sifaka.adapters.classifier.adapter**: 10 dependencies
  - sifaka.adapters.base
  - sifaka.adapters.classifier
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.core.results
  - sifaka.rules.base
  - sifaka.utils
  - sifaka.utils.errors.base
  - sifaka.utils.errors.handling
  - sifaka.utils.logging
  - sifaka.utils.text

- **sifaka.models.providers.openai**: 9 dependencies
  - sifaka.interfaces.client
  - sifaka.interfaces.counter
  - sifaka.models.core.error_handling
  - sifaka.models.core.provider
  - sifaka.models.managers.openai_client
  - sifaka.models.managers.openai_token_counter
  - sifaka.models.providers.openai
  - sifaka.utils.config.models
  - sifaka.utils.logging

- **sifaka.models.providers.anthropic**: 9 dependencies
  - sifaka.interfaces.client
  - sifaka.interfaces.counter
  - sifaka.models.core.error_handling
  - sifaka.models.core.provider
  - sifaka.models.managers.anthropic_client
  - sifaka.models.managers.anthropic_token_counter
  - sifaka.models.providers.anthropic
  - sifaka.utils.config.models
  - sifaka.utils.logging

- **sifaka.adaptersdantic_ai.adapter**: 9 dependencies
  - sifaka.adapters.base
  - sifaka.core.validation
  - sifaka.critics.base.abstract
  - sifaka.rules.base
  - sifaka.rules.formatting.length
  - sifaka.utils.errors.base
  - sifaka.utils.errors.handling
  - sifaka.utils.logging
  - sifaka.utils.state

- **sifaka.adaptersdantic_ai.factory**: 9 dependencies
  - sifaka.adapters.base
  - sifaka.critics.base.abstract
  - sifaka.critics.implementations.prompt
  - sifaka.models.base
  - sifaka.models.factories
  - sifaka.rules.base
  - sifaka.rules.formatting.length
  - sifaka.utils.errors.handling
  - sifaka.utils.logging

- **sifaka.retrieval.core**: 9 dependencies
  - sifaka.core.base
  - sifaka.core.results
  - sifaka.utils.common
  - sifaka.utils.errors.base
  - sifaka.utils.errors.component
  - sifaka.utils.errors.handling
  - sifaka.utils.errors.safe_execution
  - sifaka.utils.state
  - sifaka.utils.text

- **sifaka.rules.managers.validation**: 9 dependencies
  - sifaka.core.base
  - sifaka.interfaces
  - sifaka.rules.base
  - sifaka.rules.formatting.format
  - sifaka.rules.formatting.format.markdown
  - sifaka.rules.formatting.length
  - sifaka.rules.managers.validation
  - sifaka.utils.logging
  - sifaka.utils.state

- **sifaka.classifiers.implementations.content.toxicity**: 9 dependencies
  - sifaka.classifiers.classifier
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.classifiers.implementations.content.toxicity_model
  - sifaka.utils.common
  - sifaka.utils.config.classifiers
  - sifaka.utils.logging
  - sifaka.utils.result_types
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

- **sifaka.chain.chain**: 8 dependencies
  - sifaka.chain
  - sifaka.critics
  - sifaka.interfaces.chain.components
  - sifaka.interfaces.chain.components.formatter
  - sifaka.models
  - sifaka.rules
  - sifaka.utils.config.chain
  - sifaka.utils.text

- **sifaka.models.factories**: 8 dependencies
  - sifaka.interfaces.model
  - sifaka.models.factories
  - sifaka.models.providers.anthropic
  - sifaka.models.providers.gemini
  - sifaka.models.providers.mock
  - sifaka.models.providers.openai
  - sifaka.utils.config.models
  - sifaka.utils.logging

- **sifaka.models.base.provider**: 8 dependencies
  - sifaka.interfaces.client
  - sifaka.interfaces.counter
  - sifaka.interfaces.model
  - sifaka.models.base.provider
  - sifaka.models.base.types
  - sifaka.utils.config.models
  - sifaka.utils.logging
  - sifaka.utils.tracing

- **sifaka.adapters.__init__**: 8 dependencies
  - sifaka.adapters
  - sifaka.adapters.base
  - sifaka.classifiers.implementations.content.toxicity
  - sifaka.core.results
  - sifaka.critics
  - sifaka.models
  - sifaka.rules
  - sifaka.utils.config.classifiers

- **sifaka.adapters.guardrails.adapter**: 8 dependencies
  - sifaka.adapters.base
  - sifaka.adapters.guardrails
  - sifaka.core.base
  - sifaka.rules.base
  - sifaka.utils.errors.base
  - sifaka.utils.errors.handling
  - sifaka.utils.logging
  - sifaka.utils.state

- **sifaka.retrieval.strategies.ranking**: 8 dependencies
  - sifaka.core.base
  - sifaka.retrieval.strategies.ranking
  - sifaka.utils.common
  - sifaka.utils.config.retrieval
  - sifaka.utils.errors.base
  - sifaka.utils.errors.component
  - sifaka.utils.errors.handling
  - sifaka.utils.logging

- **sifaka.retrieval.managers.query**: 8 dependencies
  - sifaka.core.base
  - sifaka.interfaces.retrieval
  - sifaka.utils.config.retrieval
  - sifaka.utils.errors.base
  - sifaka.utils.errors.component
  - sifaka.utils.errors.handling
  - sifaka.utils.logging
  - sifaka.utils.patterns

## Most Depended-Upon Modules

- **sifaka.utils.logging**: 75 dependents
- **sifaka.utils.state**: 44 dependents
- **sifaka.rules.base**: 28 dependents
- **sifaka.core.results**: 22 dependents
- **sifaka.utils.errors.base**: 19 dependents
- **sifaka.utils.config.models**: 19 dependents
- **sifaka.utils.errors.handling**: 17 dependents
- **sifaka.utils.errors.component**: 15 dependents
- **sifaka.utils.config.classifiers**: 15 dependents
- **sifaka.core.interfaces**: 14 dependents
- **sifaka.core.base**: 14 dependents
- **sifaka.utils.errors.safe_execution**: 13 dependents
- **sifaka.utils.config.critics**: 13 dependents
- **sifaka.utils.text**: 12 dependents
- **sifaka.utils.common**: 11 dependents
- **sifaka.interfaces.client**: 11 dependents
- **sifaka.interfaces.counter**: 11 dependents
- **sifaka.core.managers.memory**: 10 dependents
- **sifaka.classifiers.classifier**: 10 dependents
- **sifaka.rules.formatting.length**: 9 dependents
