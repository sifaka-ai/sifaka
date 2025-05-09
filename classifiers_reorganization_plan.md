# Classifiers Reorganization Plan

## Current Structure
The current classifiers directory has a flat structure with all classifier implementations in the root directory:

```
/sifaka/classifiers/
├── __init__.py
├── base.py
├── bias.py
├── genre.py
├── language.py
├── ner.py
├── profanity.py
├── readability.py
├── sentiment.py
├── spam.py
├── topic.py
└── toxicity.py
```

## Proposed Structure
The proposed structure follows the pattern used in the critics and chain directories, with a more modular organization:

```
/sifaka/classifiers/
├── __init__.py       # Public API exports
├── base.py           # Base classifier implementations
├── config.py         # Classifier configuration
├── models.py         # Classifier models
├── interfaces/       # Define protocol interfaces
│   ├── __init__.py
│   └── classifier.py # Classifier protocol
├── managers/         # Component managers
│   ├── __init__.py
│   └── state.py      # State management
├── strategies/       # Strategy implementations
│   ├── __init__.py
│   └── caching.py    # Caching strategies
├── implementations/  # Concrete classifier implementations
│   ├── __init__.py
│   ├── content/      # Content analysis classifiers
│   │   ├── __init__.py
│   │   ├── bias.py         # Bias detector
│   │   ├── profanity.py    # Profanity classifier
│   │   ├── sentiment.py    # Sentiment classifier
│   │   ├── spam.py         # Spam classifier
│   │   └── toxicity.py     # Toxicity classifier
│   ├── properties/   # Text properties classifiers
│   │   ├── __init__.py
│   │   ├── genre.py        # Genre classifier
│   │   ├── language.py     # Language classifier
│   │   ├── readability.py  # Readability classifier
│   │   └── topic.py        # Topic classifier
│   └── entities/     # Entity analysis classifiers
│       ├── __init__.py
│       └── ner.py          # Named entity recognition
└── factories.py      # Factory functions
```

## Migration Steps

1. **Create Directory Structure**
   - Create the new directory structure with all necessary subdirectories

2. **Move Base Components**
   - Extract configuration code from base.py to config.py
   - Extract model definitions to models.py
   - Update imports in base.py

3. **Create Interface Protocols**
   - Create classifier.py in interfaces/ with protocol definitions
   - Move protocol definitions from base.py

4. **Create Managers**
   - Create state.py in managers/ for state management
   - Move state management code from base.py and individual classifiers

5. **Create Strategies**
   - Create caching.py in strategies/ for caching strategies
   - Move caching logic from base.py

6. **Move Implementations**
   - Move each classifier to its appropriate category in implementations/
   - Update imports in each file

7. **Create Factory Functions**
   - Move factory functions to factories.py
   - Update imports in __init__.py

8. **Update Public API**
   - Update __init__.py to export from new locations
   - Maintain backward compatibility

## Benefits

1. **Improved Organization**: Clear separation of concerns with modular components
2. **Better Maintainability**: Easier to understand and maintain the codebase
3. **Consistency**: Consistent structure across the codebase
4. **Extensibility**: Easier to add new classifiers and components
5. **Testability**: Easier to test individual components

## Implementation Timeline

1. **Phase 1**: Create directory structure and move base components
2. **Phase 2**: Create interfaces, managers, and strategies
3. **Phase 3**: Move implementations to appropriate categories
4. **Phase 4**: Create factory functions and update public API
