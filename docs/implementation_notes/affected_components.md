# Components Using Direct State Pattern

This document lists all implementation classes that use direct `_state` instead of `_state_manager` pattern.

## Classifier Implementations

1. **ToxicityClassifierImplementation** ✅
   - Location: `/sifaka/classifiers/toxicity.py`
   - Updated to use standardized state management pattern:
     ```python
     # State management using StateManager
     _state_manager = PrivateAttr(default_factory=create_classifier_state)

     def __init__(self, config: ClassifierConfig) -> None:
         self.config = config
         # State is managed by StateManager, no need to initialize here
     ```

2. **SentimentClassifierImplementation** ✅
   - Location: `/sifaka/classifiers/sentiment.py`
   - Updated to use standardized state management pattern

3. **BiasDetectorImplementation** ✅
   - Location: `/sifaka/classifiers/bias.py`
   - Updated to use standardized state management pattern

4. **SpamClassifierImplementation** ✅
   - Location: `/sifaka/classifiers/spam.py`
   - Updated to use standardized state management pattern

5. **ProfanityClassifierImplementation** ✅
   - Location: `/sifaka/classifiers/profanity.py`
   - Updated to use standardized state management pattern

6. **NERClassifierImplementation** ✅
   - Location: `/sifaka/classifiers/ner.py`
   - Updated to use standardized state management pattern

7. **GenreClassifierImplementation** ✅
   - Location: `/sifaka/classifiers/genre.py`
   - Updated to use standardized state management pattern

8. **ReadabilityClassifierImplementation** ✅
   - Location: `/sifaka/classifiers/readability.py`
   - Updated to use standardized state management pattern

## Critic Implementations

1. **PromptCriticImplementation**
   - Location: `/sifaka/critics/implementations/prompt_implementation.py`
   - Current pattern:
     ```python
     def __init__(self, config, llm_provider, prompt_factory=None):
         self.config = config
         self._state = CriticState()
         # Store components in state
         self._state.model = llm_provider
         # ...
     ```
   - Needs to be updated to use `_state_manager = PrivateAttr(default_factory=create_critic_state)`

2. **ReflexionCriticImplementation**
   - Location: `/sifaka/critics/implementations/reflexion_implementation.py`
   - Current pattern: Uses direct `_state` initialization in `__init__`

3. **SelfRefineCriticImplementation**
   - Location: `/sifaka/critics/implementations/self_refine_implementation.py`
   - Current pattern: Uses direct `_state` initialization in `__init__`

4. **SelfRAGCriticImplementation**
   - Location: `/sifaka/critics/implementations/self_rag_implementation.py`
   - Current pattern: Uses direct `_state` initialization in `__init__`

5. **ConstitutionalCriticImplementation**
   - Location: `/sifaka/critics/implementations/constitutional_implementation.py`
   - Current pattern: Uses direct `_state` initialization in `__init__`

6. **FeedbackCriticImplementation**
   - Location: `/sifaka/critics/implementations/lac_implementation.py`
   - Current pattern: Uses direct `_state` initialization in `__init__`

7. **ValueCriticImplementation**
   - Location: `/sifaka/critics/implementations/lac_implementation.py`
   - Current pattern: Uses direct `_state` initialization in `__init__`

8. **LACCriticImplementation**
   - Location: `/sifaka/critics/implementations/lac_implementation.py`
   - Current pattern: Uses direct `_state` initialization in `__init__`

## Rule Implementations

1. **LengthRuleValidator**
   - Location: `/sifaka/rules/formatting/length.py`
   - Current pattern: May use direct `_state` initialization

2. **FormatRuleValidator**
   - Location: `/sifaka/rules/formatting/format.py`
   - Current pattern: May use direct `_state` initialization

3. **ProhibitedContentRuleValidator**
   - Location: `/sifaka/rules/content/prohibited.py`
   - Current pattern: May use direct `_state` initialization

## Inconsistent Initialization Timing

Components with inconsistent initialization timing:

1. Some components initialize state in `__init__`:
   ```python
   def __init__(self, config):
       self.config = config
       self._state = ClassifierState()
       self._state.initialized = False
   ```

2. Others initialize in `warm_up()`:
   ```python
   def warm_up(self):
       if not self._state_manager.is_initialized:
           state = self._state_manager.initialize()
           # Initialize resources
           state.initialized = True
   ```

## State Access Patterns

Variations of state access patterns:

1. Direct state access:
   ```python
   self._state.model = self._load_model()
   self._state.initialized = True
   ```

2. State manager access:
   ```python
   state = self._state_manager.get_state()
   state.model = self._load_model()
   state.initialized = True
   ```

3. Caching patterns:
   ```python
   # Direct state caching
   self._state.cache[key] = value

   # State manager caching
   state = self._state_manager.get_state()
   state.cache[key] = value
   ```
