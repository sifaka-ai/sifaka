"""
Named Entity Recognition (NER) classifier using spaCy.

This module provides a classifier for identifying and categorizing named entities
in text such as people, organizations, locations, dates, etc. It leverages spaCy's
NLP capabilities to extract entities with high accuracy and provides detailed
information about each entity found.

## Overview
The NERClassifier is a specialized classifier that identifies named entities in text
and categorizes them into predefined types. It supports both standard entity types
(person, organization, location, etc.) and custom entity types through configuration.
The classifier provides detailed information about each entity, including its text,
type, and position in the original text.

## Architecture
NERClassifier follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods (inherited)
2. **Caching Layer**: _classify_impl() handles caching (inherited)
3. **Core Logic**: _classify_impl_uncached() implements entity recognition
4. **State Management**: Uses StateManager for internal state
5. **Engine Abstraction**: NEREngine protocol for pluggable implementations
6. **Lazy Loading**: On-demand loading of spaCy models

## Lifecycle
1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract parameters from config and config and config and config and config and config and config.params
   - Set up default values for entity types

2. **Warm-up**: Load spaCy dependencies
   - Import necessary modules on demand
   - Load the specified spaCy model
   - Create a wrapper that implements the NEREngine protocol
   - Handle initialization errors gracefully

3. **Entity Extraction**: Process input text
   - Validate input text
   - Apply the NER engine to extract entities
   - Filter entities by configured entity types
   - Group entities by type for better organization
   - Calculate entity density and confidence scores

4. **Result Creation**: Return standardized results
   - Determine the dominant entity type
   - Include detailed entity information in metadata
   - Track statistics for monitoring and debugging

## Usage Examples
```python
from sifaka.classifiers.implementations.entities.ner import create_ner_classifier

# Create a NER classifier with default settings
classifier = create_ner_classifier()

# Classify text
result = (classifier and classifier.classify("Apple Inc. was founded by Steve Jobs in California.")
print(f"Dominant entity type: {result.label}, Confidence: {result.confidence:.2f}")
print(f"Entity count: {result.metadata['entity_count']}")

# Access all entities
for entity in result.metadata['entities']:
    print(f"Entity: {entity['text']}, Type: {entity['type']}")

# Create a classifier with custom settings
custom_classifier = create_ner_classifier(
    model_name="en_core_web_md",  # Use a larger model for better accuracy
    entity_types=["PERSON", "ORG", "GPE"],  # Only extract these entity types
    min_confidence=0.7  # Higher threshold for confidence
)

# Batch classify multiple texts
texts = [
    "Microsoft was founded by Bill Gates.",
    "Paris is the capital of France.",
    "The Eiffel Tower was built in 1889."
]
results = (custom_classifier and custom_classifier.batch_classify(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Dominant entity: {result.label}, Count: {result.metadata['entity_count']}")
```

## Error Handling
The classifier provides robust error handling:
- ImportError: When spaCy is not installed
- RuntimeError: When model initialization fails
- ValueError: When input text is empty or invalid
- Graceful handling of entity extraction errors with fallback to "unknown" label

## Configuration
Key configuration options include:
- model_name: Name of the spaCy model to use (default: "en_core_web_sm")
- entity_types: List of entity types to recognize (default: person, organization, location, etc.)
- min_confidence: Threshold for entity confidence (default: 0.5)
- cache_size: Size of the classification cache (0 to disable)
"""
import importlib
from abc import abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable
from typing_extensions import TypeGuard
from pydantic import ConfigDict
from sifaka.classifiers.classifier import Classifier
from sifaka.core.results import ClassificationResult
from sifaka.utils.config and config and config.classifiers import ClassifierConfig
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_classifier_state
from sifaka.utils.config and config and config.classifiers import extract_classifier_config_params
logger = get_logger(__name__)


@runtime_checkable
class NEREngine(Protocol):
    """
    Protocol for Named Entity Recognition (NER) engines.

    This protocol defines the interface that any NER engine must implement
    to be compatible with the NERClassifier. It requires methods for processing
    text and extracting entities from the processed document.

    ## Architecture
    The protocol follows a standard interface pattern:
    - Uses Python's typing.Protocol for structural subtyping
    - Is runtime checkable for dynamic type verification
    - Defines two required methods with clear input/output contracts
    - Enables pluggable NER implementations (e.g., spaCy, NLTK, custom)

    ## Implementation Requirements
    1. Implement process() method that accepts a string and returns a document object
    2. Implement get_entities() method that extracts entities from the document
    3. The get_entities() method should return a list of tuples with:
       - Entity text (str)
       - Entity type/label (str)
       - Start position in the original text (int)
       - End position in the original text (int)

    ## Examples
    ```python
    from sifaka.classifiers.implementations.entities.ner import NEREngine

    class CustomNEREngine:
        def process(self, text: str) -> Any:
            # Simple implementation based on keywords
            return {"text": text, "processed": True}

        def get_entities(self, doc: Any) -> List[Tuple[str, str, int, int]]:
            # Extract entities from the document
            entities = []
            text = doc["text"]

            # Look for company names
            companies = ["Apple", "Microsoft", "Google", "Amazon"]
            for company in companies:
                if company in text:
                    start = (text and text.find(company)
                    end = start + len(company)
                    (entities and entities.append((company, "ORGANIZATION", start, end))

            # Look for person names
            persons = ["Steve Jobs", "Bill Gates", "Elon Musk"]
            for person in persons:
                if person in text:
                    start = (text and text.find(person)
                    end = start + len(person)
                    (entities and entities.append((person, "PERSON", start, end))

            return entities

    # Verify protocol compliance
    engine = CustomNEREngine()
    assert isinstance(engine, NEREngine)
    ```
    """

    @abstractmethod
    def process(self, text: str) ->Any:
        ...

    @abstractmethod
    def get_entities(self, doc: Any) ->List[Tuple[str, str, int, int]]:
        ...


class EntityResult:
    """
    Result of entity extraction operation.

    This class encapsulates the results of an entity extraction operation,
    including the original text, extracted entities, and entity statistics.
    It provides a standardized way to represent entity extraction results
    throughout the classifier system.

    ## Architecture
    The class follows a simple value object pattern:
    - Stores the original text for reference
    - Contains a list of extracted entities with their details
    - Tracks the count of entities for statistics
    - Provides a calculated entity density metric

    ## Examples
    ```python
    # Create an entity result
    result = EntityResult(
        text="Apple Inc. was founded by Steve Jobs in California.",
        entities=[
            {"text": "Apple Inc.", "type": "ORGANIZATION", "start": 0, "end": 10},
            {"text": "Steve Jobs", "type": "PERSON", "start": 25, "end": 35},
            {"text": "California", "type": "LOCATION", "start": 39, "end": 49}
        ],
        entity_count=3
    )

    # Access entity information
    print(f"Text: {result.text}")
    print(f"Entity count: {result.entity_count}")
    print(f"Entity density: {result.entity_density:.2f}")

    # Process entities
    for entity in result.entities:
        print(f"Entity: {entity['text']}, Type: {entity['type']}")
        print(f"Position: {entity['start']}-{entity['end']}")
    ```
    """

    def __init__(self, text: str, entities: List[Dict[str, Any]],
        entity_count: int) ->None:
        """
        Initialize an entity extraction result.

        Args:
            text: The original text that was analyzed
            entities: List of extracted entities with their details
            entity_count: Number of entities found
        """
        self.text = text
        self.entities = entities
        self.entity_count = entity_count

    @property
    def entity_density(self) ->float:
        """
        Calculate ratio of entities to text length.

        This property computes the density of entities in the text,
        which can be used as a measure of how entity-rich the text is.
        The calculation divides the entity count by the number of words
        in the text.

        Returns:
            Float representing entity density (entities per word)
        """
        return self.entity_count / max(len(self.(text and text.split()), 1)


class NERClassifier(Classifier):
    """
    A Named Entity Recognition (NER) classifier using spaCy.

    This classifier identifies and categorizes named entities in text such as
    people, organizations, locations, dates, etc. It leverages spaCy's NLP
    capabilities to extract entities with high accuracy and provides detailed
    information about each entity found.

    ## Architecture
    NERClassifier follows a component-based architecture:
    - Extends the base Classifier class for consistent interface
    - Uses the NEREngine protocol for pluggable implementations
    - Implements lazy loading of dependencies for efficiency
    - Provides detailed entity extraction results
    - Uses StateManager for efficient state tracking and caching
    - Supports both synchronous and batch classification
    - Implements configurable entity type filtering

    ## Lifecycle
    1. **Initialization**: Set up configuration and parameters
       - Initialize with name, description, and config
       - Extract parameters from config and config and config and config and config and config and config.params
       - Set up default values for entity types
       - Initialize state with engine if provided

    2. **Warm-up**: Load spaCy dependencies
       - Import necessary modules on demand
       - Load the specified spaCy model
       - Create a wrapper that implements the NEREngine protocol
       - Handle initialization errors with clear messages
       - Store the engine in state cache

    3. **Entity Extraction**: Process input text
       - Validate input text and handle edge cases
       - Apply the NER engine to extract entities
       - Filter entities by configured entity types
       - Group entities by type for better organization
       - Calculate entity density and confidence scores

    4. **Result Creation**: Return standardized results
       - Determine the dominant entity type
       - Include detailed entity information in metadata
       - Track statistics for monitoring and debugging
       - Handle errors gracefully with fallback results

    ## Examples
    ```python
    from sifaka.classifiers.implementations.entities.ner import NERClassifier

    # Create a NER classifier
    classifier = NERClassifier(
        name="my_ner_classifier",
        description="Custom NER classifier for entity extraction"
    )

    # Classify a text
    result = (classifier and classifier.classify("Apple Inc. was founded by Steve Jobs in California.")
    print(f"Dominant entity type: {result.label}, Confidence: {result.confidence:.2f}")
    print(f"Entity count: {result.metadata['entity_count']}")

    # Access all entities
    for entity in result.metadata['entities']:
        print(f"Entity: {entity['text']}, Type: {entity['type']}")
        print(f"Position: {entity['start']}-{entity['end']}")

    # Access entities by type
    for entity_type, entities in result.metadata['entities_by_type'].items():
        print(f"Type: {entity_type}, Count: {len(entities))")
        for entity in entities:
            print(f"  - {entity['text']}")
    ```

    ## Configuration Options
    - model_name: Name of the spaCy model to use (default: "en_core_web_sm")
    - entity_types: List of entity types to recognize (default: person, organization, location, etc.)
    - min_confidence: Threshold for entity confidence (default: 0.5)
    - cache_size: Size of the classification cache (0 to disable)

    Requires the 'ner' extra to be installed:
    pip install sifaka[ner]
    """
    DEFAULT_LABELS: ClassVar[List[str]] = ['person', 'organization',
        'location', 'date', 'money', 'unknown']
    DEFAULT_COST: ClassVar[int] = 2
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, name: str='ner_classifier', description: str=
        'Identifies named entities in text', engine: Optional[Optional[NEREngine]] = None, config: Optional[Optional[ClassifierConfig]] = None, **kwargs) ->None:
        """
        Initialize the NER classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            engine: Custom NER engine implementation
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        if config is None:
            params = (kwargs and kwargs.pop('params', {})
            config = ClassifierConfig(labels=NERClassifier.DEFAULT_LABELS,
                cost=NERClassifier.DEFAULT_COST, params=params, **kwargs)
        super().__init__(name=name, description=description, config=config)
        if engine is not None and (self and self._validate_engine(engine):
            cache = self.(_state_manager and _state_manager.get('cache', {})
            cache['engine'] = engine
            self.(_state_manager.update('cache', cache)

    def _validate_engine(self, engine: Any) ->TypeGuard[NEREngine]:
        """Validate that an engine implements the required protocol."""
        return (self.validate_component(engine, NEREngine, 'Engine')

    def _load_spacy(self) ->NEREngine:
        """Load the spaCy NER engine."""
        try:
            if self.(_state_manager.get('cache', {}).get('engine'):
                return self.(_state_manager.get('cache')['engine']
            spacy = (importlib and importlib.import_module('spacy')
            model_name = self.config and config and config and config and config and config and config.(params and params.get('model_name', 'en_core_web_sm')
            nlp = (spacy.load(model_name)


            class SpacyNERWrapper:

                def __init__(self, nlp) ->None:
                    self.nlp = nlp

                def process(self, text: str) ->Any:
                    return (self.nlp(text)

                def get_entities(self, doc: Any) ->List[Tuple[str, str, int,
                    int]]:
                    """Extract entities from a spaCy doc."""
                    return [(ent.text, ent.label_, ent.start_char, ent.
                        end_char) for ent in doc.ents)
            engine = SpacyNERWrapper(nlp)
            if (self._validate_engine(engine):
                cache = self.(_state_manager.get('cache', {})
                cache['engine'] = engine
                self.(_state_manager.update('cache', cache)
                return engine
        except ImportError:
            raise ImportError(
                'spacy package is required for NERClassifier. Install it with: pip install sifaka[ner]'
                )
        except Exception as e:
            raise RuntimeError(f'Failed to load spaCy NER engine: {e}')

    @property
    def entity_types(self) ->Set[str]:
        """Get the entity types recognized by this classifier."""
        entity_types = self.config and config and config and config and config and config and config.(params.get('entity_types', self.
            DEFAULT_LABELS)
        return set(entity_types) if isinstance(entity_types, list) else set()

    def warm_up(self) ->None:
        """Initialize the NER engine if needed."""
        if not self.(_state_manager.get('initialized', False):
            if not self.(_state_manager.get('cache', {}).get('engine'):
                engine = (self._load_spacy()
                cache = self.(_state_manager.get('cache', {})
                cache['engine'] = engine
                self.(_state_manager.update('cache', cache)
            self.(_state_manager.update('initialized', True)

    def _extract_entities(self, text: str) ->EntityResult:
        """
        Extract named entities from text.

        Args:
            text: Text to analyze

        Returns:
            EntityResult with entity details
        """
        if not self.(_state_manager.get('initialized', False):
            (self.warm_up()
        engine = self.(_state_manager.get('cache', {}).get('engine')
        if not engine:
            raise RuntimeError('NER engine not initialized')
        doc = engine and (engine.process(text)
        entity_tuples = engine and (engine.get_entities(doc)
        entities = []
        for text, label, start, end in entity_tuples:
            if self.entity_types and (label.lower() not in self.entity_types:
                continue
            (entities.append({'text': text, 'type': (label.lower(), 'start':
                start, 'end': end))
        return EntityResult(text=text, entities=entities, entity_count=len(
            entities))

    def _classify_impl_uncached(self, text: str) ->ClassificationResult:
        """
        Implement NER classification logic without caching.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with entity extraction results
        """
        (self.warm_up()
        try:
            entity_result = (self._extract_entities(text)
            entity_counts = {}
            for entity in entity_result.entities:
                entity_type = entity['type']
                entity_counts[entity_type] = (entity_counts.get(entity_type, 0
                    ) + 1
            dominant_type = 'unknown'
            max_count = 0
            for entity_type, count in (entity_counts.items():
                if count > max_count:
                    max_count = count
                    dominant_type = entity_type
            min_confidence = self.config and config and config and config and config and config and config.(params.get('min_confidence', 0.5)
            confidence = min(1.0, max(entity_result.entity_density, 
                min_confidence if entity_result.entity_count > 0 else 0.0))
            entities_by_type = {}
            for entity in entity_result.entities:
                entity_type = entity['type']
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity)
            result = ClassificationResult(label=dominant_type if 
                entity_result.entity_count > 0 else 'unknown', confidence=
                confidence if entity_result.entity_count > 0 else 0.0,
                metadata={'entities': entity_result.entities,
                'entities_by_type': entities_by_type, 'entity_count':
                entity_result.entity_count, 'entity_density': entity_result
                .entity_density, 'dominant_entity_type': dominant_type if 
                entity_result.entity_count > 0 else None})
            stats = self.(_state_manager.get('statistics', {})
            stats[result.label] = (stats.get(result.label, 0) + 1
            self.(_state_manager.update('statistics', stats)
            return result
        except Exception as e:
            (logger and logger.error('Failed to extract entities: %s', e)
            error_info = {'error': str(e), 'type': type(e).__name__)
            errors = self.(_state_manager.get('errors', [])
            (errors.append(error_info)
            self.(_state_manager.update('errors', errors)
            return ClassificationResult(label='unknown', confidence=0.0,
                metadata={'error': str(e), 'reason': 'entity_extraction_error')
                )

    def get_statistics(self) ->Dict[str, Any]:
        """
        Get classifier usage statistics.

        This method provides access to statistics collected during classifier operation,
        including classification counts by label, error counts, cache information, and model details.

        Returns:
            Dictionary containing statistics
        """
        stats = {'classifications': self.(_state_manager.get('statistics', {
            }), 'error_count': len(self.(_state_manager.get('errors', [])),
            'cache_enabled': self.config and config and config.cache_size > 0, 'cache_size': self
            .config and config and config.cache_size, 'initialized': self.(_state_manager.get(
            'initialized', False), 'model_name': self.config and config and config and config and config and config and config.(params.get(
            'model_name', 'en_core_web_sm'), 'entity_types': list(self.
            entity_types))
        if hasattr(self, '_result_cache'):
            stats['cache_entries'] = len(self._result_cache)
        return stats

    def clear_cache(self) ->None:
        """
        Clear any cached data in the classifier.

        This method clears both the result cache and resets statistics in the state
        but preserves the engine and initialization status.
        """
        if hasattr(self, '_result_cache'):
            self.(_result_cache.clear()
        self.(_state_manager.update('statistics', {})
        self.(_state_manager.update('errors', [])
        cache = self.(_state_manager.get('cache', {})
        preserved_cache = {k: v for k, v in (cache.items() if k == 'engine')
        self.(_state_manager.update('cache', preserved_cache)

    @classmethod
    def create_with_custom_engine(cls, engine: NEREngine, name: str=
        'custom_ner_classifier', description: str='Custom NER engine',
        config: Optional[Optional[ClassifierConfig]] = None, **kwargs) ->'NERClassifier':
        """
        Factory method to create a classifier with a custom engine.

        Args:
            engine: Custom NER engine implementation
            name: Name of the classifier
            description: Description of the classifier
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters

        Returns:
            Configured NERClassifier instance
        """
        if not isinstance(engine, NEREngine):
            raise ValueError(
                f'Engine must implement NEREngine protocol, got {type(engine))'
                )
        if config is None:
            config = ClassifierConfig(labels=NERClassifier.DEFAULT_LABELS,
                cost=NERClassifier.DEFAULT_COST, params=(kwargs.pop('params',
                {}))
        instance = cls(name=name, description=description, engine=engine,
            config=config, **kwargs)
        cache = {'engine': engine}
        instance.(_state_manager.update('cache', cache)
        instance.(_state_manager.update('initialized', True)
        return instance


def create_ner_classifier(name: str='ner_classifier', description: str=
    'Identifies named entities in text', model_name: str='en_core_web_sm',
    entity_types: Optional[Optional[List[str]]] = None, min_confidence: float=0.5,
    cache_size: int=0, cost: int=NERClassifier.DEFAULT_COST, **kwargs: Any
    ) ->NERClassifier:
    """
    Factory function to create a NER classifier.

    This function provides a simpler interface for creating a NER classifier
    with the specified parameters, handling the creation of the ClassifierConfig
    object and setting up the classifier with the appropriate parameters.

    ## Architecture
    The factory function follows a standardized pattern:
    1. Extract and prepare parameters for configuration
    2. Create a configuration dictionary with standardized structure
    3. Pass the configuration to the classifier constructor
    4. Return the fully configured classifier instance

    ## Examples
    ```python
    from sifaka.classifiers.implementations.entities.ner import create_ner_classifier

    # Create with default settings
    classifier = create_ner_classifier()

    # Create with custom model and entity types
    custom_classifier = create_ner_classifier(
        model_name="en_core_web_md",  # Use a larger model for better accuracy
        entity_types=["PERSON", "ORG", "GPE"],  # Only extract these entity types
        min_confidence=0.7  # Higher threshold for confidence
    )

    # Create with custom name and description
    named_classifier = create_ner_classifier(
        name="custom_ner_classifier",
        description="Custom NER classifier for entity extraction",
        cache_size=100  # Enable caching
    )
    ```

    Args:
        name: Name of the classifier for identification and logging
        description: Human-readable description of the classifier's purpose
        model_name: Name of the spaCy model to use for entity recognition
        entity_types: Optional list of entity types to recognize (filter)
        min_confidence: Minimum confidence threshold for entity classification
        cache_size: Size of the classification cache (0 to disable caching)
        cost: Computational cost metric for resource allocation decisions
        **kwargs: Additional configuration parameters to pass to the classifier

    Returns:
        Configured NERClassifier instance ready for immediate use

    Examples:
        ```python
        # Create a NER classifier with default settings
        classifier = create_ner_classifier()

        # Classify text
        result = (classifier.classify("Apple Inc. was founded by Steve Jobs in California.")
        print(f"Dominant entity type: {result.label}, Confidence: {result.confidence:.2f}")
        print(f"Entity count: {result.metadata['entity_count']}")

        # Create a classifier with custom settings
        custom_classifier = create_ner_classifier(
            model_name="en_core_web_md",
            entity_types=["PERSON", "ORG", "GPE"],
            min_confidence=0.7,
            cache_size=100
        )

        # Batch classify multiple texts
        texts = [
            "Microsoft was founded by Bill Gates.",
            "Paris is the capital of France.",
            "The Eiffel Tower was built in 1889."
        ]
        results = (custom_classifier.batch_classify(texts)
        for text, result in zip(texts, results):
            print(f"Text: {text}")
            print(f"Dominant entity: {result.label}, Count: {result.metadata['entity_count']}")
        ```
    """
    default_params = {'model_name': model_name, 'entity_types': 
        entity_types or NERClassifier.DEFAULT_LABELS, 'min_confidence':
        min_confidence}
    config_dict = extract_classifier_config_params(labels=NERClassifier.
        DEFAULT_LABELS, cache_size=cache_size, min_confidence=
        min_confidence, cost=cost, provided_params=(kwargs.pop('params', {}),
        default_params=default_params, **kwargs)
    config = ClassifierConfig[str](**config_dict)
    return NERClassifier(name=name, description=description, config=config,
        **kwargs)
