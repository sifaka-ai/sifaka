"""
Named Entity Recognition (NER) classifier using spaCy.

This classifier identifies and categorizes named entities in text such as
people, organizations, locations, dates, etc.

## Architecture

NERClassifier follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods (inherited)
2. **Caching Layer**: _classify_impl() handles caching (inherited)
3. **Core Logic**: _classify_impl_uncached() implements entity recognition
4. **State Management**: Uses StateManager for internal state
"""

import importlib
from abc import abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

from typing_extensions import TypeGuard
from pydantic import PrivateAttr

from sifaka.classifiers.base import BaseClassifier
from sifaka.classifiers.models import ClassificationResult
from sifaka.classifiers.config import ClassifierConfig
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_classifier_state

logger = get_logger(__name__)


@runtime_checkable
class NEREngine(Protocol):
    """Protocol for NER engines."""

    @abstractmethod
    def process(self, text: str) -> Any: ...

    @abstractmethod
    def get_entities(self, doc: Any) -> List[Tuple[str, str, int, int]]: ...


class EntityResult:
    """Result of entity extraction operation."""

    def __init__(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        entity_count: int,
    ):
        self.text = text
        self.entities = entities
        self.entity_count = entity_count

    @property
    def entity_density(self) -> float:
        """Calculate ratio of entities to text length."""
        return self.entity_count / max(len(self.text.split()), 1)  # Avoid division by zero


class NERClassifier(BaseClassifier):
    """
    A Named Entity Recognition (NER) classifier using spaCy.

    This classifier identifies and categorizes named entities in text such as
    people, organizations, locations, dates, etc.

    Requires the 'ner' extra to be installed:
    pip install sifaka[ner]
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)

    # Define class-level constants with ClassVar annotation
    DEFAULT_LABELS: ClassVar[List[str]] = [
        "person",
        "organization",
        "location",
        "date",
        "money",
        "unknown",
    ]
    DEFAULT_COST: ClassVar[int] = 2  # Moderate cost for NLP model

    def __init__(
        self,
        name: str = "ner_classifier",
        description: str = "Identifies named entities in text",
        engine: Optional[NEREngine] = None,
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the NER classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            engine: Custom NER engine implementation
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Create config if not provided
        if config is None:
            # Extract params from kwargs if present
            params = kwargs.pop("params", {})

            # Create config with remaining kwargs
            config = ClassifierConfig(
                labels=NERClassifier.DEFAULT_LABELS,
                cost=NERClassifier.DEFAULT_COST,
                params=params,
                **kwargs,
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

        # Initialize state
        state = self._state_manager.get_state()
        state.initialized = False

        # Store engine in state if provided
        if engine is not None and self._validate_engine(engine):
            state.cache["engine"] = engine

    def _validate_engine(self, engine: Any) -> TypeGuard[NEREngine]:
        """Validate that an engine implements the required protocol."""
        if not isinstance(engine, NEREngine):
            raise ValueError(f"Engine must implement NEREngine protocol, got {type(engine)}")
        return True

    def _load_spacy(self) -> NEREngine:
        """Load the spaCy NER engine."""
        try:
            # Get state
            state = self._state_manager.get_state()

            # Check if engine is already in state
            if "engine" in state.cache:
                return state.cache["engine"]

            spacy = importlib.import_module("spacy")

            # Get model name from params or use default
            model_name = self.config.params.get("model_name", "en_core_web_sm")

            # Load the spaCy model
            nlp = spacy.load(model_name)

            # Create a wrapper that implements the NEREngine protocol
            class SpacyNERWrapper:
                def __init__(self, nlp):
                    self.nlp = nlp

                def process(self, text: str) -> Any:
                    return self.nlp(text)

                def get_entities(self, doc: Any) -> List[Tuple[str, str, int, int]]:
                    """Extract entities from a spaCy doc."""
                    return [
                        (ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents
                    ]

            # Create wrapper with spaCy model
            engine = SpacyNERWrapper(nlp)

            # Validate and store in state
            if self._validate_engine(engine):
                state.cache["engine"] = engine
                return engine

        except ImportError:
            raise ImportError(
                "spacy package is required for NERClassifier. "
                "Install it with: pip install sifaka[ner]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load spaCy NER engine: {e}")

    @property
    def entity_types(self) -> Set[str]:
        """Get the entity types recognized by this classifier."""
        # Get from config.params for consistency
        entity_types = self.config.params.get("entity_types", self.DEFAULT_LABELS)
        return set(entity_types) if isinstance(entity_types, list) else set()

    def warm_up(self) -> None:
        """Initialize the NER engine if needed."""
        # Get state
        state = self._state_manager.get_state()

        if not state.initialized:
            # Load engine if not already in state
            if "engine" not in state.cache:
                engine = self._load_spacy()
                state.cache["engine"] = engine

            # Mark as initialized
            state.initialized = True

    def _extract_entities(self, text: str) -> EntityResult:
        """
        Extract named entities from text.

        Args:
            text: Text to analyze

        Returns:
            EntityResult with entity details
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure resources are initialized
        if not state.initialized:
            self.warm_up()

        # Get engine from state
        engine = state.cache.get("engine")
        if not engine:
            raise RuntimeError("NER engine not initialized")

        # Process the text with the NER engine
        doc = engine.process(text)

        # Extract entities
        entity_tuples = engine.get_entities(doc)

        # Convert to structured format
        entities = []
        for text, label, start, end in entity_tuples:
            # Filter by entity types if specified
            if self.entity_types and label.lower() not in self.entity_types:
                continue

            entities.append(
                {
                    "text": text,
                    "type": label.lower(),
                    "start": start,
                    "end": end,
                }
            )

        return EntityResult(
            text=text,
            entities=entities,
            entity_count=len(entities),
        )

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """
        Implement NER classification logic without caching.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with entity extraction results
        """
        self.warm_up()

        try:
            # Note: Empty text is handled by BaseClassifier.classify
            # so we don't need to handle it here

            # Extract entities
            entity_result = self._extract_entities(text)

            # Determine dominant entity type
            entity_counts = {}
            for entity in entity_result.entities:
                entity_type = entity["type"]
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

            # Find the most common entity type
            dominant_type = "unknown"
            max_count = 0

            for entity_type, count in entity_counts.items():
                if count > max_count:
                    max_count = count
                    dominant_type = entity_type

            # Calculate confidence based on proportion of entities
            # Use min_confidence from config.params for consistency
            min_confidence = self.config.params.get("min_confidence", 0.5)
            # Ensure confidence is between 0 and 1
            confidence = min(
                1.0,
                max(
                    entity_result.entity_density,
                    min_confidence if entity_result.entity_count > 0 else 0.0,
                ),
            )

            # Group entities by type for better organization
            entities_by_type = {}
            for entity in entity_result.entities:
                entity_type = entity["type"]
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity)

            return ClassificationResult(
                label=dominant_type if entity_result.entity_count > 0 else "unknown",
                confidence=confidence if entity_result.entity_count > 0 else 0.0,
                metadata={
                    "entities": entity_result.entities,
                    "entities_by_type": entities_by_type,
                    "entity_count": entity_result.entity_count,
                    "entity_density": entity_result.entity_density,
                    "dominant_entity_type": (
                        dominant_type if entity_result.entity_count > 0 else None
                    ),
                },
            )

        except Exception as e:
            logger.error("Failed to extract entities: %s", e)
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "reason": "entity_extraction_error",
                },
            )

    @classmethod
    def create_with_custom_engine(
        cls,
        engine: NEREngine,
        name: str = "custom_ner_classifier",
        description: str = "Custom NER engine",
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> "NERClassifier":
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
        # Validate engine first
        if not isinstance(engine, NEREngine):
            raise ValueError(f"Engine must implement NEREngine protocol, got {type(engine)}")

        # Create config if not provided
        if config is None:
            config = ClassifierConfig(
                labels=NERClassifier.DEFAULT_LABELS,
                cost=NERClassifier.DEFAULT_COST,
                params=kwargs.pop("params", {}),
            )

        # Create instance with validated engine
        instance = cls(
            name=name,
            description=description,
            engine=engine,
            config=config,
            **kwargs,
        )

        # Initialize state
        state = instance._state.get_state()
        state.cache["engine"] = engine
        state.initialized = True

        return instance


def create_ner_classifier(
    name: str = "ner_classifier",
    description: str = "Identifies named entities in text",
    model_name: str = "en_core_web_sm",
    entity_types: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    cache_size: int = 0,
    cost: int = 2,
    **kwargs,
) -> NERClassifier:
    """
    Factory function to create a NER classifier.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        model_name: Name of the spaCy model to use
        entity_types: Optional list of entity types to recognize (filter)
        min_confidence: Minimum confidence for entity classification
        cache_size: Size of the classification cache (0 to disable)
        cost: Computational cost of this classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured NERClassifier instance
    """
    # Prepare params
    params = kwargs.pop("params", {})
    params.update(
        {
            "model_name": model_name,
            "entity_types": entity_types or NERClassifier.DEFAULT_LABELS,
            "min_confidence": min_confidence,
        }
    )

    # Create config
    config = ClassifierConfig(
        labels=NERClassifier.DEFAULT_LABELS,
        cache_size=cache_size,
        cost=cost,
        params=params,
    )

    # Create and return classifier
    return NERClassifier(
        name=name,
        description=description,
        config=config,
        **kwargs,
    )
