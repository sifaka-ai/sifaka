"""
Tests for the NER classifier.
"""

import pytest
from unittest.mock import MagicMock, patch

from sifaka.classifiers.ner import (
    NERClassifier,
    NEREngine,
    EntityResult,
    create_ner_classifier,
)
from sifaka.classifiers.base import ClassificationResult


class MockNEREngine:
    """Mock NER engine for testing."""

    def process(self, text):
        """Process text and return a mock document."""
        return text

    def get_entities(self, doc):
        """Get entities from a document."""
        # Return some mock entities
        if "person" in doc.lower():
            return [("John Doe", "PERSON", 0, 8)]
        elif "organization" in doc.lower():
            return [("Acme Corp", "ORG", 0, 9)]
        elif "location" in doc.lower():
            return [("New York", "LOC", 0, 8)]
        else:
            return []


def test_ner_classifier_initialization():
    """Test NER classifier initialization."""
    # Create a classifier
    classifier = NERClassifier(name="test_ner")
    
    # Check that state is initialized correctly
    assert classifier._state is not None
    
    # State should not be initialized yet
    state = classifier._state.get_state()
    assert not state.initialized
    
    # Cache should be empty
    assert "engine" not in state.cache


def test_ner_classifier_warm_up():
    """Test NER classifier warm up."""
    # Create a mock engine
    mock_engine = MockNEREngine()
    
    # Create a classifier with the mock engine
    classifier = NERClassifier(name="test_ner", engine=mock_engine)
    
    # Warm up the classifier
    classifier.warm_up()
    
    # Check that state is initialized
    state = classifier._state.get_state()
    assert state.initialized
    
    # Check that engine is in cache
    assert "engine" in state.cache
    assert state.cache["engine"] is mock_engine


def test_extract_entities():
    """Test entity extraction."""
    # Create a mock engine
    mock_engine = MockNEREngine()
    
    # Create a classifier with the mock engine
    classifier = NERClassifier(name="test_ner", engine=mock_engine)
    
    # Extract entities
    result = classifier._extract_entities("This text mentions a person named John Doe")
    
    # Check result
    assert isinstance(result, EntityResult)
    assert result.entity_count == 1
    assert result.entities[0]["text"] == "John Doe"
    assert result.entities[0]["type"] == "person"


def test_classify():
    """Test classification."""
    # Create a mock engine
    mock_engine = MockNEREngine()
    
    # Create a classifier with the mock engine
    classifier = NERClassifier(name="test_ner", engine=mock_engine)
    
    # Classify text
    result = classifier.classify("This text mentions a person named John Doe")
    
    # Check result
    assert isinstance(result, ClassificationResult)
    assert result.label == "person"
    assert result.confidence > 0
    assert "entities" in result.metadata
    assert result.metadata["entity_count"] == 1


def test_empty_text():
    """Test classification with empty text."""
    # Create a mock engine
    mock_engine = MockNEREngine()
    
    # Create a classifier with the mock engine
    classifier = NERClassifier(name="test_ner", engine=mock_engine)
    
    # Classify empty text
    result = classifier.classify("")
    
    # Check result
    assert isinstance(result, ClassificationResult)
    assert result.label == "unknown"
    assert result.confidence == 0.0


def test_factory_function():
    """Test factory function."""
    # Mock the _load_spacy method to avoid actual loading
    with patch.object(NERClassifier, '_load_spacy', return_value=MockNEREngine()):
        # Create a classifier using the factory function
        classifier = create_ner_classifier(
            name="test_factory",
            model_name="en_core_web_sm",
            entity_types=["person", "organization"],
            min_confidence=0.7,
        )
        
        # Check classifier
        assert classifier.name == "test_factory"
        assert "model_name" in classifier.config.params
        assert classifier.config.params["model_name"] == "en_core_web_sm"
        assert "entity_types" in classifier.config.params
        assert "min_confidence" in classifier.config.params
        assert classifier.config.params["min_confidence"] == 0.7
