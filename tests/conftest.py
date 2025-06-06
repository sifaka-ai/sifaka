"""Global test configuration and fixtures for Sifaka tests.

This module provides shared fixtures and configuration for all Sifaka tests,
including mock objects, sample data, and test utilities.
"""

from pathlib import Path
import sys
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional

# To obtain modules from sifaka
if (Path(__file__).resolve().parents[1]).exists():
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from sifaka.classifiers.base import ClassificationResult
from sifaka.core.thought import SifakaThought


# Test event loop configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Sample text data for testing classifiers
@pytest.fixture
def sample_texts():
    """Provide sample texts for classifier testing."""
    return {
        "neutral": "This is a neutral statement about the weather today.",
        "positive": "I absolutely love this amazing product! It's fantastic and wonderful!",
        "negative": "This is terrible and I hate it completely. Worst experience ever.",
        "toxic": "You are stupid and worthless, go kill yourself.",
        "non_toxic": "I disagree with your opinion, but I respect your right to have it.",
        "spam": "URGENT!!! WIN $1000000 NOW!!! CLICK HERE!!! FREE MONEY!!!",
        "ham": "Hi John, can we schedule a meeting for tomorrow at 2 PM?",
        "biased": "All women are bad drivers and shouldn't be allowed on the road.",
        "unbiased": "Traffic safety requires attention from all drivers regardless of gender.",
        "english": "Hello, how are you doing today?",
        "spanish": "Hola, ¿cómo estás hoy?",
        "french": "Bonjour, comment allez-vous aujourd'hui?",
        "simple": "The cat sat on the mat.",
        "complex": "The multifaceted implications of quantum mechanical phenomena necessitate comprehensive analysis.",
        "happy": "I'm so excited and joyful about this wonderful news!",
        "sad": "I feel devastated and heartbroken about this terrible loss.",
        "angry": "I'm furious and outraged by this completely unacceptable behavior!",
        "question": "What time is the meeting scheduled for tomorrow?",
        "statement": "The meeting is scheduled for tomorrow at 2 PM.",
        "request": "Please send me the report by end of day.",
        "greeting": "Hello there! How are you doing today?",
        "empty": "",
        "whitespace": "   \n\t   ",
        "very_long": "This is a very long text. " * 100,
        "special_chars": "Hello! @#$%^&*()_+ 123 ñáéíóú",
    }


# Mock classification results
@pytest.fixture
def mock_classification_results():
    """Provide mock classification results for testing."""
    return {
        "positive_sentiment": ClassificationResult(
            label="positive",
            confidence=0.85,
            metadata={"method": "test", "model": "mock"},
            processing_time_ms=10.5,
        ),
        "negative_sentiment": ClassificationResult(
            label="negative",
            confidence=0.92,
            metadata={"method": "test", "model": "mock"},
            processing_time_ms=12.3,
        ),
        "toxic": ClassificationResult(
            label="toxic",
            confidence=0.95,
            metadata={"method": "test", "toxicity_score": 0.95},
            processing_time_ms=15.2,
        ),
        "spam": ClassificationResult(
            label="spam",
            confidence=0.88,
            metadata={"method": "test", "spam_indicators": 5},
            processing_time_ms=8.7,
        ),
    }


# Mock external dependencies
@pytest.fixture
def mock_transformers():
    """Mock transformers library for testing."""
    mock_pipeline = Mock()
    mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.85}]

    mock_transformers = Mock()
    mock_transformers.pipeline = Mock(return_value=mock_pipeline)
    mock_transformers.AutoTokenizer = Mock()
    mock_transformers.AutoModel = Mock()

    return mock_transformers


@pytest.fixture
def mock_detoxify():
    """Mock detoxify library for testing."""
    mock_detoxify = Mock()
    mock_detoxify.Detoxify = Mock()
    mock_instance = Mock()
    mock_instance.predict.return_value = {
        "toxicity": 0.1,
        "severe_toxicity": 0.05,
        "obscene": 0.02,
        "threat": 0.01,
        "insult": 0.03,
        "identity_attack": 0.02,
    }
    mock_detoxify.Detoxify.return_value = mock_instance
    return mock_detoxify


@pytest.fixture
def mock_textstat():
    """Mock textstat library for testing."""
    mock_textstat = Mock()
    mock_textstat.flesch_reading_ease.return_value = 65.0
    mock_textstat.flesch_kincaid_grade.return_value = 8.5
    mock_textstat.automated_readability_index.return_value = 9.2
    mock_textstat.coleman_liau_index.return_value = 10.1
    mock_textstat.gunning_fog.return_value = 11.3
    mock_textstat.smog_index.return_value = 9.8
    mock_textstat.text_standard.return_value = "9th and 10th grade"
    return mock_textstat


@pytest.fixture
def mock_langdetect():
    """Mock langdetect library for testing."""
    mock_lang_prob = Mock()
    mock_lang_prob.lang = "en"
    mock_lang_prob.prob = 0.95

    mock_detector = Mock()
    mock_detector.detect_langs.return_value = [mock_lang_prob]

    mock_langdetect = Mock()
    mock_langdetect.DetectorFactory = Mock()
    mock_langdetect.DetectorFactory.create.return_value = mock_detector

    return mock_langdetect


# Sample SifakaThought for testing
@pytest.fixture
def sample_thought():
    """Provide a sample SifakaThought for testing."""
    return SifakaThought(
        prompt="Test prompt",
        final_text="Test response",
        iteration=1,
        max_iterations=3,
    )
