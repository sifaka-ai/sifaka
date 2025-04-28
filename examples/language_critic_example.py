#!/usr/bin/env python3
"""
Language Classifier with Critic Example using Sifaka and Claude.

This example demonstrates:
1. Using Claude to generate text in Spanish
2. Using a language classifier to detect if text is English
3. Using a critic to provide feedback to Claude until the text is in English

Usage:
    python language_critic_example.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
    - Anthropic API key set as ANTHROPIC_API_KEY environment variable
"""

import argparse
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import Field

# Add parent directory to system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import anthropic
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install required packages: pip install anthropic scikit-learn")
    sys.exit(1)

from sifaka.classifiers import ClassifierConfig
from sifaka.classifiers.base import BaseClassifier, ClassificationResult
from sifaka.utils.logging import get_logger

# Initialize logger from Sifaka
logger = get_logger(__name__)

# Language samples for classifier training (expanded set)
LANGUAGE_SAMPLES = {
    "english": [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm.",
        "You can't judge a book by its cover.",
        "Life is what happens while you're busy making other plans.",
        "Where there's a will, there's a way.",
        "Actions speak louder than words.",
        "Knowledge is power, and learning is the key.",
        "The future belongs to those who believe in the beauty of their dreams.",
    ],
    "spanish": [
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "Ser o no ser, esa es la cuestión.",
        "No es oro todo lo que reluce.",
        "A quien madruga, Dios le ayuda.",
        "No juzgues un libro por su portada.",
        "La vida es lo que pasa mientras estás ocupado haciendo otros planes.",
        "Donde hay voluntad, hay un camino.",
        "Las acciones hablan más fuerte que las palabras.",
        "El conocimiento es poder, y aprender es la clave.",
        "El futuro pertenece a quienes creen en la belleza de sus sueños.",
    ],
    "french": [
        "Le rapide renard brun saute par-dessus le chien paresseux.",
        "Être ou ne pas être, telle est la question.",
        "Tout ce qui brille n'est pas or.",
        "L'oiseau matinal attrape le ver.",
        "Il ne faut pas juger un livre à sa couverture.",
        "La vie est ce qui arrive pendant que vous faites d'autres projets.",
        "Là où il y a une volonté, il y a un chemin.",
        "Les actes parlent plus fort que les mots.",
        "Le savoir est le pouvoir, et l'apprentissage est la clé.",
        "L'avenir appartient à ceux qui croient en la beauté de leurs rêves.",
    ],
}

# Test prompts for Claude
TEST_PROMPTS = [
    "Describe the benefits of exercise",
    "Write a short story about a magic forest",
    "Explain the concept of artificial intelligence",
]

@dataclass(frozen=True)
class LanguageConfig:
    """Configuration for language classification."""

    min_confidence: float = 0.6
    max_features: int = 2000
    random_state: int = 42
    supported_languages: List[str] = field(default_factory=lambda: ["english", "spanish", "french"])

class LanguageClassifier(BaseClassifier):
    """A classifier that detects the language of text."""

    language_config: LanguageConfig = Field(
        default_factory=LanguageConfig,
        description="Configuration for language classification",
    )

    def __init__(
        self,
        name: str = "language_classifier",
        description: str = "Classifies text into different languages",
        language_config: Optional[LanguageConfig] = None,
    ):
        # Initialize base class first with config
        config = ClassifierConfig(
            labels=(language_config or LanguageConfig()).supported_languages,
            cost=1.5,
            min_confidence=(language_config or LanguageConfig()).min_confidence,
        )
        super().__init__(name=name, description=description, config=config)

        # Set language config
        if language_config is not None:
            self.language_config = language_config

        # Create character-level TF-IDF vectorizer (important for language detection)
        self._vectorizer = TfidfVectorizer(
            max_features=self.language_config.max_features,
            analyzer="char",
            ngram_range=(1, 4),
            min_df=2,
            max_df=0.95,
        )

        # Create RandomForest model with improved parameters
        self._model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=self.language_config.random_state,
            class_weight="balanced",
        )

        # Create pipeline
        self._pipeline = Pipeline(
            [
                ("vectorizer", self._vectorizer),
                ("classifier", self._model),
            ]
        )

    def fit(self, texts: List[str], labels: List[str]) -> "LanguageClassifier":
        """Fit the language classifier on training data."""
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

        # Fit the pipeline
        self._pipeline.fit(texts, labels)
        return self

    def _classify_impl(self, text: str) -> ClassificationResult:
        """Classify text language."""
        # Predict probability
        proba = self._pipeline.predict_proba([text])[0]

        # Get dominant class
        dominant_class_idx = proba.argmax()
        confidence = float(proba[dominant_class_idx])

        # Get all language probabilities
        all_probs = {self.config.labels[i]: float(prob) for i, prob in enumerate(proba)}

        metadata = {
            "probabilities": all_probs,
            "threshold": self.language_config.min_confidence,
            "is_confident": confidence >= self.language_config.min_confidence,
        }

        return ClassificationResult(
            label=self.config.labels[dominant_class_idx],
            confidence=confidence,
            metadata=metadata,
        )

class ClaudeGenerator:
    """Text generator using Claude API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 500) -> str:
        """Generate text using Claude."""
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating text with Claude: {e}")
            return f"Error generating text: {str(e)}"

def main():
    """Run the language critic example."""
    parser = argparse.ArgumentParser(description="Language Critic Example")
    parser.add_argument(
        "--api-key", help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument("--prompt", help="Custom prompt to use")
    args = parser.parse_args()

    logger.info("Starting language critic example...")

    # Prepare training data
    texts, labels = [], []
    for language, samples in LANGUAGE_SAMPLES.items():
        texts.extend(samples)
        labels.extend([language] * len(samples))

    # Train language classifier
    classifier = LanguageClassifier()
    classifier.fit(texts, labels)

    # Create Claude generator
    generator = ClaudeGenerator(api_key=args.api_key)

    # Select a prompt
    prompt = args.prompt or random.choice(TEST_PROMPTS)

    # Generate Spanish text
    logger.info(f"Generating Spanish text for prompt: '{prompt}'")
    system_prompt = (
        "You are a helpful assistant that responds in Spanish. Always answer in Spanish."
    )
    spanish_text = generator.generate(prompt, system_prompt=system_prompt)
    logger.info(f"Generated Spanish text (preview): '{spanish_text[:100]}...'")

    # Iterate until English
    current_text = spanish_text
    target_language = "english"
    max_iterations = 3

    logger.info("\nStarting iteration process to convert to English...")

    for i in range(max_iterations):
        logger.info(f"\nIteration {i+1}/{max_iterations}")

        # Classify the current text
        result = classifier.classify(current_text)
        detected_language = result.label
        confidence = result.confidence

        logger.info(f"Detected language: {detected_language} (confidence: {confidence:.2f})")

        # Check if the text is already in English
        if detected_language == target_language and confidence >= classifier.config.min_confidence:
            logger.info("✓ Text is correctly in English!")
            break

        # If not English, ask Claude to translate it
        logger.info(f"× Text needs to be translated from {detected_language} to English")

        fix_prompt = f"Please translate the following text to English:\n\n{current_text}"
        fixed_text = generator.generate(fix_prompt)
        current_text = fixed_text

        logger.info(f"Translated text (preview): '{current_text[:100]}...'")

    # Print final results
    logger.info("\n" + "=" * 40)
    logger.info("FINAL RESULTS")
    logger.info("=" * 40)
    logger.info(f"Original prompt: '{prompt}'")

    # Final check if text is in English
    final_result = classifier.classify(current_text)
    success = final_result.label == target_language

    logger.info(f"Final language: {final_result.label}")
    logger.info(f"Success: {success}")

    logger.info("\nFinal text:")
    logger.info("-" * 40)
    logger.info(current_text)
    logger.info("-" * 40)

    logger.info("\nLanguage critic example completed.")

if __name__ == "__main__":
    main()
