"""
Example demonstrating the Named Entity Recognition (NER) Classifier.

This example shows how to use the NERClassifier to identify named entities
in text, such as people, organizations, locations, etc.

Requirements:
    pip install sifaka[ner]
    python -m spacy download en_core_web_sm
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import sifaka
sys.path.append(str(Path(__file__).parent.parent))

from sifaka.classifiers import NERClassifier


def main():
    """Run the NER classifier example."""
    # Create a NER classifier
    ner = NERClassifier(
        name="entity_detector",
        description="Identifies named entities in text",
        params={
            "model_name": "en_core_web_sm",  # Use the small English model
            "entity_types": [
                "person", "organization", "location", "date", "money", "gpe"
            ],  # Filter to these entity types
            "min_confidence": 0.5,
        },
    )

    # Sample texts with different entity types
    texts = [
        "Apple Inc. is planning to open a new store in New York City next month.",
        "Jeff Bezos founded Amazon in 1994 and stepped down as CEO in 2021.",
        "The Eiffel Tower in Paris, France attracts millions of tourists every year.",
        "Microsoft announced a $10 billion investment in OpenAI on January 15, 2023.",
        "The United Nations headquarters is located in Manhattan.",
    ]

    # Process each text
    for i, text in enumerate(texts):
        print(f"\n=== Example {i+1} ===")
        print(f"Text: {text}")
        
        # Classify the text
        result = ner.classify(text)
        
        # Print the classification result
        print(f"Dominant entity type: {result.label}")
        print(f"Confidence: {result.confidence:.2f}")
        
        # Print all detected entities
        print("\nDetected entities:")
        if "entities" in result.metadata:
            for entity in result.metadata["entities"]:
                print(f"  - {entity['text']} ({entity['type']})")
        
        # Print entities grouped by type
        if "entities_by_type" in result.metadata:
            print("\nEntities by type:")
            for entity_type, entities in result.metadata["entities_by_type"].items():
                print(f"  {entity_type.upper()}:")
                for entity in entities:
                    print(f"    - {entity['text']}")


if __name__ == "__main__":
    main()
