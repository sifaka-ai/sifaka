"""
Simple test script for the NER classifier.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import sifaka
sys.path.append(str(Path(__file__).parent))

from sifaka.classifiers.ner import NERClassifier

def main():
    """Test the NER classifier."""
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

    # Sample text with different entity types
    text = "Apple Inc. is planning to open a new store in New York City next month."
    
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
