# Named Entity Recognition (NER) Classifier

The NER Classifier is a component that identifies and categorizes named entities in text, such as people, organizations, locations, dates, and more. It uses spaCy as the underlying NLP engine.

## Installation

To use the NER Classifier, you need to install Sifaka with the `ner` extra:

```bash
pip install "sifaka[ner]"
```

You also need to download at least one spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Basic Usage

```python
from sifaka.classifiers import create_ner_classifier

# Create a NER classifier with default settings
ner = create_ner_classifier()

# Classify text
result = ner.classify("Apple Inc. was founded by Steve Jobs in California on April 1, 1976.")

# Print the results
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Entities: {result.metadata['entities']}")
```

## Configuration Options

The NER Classifier can be configured with various options:

```python
from sifaka.classifiers import create_ner_classifier

# Create a NER classifier with custom settings
ner = create_ner_classifier(
    name="custom_ner",
    description="Custom NER classifier for financial entities",
    model_name="en_core_web_lg",  # Larger, more accurate model
    entity_types=["PERSON", "ORG", "MONEY", "DATE"],  # Only detect these entity types
    min_confidence=0.7,  # Higher confidence threshold
    cache_size=100,  # Cache up to 100 results
)
```

## Advanced Usage

### Using with a Chain

```python
from sifaka.chain import create_simple_chain
from sifaka.models import create_openai_provider
from sifaka.classifiers import create_ner_classifier
from sifaka.adapters.rules import create_classifier_rule

# Create a NER classifier
ner = create_ner_classifier(
    model_name="en_core_web_lg",
    entity_types=["PERSON", "ORG", "GPE"],  # GPE = Geopolitical Entity (countries, cities, etc.)
)

# Create a rule that requires at least one organization to be mentioned
ner_rule = create_classifier_rule(
    classifier=ner,
    name="org_mention_rule",
    description="Requires at least one organization to be mentioned",
    valid_labels=["organization"],  # Only pass if an organization is detected
    threshold=0.7,  # With at least 70% confidence
)

# Create a model provider
model = create_openai_provider("gpt-4")

# Create a chain with the NER rule
chain = create_simple_chain(
    model=model,
    rules=[ner_rule],
    max_attempts=3,
)

# Run the chain
result = chain.run("Write a short paragraph about a tech company.")
```

### Custom Entity Processing

```python
from sifaka.classifiers import NERClassifier, ClassifierConfig
from typing import Dict, List, Any

class CustomNERClassifier(NERClassifier):
    """Custom NER classifier with specialized entity processing."""
    
    def _process_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Custom processing of detected entities."""
        # Call the parent method first
        processed = super()._process_entities(entities)
        
        # Add custom processing
        for entity in processed:
            # Add a custom field for entity importance based on length
            entity["importance"] = len(entity["text"]) / 10
            
            # Normalize organization names
            if entity["type"] == "ORG" or entity["type"] == "organization":
                entity["normalized_name"] = entity["text"].replace(".", "").strip().upper()
        
        return processed

# Create the custom classifier
config = ClassifierConfig(
    labels=["person", "organization", "location", "date", "money", "unknown"],
    cache_size=100,
    params={"model_name": "en_core_web_lg"}
)

custom_ner = CustomNERClassifier(
    name="custom_ner",
    description="NER with custom entity processing",
    config=config
)

# Use the custom classifier
result = custom_ner.classify("Apple Inc. was founded by Steve Jobs in California.")
entities = result.metadata["entities"]

# Access custom fields
for entity in entities:
    if "importance" in entity:
        print(f"{entity['text']} - Importance: {entity['importance']:.2f}")
    if "normalized_name" in entity:
        print(f"{entity['text']} -> {entity['normalized_name']}")
```

## Entity Types

The NER Classifier can detect the following entity types (depending on the spaCy model used):

| Entity Type | Description | Example |
|-------------|-------------|---------|
| PERSON | People, including fictional | "Steve Jobs", "Harry Potter" |
| ORG | Companies, agencies, institutions | "Apple Inc.", "United Nations" |
| GPE | Countries, cities, states | "United States", "Paris" |
| LOC | Non-GPE locations, mountain ranges, bodies of water | "the Sahara", "Pacific Ocean" |
| DATE | Absolute or relative dates or periods | "January 2022", "yesterday" |
| TIME | Times smaller than a day | "3pm", "noon" |
| MONEY | Monetary values, including unit | "$10 million", "€50" |
| PERCENT | Percentage | "10%", "one-third" |
| PRODUCT | Objects, vehicles, foods, etc. (not services) | "iPhone", "Boeing 747" |
| EVENT | Named hurricanes, battles, wars, sports events, etc. | "World War II", "Super Bowl" |
| WORK_OF_ART | Titles of books, songs, etc. | "The Great Gatsby" |
| LAW | Named documents made into laws | "Constitution" |
| LANGUAGE | Any named language | "English", "Spanish" |
| FAC | Buildings, airports, highways, bridges, etc. | "O'Hare Airport", "Golden Gate Bridge" |
| NORP | Nationalities or religious or political groups | "Americans", "Republicans" |
| CARDINAL | Numerals that do not fall under another type | "ten", "4" |
| ORDINAL | "first", "second", etc. | "1st", "second" |
| QUANTITY | Measurements, as of weight or distance | "10km", "5 pounds" |

## Performance Considerations

- The NER Classifier has a moderate computational cost (default: 2)
- Larger spaCy models (e.g., `en_core_web_lg`) provide better accuracy but use more memory and are slower
- Consider using caching (`cache_size` parameter) for repeated classifications of the same text
- For high-throughput applications, consider using batch classification with `batch_classify()`

## State Management

The NER Classifier uses Sifaka's standardized state management approach:

```python
from sifaka.classifiers import create_ner_classifier

# Create a NER classifier
ner = create_ner_classifier()

# State is managed internally using StateManager
# The state includes:
# - The spaCy model (loaded lazily)
# - The classification cache
# - Initialization status

# Warm up the classifier (load the model)
ner.warm_up()

# The model is now loaded and ready for fast classification
```
