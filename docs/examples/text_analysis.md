# Text Analysis Examples

This document demonstrates how to use Sifaka for comprehensive text analysis, extracting insights and understanding from text content.

## Basic Text Analysis

```python
from sifaka.classifiers import (
    create_sentiment_classifier,
    create_language_classifier,
    create_readability_classifier
)

def analyze_text(text):
    # Create classifiers
    sentiment = create_sentiment_classifier()
    language = create_language_classifier()
    readability = create_readability_classifier()

    # Analyze text
    sentiment_result = sentiment.classify(text)
    language_result = language.classify(text)
    readability_result = readability.classify(text)

    # Combine results
    return {
        "sentiment": {
            "label": sentiment_result.label,
            "confidence": sentiment_result.confidence,
            "scores": sentiment_result.metadata.get("all_scores", {})
        },
        "language": {
            "detected": language_result.label,
            "confidence": language_result.confidence
        },
        "readability": {
            "level": readability_result.label,
            "scores": {
                "flesch_kincaid": readability_result.metadata.get("flesch_kincaid"),
                "coleman_liau": readability_result.metadata.get("coleman_liau")
            }
        }
    }

# Usage example
text = "This is a sample text for analysis. It demonstrates how to extract multiple insights from content."
analysis = analyze_text(text)

print(f"Sentiment: {analysis['sentiment']['label']} ({analysis['sentiment']['confidence']:.2f})")
print(f"Language: {analysis['language']['detected']}")
print(f"Reading Level: {analysis['readability']['level']}")
```

## Advanced Text Analysis

```python
from sifaka.classifiers import (
    create_sentiment_classifier,
    create_toxicity_classifier,
    create_topic_classifier,
    create_ner_classifier
)

class TextAnalyzer:
    def __init__(self):
        # Initialize classifiers
        self.sentiment_classifier = create_sentiment_classifier(cache_size=100)
        self.toxicity_classifier = create_toxicity_classifier(cache_size=100)
        self.topic_classifier = create_topic_classifier(cache_size=100)
        self.ner_classifier = create_ner_classifier(cache_size=100)

    def analyze(self, text):
        # Run all classifiers
        sentiment = self.sentiment_classifier.classify(text)
        toxicity = self.toxicity_classifier.classify(text)
        topic = self.topic_classifier.classify(text)
        entities = self.ner_classifier.classify(text)

        # Format entity data from NER classifier
        extracted_entities = entities.metadata.get("entities", [])
        formatted_entities = {}

        for entity in extracted_entities:
            entity_type = entity.get("type", "unknown")
            entity_text = entity.get("text", "")
            if entity_type not in formatted_entities:
                formatted_entities[entity_type] = []
            formatted_entities[entity_type].append(entity_text)

        # Construct comprehensive analysis
        analysis = {
            "content_type": {
                "sentiment": sentiment.label,
                "sentiment_confidence": sentiment.confidence,
                "toxicity": toxicity.label,
                "toxicity_confidence": toxicity.confidence,
                "topics": topic.metadata.get("topics", []),
                "primary_topic": topic.label,
                "topic_confidence": topic.confidence
            },
            "entities": formatted_entities,
            "metadata": {
                "text_length": len(text),
                "word_count": len(text.split()),
                "analysis_timestamp": self._get_timestamp()
            },
            "summary": self._generate_summary(
                sentiment.label,
                toxicity.label,
                topic.label,
                formatted_entities
            )
        }

        return analysis

    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()

    def _generate_summary(self, sentiment, toxicity, topic, entities):
        # Create a brief summary of the analysis
        entity_summary = ""
        if entities:
            entity_types = list(entities.keys())
            if entity_types:
                entity_summary = f" Contains {', '.join(entity_types)}."

        return f"{sentiment.capitalize()} {toxicity} content about {topic}.{entity_summary}"

# Usage example
analyzer = TextAnalyzer()
texts = [
    "I absolutely love this new phone! The camera quality is outstanding and the battery lasts all day.",
    "This restaurant was terrible. The service was slow and the food was cold when it arrived.",
    "According to NASA, the James Webb Space Telescope has discovered new galaxies forming at the edge of the observable universe."
]

for i, text in enumerate(texts):
    print(f"\nAnalysis {i+1}: {text[:50]}...")
    analysis = analyzer.analyze(text)

    print(f"Summary: {analysis['summary']}")
    print(f"Sentiment: {analysis['content_type']['sentiment']} ({analysis['content_type']['sentiment_confidence']:.2f})")
    print(f"Toxicity: {analysis['content_type']['toxicity']} ({analysis['content_type']['toxicity_confidence']:.2f})")
    print(f"Primary topic: {analysis['content_type']['primary_topic']}")

    if analysis['entities']:
        print("Entities found:")
        for entity_type, items in analysis['entities'].items():
            print(f"  {entity_type}: {', '.join(items)}")
```

## Batch Text Analysis

```python
from sifaka.classifiers import create_sentiment_classifier
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import time

def analyze_batch_sentiment(texts, max_workers=4):
    # Create classifier
    classifier = create_sentiment_classifier(cache_size=1000)

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(classifier.classify, text) for text in texts]
        for i, future in enumerate(futures):
            result = future.result()
            results.append({
                'text': texts[i],
                'sentiment': result.label,
                'confidence': result.confidence,
                'compound_score': result.metadata.get('compound_score', 0),
                'pos_score': result.metadata.get('pos_score', 0),
                'neg_score': result.metadata.get('neg_score', 0),
                'neu_score': result.metadata.get('neu_score', 0)
            })

    end_time = time.time()
    processing_time = end_time - start_time

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Calculate sentiment distribution
    sentiment_counts = df['sentiment'].value_counts().to_dict()

    # Calculate average confidence by sentiment
    avg_confidence = df.groupby('sentiment')['confidence'].mean().to_dict()

    return {
        'results': df,
        'summary': {
            'total_texts': len(texts),
            'processing_time': processing_time,
            'texts_per_second': len(texts) / processing_time,
            'sentiment_distribution': sentiment_counts,
            'avg_confidence': avg_confidence
        }
    }

# Example usage with customer feedback data
customer_feedback = [
    "The product exceeded my expectations. I'm very satisfied!",
    "Delivery was quick and the item was as described.",
    "Customer service was unhelpful and rude.",
    "I've been waiting for my refund for 2 weeks now.",
    "Average product, nothing special but does the job.",
    "This is the best purchase I've made all year!",
    "The quality is poor and it broke after just a few uses.",
    "Shipping took forever but the product is good.",
    "I would recommend this to anyone looking for a reliable option.",
    "Not worth the money at all."
]

# Run batch analysis
analysis = analyze_batch_sentiment(customer_feedback)

# Print summary
print(f"Analyzed {analysis['summary']['total_texts']} texts in {analysis['summary']['processing_time']:.2f} seconds")
print(f"Processing rate: {analysis['summary']['texts_per_second']:.2f} texts per second")

print("\nSentiment distribution:")
for sentiment, count in analysis['summary']['sentiment_distribution'].items():
    print(f"  {sentiment}: {count} texts ({count/len(customer_feedback)*100:.1f}%)")

print("\nAverage confidence by sentiment:")
for sentiment, confidence in analysis['summary']['avg_confidence'].items():
    print(f"  {sentiment}: {confidence:.2f}")

# Print individual results
print("\nIndividual results:")
for index, row in analysis['results'].iterrows():
    print(f"{index+1}. \"{row['text'][:40]}...\"")
    print(f"   Sentiment: {row['sentiment']} (confidence: {row['confidence']:.2f})")
```

## Integration with Text Analysis Service

```python
from flask import Flask, request, jsonify
from sifaka.classifiers import (
    create_sentiment_classifier,
    create_topic_classifier
)
import uuid
import json
from datetime import datetime

app = Flask(__name__)

# Initialize classifiers
sentiment_classifier = create_sentiment_classifier(cache_size=1000)
topic_classifier = create_topic_classifier(cache_size=1000)

# In-memory storage for analysis results
analysis_store = {}

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    data = request.json

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    # Create a unique ID for this analysis
    analysis_id = str(uuid.uuid4())

    # Analyze sentiment
    sentiment_result = sentiment_classifier.classify(text)

    # Analyze topics
    topic_result = topic_classifier.classify(text)

    # Create analysis result
    analysis = {
        'id': analysis_id,
        'timestamp': datetime.now().isoformat(),
        'text': text,
        'sentiment': {
            'label': sentiment_result.label,
            'confidence': sentiment_result.confidence,
            'compound_score': sentiment_result.metadata.get('compound_score', 0),
        },
        'topic': {
            'primary': topic_result.label,
            'confidence': topic_result.confidence,
            'all_topics': topic_result.metadata.get('topics', []),
        }
    }

    # Store for later retrieval
    analysis_store[analysis_id] = analysis

    return jsonify({
        'analysis_id': analysis_id,
        'sentiment': sentiment_result.label,
        'topic': topic_result.label,
        'url': f'/api/analysis/{analysis_id}'
    })

@app.route('/api/analysis/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    if analysis_id not in analysis_store:
        return jsonify({'error': 'Analysis not found'}), 404

    return jsonify(analysis_store[analysis_id])

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    data = request.json

    if not data or 'texts' not in data:
        return jsonify({'error': 'No texts provided'}), 400

    texts = data['texts']

    if not isinstance(texts, list):
        return jsonify({'error': 'Texts must be provided as a list'}), 400

    # For larger batches, this should use background processing
    results = []
    for text in texts:
        sentiment_result = sentiment_classifier.classify(text)
        results.append({
            'text': text,
            'sentiment': sentiment_result.label,
            'confidence': sentiment_result.confidence
        })

    batch_id = str(uuid.uuid4())

    return jsonify({
        'batch_id': batch_id,
        'count': len(results),
        'results': results
    })

if __name__ == '__main__':
    app.run(debug=True)
```

## Sentiment Analysis Dashboard

This example shows how you might integrate Sifaka's text analysis with a data visualization tool:

```python
from sifaka.classifiers import create_sentiment_classifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample data
df = pd.read_csv('customer_feedback.csv')

# Initialize sentiment classifier
sentiment_classifier = create_sentiment_classifier(cache_size=1000)

# Analyze sentiment for each review
def analyze_sentiment(text):
    result = sentiment_classifier.classify(text)
    return {
        'sentiment': result.label,
        'confidence': result.confidence,
        'compound_score': result.metadata.get('compound_score', 0)
    }

# Apply sentiment analysis to each review
sentiments = df['feedback'].apply(analyze_sentiment)

# Extract sentiment data into dataframe
df['sentiment'] = sentiments.apply(lambda x: x['sentiment'])
df['confidence'] = sentiments.apply(lambda x: x['confidence'])
df['compound_score'] = sentiments.apply(lambda x: x['compound_score'])

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution in Customer Feedback')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('sentiment_distribution.png')

# Sentiment over time
plt.figure(figsize=(12, 6))
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')
sentiment_by_month = df.groupby('month')['compound_score'].mean()

sentiment_by_month.plot(kind='line', marker='o')
plt.title('Average Sentiment Score by Month')
plt.xlabel('Month')
plt.ylabel('Average Compound Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sentiment_trend.png')

# Generate summary statistics
sentiment_counts = df['sentiment'].value_counts()
avg_confidence = df.groupby('sentiment')['confidence'].mean()

print("Sentiment Analysis Summary:")
print("==========================")
print(f"Total reviews analyzed: {len(df)}")
print("\nSentiment distribution:")
for sentiment, count in sentiment_counts.items():
    print(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")

print("\nAverage confidence by sentiment:")
for sentiment, confidence in avg_confidence.items():
    print(f"  {sentiment}: {confidence:.2f}")

# Identify most positive and negative reviews
most_positive = df.loc[df['compound_score'].idxmax()]
most_negative = df.loc[df['compound_score'].idxmin()]

print("\nMost positive review:")
print(f"  Score: {most_positive['compound_score']:.3f}")
print(f"  \"{most_positive['feedback'][:100]}...\"")

print("\nMost negative review:")
print(f"  Score: {most_negative['compound_score']:.3f}")
print(f"  \"{most_negative['feedback'][:100]}...\"")
```

These examples demonstrate various ways to use Sifaka for text analysis, including sentiment analysis, topic identification, named entity recognition, batch processing, and integration with web services and visualization tools.