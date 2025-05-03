# Content Moderation Example

This document demonstrates how to use Sifaka for content moderation, helping identify and filter potentially problematic content.

## Basic Content Moderation

```python
from sifaka.classifiers import create_toxicity_classifier
from sifaka.rules.content import create_prohibited_content_rule

def setup_moderation_pipeline():
    # Create a toxicity classifier
    toxicity_classifier = create_toxicity_classifier(
        model_name="original",
        cache_size=1000
    )

    # Create a rule using the classifier
    prohibited_rule = create_prohibited_content_rule(
        classifier=toxicity_classifier,
        valid_labels=["non_toxic"],
        min_confidence=0.8
    )

    return prohibited_rule

# Setup the moderation rule
moderation_rule = setup_moderation_pipeline()

# Test with some content
content_examples = [
    "This is a friendly message about collaboration.",
    "You are so stupid, I hate you!",
    "I'm concerned about the quality of this work.",
]

for i, content in enumerate(content_examples):
    result = moderation_rule.validate(content)
    print(f"Example {i+1}: {content}")
    print(f"   Passed: {result.passed}")
    print(f"   Message: {result.message}")
    print(f"   Score: {result.metadata.get('score', 'N/A')}")
    print()
```

## Multi-rule Content Moderation

```python
from sifaka.classifiers import create_toxicity_classifier, create_sentiment_classifier
from sifaka.rules.content import create_prohibited_content_rule, create_sentiment_rule
from sifaka.chain import create_validation_chain

def setup_comprehensive_moderation():
    # Create classifiers
    toxicity_classifier = create_toxicity_classifier(
        model_name="original",
        cache_size=1000
    )

    sentiment_classifier = create_sentiment_classifier(
        cache_size=1000
    )

    # Create rules
    toxicity_rule = create_prohibited_content_rule(
        name="toxicity_check",
        classifier=toxicity_classifier,
        valid_labels=["non_toxic"],
        min_confidence=0.8
    )

    sentiment_rule = create_sentiment_rule(
        name="sentiment_check",
        classifier=sentiment_classifier,
        valid_labels=["positive", "neutral"],
        min_confidence=0.7
    )

    # Create a chain for sequential processing
    chain = create_validation_chain(
        name="content_moderation_chain",
        rules=[toxicity_rule, sentiment_rule]
    )

    return chain

# Setup the moderation chain
moderation_chain = setup_comprehensive_moderation()

# Test with some content
content_examples = [
    "This is a friendly message about collaboration.",
    "You are so stupid, I hate you!",
    "I'm concerned about the quality of this work.",
    "This product is terrible and I want a refund!"
]

for i, content in enumerate(content_examples):
    result = moderation_chain.process(content)
    print(f"Example {i+1}: {content}")
    print(f"   All checks passed: {result.all_passed}")

    for rule_result in result.rule_results:
        print(f"   {rule_result.rule_name}: {rule_result.passed}")
        if not rule_result.passed:
            print(f"      Reason: {rule_result.message}")
    print()
```

## Content Moderation with Improvement

```python
from sifaka.classifiers import create_toxicity_classifier
from sifaka.rules.content import create_prohibited_content_rule
from sifaka.critics import create_content_critic
from sifaka.chain import create_improvement_chain

def setup_moderation_with_improvement():
    # Create classifier and rule
    toxicity_classifier = create_toxicity_classifier()
    toxicity_rule = create_prohibited_content_rule(
        classifier=toxicity_classifier,
        valid_labels=["non_toxic"]
    )

    # Create content critic
    content_critic = create_content_critic()

    # Create improvement chain
    chain = create_improvement_chain(
        name="content_improvement_chain",
        rules=[toxicity_rule],
        critic=content_critic,
        max_attempts=2
    )

    return chain

# Setup the moderation and improvement chain
improvement_chain = setup_moderation_with_improvement()

# Test with problematic content
problematic_content = "This service is terrible and I hate it!"

result = improvement_chain.process(problematic_content)

print(f"Original content: {problematic_content}")
print(f"Improved content: {result.improved_text}")
print(f"Validation passed: {result.passed}")
print(f"Improvement attempts: {result.attempt_count}")

if result.critic_metadata:
    print("\nCritic feedback:")
    print(f"   Score: {result.critic_metadata.score}")
    print(f"   Feedback: {result.critic_metadata.feedback}")
```

## Batch Processing for Content Moderation

```python
from sifaka.classifiers import create_toxicity_classifier
from sifaka.rules.content import create_prohibited_content_rule
from concurrent.futures import ThreadPoolExecutor
import time

def setup_moderation_rule():
    toxicity_classifier = create_toxicity_classifier(cache_size=1000)
    return create_prohibited_content_rule(
        classifier=toxicity_classifier,
        valid_labels=["non_toxic"],
        min_confidence=0.7
    )

# Setup the rule
rule = setup_moderation_rule()

# Example content items
content_items = [
    "I love working with this team!",
    "You're completely useless.",
    "The service was satisfactory.",
    "This is the worst product I've ever used.",
    "Thanks for your help on this project.",
    "Die in a fire, you terrible person.",
    "I appreciate your timely response.",
    "Let me know if you need any other information.",
    "I'm disappointed with the outcome.",
    "This is absolutely unacceptable."
]

# Sequential processing
def process_sequentially():
    start_time = time.time()
    results = []

    for content in content_items:
        result = rule.validate(content)
        results.append((content, result))

    end_time = time.time()
    return results, end_time - start_time

# Parallel processing
def process_in_parallel():
    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(rule.validate, content) for content in content_items]
        for i, future in enumerate(futures):
            results.append((content_items[i], future.result()))

    end_time = time.time()
    return results, end_time - start_time

# Run sequential processing
seq_results, seq_time = process_sequentially()

# Run parallel processing
par_results, par_time = process_in_parallel()

# Print results
print(f"Sequential processing time: {seq_time:.4f} seconds")
print(f"Parallel processing time: {par_time:.4f} seconds")
print(f"Speedup: {seq_time/par_time:.2f}x")

print("\nModeration results:")
for content, result in seq_results:
    print(f"Content: {content[:30]}..." if len(content) > 30 else f"Content: {content}")
    print(f"   Passed: {result.passed}")
    if not result.passed:
        print(f"   Reason: {result.message}")
```

## Integration with Web Application

Here's an example of integrating Sifaka content moderation with a Flask application:

```python
from flask import Flask, request, jsonify
from sifaka.classifiers import create_toxicity_classifier
from sifaka.rules.content import create_prohibited_content_rule

app = Flask(__name__)

# Initialize moderation components
toxicity_classifier = create_toxicity_classifier(cache_size=1000)
moderation_rule = create_prohibited_content_rule(
    classifier=toxicity_classifier,
    valid_labels=["non_toxic"],
    min_confidence=0.7
)

@app.route('/api/comments', methods=['POST'])
def submit_comment():
    data = request.json

    if not data or 'content' not in data:
        return jsonify({'error': 'No content provided'}), 400

    content = data['content']

    # Validate content
    result = moderation_rule.validate(content)

    if not result.passed:
        return jsonify({
            'error': 'Content moderation failed',
            'reason': result.message,
            'status': 'rejected'
        }), 400

    # If content passes moderation, store it in the database
    # (database code would go here)

    return jsonify({
        'status': 'approved',
        'message': 'Comment accepted'
    })

@app.route('/api/comments/analyze', methods=['POST'])
def analyze_content():
    data = request.json

    if not data or 'content' not in data:
        return jsonify({'error': 'No content provided'}), 400

    content = data['content']

    # Validate content
    result = moderation_rule.validate(content)

    # Return detailed analysis
    return jsonify({
        'passed': result.passed,
        'message': result.message,
        'score': result.metadata.get('score', None),
        'classification': {
            'label': result.metadata.get('label', 'unknown'),
            'confidence': result.metadata.get('confidence', 0)
        },
        'recommendation': 'approve' if result.passed else 'reject'
    })

if __name__ == '__main__':
    app.run(debug=True)
```

This Flask application demonstrates:
1. Content validation for user-submitted comments
2. Detailed content analysis endpoint
3. Integration of Sifaka's moderation capabilities into a web application

These examples showcase various ways to use Sifaka for content moderation, from simple validation to multi-rule checking, content improvement, batch processing, and web application integration.