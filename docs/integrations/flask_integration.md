# Flask Integration Guide

This guide demonstrates how to integrate Sifaka with Flask applications for text analysis, content moderation, and validation.

## Setup

First, install Flask and Sifaka:

```bash
pip install flask
pip install sifaka
```

## Basic Flask Integration

### Project Structure

Create a basic Flask project structure:

```
sifaka_flask/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── content.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── sifaka_service.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   └── moderator.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       ├── base.html
│       ├── index.html
│       └── analyze.html
├── instance/
│   └── config.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
└── run.py
```

### Configuration

Set up the Flask configuration:

```python
# app/config.py
import os

class Config:
    """Base configuration class."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    SIFAKA_CACHE_SIZE = 1000
    SIFAKA_TOXICITY_MODEL = 'original'
    ENABLE_CONTENT_MODERATION = True
    TOXICITY_THRESHOLD = 0.7

    # API keys
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    WTF_CSRF_ENABLED = False
    SIFAKA_CACHE_SIZE = 10  # Smaller cache for testing

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

    # Use environment variables in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SIFAKA_CACHE_SIZE = int(os.environ.get('SIFAKA_CACHE_SIZE', 1000))
    ENABLE_CONTENT_MODERATION = os.environ.get('ENABLE_CONTENT_MODERATION', 'True') == 'True'
    TOXICITY_THRESHOLD = float(os.environ.get('TOXICITY_THRESHOLD', 0.7))

# Config dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
```

### Initialize the Application

Create the Flask application factory:

```python
# app/__init__.py
from flask import Flask
from app.config import config

def create_app(config_name='default'):
    """Create a Flask application."""
    app = Flask(__name__, instance_relative_config=True)

    # Load the config
    app.config.from_object(config[config_name])
    app.config.from_pyfile('config.py', silent=True)  # Instance config

    # Initialize extensions
    init_extensions(app)

    # Register blueprints
    register_blueprints(app)

    # Register error handlers
    register_error_handlers(app)

    # Initialize the Sifaka service
    from app.services.sifaka_service import init_sifaka
    init_sifaka(app)

    return app

def init_extensions(app):
    """Initialize Flask extensions."""
    # If you have Flask extensions, initialize them here
    pass

def register_blueprints(app):
    """Register Flask blueprints."""
    from app.routes.analyzer import analyzer_bp
    from app.routes.moderator import moderator_bp

    app.register_blueprint(analyzer_bp, url_prefix='/api/analyzer')
    app.register_blueprint(moderator_bp, url_prefix='/api/moderator')

def register_error_handlers(app):
    """Register error handlers."""
    @app.errorhandler(404)
    def page_not_found(e):
        return {'error': 'Not found'}, 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return {'error': 'Internal server error'}, 500
```

### Sifaka Service

Create a service for Sifaka components:

```python
# app/services/sifaka_service.py
from sifaka.classifiers import (
    create_toxicity_classifier,
    create_sentiment_classifier,
    create_topic_classifier,
    create_ner_classifier
)
from sifaka.rules.content import (
    create_prohibited_content_rule,
    create_sentiment_rule
)
from sifaka.critics import create_content_critic
from flask import current_app, g

# Global variables to store initialized components
_toxicity_classifier = None
_sentiment_classifier = None
_topic_classifier = None
_ner_classifier = None
_toxicity_rule = None
_sentiment_rule = None
_content_critic = None

def init_sifaka(app):
    """Initialize Sifaka components."""
    # No initialization here, just setting up the application context
    pass

def get_toxicity_classifier():
    """Get or create the toxicity classifier."""
    global _toxicity_classifier
    if _toxicity_classifier is None:
        _toxicity_classifier = create_toxicity_classifier(
            model_name=current_app.config['SIFAKA_TOXICITY_MODEL'],
            cache_size=current_app.config['SIFAKA_CACHE_SIZE']
        )
    return _toxicity_classifier

def get_sentiment_classifier():
    """Get or create the sentiment classifier."""
    global _sentiment_classifier
    if _sentiment_classifier is None:
        _sentiment_classifier = create_sentiment_classifier(
            cache_size=current_app.config['SIFAKA_CACHE_SIZE']
        )
    return _sentiment_classifier

def get_topic_classifier():
    """Get or create the topic classifier."""
    global _topic_classifier
    if _topic_classifier is None:
        _topic_classifier = create_topic_classifier(
            cache_size=current_app.config['SIFAKA_CACHE_SIZE']
        )
    return _topic_classifier

def get_ner_classifier():
    """Get or create the NER classifier."""
    global _ner_classifier
    if _ner_classifier is None:
        _ner_classifier = create_ner_classifier(
            cache_size=current_app.config['SIFAKA_CACHE_SIZE']
        )
    return _ner_classifier

def get_toxicity_rule():
    """Get or create the toxicity rule."""
    global _toxicity_rule
    if _toxicity_rule is None:
        toxicity_classifier = get_toxicity_classifier()
        _toxicity_rule = create_prohibited_content_rule(
            classifier=toxicity_classifier,
            valid_labels=["non_toxic"],
            min_confidence=current_app.config['TOXICITY_THRESHOLD']
        )
    return _toxicity_rule

def get_sentiment_rule():
    """Get or create the sentiment rule."""
    global _sentiment_rule
    if _sentiment_rule is None:
        sentiment_classifier = get_sentiment_classifier()
        _sentiment_rule = create_sentiment_rule(
            classifier=sentiment_classifier,
            valid_labels=["positive", "neutral"]
        )
    return _sentiment_rule

def get_content_critic():
    """Get or create the content critic."""
    global _content_critic
    if _content_critic is None:
        _content_critic = create_content_critic()
    return _content_critic

def analyze_text(text):
    """Analyze a piece of text."""
    sentiment_classifier = get_sentiment_classifier()
    toxicity_classifier = get_toxicity_classifier()
    topic_classifier = get_topic_classifier()
    ner_classifier = get_ner_classifier()

    # Perform classification
    sentiment_result = sentiment_classifier.classify(text)
    toxicity_result = toxicity_classifier.classify(text)
    topic_result = topic_classifier.classify(text)
    entities_result = ner_classifier.classify(text)

    # Extract entities
    entities = {}
    for entity in entities_result.metadata.get("entities", []):
        entity_type = entity.get("type", "unknown")
        entity_text = entity.get("text", "")
        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append(entity_text)

    # Compile the results
    return {
        "sentiment": {
            "label": sentiment_result.label,
            "confidence": sentiment_result.confidence,
            "compound_score": sentiment_result.metadata.get("compound_score", 0),
        },
        "toxicity": {
            "label": toxicity_result.label,
            "confidence": toxicity_result.confidence,
        },
        "topic": {
            "label": topic_result.label,
            "confidence": topic_result.confidence,
            "topics": topic_result.metadata.get("topics", []),
        },
        "entities": entities,
        "text": text
    }

def moderate_content(content):
    """Moderate content."""
    rule = get_toxicity_rule()
    result = rule.validate(content)

    return {
        "passed": result.passed,
        "message": result.message,
        "label": result.metadata.get("label", "unknown"),
        "confidence": result.metadata.get("confidence", 0),
        "content": content
    }

def improve_content(content, feedback):
    """Improve content based on feedback."""
    critic = get_content_critic()
    improved = critic.improve_with_feedback(content, feedback)

    return {
        "original": content,
        "improved": improved,
        "feedback": feedback
    }
```

### Content Models

Define your content models:

```python
# app/models/content.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class AnalysisRequest:
    """Request for text analysis."""
    text: str

    def validate(self):
        """Validate the request."""
        if not self.text or not self.text.strip():
            return False, "Text cannot be empty"
        if len(self.text) > 10000:
            return False, "Text is too long (max 10,000 characters)"
        return True, ""

@dataclass
class ModerationRequest:
    """Request for content moderation."""
    content: str

    def validate(self):
        """Validate the request."""
        if not self.content or not self.content.strip():
            return False, "Content cannot be empty"
        if len(self.content) > 10000:
            return False, "Content is too long (max 10,000 characters)"
        return True, ""

@dataclass
class ImprovementRequest:
    """Request for content improvement."""
    content: str
    feedback: str

    def validate(self):
        """Validate the request."""
        if not self.content or not self.content.strip():
            return False, "Content cannot be empty"
        if len(self.content) > 10000:
            return False, "Content is too long (max 10,000 characters)"
        if not self.feedback or not self.feedback.strip():
            return False, "Feedback cannot be empty"
        if len(self.feedback) > 1000:
            return False, "Feedback is too long (max 1,000 characters)"
        return True, ""

@dataclass
class BatchAnalysisRequest:
    """Request for batch text analysis."""
    texts: List[str]

    def validate(self):
        """Validate the request."""
        if not self.texts:
            return False, "Texts list cannot be empty"
        if len(self.texts) > 100:
            return False, "Too many texts (max 100)"
        for i, text in enumerate(self.texts):
            if not text or not text.strip():
                return False, f"Text {i+1} cannot be empty"
            if len(text) > 10000:
                return False, f"Text {i+1} is too long (max 10,000 characters)"
        return True, ""

@dataclass
class ApiResponse:
    """Generic API response."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
```

### API Routes

Create the API routes for the analyzer:

```python
# app/routes/analyzer.py
from flask import Blueprint, request, jsonify, current_app
from app.models.content import (
    AnalysisRequest,
    BatchAnalysisRequest,
    ApiResponse
)
from app.services.sifaka_service import analyze_text
import json
from concurrent.futures import ThreadPoolExecutor
import threading

# Thread-local storage
thread_local = threading.local()

# Blueprint definition
analyzer_bp = Blueprint('analyzer', __name__)

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

@analyzer_bp.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a single text."""
    try:
        data = request.get_json()
        if not data:
            return jsonify(ApiResponse(
                success=False,
                error="Invalid JSON data"
            ).__dict__), 400

        # Create and validate the request
        analysis_request = AnalysisRequest(text=data.get('text', ''))
        valid, error = analysis_request.validate()
        if not valid:
            return jsonify(ApiResponse(
                success=False,
                error=error
            ).__dict__), 400

        # Analyze the text
        analysis = analyze_text(analysis_request.text)

        return jsonify(ApiResponse(
            success=True,
            data=analysis
        ).__dict__)

    except Exception as e:
        current_app.logger.error(f"Error in analyze: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        ).__dict__), 500

@analyzer_bp.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple texts."""
    try:
        data = request.get_json()
        if not data:
            return jsonify(ApiResponse(
                success=False,
                error="Invalid JSON data"
            ).__dict__), 400

        # Create and validate the request
        batch_request = BatchAnalysisRequest(texts=data.get('texts', []))
        valid, error = batch_request.validate()
        if not valid:
            return jsonify(ApiResponse(
                success=False,
                error=error
            ).__dict__), 400

        # For small batches, process synchronously
        if len(batch_request.texts) <= 5:
            results = []
            for text in batch_request.texts:
                analysis = analyze_text(text)
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "analysis": analysis
                })

            return jsonify(ApiResponse(
                success=True,
                data={
                    "results": results,
                    "meta": {
                        "count": len(results),
                        "processed": "synchronously"
                    }
                }
            ).__dict__)

        # For larger batches, use async processing with thread pool
        task_id = f"batch_{hash(tuple(batch_request.texts))}"

        def process_batch(texts):
            """Process a batch of texts."""
            results = []
            for text in texts:
                analysis = analyze_text(text)
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "analysis": analysis
                })

            # In a real app, you would store these results somewhere
            current_app.logger.info(f"Completed batch analysis {task_id} with {len(results)} results")

        # Submit the task to the executor
        executor.submit(process_batch, batch_request.texts)

        return jsonify(ApiResponse(
            success=True,
            data={
                "task_id": task_id,
                "message": f"Processing {len(batch_request.texts)} texts in the background",
                "meta": {
                    "count": len(batch_request.texts),
                    "processed": "asynchronously"
                }
            }
        ).__dict__)

    except Exception as e:
        current_app.logger.error(f"Error in batch_analyze: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=f"Batch analysis failed: {str(e)}"
        ).__dict__), 500
```

Create the API routes for the moderator:

```python
# app/routes/moderator.py
from flask import Blueprint, request, jsonify, current_app
from app.models.content import (
    ModerationRequest,
    ImprovementRequest,
    ApiResponse
)
from app.services.sifaka_service import moderate_content, improve_content

# Blueprint definition
moderator_bp = Blueprint('moderator', __name__)

@moderator_bp.route('/moderate', methods=['POST'])
def moderate():
    """Moderate content."""
    try:
        data = request.get_json()
        if not data:
            return jsonify(ApiResponse(
                success=False,
                error="Invalid JSON data"
            ).__dict__), 400

        # Create and validate the request
        moderation_request = ModerationRequest(content=data.get('content', ''))
        valid, error = moderation_request.validate()
        if not valid:
            return jsonify(ApiResponse(
                success=False,
                error=error
            ).__dict__), 400

        # Moderate the content
        moderation = moderate_content(moderation_request.content)

        return jsonify(ApiResponse(
            success=True,
            data=moderation
        ).__dict__)

    except Exception as e:
        current_app.logger.error(f"Error in moderate: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=f"Moderation failed: {str(e)}"
        ).__dict__), 500

@moderator_bp.route('/improve', methods=['POST'])
def improve():
    """Improve content."""
    try:
        data = request.get_json()
        if not data:
            return jsonify(ApiResponse(
                success=False,
                error="Invalid JSON data"
            ).__dict__), 400

        # Create and validate the request
        improvement_request = ImprovementRequest(
            content=data.get('content', ''),
            feedback=data.get('feedback', '')
        )
        valid, error = improvement_request.validate()
        if not valid:
            return jsonify(ApiResponse(
                success=False,
                error=error
            ).__dict__), 400

        # Improve the content
        improvement = improve_content(
            improvement_request.content,
            improvement_request.feedback
        )

        return jsonify(ApiResponse(
            success=True,
            data=improvement
        ).__dict__)

    except Exception as e:
        current_app.logger.error(f"Error in improve: {str(e)}")
        return jsonify(ApiResponse(
            success=False,
            error=f"Improvement failed: {str(e)}"
        ).__dict__), 500
```

### Main Application Runner

Create the main application runner:

```python
# run.py
import os
from app import create_app

app = create_app(os.getenv('FLASK_CONFIG', 'default'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Advanced: Implementing a Web Interface

Add HTML templates for the web interface:

```html
<!-- app/templates/base.html -->
<!DOCTYPE html>
<html>
<head>
  <title>{% block title %}Sifaka Flask{% endblock %}</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
  <header>
    <h1>Sifaka Flask</h1>
    <nav>
      <ul>
        <li><a href="{{ url_for('web.index') }}">Home</a></li>
        <li><a href="{{ url_for('web.analyze_form') }}">Analyzer</a></li>
        <li><a href="{{ url_for('web.moderate_form') }}">Moderator</a></li>
      </ul>
    </nav>
  </header>

  <main>
    {% for message in get_flashed_messages() %}
      <div class="flash">{{ message }}</div>
    {% endfor %}

    {% block content %}{% endblock %}
  </main>

  <footer>
    <p>Powered by Sifaka</p>
  </footer>
</body>
</html>
```

```html
<!-- app/templates/index.html -->
{% extends "base.html" %}

{% block title %}Sifaka Flask - Home{% endblock %}

{% block content %}
  <h2>Welcome to Sifaka Flask</h2>
  <p>This is a demonstration of integrating Sifaka with Flask for text analysis and content moderation.</p>

  <div class="features">
    <div class="feature-card">
      <h3>Text Analysis</h3>
      <p>Analyze text for sentiment, toxicity, topics, and entities.</p>
      <a href="{{ url_for('web.analyze_form') }}" class="button">Try It</a>
    </div>

    <div class="feature-card">
      <h3>Content Moderation</h3>
      <p>Moderate content for toxicity and other issues.</p>
      <a href="{{ url_for('web.moderate_form') }}" class="button">Try It</a>
    </div>

    <div class="feature-card">
      <h3>Content Improvement</h3>
      <p>Improve content based on feedback.</p>
      <a href="{{ url_for('web.improve_form') }}" class="button">Try It</a>
    </div>
  </div>
{% endblock %}
```

```html
<!-- app/templates/analyze.html -->
{% extends "base.html" %}

{% block title %}Sifaka Flask - Text Analysis{% endblock %}

{% block content %}
  <h2>Text Analysis</h2>
  <p>Enter text to analyze for sentiment, toxicity, topics, and entities.</p>

  <form method="post" action="{{ url_for('web.analyze_submit') }}">
    <div class="form-group">
      <label for="text">Text to Analyze:</label>
      <textarea id="text" name="text" rows="5" required></textarea>
    </div>

    <button type="submit" class="button">Analyze</button>
  </form>

  {% if result %}
    <div class="result">
      <h3>Analysis Results</h3>

      <div class="result-section">
        <h4>Sentiment</h4>
        <p>Label: <strong>{{ result.sentiment.label }}</strong></p>
        <p>Confidence: {{ "%.2f"|format(result.sentiment.confidence) }}</p>
        {% if result.sentiment.compound_score %}
          <p>Compound Score: {{ "%.2f"|format(result.sentiment.compound_score) }}</p>
        {% endif %}
      </div>

      <div class="result-section">
        <h4>Toxicity</h4>
        <p>Label: <strong>{{ result.toxicity.label }}</strong></p>
        <p>Confidence: {{ "%.2f"|format(result.toxicity.confidence) }}</p>
      </div>

      <div class="result-section">
        <h4>Topic</h4>
        <p>Primary Topic: <strong>{{ result.topic.label }}</strong></p>
        <p>Confidence: {{ "%.2f"|format(result.topic.confidence) }}</p>
        {% if result.topic.topics %}
          <p>Related Topics:</p>
          <ul>
            {% for topic in result.topic.topics %}
              <li>{{ topic }}</li>
            {% endfor %}
          </ul>
        {% endif %}
      </div>

      {% if result.entities %}
        <div class="result-section">
          <h4>Entities</h4>
          {% for entity_type, entities in result.entities.items() %}
            <p>{{ entity_type|title }}:</p>
            <ul>
              {% for entity in entities %}
                <li>{{ entity }}</li>
              {% endfor %}
            </ul>
          {% endfor %}
        </div>
      {% endif %}
    </div>
  {% endif %}
{% endblock %}
```

Add the web routes:

```python
# app/routes/web.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.services.sifaka_service import analyze_text, moderate_content, improve_content

# Blueprint definition
web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@web_bp.route('/analyze', methods=['GET'])
def analyze_form():
    """Text analysis form."""
    return render_template('analyze.html')

@web_bp.route('/analyze', methods=['POST'])
def analyze_submit():
    """Process text analysis form submission."""
    text = request.form.get('text', '')
    if not text or not text.strip():
        flash('Please enter text to analyze.')
        return redirect(url_for('web.analyze_form'))

    result = analyze_text(text)
    return render_template('analyze.html', result=result)

@web_bp.route('/moderate', methods=['GET'])
def moderate_form():
    """Content moderation form."""
    return render_template('moderate.html')

@web_bp.route('/moderate', methods=['POST'])
def moderate_submit():
    """Process content moderation form submission."""
    content = request.form.get('content', '')
    if not content or not content.strip():
        flash('Please enter content to moderate.')
        return redirect(url_for('web.moderate_form'))

    result = moderate_content(content)
    return render_template('moderate.html', result=result)

@web_bp.route('/improve', methods=['GET'])
def improve_form():
    """Content improvement form."""
    return render_template('improve.html')

@web_bp.route('/improve', methods=['POST'])
def improve_submit():
    """Process content improvement form submission."""
    content = request.form.get('content', '')
    feedback = request.form.get('feedback', '')

    if not content or not content.strip():
        flash('Please enter content to improve.')
        return redirect(url_for('web.improve_form'))

    if not feedback or not feedback.strip():
        flash('Please enter feedback for improvement.')
        return redirect(url_for('web.improve_form'))

    result = improve_content(content, feedback)
    return render_template('improve.html', result=result)
```

Add the web blueprint to the application:

```python
# app/__init__.py (updated)
def register_blueprints(app):
    """Register Flask blueprints."""
    from app.routes.analyzer import analyzer_bp
    from app.routes.moderator import moderator_bp
    from app.routes.web import web_bp

    app.register_blueprint(analyzer_bp, url_prefix='/api/analyzer')
    app.register_blueprint(moderator_bp, url_prefix='/api/moderator')
    app.register_blueprint(web_bp)  # No prefix for web routes
```

## Middleware for Content Moderation

Add a middleware for automatic content moderation:

```python
# app/services/middleware.py
from functools import wraps
from flask import request, jsonify, current_app
from app.services.sifaka_service import moderate_content
from app.models.content import ApiResponse

def moderate_content_middleware(f):
    """Middleware to automatically moderate content in requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Only check for POST requests with JSON content
        if request.method == 'POST' and request.is_json:
            if not current_app.config['ENABLE_CONTENT_MODERATION']:
                return f(*args, **kwargs)

            data = request.get_json()

            # Check if there's content to moderate
            if 'content' in data:
                content = data['content']

                # Moderate the content
                moderation = moderate_content(content)

                # If moderation failed, return an error
                if not moderation['passed']:
                    return jsonify(ApiResponse(
                        success=False,
                        error='Content moderation failed',
                        data=moderation
                    ).__dict__), 400

        # Continue to the endpoint function
        return f(*args, **kwargs)

    return decorated_function
```

Apply the middleware to the routes:

```python
# app/routes/moderator.py (updated)
from app.services.middleware import moderate_content_middleware

@moderator_bp.route('/improve', methods=['POST'])
@moderate_content_middleware  # Apply the middleware
def improve():
    # ... (existing code)
```

## Testing the Application

Create tests for the Flask application:

```python
# tests/test_api.py
import pytest
from app import create_app

@pytest.fixture
def app():
    app = create_app('testing')
    yield app

@pytest.fixture
def client(app):
    return app.test_client()

def test_analyze_endpoint(client):
    """Test the text analysis endpoint."""
    response = client.post(
        '/api/analyzer/analyze',
        json={'text': 'This is a test of the analyzer.'}
    )

    assert response.status_code == 200
    data = response.get_json()

    assert data['success'] is True
    assert 'data' in data

    result = data['data']
    assert 'sentiment' in result
    assert 'toxicity' in result
    assert 'topic' in result

def test_moderate_endpoint(client):
    """Test the content moderation endpoint."""
    response = client.post(
        '/api/moderator/moderate',
        json={'content': 'This is a test of the moderator.'}
    )

    assert response.status_code == 200
    data = response.get_json()

    assert data['success'] is True
    assert 'data' in data

    result = data['data']
    assert 'passed' in result
    assert 'message' in result
    assert 'label' in result
    assert 'confidence' in result

def test_improve_endpoint(client):
    """Test the content improvement endpoint."""
    response = client.post(
        '/api/moderator/improve',
        json={
            'content': 'This is a test of the improver.',
            'feedback': 'Make it more formal.'
        }
    )

    assert response.status_code == 200
    data = response.get_json()

    assert data['success'] is True
    assert 'data' in data

    result = data['data']
    assert 'original' in result
    assert 'improved' in result
    assert 'feedback' in result

def test_empty_text(client):
    """Test with empty text."""
    response = client.post(
        '/api/analyzer/analyze',
        json={'text': ''}
    )

    assert response.status_code == 400
    data = response.get_json()

    assert data['success'] is False
    assert 'error' in data
```

## Deployment Configuration

Create a WSGI file for deployment:

```python
# wsgi.py
from app import create_app

application = create_app('production')

if __name__ == '__main__':
    application.run()
```

Create a Dockerfile:

```Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV FLASK_APP=wsgi.py
ENV FLASK_ENV=production
ENV FLASK_CONFIG=production

# Run gunicorn
CMD ["gunicorn", "wsgi:application", "--bind", "0.0.0.0:5000", "--workers", "4"]
```

Docker Compose configuration:

```yaml
# docker-compose.yml
version: '3'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_APP=wsgi.py
      - FLASK_ENV=production
      - FLASK_CONFIG=production
      - SECRET_KEY=${SECRET_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - SIFAKA_CACHE_SIZE=1000
      - ENABLE_CONTENT_MODERATION=True
    volumes:
      - ./instance:/app/instance
    restart: always
```

## Summary

This integration guide demonstrates how to use Sifaka with Flask for:

1. Text analysis and classification
2. Content moderation
3. Content improvement
4. Batch processing
5. Web interface for user interaction
6. Middleware for automatic content moderation
7. Deployment configuration

Flask's simplicity and flexibility make it a great choice for building applications with Sifaka's text analysis capabilities. The combination allows for both API-based services and web applications with minimal boilerplate code.