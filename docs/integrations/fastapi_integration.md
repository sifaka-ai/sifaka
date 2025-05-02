# FastAPI Integration Guide

This guide demonstrates how to integrate Sifaka with FastAPI applications for text analysis, content moderation, and validation.

## Setup

First, install FastAPI and Sifaka:

```bash
pip install fastapi uvicorn
pip install sifaka
```

## Basic FastAPI Integration

### Project Structure

Create a basic FastAPI project structure:

```
sifaka_fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── analyzer.py
│   │   │   └── moderation.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── sifaka.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── content.py
│   │   └── response.py
│   └── services/
│       ├── __init__.py
│       └── analysis_service.py
└── requirements.txt
```

### Core Configuration

First, let's set up the configuration:

```python
# app/core/config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings."""

    APP_NAME: str = "Sifaka FastAPI"
    API_V1_PREFIX: str = "/api/v1"

    # Sifaka configuration
    SIFAKA_CACHE_SIZE: int = 1000
    SIFAKA_DEFAULT_TOXICITY_MODEL: str = "original"
    SIFAKA_OPENAI_API_KEY: str = ""
    SIFAKA_ANTHROPIC_API_KEY: str = ""

    # Additional settings
    ENABLE_CONTENT_MODERATION: bool = True
    TOXICITY_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"

# Create global settings object
settings = Settings()
```

### Sifaka Initialization

Next, we'll create a module for initializing Sifaka components:

```python
# app/core/sifaka.py
from typing import Dict, Any

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

from .config import settings


class SifakaComponents:
    """Container for Sifaka components."""

    def __init__(self) -> None:
        # Initialize classifiers
        self.toxicity_classifier = create_toxicity_classifier(
            model_name=settings.SIFAKA_DEFAULT_TOXICITY_MODEL,
            cache_size=settings.SIFAKA_CACHE_SIZE
        )

        self.sentiment_classifier = create_sentiment_classifier(
            cache_size=settings.SIFAKA_CACHE_SIZE
        )

        self.topic_classifier = create_topic_classifier(
            cache_size=settings.SIFAKA_CACHE_SIZE
        )

        self.ner_classifier = create_ner_classifier(
            cache_size=settings.SIFAKA_CACHE_SIZE
        )

        # Initialize rules
        self.toxicity_rule = create_prohibited_content_rule(
            classifier=self.toxicity_classifier,
            valid_labels=["non_toxic"],
            min_confidence=settings.TOXICITY_THRESHOLD
        )

        self.sentiment_rule = create_sentiment_rule(
            classifier=self.sentiment_classifier,
            valid_labels=["positive", "neutral"]
        )

        # Initialize critic
        self.content_critic = create_content_critic()


    def classify_text(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive text analysis."""
        sentiment_result = self.sentiment_classifier.classify(text)
        toxicity_result = self.toxicity_classifier.classify(text)
        topic_result = self.topic_classifier.classify(text)
        entities_result = self.ner_classifier.classify(text)

        # Extract entities
        entities = {}
        for entity in entities_result.metadata.get("entities", []):
            entity_type = entity.get("type", "unknown")
            entity_text = entity.get("text", "")
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entity_text)

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
        }

    def moderate_content(self, content: str) -> Dict[str, Any]:
        """Moderate content using toxicity rule."""
        result = self.toxicity_rule.validate(content)

        return {
            "passed": result.passed,
            "message": result.message,
            "label": result.metadata.get("label", "unknown"),
            "confidence": result.metadata.get("confidence", 0),
        }

    def improve_content(self, content: str, feedback: str) -> Dict[str, Any]:
        """Improve content using critic."""
        result = self.content_critic.improve_with_feedback(content, feedback)

        return {
            "original": content,
            "improved": result,
        }


# Create global components instance
sifaka_components = SifakaComponents()
```

### Pydantic Models

Define Pydantic models for request and response validation:

```python
# app/models/content.py
from pydantic import BaseModel, Field, validator


class ContentAnalysisRequest(BaseModel):
    """Request model for content analysis."""

    text: str = Field(..., min_length=1, max_length=10000, example="This is a sample text for analysis.")

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v


class ContentModerationRequest(BaseModel):
    """Request model for content moderation."""

    content: str = Field(..., min_length=1, max_length=10000, example="This is user-generated content to moderate.")

    @validator("content")
    def content_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v


class ContentImprovementRequest(BaseModel):
    """Request model for content improvement."""

    content: str = Field(..., min_length=1, max_length=10000, example="This is poorly written content.")
    feedback: str = Field(..., min_length=1, max_length=1000, example="Improve grammar and clarity.")

    @validator("content", "feedback")
    def fields_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Fields cannot be empty or whitespace only")
        return v


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""

    texts: list[str] = Field(..., min_items=1, max_items=100, example=["Text 1", "Text 2"])

    @validator("texts")
    def texts_must_not_be_empty(cls, v):
        if not all(t.strip() for t in v):
            raise ValueError("Texts cannot contain empty items")
        return v
```

```python
# app/models/response.py
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class StandardResponse(BaseModel):
    """Standard API response model."""

    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if any")


class AnalysisResult(BaseModel):
    """Result of text analysis."""

    sentiment: Dict[str, Any] = Field(..., description="Sentiment analysis results")
    toxicity: Dict[str, Any] = Field(..., description="Toxicity analysis results")
    topic: Dict[str, Any] = Field(..., description="Topic analysis results")
    entities: Dict[str, List[str]] = Field(default_factory=dict, description="Named entities found")


class ModerationResult(BaseModel):
    """Result of content moderation."""

    passed: bool = Field(..., description="Whether the content passed moderation")
    message: str = Field(..., description="Moderation message")
    label: str = Field(..., description="Classification label")
    confidence: float = Field(..., description="Classification confidence")


class ImprovementResult(BaseModel):
    """Result of content improvement."""

    original: str = Field(..., description="Original content")
    improved: str = Field(..., description="Improved content")


class BatchAnalysisResult(BaseModel):
    """Result of batch analysis."""

    results: List[Dict[str, Any]] = Field(..., description="Analysis results for each text")
    meta: Dict[str, Any] = Field(..., description="Metadata about the batch operation")
```

### Dependencies

Set up FastAPI dependencies for easier access to Sifaka components:

```python
# app/api/dependencies.py
from typing import Callable, Any
from functools import lru_cache

from fastapi import Depends

from app.core.config import settings
from app.core.sifaka import sifaka_components


@lru_cache()
def get_settings():
    """Dependency for application settings."""
    return settings


def get_sifaka_components():
    """Dependency for Sifaka components."""
    return sifaka_components


def get_moderation_middleware(components=Depends(get_sifaka_components)):
    """Factory for content moderation middleware."""

    async def moderate_content(content: str) -> dict:
        """Moderate content and return the result."""
        if not settings.ENABLE_CONTENT_MODERATION:
            return {"passed": True}

        return components.moderate_content(content)

    return moderate_content
```

### Endpoint Implementation

Now, let's implement the API endpoints:

```python
# app/api/endpoints/analyzer.py
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.sifaka import sifaka_components
from app.api.dependencies import get_sifaka_components
from app.models.content import (
    ContentAnalysisRequest,
    BatchAnalysisRequest
)
from app.models.response import (
    StandardResponse,
    AnalysisResult
)


router = APIRouter()


@router.post("/analyze", response_model=StandardResponse)
async def analyze_text(
    request: ContentAnalysisRequest,
    sifaka: sifaka_components = Depends(get_sifaka_components)
):
    """Analyze text for sentiment, toxicity, topics and entities."""
    try:
        analysis = sifaka.classify_text(request.text)

        return StandardResponse(
            success=True,
            data=analysis
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )


@router.post("/batch-analyze", response_model=StandardResponse)
async def batch_analyze(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    sifaka: sifaka_components = Depends(get_sifaka_components)
):
    """Analyze multiple texts in a batch."""
    try:
        # For small batches, we can process synchronously
        if len(request.texts) <= 5:
            results = []
            for text in request.texts:
                analysis = sifaka.classify_text(text)
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "analysis": analysis
                })

            return StandardResponse(
                success=True,
                data={
                    "results": results,
                    "meta": {
                        "count": len(results),
                        "processed": "synchronously"
                    }
                }
            )

        # For larger batches, we should process asynchronously
        # In a real app, you'd use a task queue like Celery
        # Here we'll use FastAPI's background tasks as a simple example

        # This is a placeholder for the results
        task_id = f"batch_{hash(tuple(request.texts))}"

        # Define background task
        async def process_batch():
            results = []
            for text in request.texts:
                analysis = sifaka.classify_text(text)
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "analysis": analysis
                })

            # In a real app, you'd store these results somewhere
            print(f"Completed batch analysis {task_id} with {len(results)} results")

        # Add the task to the background tasks
        background_tasks.add_task(process_batch)

        return StandardResponse(
            success=True,
            data={
                "task_id": task_id,
                "message": f"Processing {len(request.texts)} texts in the background",
                "meta": {
                    "count": len(request.texts),
                    "processed": "asynchronously"
                }
            }
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"Batch analysis failed: {str(e)}"
        )
```

```python
# app/api/endpoints/moderation.py
from fastapi import APIRouter, Depends, HTTPException

from app.core.config import settings
from app.core.sifaka import sifaka_components
from app.api.dependencies import get_sifaka_components
from app.models.content import (
    ContentModerationRequest,
    ContentImprovementRequest
)
from app.models.response import (
    StandardResponse,
    ModerationResult,
    ImprovementResult
)


router = APIRouter()


@router.post("/moderate", response_model=StandardResponse)
async def moderate_content(
    request: ContentModerationRequest,
    sifaka: sifaka_components = Depends(get_sifaka_components)
):
    """Moderate content for toxicity and other issues."""
    try:
        moderation = sifaka.moderate_content(request.content)

        return StandardResponse(
            success=True,
            data=moderation
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"Moderation failed: {str(e)}"
        )


@router.post("/improve", response_model=StandardResponse)
async def improve_content(
    request: ContentImprovementRequest,
    sifaka: sifaka_components = Depends(get_sifaka_components)
):
    """Improve content based on feedback."""
    try:
        improvement = sifaka.improve_content(request.content, request.feedback)

        return StandardResponse(
            success=True,
            data=improvement
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"Improvement failed: {str(e)}"
        )
```

### Main Application

Finally, let's create the main application file:

```python
# app/main.py
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json

from app.core.config import settings
from app.api.endpoints import analyzer, moderation
from app.api.dependencies import get_moderation_middleware


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="FastAPI application with Sifaka integration for text analysis and moderation",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Content moderation middleware
@app.middleware("http")
async def content_moderation_middleware(request: Request, call_next):
    # Only check POST requests with JSON content
    if (request.method == "POST" and
            request.headers.get("content-type") == "application/json" and
            settings.ENABLE_CONTENT_MODERATION):

        # Get the moderation function
        moderate_content = get_moderation_middleware()

        try:
            # Read request body
            body = await request.body()
            if body:
                data = json.loads(body)

                # Check for content field
                if "content" in data:
                    content = data["content"]

                    # Moderate content
                    result = await moderate_content(content)

                    # If moderation failed, return error response
                    if not result["passed"]:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "success": False,
                                "error": "Content moderation failed",
                                "details": result
                            }
                        )

            # Reconstruct the request body
            # https://github.com/tiangolo/fastapi/issues/394#issuecomment-730108949
            async def receive():
                return {"type": "http.request", "body": body}

            request._receive = receive
        except json.JSONDecodeError:
            # Not valid JSON, proceed normally
            pass

    # Continue with the request
    return await call_next(request)


# Add routers
app.include_router(
    analyzer.router,
    prefix=f"{settings.API_V1_PREFIX}/analyzer",
    tags=["analyzer"]
)

app.include_router(
    moderation.router,
    prefix=f"{settings.API_V1_PREFIX}/moderation",
    tags=["moderation"]
)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {"message": f"Welcome to {settings.APP_NAME}"}


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

## Advanced: Implementing a Service Layer

For better separation of concerns, you might want to add a service layer:

```python
# app/services/analysis_service.py
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.sifaka import sifaka_components


class AnalysisService:
    """Service for text analysis operations."""

    def __init__(self, components: sifaka_components):
        self.components = components
        self.executor = ThreadPoolExecutor(max_workers=4)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze a single text."""
        return self.components.classify_text(text)

    async def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze a batch of texts asynchronously."""
        loop = asyncio.get_event_loop()

        # Create tasks for each text
        tasks = []
        for text in texts:
            task = loop.run_in_executor(
                self.executor,
                self.analyze_text,
                text
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        return [
            {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "analysis": result
            }
            for text, result in zip(texts, results)
        ]

    def moderate_content(self, content: str) -> Dict[str, Any]:
        """Moderate content."""
        return self.components.moderate_content(content)

    def improve_content(self, content: str, feedback: str) -> Dict[str, Any]:
        """Improve content."""
        return self.components.improve_content(content, feedback)


# Create analysis service factory
def get_analysis_service(components: sifaka_components) -> AnalysisService:
    """Create an analysis service."""
    return AnalysisService(components)
```

Then update your dependency injection:

```python
# app/api/dependencies.py (updated)
from typing import Callable, Any
from functools import lru_cache

from fastapi import Depends

from app.core.config import settings
from app.core.sifaka import sifaka_components
from app.services.analysis_service import AnalysisService, get_analysis_service


@lru_cache()
def get_settings():
    """Dependency for application settings."""
    return settings


def get_sifaka_components():
    """Dependency for Sifaka components."""
    return sifaka_components


def get_analysis_service(
    components: sifaka_components = Depends(get_sifaka_components)
) -> AnalysisService:
    """Dependency for analysis service."""
    return AnalysisService(components)


def get_moderation_middleware(service: AnalysisService = Depends(get_analysis_service)):
    """Factory for content moderation middleware."""

    async def moderate_content(content: str) -> dict:
        """Moderate content and return the result."""
        if not settings.ENABLE_CONTENT_MODERATION:
            return {"passed": True}

        return service.moderate_content(content)

    return moderate_content
```

And update your endpoint implementation:

```python
# app/api/endpoints/analyzer.py (updated)
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.dependencies import get_analysis_service
from app.services.analysis_service import AnalysisService
from app.models.content import (
    ContentAnalysisRequest,
    BatchAnalysisRequest
)
from app.models.response import (
    StandardResponse,
    AnalysisResult
)


router = APIRouter()


@router.post("/analyze", response_model=StandardResponse)
async def analyze_text(
    request: ContentAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
):
    """Analyze text for sentiment, toxicity, topics and entities."""
    try:
        analysis = service.analyze_text(request.text)

        return StandardResponse(
            success=True,
            data=analysis
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )


@router.post("/batch-analyze", response_model=StandardResponse)
async def batch_analyze(
    request: BatchAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
):
    """Analyze multiple texts in a batch."""
    try:
        # Process asynchronously
        results = await service.analyze_batch(request.texts)

        return StandardResponse(
            success=True,
            data={
                "results": results,
                "meta": {
                    "count": len(results),
                    "processed": "asynchronously"
                }
            }
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"Batch analysis failed: {str(e)}"
        )
```

## Testing

Create tests for your FastAPI application:

```python
# tests/test_analyzer.py
from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


def test_analyze_text():
    """Test the text analysis endpoint."""
    response = client.post(
        "/api/v1/analyzer/analyze",
        json={"text": "This is a test of the analyzer."}
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert "data" in result

    data = result["data"]
    assert "sentiment" in data
    assert "toxicity" in data
    assert "topic" in data
    assert "entities" in data


def test_batch_analyze():
    """Test the batch analysis endpoint."""
    response = client.post(
        "/api/v1/analyzer/batch-analyze",
        json={"texts": ["This is test 1.", "This is test 2."]}
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert "data" in result
    assert "results" in result["data"]
    assert len(result["data"]["results"]) == 2


def test_empty_text():
    """Test with empty text."""
    response = client.post(
        "/api/v1/analyzer/analyze",
        json={"text": ""}
    )

    assert response.status_code == 422  # Validation error
```

## Deployment

For deploying your FastAPI application with Sifaka:

```Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Set environment variables
ENV APP_NAME="Sifaka FastAPI"
ENV SIFAKA_CACHE_SIZE=1000
ENV ENABLE_CONTENT_MODERATION=true

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Docker Compose configuration:

```yaml
# docker-compose.yml
version: '3'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APP_NAME=Sifaka FastAPI
      - SIFAKA_CACHE_SIZE=1000
      - ENABLE_CONTENT_MODERATION=true
      - SIFAKA_OPENAI_API_KEY=${OPENAI_API_KEY}
      - SIFAKA_ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./app:/app/app
```

## Summary

This integration guide demonstrates how to use Sifaka with FastAPI for:

1. Text analysis and classification
2. Content moderation
3. Content improvement
4. Batch processing
5. Middleware for automatic content moderation
6. Dependency injection for easier testing
7. Service layer for better separation of concerns
8. Deployment configuration

FastAPI's modern async architecture works well with Sifaka's comprehensive text analysis capabilities, making it an excellent choice for building high-performance text analysis APIs.