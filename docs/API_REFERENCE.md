# Sifaka API Reference

This document provides a comprehensive reference for the Sifaka API, including all major classes, methods, and functions.

## Table of Contents

- [Core Components](#core-components)
  - [Chain](#chain)
  - [Models](#models)
  - [Validators](#validators)
  - [Critics](#critics)
  - [Retrievers](#retrievers)
  - [Classifiers](#classifiers)
- [Configuration](#configuration)
- [Registry System](#registry-system)
- [Error Handling](#error-handling)

## Core Components

### Chain

The central orchestrator that coordinates the entire process.

#### Classes

- **Chain**
  - **Methods**:
    - `__init__(config: Optional[SifakaConfig] = None, model_factory: Optional[Callable[[str, str], Model]] = None)`: Initialize a new Chain instance
    - `with_model(model: Union[str, Model]) -> Chain`: Set the model to use for text generation
    - `with_prompt(prompt: str) -> Chain`: Set the prompt to use for text generation
    - `validate_with(validator: Validator) -> Chain`: Add a validator to the chain
    - `improve_with(improver: Improver) -> Chain`: Add an improver (critic) to the chain
    - `with_options(**options: Any) -> Chain`: Set options to pass to the model during generation
    - `with_config(config: SifakaConfig) -> Chain`: Set the configuration for the chain
    - `run() -> Result`: Execute the chain and return the result

### Models

Language model integrations for text generation.

#### Base Classes

- **Model** (Protocol)
  - **Methods**:
    - `generate(prompt: str, **options: Any) -> str`: Generate text from a prompt
    - `count_tokens(text: str) -> int`: Count the number of tokens in text
    - `get_model_name() -> str`: Get the name of the model

#### Implementations

- **OpenAIModel**
  - **Methods**:
    - `__init__(model_name: str, api_key: Optional[str] = None, **options: Any)`: Initialize a new OpenAI model
    - `generate(prompt: str, **options: Any) -> str`: Generate text using the OpenAI API
    - `count_tokens(text: str) -> int`: Count tokens using the OpenAI tokenizer
    - `get_model_name() -> str`: Get the OpenAI model name

- **AnthropicModel**
  - **Methods**:
    - `__init__(model_name: str, api_key: Optional[str] = None, **options: Any)`: Initialize a new Anthropic model
    - `generate(prompt: str, **options: Any) -> str`: Generate text using the Anthropic API
    - `count_tokens(text: str) -> int`: Count tokens using the Anthropic tokenizer
    - `get_model_name() -> str`: Get the Anthropic model name

- **GeminiModel**
  - **Methods**:
    - `__init__(model_name: str, api_key: Optional[str] = None, **options: Any)`: Initialize a new Gemini model
    - `generate(prompt: str, **options: Any) -> str`: Generate text using the Gemini API
    - `count_tokens(text: str) -> int`: Count tokens using the Gemini tokenizer
    - `get_model_name() -> str`: Get the Gemini model name

### Validators

Components that check if text meets specific criteria.

#### Base Classes

- **Validator** (Protocol)
  - **Methods**:
    - `validate(text: str) -> ValidationResult`: Validate text against criteria

#### Implementations

- **LengthValidator**
  - **Factory Function**: `length(min_words: Optional[int] = None, max_words: Optional[int] = None, min_chars: Optional[int] = None, max_chars: Optional[int] = None) -> Validator`
  - **Methods**:
    - `validate(text: str) -> ValidationResult`: Validate text length

- **FormatValidator**
  - **Factory Function**: `format(format_type: str, schema: Optional[Dict[str, Any]] = None, custom_validator: Optional[Callable[[str], Dict[str, Any]]] = None) -> Validator`
  - **Methods**:
    - `validate(text: str) -> ValidationResult`: Validate text format (JSON, Markdown, etc.)

- **ProhibitedContentValidator**
  - **Factory Function**: `prohibited_content(prohibited: List[str]) -> Validator`
  - **Methods**:
    - `validate(text: str) -> ValidationResult`: Check if text contains prohibited content

- **ClassifierValidator**
  - **Factory Function**: `classifier_validator(classifier: Classifier, threshold: float = 0.5, pass_on: str = "positive") -> Validator`
  - **Methods**:
    - `validate(text: str) -> ValidationResult`: Validate text using a classifier

### Critics

Components that improve text quality through various techniques.

#### Base Classes

- **Critic** (Abstract Base Class)
  - **Methods**:
    - `__init__(model: Model, name: Optional[str] = None, **options: Any)`: Initialize a critic
    - `improve(text: str) -> tuple[str, ImprovementResult]`: Improve text using this critic
    - `_critique(text: str) -> Dict[str, Any]`: Critique text (to be implemented by subclasses)
    - `_improve(text: str, critique: Dict[str, Any]) -> str`: Improve text based on critique (to be implemented by subclasses)

#### Implementations

- **ReflexionCritic**
  - **Factory Function**: `create_reflexion_critic(model: Union[str, Model], **options: Any) -> Critic`
  - **Methods**:
    - `_critique(text: str) -> Dict[str, Any]`: Critique text using self-reflection
    - `_improve(text: str, critique: Dict[str, Any]) -> str`: Improve text based on self-reflection

- **NCriticsCritic**
  - **Factory Function**: `create_n_critics_critic(model: Union[str, Model], num_critics: int = 3, **options: Any) -> Critic`
  - **Methods**:
    - `_critique(text: str) -> Dict[str, Any]`: Generate multiple independent critiques
    - `_improve(text: str, critique: Dict[str, Any]) -> str`: Synthesize improvements from multiple critiques

- **SelfRAGCritic**
  - **Factory Function**: `create_self_rag_critic(model: Union[str, Model], retriever: Retriever, **options: Any) -> Critic`
  - **Methods**:
    - `_critique(text: str) -> Dict[str, Any]`: Critique text using retrieved information
    - `_improve(text: str, critique: Dict[str, Any]) -> str`: Improve text using retrieved information

- **ConstitutionalCritic**
  - **Factory Function**: `create_constitutional_critic(model: Union[str, Model], principles: List[str], **options: Any) -> Critic`
  - **Methods**:
    - `_critique(text: str) -> Dict[str, Any]`: Critique text against constitutional principles
    - `_improve(text: str, critique: Dict[str, Any]) -> str`: Improve text to adhere to constitutional principles

### Retrievers

Components that retrieve relevant information for queries.

#### Base Classes

- **Retriever** (Abstract Base Class)
  - **Methods**:
    - `retrieve(query: str, top_k: int = 5, **kwargs: Any) -> List[str]`: Retrieve relevant documents for a query

#### Implementations

- **SimpleRetriever**
  - **Methods**:
    - `__init__(documents: List[str], **kwargs: Any)`: Initialize with a list of documents
    - `retrieve(query: str, top_k: int = 5, **kwargs: Any) -> List[str]`: Retrieve documents using simple keyword matching

- **ElasticsearchRetriever**
  - **Methods**:
    - `__init__(index: str, host: str = "localhost", port: int = 9200, **kwargs: Any)`: Initialize with Elasticsearch connection details
    - `retrieve(query: str, top_k: int = 5, **kwargs: Any) -> List[str]`: Retrieve documents using Elasticsearch

- **MilvusRetriever**
  - **Methods**:
    - `__init__(collection_name: str, embedding_field: str = "embedding", text_field: str = "text", **kwargs: Any)`: Initialize with Milvus connection details
    - `retrieve(query: str, top_k: int = 5, **kwargs: Any) -> List[str]`: Retrieve documents using Milvus vector database

### Classifiers

Components that categorize text into specific classes.

#### Base Classes

- **Classifier** (Protocol)
  - **Methods**:
    - `classify(text: str, **options: Any) -> ClassificationResult`: Classify text into categories
    - `get_classifier_name() -> str`: Get the name of the classifier

#### Implementations

- **ProfanityClassifier**
  - **Methods**:
    - `__init__(custom_words: Optional[List[str]] = None, censor_char: str = "*")`: Initialize with optional custom profane words
    - `classify(text: str, **options: Any) -> ClassificationResult`: Classify text for profanity
    - `get_classifier_name() -> str`: Get the classifier name

- **SentimentClassifier**
  - **Methods**:
    - `__init__(**options: Any)`: Initialize the sentiment classifier
    - `classify(text: str, **options: Any) -> ClassificationResult`: Classify text sentiment
    - `get_classifier_name() -> str`: Get the classifier name

## Configuration

- **SifakaConfig**
  - **Properties**:
    - `model: ModelConfig`: Configuration for models
    - `debug: bool`: Enable debug mode
    - `registry: RegistryConfig`: Configuration for the registry system

- **ModelConfig**
  - **Properties**:
    - `temperature: float`: Temperature for text generation
    - `max_tokens: int`: Maximum number of tokens to generate
    - `top_p: float`: Top-p sampling parameter

## Registry System

- **Registry**
  - **Methods**:
    - `register(component_type: str, name: str, factory: Callable[..., Any]) -> None`: Register a component factory
    - `create(component_type: str, name: str, **kwargs: Any) -> Any`: Create a component instance
    - `get_factory(component_type: str, name: str) -> Callable[..., Any]`: Get a component factory

## Error Handling

- **ValidationError**
  - **Properties**:
    - `message: str`: Error message
    - `component: str`: Component that raised the error
    - `operation: str`: Operation that failed
    - `suggestions: List[str]`: Suggestions for fixing the error

- **ModelError**
  - **Properties**:
    - `message: str`: Error message
    - `model_name: str`: Name of the model that raised the error
    - `operation: str`: Operation that failed
    - `provider: str`: Provider of the model (e.g., OpenAI, Anthropic)

- **RetrieverError**
  - **Properties**:
    - `message: str`: Error message
    - `retriever_name: str`: Name of the retriever that raised the error
    - `operation: str`: Operation that failed
