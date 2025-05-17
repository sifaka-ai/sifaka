# Retrievers in Sifaka

Retrievers are components in the Sifaka framework that retrieve relevant documents or information for a given query. They are primarily used by retrieval-augmented critics like Self-RAG to enhance text generation with external knowledge.

## Overview

Retrievers implement a common interface defined by the `Retriever` abstract base class, which requires implementing a `retrieve` method that takes a query string and returns a list of relevant document texts.

The Sifaka framework provides several retriever implementations:

1. **SimpleRetriever** - A basic retriever that searches through a predefined collection of documents
2. **ElasticsearchRetriever** - A retriever that uses Elasticsearch for document retrieval
3. **MilvusRetriever** - A retriever that uses Milvus vector database for semantic search
4. **RetrievalAugmenter** - A utility that combines a retriever with a model to augment text generation

## Base Retriever Interface

All retrievers in Sifaka implement the following interface:

```python
class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.
        """
        pass
```

## SimpleRetriever

The `SimpleRetriever` is a basic implementation that searches through a predefined collection of documents. It's useful for simple use cases or testing.

```python
from sifaka.retrievers import SimpleRetriever

# Create a simple retriever with a collection of documents
documents = [
    "Sifaka is a framework for text generation and improvement.",
    "Retrievers are used to find relevant information for a query.",
    "Critics improve text quality through various techniques."
]

retriever = SimpleRetriever(documents)

# Retrieve documents relevant to a query
results = retriever.retrieve("What is Sifaka?")
print(results)  # Will return the most relevant documents
```

## ElasticsearchRetriever

The `ElasticsearchRetriever` uses Elasticsearch for document retrieval. It requires the Elasticsearch Python client to be installed.

```python
from sifaka.retrievers import create_elasticsearch_retriever

# Create an Elasticsearch retriever
retriever = create_elasticsearch_retriever(
    hosts=["http://localhost:9200"],
    index_name="documents",
    field_name="content"
)

# Retrieve documents relevant to a query
results = retriever.retrieve("What is Sifaka?")
```

### Installation

To use the ElasticsearchRetriever, you need to install the Elasticsearch Python client:

```bash
pip install elasticsearch
```

## MilvusRetriever

The `MilvusRetriever` uses the Milvus vector database for semantic search. It requires the Milvus Python client to be installed.

```python
from sifaka.retrievers import create_milvus_retriever

# Create a Milvus retriever
retriever = create_milvus_retriever(
    uri="http://localhost:19530",
    collection_name="documents",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Retrieve documents relevant to a query
results = retriever.retrieve("What is Sifaka?")
```

### Installation

To use the MilvusRetriever, you need to install the Milvus Python client and a sentence transformer model:

```bash
pip install pymilvus sentence-transformers
```

## RetrievalAugmenter

The `RetrievalAugmenter` is a utility that combines a retriever with a model to augment text generation with retrieved information.

```python
from sifaka.retrievers import create_retrieval_augmenter
from sifaka.models import create_model
from sifaka.retrievers import SimpleRetriever

# Create a model
model = create_model("openai:gpt-4")

# Create a simple retriever
documents = [
    "Sifaka is a framework for text generation and improvement.",
    "Retrievers are used to find relevant information for a query.",
    "Critics improve text quality through various techniques."
]
retriever = SimpleRetriever(documents)

# Create a retrieval augmenter
augmenter = create_retrieval_augmenter(
    model=model,
    retriever=retriever
)

# Generate text with retrieval augmentation
result = augmenter.generate("What is Sifaka and how does it use retrievers?")
print(result)
```

## Using Retrievers with Critics

Retrievers are commonly used with critics that implement retrieval-augmented generation, such as the Self-RAG critic:

```python
from sifaka import Chain
from sifaka.models import create_model
from sifaka.critics.self_rag import create_self_rag_critic
from sifaka.retrievers import SimpleRetriever

# Create a model
model = create_model("openai:gpt-4")

# Create a retriever
documents = [
    "Sifaka is a framework for text generation and improvement.",
    "Retrievers are used to find relevant information for a query.",
    "Critics improve text quality through various techniques."
]
retriever = SimpleRetriever(documents)

# Create a Self-RAG critic with the retriever
critic = create_self_rag_critic(
    model=model,
    retriever=retriever
)

# Use the critic in a chain
result = (Chain()
    .with_model(model)
    .with_prompt("What is Sifaka and how does it use retrievers?")
    .improve_with(critic)
    .run())

print(result.text)
```

## Creating Custom Retrievers

You can create custom retrievers by implementing the `Retriever` interface:

```python
from sifaka.retrievers import Retriever
from typing import List

class CustomRetriever(Retriever):
    def __init__(self, data_source):
        self.data_source = data_source
        
    def retrieve(self, query: str) -> List[str]:
        # Implement your custom retrieval logic here
        # This could involve querying a database, API, or any other data source
        results = []
        # ... your retrieval logic ...
        return results
```

## Next Steps

- Learn more about [Critics](CRITICS.md) that use retrievers
- Explore the [Self-RAG](CRITICS.md#self-rag) critic implementation
- Check out the [Retrieval-Enhanced](CRITICS.md#retrieval-enhanced) critic
