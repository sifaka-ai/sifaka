"""
Example of using Self-RAG with Elasticsearch and Milvus retrievers.

This example demonstrates how to use the Self-RAG critic with Elasticsearch and Milvus retrievers.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from sifaka.models import get_model
from sifaka.critics.self_rag import create_self_rag_critic
from sifaka.retrievers.factory import create_elasticsearch_retriever, create_milvus_retriever

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def dummy_embedding_model(text: str) -> List[float]:
    """
    Dummy embedding model for demonstration purposes.
    In a real application, you would use a proper embedding model.
    
    Args:
        text: Text to embed
        
    Returns:
        A list of floats representing the embedding
    """
    # This is just a dummy implementation
    # In a real application, you would use a proper embedding model
    # such as sentence-transformers, OpenAI embeddings, etc.
    return [0.1] * 768  # Return a 768-dimensional vector of 0.1


def dummy_retriever(query: str) -> List[str]:
    """
    Dummy retriever for demonstration purposes.
    In a real application, you would use a proper retriever.
    
    Args:
        query: Query to retrieve documents for
        
    Returns:
        A list of retrieved documents
    """
    # This is just a dummy implementation
    # In a real application, you would use a proper retriever
    logger.info(f"Retrieving documents for query: {query}")
    
    # Return some dummy documents based on the query
    if "history" in query.lower():
        return [
            "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
            "artificial beings endowed with intelligence or consciousness by master craftsmen.",
            "The field of AI research was founded at a workshop held on the campus of Dartmouth College "
            "during the summer of 1956."
        ]
    elif "application" in query.lower():
        return [
            "AI applications include advanced web search engines, recommendation systems, "
            "understanding human speech, self-driving cars, automated decision-making and "
            "competing at the highest level in strategic game systems.",
            "Many tools are used in AI, including versions of search and mathematical optimization, "
            "artificial neural networks, and methods based on statistics, probability and economics."
        ]
    else:
        return [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, "
            "as opposed to natural intelligence displayed by animals including humans.",
            "Leading AI textbooks define the field as the study of 'intelligent agents': "
            "any system that perceives its environment and takes actions that maximize "
            "its chance of achieving its goals."
        ]


def example_elasticsearch_retriever():
    """Example of using Self-RAG with Elasticsearch retriever."""
    logger.info("Example of using Self-RAG with Elasticsearch retriever")
    
    try:
        # Create an Elasticsearch retriever
        es_retriever = create_elasticsearch_retriever(
            es_host="http://localhost:9200",
            es_index="documents",
            embedding_model=dummy_embedding_model,
            hybrid_search=True,
            top_k=3
        )
        
        # Create a Self-RAG critic with the Elasticsearch retriever
        model = get_model("gpt-4")
        self_rag_critic = create_self_rag_critic(
            model=model,
            retriever=es_retriever,
            reflection_enabled=True,
            max_passages=3
        )
        
        # Use the critic to improve text
        text = "Artificial intelligence is a technology that can solve many problems."
        improved_text = self_rag_critic.improve(text)
        
        logger.info(f"Original text: {text}")
        logger.info(f"Improved text: {improved_text}")
        
    except Exception as e:
        logger.error(f"Error in Elasticsearch example: {str(e)}")
        logger.info("Falling back to dummy retriever")
        
        # Fall back to dummy retriever
        model = get_model("gpt-4")
        self_rag_critic = create_self_rag_critic(
            model=model,
            retriever=dummy_retriever,
            reflection_enabled=True,
            max_passages=3
        )
        
        # Use the critic to improve text
        text = "Artificial intelligence is a technology that can solve many problems."
        improved_text = self_rag_critic.improve(text)
        
        logger.info(f"Original text: {text}")
        logger.info(f"Improved text: {improved_text}")


def example_milvus_retriever():
    """Example of using Self-RAG with Milvus retriever."""
    logger.info("Example of using Self-RAG with Milvus retriever")
    
    try:
        # Create a Milvus retriever
        milvus_retriever = create_milvus_retriever(
            milvus_host="localhost",
            milvus_port="19530",
            collection_name="documents",
            embedding_model=dummy_embedding_model,
            top_k=3
        )
        
        # Create a Self-RAG critic with the Milvus retriever
        model = get_model("gpt-4")
        self_rag_critic = create_self_rag_critic(
            model=model,
            retriever=milvus_retriever,
            reflection_enabled=True,
            max_passages=3
        )
        
        # Use the critic to improve text
        text = "Artificial intelligence is a technology that can solve many problems."
        improved_text = self_rag_critic.improve(text)
        
        logger.info(f"Original text: {text}")
        logger.info(f"Improved text: {improved_text}")
        
    except Exception as e:
        logger.error(f"Error in Milvus example: {str(e)}")
        logger.info("Falling back to dummy retriever")
        
        # Fall back to dummy retriever
        model = get_model("gpt-4")
        self_rag_critic = create_self_rag_critic(
            model=model,
            retriever=dummy_retriever,
            reflection_enabled=True,
            max_passages=3
        )
        
        # Use the critic to improve text
        text = "Artificial intelligence is a technology that can solve many problems."
        improved_text = self_rag_critic.improve(text)
        
        logger.info(f"Original text: {text}")
        logger.info(f"Improved text: {improved_text}")


if __name__ == "__main__":
    # Run the examples
    example_elasticsearch_retriever()
    print("\n" + "-" * 80 + "\n")
    example_milvus_retriever()
