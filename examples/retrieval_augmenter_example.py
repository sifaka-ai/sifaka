"""
Example of using the RetrievalAugmenter with different critics.

This example demonstrates how to use the RetrievalAugmenter with different critics
to enhance their capabilities with retrieval-based information.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from sifaka.models import get_model
from sifaka.critics.self_rag import create_self_rag_critic
from sifaka.critics.self_refine import create_self_refine_critic
from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.critics.n_critics import create_n_critics_critic
from sifaka.retrievers import create_retrieval_augmenter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    elif "ethics" in query.lower():
        return [
            "AI ethics is a set of values, principles, and techniques that employ widely accepted standards "
            "of right and wrong to guide moral conduct in the development and use of AI technologies.",
            "The ethics of artificial intelligence is the branch of the ethics of technology specific to "
            "artificially intelligent systems. It is sometimes divided into a concern with the moral behavior "
            "of humans as they design, make, use and treat artificially intelligent systems, and a concern "
            "with the behavior of machines, in machine ethics."
        ]
    else:
        return [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, "
            "as opposed to natural intelligence displayed by animals including humans.",
            "Leading AI textbooks define the field as the study of 'intelligent agents': "
            "any system that perceives its environment and takes actions that maximize "
            "its chance of achieving its goals."
        ]


def example_self_rag_with_augmenter():
    """Example of using the RetrievalAugmenter with Self-RAG critic."""
    logger.info("Example of using the RetrievalAugmenter with Self-RAG critic")
    
    # Create a model
    model = get_model("gpt-4")
    
    # Create a retrieval augmenter
    retrieval_augmenter = create_retrieval_augmenter(
        retriever=dummy_retriever,
        model=model,
        max_passages=3,
        max_queries=3,
        query_temperature=0.3,
        include_query_context=True,
    )
    
    # Create a Self-RAG critic with the retrieval augmenter
    self_rag_critic = create_self_rag_critic(
        model=model,
        retriever=retrieval_augmenter.retrieve,
        reflection_enabled=True,
        max_passages=3
    )
    
    # Use the critic to improve text
    text = "Artificial intelligence is a technology that can solve many problems."
    improved_text = self_rag_critic.improve(text)
    
    logger.info(f"Original text: {text}")
    logger.info(f"Improved text: {improved_text}")


def example_self_refine_with_augmenter():
    """Example of using the RetrievalAugmenter with Self-Refine critic."""
    logger.info("Example of using the RetrievalAugmenter with Self-Refine critic")
    
    # Create a model
    model = get_model("gpt-4")
    
    # Create a retrieval augmenter
    retrieval_augmenter = create_retrieval_augmenter(
        retriever=dummy_retriever,
        model=model,
        max_passages=3,
        max_queries=3
    )
    
    # Create a Self-Refine critic
    self_refine_critic = create_self_refine_critic(
        model=model,
        max_iterations=2
    )
    
    # Use the critic to improve text with retrieval augmentation
    text = "Artificial intelligence is a technology that can solve many problems."
    
    # Get retrieval context
    retrieval_context = retrieval_augmenter.get_retrieval_context(text)
    
    # Create a custom critique function that incorporates retrieved information
    def custom_critique(text_to_critique):
        # Get the base critique from the critic
        base_critique = self_refine_critic._critique(text_to_critique)
        
        # Add retrieval context to the critique
        base_critique["retrieved_passages"] = retrieval_context["passages"]
        base_critique["formatted_passages"] = retrieval_context["formatted_passages"]
        
        # Add a suggestion to incorporate retrieved information
        if retrieval_context["passage_count"] > 0:
            base_critique["suggestions"].append("Incorporate information from retrieved passages")
        
        return base_critique
    
    # Create a custom improve function that incorporates retrieved information
    def custom_improve(text_to_improve, critique):
        # Create a prompt that includes retrieved information
        prompt = f"""
        Please improve the following text based on the critique and retrieved information:
        
        Original text:
        ```
        {text_to_improve}
        ```
        
        Critique:
        - {' '.join(critique.get('issues', []))}
        
        Suggestions:
        - {' '.join(critique.get('suggestions', []))}
        
        Retrieved information:
        {retrieval_context["formatted_passages"]}
        
        Instructions:
        1. Address all issues mentioned in the critique
        2. Incorporate relevant information from the retrieved passages
        3. Maintain the original style and tone
        4. Ensure factual accuracy and logical coherence
        
        Improved text:
        """
        
        # Generate improved text
        response = model.generate(prompt, temperature=0.7)
        
        # Extract improved text from response
        improved_text = response.strip()
        
        # Remove any markdown code block markers
        if improved_text.startswith("```") and improved_text.endswith("```"):
            improved_text = improved_text[3:-3].strip()
        
        return improved_text
    
    # Override the critique and improve methods
    original_critique = self_refine_critic._critique
    original_improve = self_refine_critic._improve
    
    self_refine_critic._critique = custom_critique
    self_refine_critic._improve = custom_improve
    
    # Use the critic to improve text
    improved_text = self_refine_critic.improve(text)
    
    # Restore original methods
    self_refine_critic._critique = original_critique
    self_refine_critic._improve = original_improve
    
    logger.info(f"Original text: {text}")
    logger.info(f"Improved text: {improved_text}")


def example_constitutional_with_augmenter():
    """Example of using the RetrievalAugmenter with Constitutional critic."""
    logger.info("Example of using the RetrievalAugmenter with Constitutional critic")
    
    # Create a model
    model = get_model("gpt-4")
    
    # Create a retrieval augmenter
    retrieval_augmenter = create_retrieval_augmenter(
        retriever=dummy_retriever,
        model=model,
        max_passages=3,
        max_queries=3
    )
    
    # Define principles that incorporate retrieval
    principles = [
        "Ensure factual accuracy by incorporating relevant information from retrieved sources.",
        "Provide comprehensive context by including information from retrieved passages when relevant.",
        "Maintain clarity and coherence in the text while integrating new information.",
        "Cite or reference external information appropriately.",
    ]
    
    # Create a Constitutional critic
    constitutional_critic = create_constitutional_critic(
        model=model,
        principles=principles
    )
    
    # Use the critic to improve text with retrieval augmentation
    text = "Artificial intelligence is a technology that can solve many problems."
    
    # Get retrieval context
    retrieval_context = retrieval_augmenter.get_retrieval_context(text)
    
    # Create a custom critique function that incorporates retrieved information
    def custom_critique(text_to_critique):
        # Get the base critique from the critic
        base_critique = constitutional_critic._critique(text_to_critique)
        
        # Add retrieval context to the critique
        base_critique["retrieved_passages"] = retrieval_context["passages"]
        base_critique["formatted_passages"] = retrieval_context["formatted_passages"]
        
        # Check if there are factual issues that could be addressed with retrieved information
        if retrieval_context["passage_count"] > 0:
            base_critique["violations"].append({
                "principle": "Ensure factual accuracy by incorporating relevant information from retrieved sources.",
                "explanation": "The text could be improved by incorporating factual information from retrieved sources."
            })
        
        return base_critique
    
    # Create a custom improve function that incorporates retrieved information
    def custom_improve(text_to_improve, critique):
        # Add retrieved information to the prompt
        prompt = f"""
        Please revise the following text to address the constitutional violations and incorporate relevant information from retrieved sources:
        
        Original text:
        ```
        {text_to_improve}
        ```
        
        Constitutional violations:
        {constitutional_critic._format_violations(critique.get('violations', []))}
        
        Retrieved information:
        {retrieval_context["formatted_passages"]}
        
        Instructions:
        1. Address all constitutional violations
        2. Incorporate relevant information from the retrieved passages
        3. Maintain the original style and tone
        4. Ensure factual accuracy and logical coherence
        
        Revised text:
        """
        
        # Generate improved text
        response = model.generate(prompt, temperature=0.7)
        
        # Extract improved text from response
        improved_text = response.strip()
        
        # Remove any markdown code block markers
        if improved_text.startswith("```") and improved_text.endswith("```"):
            improved_text = improved_text[3:-3].strip()
        
        return improved_text
    
    # Override the critique and improve methods
    original_critique = constitutional_critic._critique
    original_improve = constitutional_critic._improve
    
    constitutional_critic._critique = custom_critique
    constitutional_critic._improve = custom_improve
    
    # Use the critic to improve text
    improved_text = constitutional_critic.improve(text)
    
    # Restore original methods
    constitutional_critic._critique = original_critique
    constitutional_critic._improve = original_improve
    
    logger.info(f"Original text: {text}")
    logger.info(f"Improved text: {improved_text}")


if __name__ == "__main__":
    # Run the examples
    example_self_rag_with_augmenter()
    print("\n" + "-" * 80 + "\n")
    example_self_refine_with_augmenter()
    print("\n" + "-" * 80 + "\n")
    example_constitutional_with_augmenter()
