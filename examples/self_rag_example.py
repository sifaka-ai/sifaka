"""
Example of using the SelfRAGCritic in Sifaka.

This example demonstrates how to use the SelfRAGCritic to improve text
through self-reflective retrieval-augmented generation.
"""

import os
from sifaka.critics.self_rag import create_self_rag_critic
from sifaka.models import OpenAIProvider, AnthropicProvider
from sifaka.retrieval import SimpleRetriever
from sifaka.utils.config import standardize_model_config


def main():
    """Run the example."""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_api_key and not anthropic_api_key:
        print(
            "Error: No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables."
        )
        return

    # Create a model provider
    if openai_api_key:
        config = standardize_model_config(api_key=openai_api_key)
        provider = OpenAIProvider(model_name="gpt-4", config=config)
        print("Using OpenAI provider")
    else:
        config = standardize_model_config(api_key=anthropic_api_key)
        provider = AnthropicProvider(model_name="claude-3-opus-20240229", config=config)
        print("Using Anthropic provider")

    # Create a simple retriever with a document collection
    documents = {
        "health_insurance": """
        To file a claim for health reimbursement, follow these steps:
        1. Complete the claim form with your personal and policy information
        2. Attach all original receipts and medical documentation
        3. Make copies of all documents for your records
        4. Submit the claim through the online portal, mobile app, or by mail
        5. Track your claim status using the provided claim number
        6. Expect processing within 14-30 days depending on complexity
        """,
        "travel_insurance": """
        For travel insurance claims, you need to provide:
        1. Proof of travel (boarding passes, itinerary)
        2. Incident report or documentation of the event
        3. Original receipts for expenses being claimed
        4. Completed claim form with policy number
        5. Medical reports if claiming for health issues
        6. Police reports for theft or loss claims
        """,
        "auto_insurance": """
        When filing an auto insurance claim:
        1. Report the incident to your insurance company immediately
        2. Document the damage with photos and notes
        3. Get the police report if applicable
        4. Collect contact information from any witnesses
        5. Keep receipts for all repair costs and related expenses
        6. Follow up regularly on the status of your claim
        """,
    }

    retriever = SimpleRetriever(documents=documents)
    print("Created simple retriever with insurance documents")

    # Create a self-rag critic
    critic = create_self_rag_critic(
        llm_provider=provider,
        retriever=retriever,
        name="insurance_rag_critic",
        description="A critic for insurance-related queries",
        system_prompt="You are an expert at retrieving and using insurance information.",
        temperature=0.7,
        max_tokens=1000,
    )
    print("Created SelfRAGCritic")

    # Example tasks
    tasks = [
        "What are the steps to file a claim for health reimbursement?",
        "What documents do I need for a travel insurance claim?",
        "How do I file an auto insurance claim?",
        "What's the process for filing a home insurance claim?",  # Not in our documents
    ]

    # Process each task
    for i, task in enumerate(tasks, 1):
        print(f"\n\n--- Task {i}: {task} ---")

        # Run the critic
        result = critic.run(task)

        # Print results
        print(f"\nRetrieval Query: {result['retrieval_query']}")
        print(f"\nRetrieved Context:\n{result['retrieved_context']}")
        print(f"\nResponse:\n{result['response']}")
        print(f"\nReflection:\n{result['reflection']}")
        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()
