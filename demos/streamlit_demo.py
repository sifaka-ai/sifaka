"""
Streamlit-based interactive demo for Sifaka.

This demo showcases the capabilities of the Sifaka framework through an
interactive web interface built with Streamlit.

To run this demo:
1. Install Streamlit: pip install streamlit
2. Run the demo: streamlit run demos/streamlit_demo.py
"""

import sys
import os
import time
import streamlit as st
from typing import Dict, Any, List, Optional

# Add the project root to the path so we can import sifaka
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sifaka.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators import length, factual_accuracy
from sifaka.critics.lac import create_lac_critic
from sifaka.critics.reflexion import create_reflexion_critic
from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.critics.self_rag import create_self_rag_critic
from sifaka.critics.self_refine import create_self_refine_critic


# Set page configuration
st.set_page_config(
    page_title="Sifaka Demo",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def create_chain(
    model_name: str,
    prompt: str,
    validators: List[Dict[str, Any]],
    improvers: List[Dict[str, Any]],
    options: Dict[str, Any],
) -> Chain:
    """Create a Sifaka chain with the specified configuration.

    Args:
        model_name: The model to use (e.g., "openai:gpt-4")
        prompt: The prompt to use for generation
        validators: List of validator configurations
        improvers: List of improver configurations
        options: Options for the model

    Returns:
        A configured Chain instance
    """
    # Create a chain
    chain = Chain()

    # Configure the chain
    chain.with_model(model_name)
    chain.with_prompt(prompt)

    # Add validators
    for validator_config in validators:
        validator_type = validator_config["type"]

        if validator_type == "length":
            min_words = validator_config.get("min_words", 0)
            max_words = validator_config.get("max_words", 1000)
            chain.validate_with(length(min_words=min_words, max_words=max_words))

        elif validator_type == "factual_accuracy":
            chain.validate_with(factual_accuracy(model_name))

    # Add improvers
    for improver_config in improvers:
        improver_type = improver_config["type"]

        # Create a model instance for critics
        provider, model_id = model_name.split(":", 1)
        model = create_model(provider, model_id)

        if improver_type == "lac":
            lac_critic = create_lac_critic(
                model=model,
                temperature=options.get("temperature", 0.7),
                feedback_weight=improver_config.get("feedback_weight", 0.7),
                max_improvement_iterations=improver_config.get("max_iterations", 2),
            )
            chain.improve_with(lac_critic)

        elif improver_type == "reflexion":
            reflexion_critic = create_reflexion_critic(
                model=model,
                temperature=options.get("temperature", 0.7),
                max_iterations=improver_config.get("max_iterations", 2),
            )
            chain.improve_with(reflexion_critic)

        elif improver_type == "constitutional":
            constitution = improver_config.get(
                "constitution",
                [
                    "Your output should be factually accurate and not misleading.",
                    "Your output should be helpful, harmless, and honest.",
                    "Your output should be clear, concise, and well-organized.",
                ],
            )
            constitutional_critic = create_constitutional_critic(
                model=model,
                constitution=constitution,
                temperature=options.get("temperature", 0.7),
            )
            chain.improve_with(constitutional_critic)

        elif improver_type == "self_rag":
            self_rag_critic = create_self_rag_critic(
                model=model,
                temperature=options.get("temperature", 0.7),
            )
            chain.improve_with(self_rag_critic)

        elif improver_type == "self_refine":
            self_refine_critic = create_self_refine_critic(
                model=model,
                temperature=options.get("temperature", 0.7),
                max_iterations=improver_config.get("max_iterations", 2),
            )
            chain.improve_with(self_refine_critic)

    # Set options
    chain.with_options(**options)

    return chain


def run_chain(chain: Chain) -> Dict[str, Any]:
    """Run a chain and return the results with timing information.

    Args:
        chain: The chain to run

    Returns:
        A dictionary with the results and timing information
    """
    start_time = time.time()
    result = chain.run()
    elapsed_time = time.time() - start_time

    return {
        "result": result,
        "elapsed_time": elapsed_time,
    }


def main():
    """Run the Streamlit demo."""
    # Add a title and description
    st.title("🦊 Sifaka Interactive Demo")
    st.markdown(
        """
        This demo showcases the capabilities of the Sifaka framework for building reliable LLM applications.

        Sifaka provides a clean, intuitive API for working with large language models (LLMs) with built-in
        validation and improvement mechanisms.

        Configure the chain below and see how Sifaka can help improve the quality of LLM outputs.
        """
    )

    # Create a sidebar for configuration
    st.sidebar.title("Configuration")

    # Model selection
    st.sidebar.subheader("Model")
    model_provider = st.sidebar.selectbox(
        "Provider",
        ["openai", "anthropic", "gemini"],
        index=0,
    )

    # Model options based on provider
    if model_provider == "openai":
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    elif model_provider == "anthropic":
        model_options = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
    else:  # gemini
        model_options = ["gemini-pro", "gemini-ultra"]

    model_name = st.sidebar.selectbox(
        "Model",
        model_options,
        index=0,
    )

    # Combine provider and model name
    full_model_name = f"{model_provider}:{model_name}"

    # Generation options
    st.sidebar.subheader("Generation Options")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
    )
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
    )

    # Validator configuration
    st.sidebar.subheader("Validators")
    use_length_validator = st.sidebar.checkbox("Length Validator", value=True)
    min_words = st.sidebar.number_input("Min Words", value=50, min_value=0, max_value=1000, step=10)
    max_words = st.sidebar.number_input(
        "Max Words", value=500, min_value=0, max_value=2000, step=10
    )

    use_factual_validator = st.sidebar.checkbox("Factual Accuracy Validator", value=False)

    # Improver configuration
    st.sidebar.subheader("Improvers")
    improver_type = st.sidebar.selectbox(
        "Improver Type",
        ["None", "LAC", "Reflexion", "Constitutional", "Self-RAG", "Self-Refine"],
        index=1,
    )

    # Additional options based on improver type
    improver_options = {}

    if improver_type == "LAC":
        improver_options["feedback_weight"] = st.sidebar.slider(
            "Feedback Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
        )
        improver_options["max_iterations"] = st.sidebar.slider(
            "Max Iterations",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
        )

    elif improver_type == "Reflexion" or improver_type == "Self-Refine":
        improver_options["max_iterations"] = st.sidebar.slider(
            "Max Iterations",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
        )

    elif improver_type == "Constitutional":
        constitution = st.sidebar.text_area(
            "Constitution",
            value=(
                "Your output should be factually accurate and not misleading.\n"
                "Your output should be helpful, harmless, and honest.\n"
                "Your output should be clear, concise, and well-organized."
            ),
            height=100,
        )
        improver_options["constitution"] = [
            line.strip() for line in constitution.split("\n") if line.strip()
        ]

    # Main content area
    st.subheader("Input")
    prompt = st.text_area(
        "Prompt",
        value="Write a short explanation of quantum computing.",
        height=100,
    )

    # Run button
    if st.button("Run Chain"):
        # Show a spinner while running
        with st.spinner("Running chain..."):
            # Prepare validator configurations
            validators = []
            if use_length_validator:
                validators.append(
                    {
                        "type": "length",
                        "min_words": min_words,
                        "max_words": max_words,
                    }
                )
            if use_factual_validator:
                validators.append(
                    {
                        "type": "factual_accuracy",
                    }
                )

            # Prepare improver configurations
            improvers = []
            if improver_type != "None":
                improvers.append(
                    {
                        "type": improver_type.lower(),
                        **improver_options,
                    }
                )

            # Prepare options
            options = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Create and run the chain
            try:
                chain = create_chain(
                    model_name=full_model_name,
                    prompt=prompt,
                    validators=validators,
                    improvers=improvers,
                    options=options,
                )

                results = run_chain(chain)

                # Display results
                st.subheader("Results")

                # Display timing information
                st.info(f"Chain completed in {results['elapsed_time']:.2f} seconds")

                # Display the generated text
                st.subheader("Generated Text")
                st.markdown(results["result"].text)

                # Display validation results
                if validators:
                    st.subheader("Validation Results")
                    for i, validation_result in enumerate(results["result"].validation_results):
                        st.write(
                            f"**Validator {i+1}:** {'✅ Passed' if validation_result.passed else '❌ Failed'}"
                        )
                        st.write(f"Message: {validation_result.message}")

                # Display improvement results
                if improvers:
                    st.subheader("Improvement Results")
                    for i, improvement_result in enumerate(results["result"].improvement_results):
                        st.write(
                            f"**Improver {i+1}:** {'✅ Changes Made' if improvement_result.changes_made else '⚠️ No Changes'}"
                        )
                        st.write(f"Message: {improvement_result.message}")

            except Exception as e:
                st.error(f"Error running chain: {str(e)}")
                st.error(
                    "Make sure you have the necessary API keys set up for the selected model provider."
                )


if __name__ == "__main__":
    main()
