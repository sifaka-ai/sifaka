#!/usr/bin/env python3
"""Test script to check model name extraction."""

import os
import sys

sys.path.insert(0, os.path.abspath("."))


def test_model_name_extraction():
    """Test the model name extraction logic."""
    try:
        from pydantic_ai import Agent

        from sifaka.agents import create_pydantic_chain

        # Create a simple agent
        agent = Agent("openai:gpt-4o-mini")

        # Create chain
        chain = create_pydantic_chain(agent=agent)

        # Test the model name extraction
        model_name = chain._extract_model_name()
        print(f"Extracted model name: {model_name}")
        print(f"Model name type: {type(model_name)}")

        # Also check the raw model object
        raw_model = getattr(agent, "model", None)
        print(f"Raw model object: {raw_model}")
        print(f"Raw model type: {type(raw_model)}")

        if hasattr(raw_model, "_model_name"):
            print(f"Raw model._model_name: {raw_model._model_name}")

        return model_name

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_model_name_extraction()
