"""
Basic usage example for Sifaka.
"""
import os
import sys
import logging

# Add the parent directory to the path so we can import sifaka
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sifaka import Reflector, legal_citation_check
from sifaka.models import OpenAIProvider
from sifaka.rules.content import ProhibitedContentRule, ToneConsistencyRule

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    # Initialize the model provider
    # Replace with your API key or set the OPENAI_API_KEY environment variable
    model = OpenAIProvider(
        model_name="gpt-4",
        temperature=0.7
    )
    
    # Create rules
    prohibited_terms_rule = ProhibitedContentRule(
        prohibited_terms=["controversial", "inappropriate"],
        case_sensitive=False
    )
    
    formal_tone_rule = ToneConsistencyRule(
        expected_tone="formal"
    )
    
    # Create a reflector with rules and critique
    reflector = Reflector(
        rules=[legal_citation_check, prohibited_terms_rule, formal_tone_rule],
        critique=True,
        trace=True
    )
    
    # Example prompt
    prompt = """
    Write a brief summary of the landmark Supreme Court case Brown v. Board of Education.
    """
    
    # Run the reflector
    print(f"Running reflector with prompt: {prompt}")
    result = reflector.run(model, prompt)
    
    # Print the results
    print("\nOriginal output:")
    print(result["original_output"])
    
    if result["rule_violations"]:
        print("\nRule violations:")
        for violation in result["rule_violations"]:
            print(f"- {violation['rule']}: {violation['message']}")
    
    print("\nFinal output:")
    print(result["final_output"])
    
    if "trace" in result:
        print("\nTrace data:")
        for event in result["trace"]:
            print(f"- {event['stage']}")

if __name__ == "__main__":
    main()
