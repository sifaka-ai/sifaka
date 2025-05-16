#!/usr/bin/env python
"""
Fact Checker Example

This example demonstrates how to use Sifaka to build a fact-checking system
that can verify claims and provide corrections.
"""

import sys
import os
import argparse
import json
from typing import Dict, Any, List, Optional, Tuple

# Add the project root to the path so we can import sifaka
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sifaka.chain import Chain
from sifaka.validators import factual_accuracy
from sifaka.interfaces import Model
from sifaka.factories import create_model, create_model_from_string
from sifaka.results import ValidationResult, ImprovementResult
from sifaka.critics.self_refine import create_self_refine_critic


class FactChecker:
    """A fact-checking system that verifies claims and provides corrections using Self-Refine critic.

    This implementation uses the Self-Refine approach from the paper:
    "Self-Refine: Iterative Refinement with Self-Feedback"
    (Madaan et al., 2023, https://arxiv.org/abs/2303.17651)

    The Self-Refine approach involves multiple rounds of feedback and refinement,
    where the model critiques its own output and then improves it based on that critique.
    """

    def __init__(self, model: str = "openai:gpt-4", refinement_rounds: int = 2):
        """Initialize the fact checker.

        Args:
            model: The model to use for fact checking.
            refinement_rounds: Number of refinement rounds for the Self-Refine critic.
        """
        self.model = model

        # Create model instance
        if isinstance(model, str):
            self.model_instance = create_model_from_string(model)
        else:
            self.model_instance = model

        # Create both a factual accuracy validator and a self-refine critic
        self._validator = factual_accuracy(model)
        self._improver = create_self_refine_critic(
            model=self.model_instance,
            refinement_rounds=refinement_rounds,
            system_prompt=(
                "You are an expert fact-checker who specializes in identifying and correcting "
                "factual inaccuracies. Your goal is to provide detailed feedback on factual "
                "errors and iteratively improve text to make it factually accurate."
            ),
            temperature=0.3,
        )

    def check_claim(self, claim: str) -> Dict[str, Any]:
        """Check a single claim for factual accuracy.

        Args:
            claim: The claim to check.

        Returns:
            A dictionary containing the verification result.
        """
        # Validate the claim
        validation_result = self._validator.validate(claim)

        # If the claim is inaccurate, get a correction using Self-Refine
        correction = None
        if not validation_result.passed:
            # Use the Self-Refine critic to improve the claim
            corrected_text, improvement_result = self._improver.improve(claim)

            # Extract refinement history if available
            refinement_history = []
            if improvement_result.details and "refinement_history" in improvement_result.details:
                refinement_history = improvement_result.details["refinement_history"]

            correction = {
                "corrected_text": corrected_text,
                "explanation": improvement_result.message,
                "refinement_rounds": len(refinement_history),
                "refinement_history": refinement_history,
            }

        # Return the result
        return {
            "claim": claim,
            "is_accurate": validation_result.passed,
            "confidence": (
                validation_result.details.get("score", 0) / 10
                if validation_result.details
                else None
            ),
            "explanation": validation_result.message,
            "correction": correction,
        }

    def check_text(self, text: str) -> Dict[str, Any]:
        """Check a text for factual accuracy and extract claims.

        Args:
            text: The text to check.

        Returns:
            A dictionary containing the verification results for the text and individual claims.
        """
        # First, extract claims from the text
        claims = self._extract_claims(text)

        # Check each claim
        claim_results = [self.check_claim(claim) for claim in claims]

        # Check the overall text
        overall_result = self._validator.validate(text)

        # If the text is inaccurate, get a correction using Self-Refine
        correction = None
        if not overall_result.passed:
            # Use the Self-Refine critic to improve the text
            corrected_text, improvement_result = self._improver.improve(text)

            # Extract refinement history if available
            refinement_history = []
            if improvement_result.details and "refinement_history" in improvement_result.details:
                refinement_history = improvement_result.details["refinement_history"]

            correction = {
                "corrected_text": corrected_text,
                "explanation": improvement_result.message,
                "refinement_rounds": len(refinement_history),
                "refinement_history": refinement_history,
            }

        # Return the results
        return {
            "text": text,
            "is_accurate": overall_result.passed,
            "confidence": (
                overall_result.details.get("score", 0) / 10 if overall_result.details else None
            ),
            "explanation": overall_result.message,
            "correction": correction,
            "claims": claim_results,
            "claim_count": len(claims),
            "accurate_claims": sum(1 for result in claim_results if result["is_accurate"]),
            "inaccurate_claims": sum(1 for result in claim_results if not result["is_accurate"]),
        }

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from a text.

        Args:
            text: The text to extract claims from.

        Returns:
            A list of factual claims extracted from the text.
        """
        # Create a model instance
        if isinstance(self.model, str):
            model = create_model_from_string(self.model)
        else:
            model = self.model

        # Create a prompt to extract claims
        prompt = f"""
        Extract the factual claims from the following text. A factual claim is a statement
        that can be verified as true or false based on evidence.

        Text:
        ---
        {text}
        ---

        Extract each distinct factual claim and list them one per line.
        Format your response as a JSON array of strings, where each string is a claim.
        Only include the JSON array in your response, nothing else.
        """

        # Generate the response
        response = model.generate(
            prompt,
            temperature=0.3,
            system_message="You are a helpful assistant that extracts factual claims from text.",
        )

        # Parse the response as JSON
        try:
            claims = json.loads(response)
            if not isinstance(claims, list):
                claims = []
        except json.JSONDecodeError:
            # If the response is not valid JSON, try to extract claims manually
            claims = [line.strip() for line in response.split("\n") if line.strip()]
            # Remove any lines that are not claims (e.g., headers, explanations)
            claims = [line for line in claims if not line.startswith(("```", "[", "#", "Claim"))]

        return claims


class ClaimVerifier:
    """A component that verifies specific claims against a knowledge base."""

    def __init__(self, model: str = "openai:gpt-4"):
        """Initialize the claim verifier.

        Args:
            model: The model to use for verification.
        """
        self.model = model

    def verify_claim(self, claim: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Verify a claim against a knowledge base or provided context.

        Args:
            claim: The claim to verify.
            context: Optional context or knowledge to use for verification.

        Returns:
            A dictionary containing the verification result.
        """
        # Create a model instance
        if isinstance(self.model, str):
            model = create_model_from_string(self.model)
        else:
            model = self.model

        # Create a prompt for verification
        if context:
            prompt = f"""
            Verify the following claim based on the provided context:

            Claim: {claim}

            Context:
            ---
            {context}
            ---

            First, determine if the claim is supported by the context, contradicted by the context,
            or if the context doesn't provide enough information to verify the claim.

            Then, provide a detailed explanation of your reasoning.

            Format your response as a JSON object with the following fields:
            - verdict: "SUPPORTED", "CONTRADICTED", or "INSUFFICIENT_INFO"
            - confidence: A number between 0 and 1 indicating your confidence in the verdict
            - explanation: Your detailed explanation

            Only include the JSON object in your response, nothing else.
            """
        else:
            prompt = f"""
            Verify the following claim based on your knowledge:

            Claim: {claim}

            First, determine if the claim is factually accurate, inaccurate, or if you don't have
            enough information to verify the claim.

            Then, provide a detailed explanation of your reasoning.

            Format your response as a JSON object with the following fields:
            - verdict: "ACCURATE", "INACCURATE", or "INSUFFICIENT_INFO"
            - confidence: A number between 0 and 1 indicating your confidence in the verdict
            - explanation: Your detailed explanation

            Only include the JSON object in your response, nothing else.
            """

        # Generate the response
        response = model.generate(
            prompt,
            temperature=0.3,
            system_message="You are a helpful assistant that verifies factual claims.",
        )

        # Parse the response as JSON
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # If the response is not valid JSON, create a default result
            result = {
                "verdict": "INSUFFICIENT_INFO",
                "confidence": 0.5,
                "explanation": "Failed to parse the verification result.",
            }

        # Return the result
        return {
            "claim": claim,
            "context": context,
            "verdict": result.get("verdict", "INSUFFICIENT_INFO"),
            "confidence": result.get("confidence", 0.5),
            "explanation": result.get("explanation", "No explanation provided."),
        }


def main():
    """Run the fact checker example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check facts using Sifaka")
    parser.add_argument("--model", default="openai:gpt-4", help="Model to use")
    parser.add_argument("--claim", help="Single claim to check")
    parser.add_argument("--text", help="Text to check for factual accuracy")
    parser.add_argument("--file", help="File containing text to check")
    parser.add_argument("--context", help="Context or knowledge to use for verification")
    parser.add_argument("--context-file", help="File containing context for verification")
    parser.add_argument(
        "--refinement-rounds",
        type=int,
        default=2,
        help="Number of refinement rounds for Self-Refine critic",
    )
    parser.add_argument("--output", help="Output file to write results to")
    args = parser.parse_args()

    try:
        # Ensure at least one input is provided
        if not args.claim and not args.text and not args.file:
            parser.error("At least one of --claim, --text, or --file must be provided")

        # Read text from file if specified
        if args.file:
            with open(args.file, "r") as f:
                args.text = f.read()

        # Read context from file if specified
        context = None
        if args.context_file:
            with open(args.context_file, "r") as f:
                context = f.read()
        elif args.context:
            context = args.context

        # Create the appropriate checker
        if args.claim and context:
            # Use ClaimVerifier for claim with context
            checker = ClaimVerifier(model=args.model)
            print(f"Verifying claim against provided context...")
            result = checker.verify_claim(args.claim, context)
        elif args.claim:
            # Use FactChecker for a single claim
            checker = FactChecker(model=args.model, refinement_rounds=args.refinement_rounds)
            print(
                f"Checking claim for factual accuracy using Self-Refine critic with {args.refinement_rounds} refinement rounds..."
            )
            result = checker.check_claim(args.claim)
        else:
            # Use FactChecker for text
            checker = FactChecker(model=args.model, refinement_rounds=args.refinement_rounds)
            print(
                f"Checking text for factual accuracy using Self-Refine critic with {args.refinement_rounds} refinement rounds..."
            )
            result = checker.check_text(args.text)

        # Print the result
        print("\nVerification Result:")
        print("=" * 40)
        if "verdict" in result:
            print(f"Verdict: {result['verdict']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Explanation: {result['explanation']}")
        else:
            print(f"Is Accurate: {result['is_accurate']}")
            if "confidence" in result and result["confidence"] is not None:
                print(f"Confidence: {result['confidence']:.2f}")
            print(f"Explanation: {result['explanation']}")

            if "claims" in result:
                print(
                    f"\nClaims: {result['claim_count']} total, "
                    f"{result['accurate_claims']} accurate, "
                    f"{result['inaccurate_claims']} inaccurate"
                )

                for i, claim_result in enumerate(result["claims"]):
                    print(f"\nClaim {i+1}: {claim_result['claim']}")
                    print(f"Is Accurate: {claim_result['is_accurate']}")
                    print(f"Explanation: {claim_result['explanation']}")

                    if not claim_result["is_accurate"] and claim_result["correction"]:
                        print(f"Correction: {claim_result['correction']['corrected_text']}")
                        if "refinement_rounds" in claim_result["correction"]:
                            print(
                                f"Refinement Rounds: {claim_result['correction']['refinement_rounds']}"
                            )

        # Write to file if specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults written to {args.output}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
