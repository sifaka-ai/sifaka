"""
Example of using Sifaka with Pydantic models for structured data validation.
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from sifaka import Reflector
from sifaka.models import AnthropicProvider
from sifaka.rules import Rule, RuleResult
from sifaka.critique import PromptCritique

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define Pydantic models for structured data validation
class ProductReview(BaseModel):
    """
    A product review with structured fields.

    Attributes:
        title (str): The review title
        rating (int): Rating from 1-5
        pros (List[str]): List of positive points
        cons (List[str]): List of negative points
        summary (str): Brief summary of the review
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., min_length=5, max_length=100)
    rating: int = Field(..., ge=1, le=5)
    pros: List[str] = Field(default_factory=list, min_length=1)
    cons: List[str] = Field(default_factory=list, min_length=1)
    summary: str = Field(..., min_length=20, max_length=500)


class ReviewValidationRule(Rule):
    """
    Rule that validates product reviews using Pydantic models.
    """

    min_rating: int = 1
    max_rating: int = 5

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the review validation rule."""
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        config = config or {}
        if "min_rating" in config:
            self.min_rating = config["min_rating"]
        if "max_rating" in config:
            self.max_rating = config["max_rating"]

    def validate(self, output: str) -> RuleResult:
        """
        Validate that the output contains a valid product review.

        Args:
            output (str): The LLM output to validate

        Returns:
            RuleResult: The result of the validation
        """
        try:
            # Parse the output into sections
            sections = output.split("\n\n")
            if len(sections) < 4:
                return RuleResult(
                    passed=False,
                    message="Output does not contain all required sections",
                    metadata={"error": "Missing sections"},
                )

            # Extract review components
            title = sections[0].strip()
            rating_text = sections[1].strip().split(":")[1].strip() if ":" in sections[1] else "0"
            rating = int(rating_text)

            pros_section = [s for s in sections if s.lower().startswith("pros:")][0]
            cons_section = [s for s in sections if s.lower().startswith("cons:")][0]

            pros = [p.strip("- ") for p in pros_section.split("\n")[1:] if p.strip()]
            cons = [c.strip("- ") for c in cons_section.split("\n")[1:] if c.strip()]

            summary = sections[-1].strip()

            # Create and validate the review using Pydantic
            review = ProductReview(
                title=title, rating=rating, pros=pros, cons=cons, summary=summary
            )

            return RuleResult(
                passed=True,
                message="Valid product review format",
                metadata={"review": review.model_dump()},
            )

        except Exception as e:
            return RuleResult(
                passed=False, message=f"Invalid review format: {str(e)}", metadata={"error": str(e)}
            )


def main():
    """Example usage of Pydantic validation with Sifaka."""
    # Load environment variables
    load_dotenv()

    # Initialize the model provider
    model = AnthropicProvider(model_name="claude-3-haiku-20240307")

    # Create the review validation rule
    review_rule = ReviewValidationRule(
        name="review_validator",
        description="Validates product review format",
        config={"min_rating": 1, "max_rating": 5},
    )

    # Create a critic for improving outputs that fail validation
    critic = PromptCritique(model=model)

    # Create a reflector with the rule and critique
    reflector = Reflector(
        name="review_validator", model=model, rules=[review_rule], critique=True, critic=critic
    )

    # Example prompt
    prompt = """
    Write a detailed product review for a new smartphone. Include:
    - A clear title
    - A rating from 1-5
    - At least 3 pros
    - At least 2 cons
    - A brief summary
    """

    # Run the reflector
    logger.info("Running reflector with prompt: %s", prompt)
    result = reflector.reflect(prompt)

    # Print the results
    logger.info("\nOriginal output:")
    logger.info(result.original_output)

    if result.rule_violations:
        logger.info("\nRule violations:")
        for violation in result.rule_violations:
            logger.info("- %s: %s", violation.rule_name, violation.message)
    else:
        logger.info("\nNo rule violations found.")
        review = result.metadata["review"]
        logger.info("\nValidated Review:")
        logger.info("Title: %s", review["title"])
        logger.info("Rating: %d/5", review["rating"])
        logger.info("Pros:")
        for pro in review["pros"]:
            logger.info("- %s", pro)
        logger.info("Cons:")
        for con in review["cons"]:
            logger.info("- %s", con)
        logger.info("\nSummary: %s", review["summary"])

    logger.info("\nFinal output:")
    logger.info(result.final_output)


if __name__ == "__main__":
    main()
