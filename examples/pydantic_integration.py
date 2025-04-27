"""
Example of using Sifaka with Pydantic models and LangChain for structured output validation and reflection.
"""

import logging
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

from sifaka.integrations.langchain import ChainConfig, wrap_chain
from sifaka.models import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.rules import SymmetryRule, RepetitionRule
from sifaka.rules.base import RuleConfig, RulePriority

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieReview(BaseModel):
    """
    A structured movie review.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released", ge=1900, le=2024)
    rating: float = Field(..., description="Rating from 0.0 to 10.0", ge=0.0, le=10.0)
    genre: List[str] = Field(..., description="List of movie genres")
    strengths: List[str] = Field(..., description="List of movie strengths")
    weaknesses: List[str] = Field(..., description="List of movie weaknesses")
    summary: str = Field(
        ..., description="Brief summary of the movie", min_length=50, max_length=500
    )
    recommended: bool = Field(..., description="Whether the movie is recommended")


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze text using pattern rules.

    Args:
        text: The text to analyze

    Returns:
        Dict containing analysis results
    """
    results = {}

    # Create pattern detection rules
    symmetry_rule = SymmetryRule(
        name="symmetry_check",
        description="Checks for text symmetry patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            metadata={
                "mirror_mode": "both",
                "symmetry_threshold": 0.4,  # Lower threshold for movie reviews
                "preserve_whitespace": True,
                "preserve_case": True,
                "ignore_punctuation": True,
            },
        ),
    )

    repetition_rule = RepetitionRule(
        name="repetition_check",
        description="Detects repetitive patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            metadata={
                "pattern_type": "repeat",
                "pattern_length": 3,
                "case_sensitive": False,
                "allow_overlap": True,
            },
        ),
    )

    # Check for patterns
    symmetry_result = symmetry_rule._validate_impl(text)
    results["symmetry"] = {
        "passed": symmetry_result.passed,
        "message": symmetry_result.message,
        "metadata": symmetry_result.metadata,
    }

    repetition_result = repetition_rule._validate_impl(text)
    results["repetition"] = {
        "passed": repetition_result.passed,
        "message": repetition_result.message,
        "metadata": repetition_result.metadata,
    }

    return results


def main():
    # Load environment variables
    load_dotenv()

    # Initialize the model provider with configuration
    config = ModelConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7,
        max_tokens=2000,
    )

    model = AnthropicProvider(
        model_name="claude-3-haiku-20240307",
        config=config,
    )

    # Create the Pydantic parser
    parser = PydanticOutputParser(pydantic_object=MovieReview)

    # Create a prompt template that includes the parser's formatting instructions
    prompt = PromptTemplate(
        template="""
        Write a detailed movie review for the following movie: {movie_title}

        {format_instructions}

        CRITICAL SUMMARY LENGTH REQUIREMENT:
        The summary field MUST be EXACTLY 250 characters or less. Here's a template to follow:
        "[Movie Title] is a [genre] film about [brief plot]. Through [key elements], it explores [main theme]. With [notable aspects], the film [conclusion]."

        Example of appropriate length summary (exactly 250 characters):
        "The Godfather is a crime drama about the Corleone family's rise in the criminal underworld. Through masterful direction and performances, it explores themes of family and power. With iconic scenes and rich character development, the film defines the gangster genre."
        Character count: 250

        For Inception, here's an example of an appropriate length summary:
        "Inception is a mind-bending thriller about a team of dream infiltrators who must plant an idea in a CEO's mind. Through stunning visuals and layered storytelling, it explores the nature of reality and consciousness. With innovative effects and compelling performances, the film redefines the sci-fi genre."
        Character count: 249

        Additional requirements:
        1. Be objective in your assessment
        2. Provide specific examples for strengths and weaknesses
        3. Base the recommendation on clear criteria
        4. Follow the exact JSON format specified above
        5. Use proper JSON syntax with no trailing commas
        6. Ensure all text fields are properly escaped and formatted as valid JSON strings
        """,
        input_variables=["movie_title"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )

    # Create the chain using RunnableSequence
    llm = model.get_langchain_llm()
    chain = (
        {"movie_title": lambda x: x["movie_title"]}
        | prompt
        | llm
        | (lambda x: x.content if hasattr(x, "content") else x)
    )

    # Create chain configuration with the parser
    config = ChainConfig(
        output_parser=parser,
        critique=True,
    )

    # Wrap the chain with Sifaka
    sifaka_chain = wrap_chain(chain=chain, config=config)

    # Example movies to review
    movies = [
        "The Matrix",
        "Inception",
        "The Shawshank Redemption",
    ]

    # Generate and validate reviews
    for movie in movies:
        logger.info("\nGenerating review for: %s", movie)
        try:
            result = sifaka_chain.run({"movie_title": movie})
            review = MovieReview.model_validate(result)

            # Log the structured review
            logger.info("\nValidated Review:")
            logger.info("Title: %s (%d)", review.title, review.year)
            logger.info("Rating: %.1f/10", review.rating)
            logger.info("Genre: %s", ", ".join(review.genre))
            logger.info("\nStrengths:")
            for strength in review.strengths:
                logger.info("- %s", strength)
            logger.info("\nWeaknesses:")
            for weakness in review.weaknesses:
                logger.info("- %s", weakness)
            logger.info("\nSummary: %s", review.summary)
            logger.info("Recommended: %s", "Yes" if review.recommended else "No")

            # Perform pattern analysis on the summary
            pattern_results = analyze_text(review.summary)

            logger.info("\nSummary Pattern Analysis:")
            logger.info("Symmetry Analysis:")
            logger.info("- Score: %.2f", pattern_results["symmetry"]["metadata"]["symmetry_score"])
            logger.info("- Message: %s", pattern_results["symmetry"]["message"])

            logger.info("\nRepetition Analysis:")
            logger.info("- Message: %s", pattern_results["repetition"]["message"])
            if "patterns" in pattern_results["repetition"]["metadata"]:
                logger.info("- Notable patterns:")
                patterns = pattern_results["repetition"]["metadata"]["patterns"]
                if isinstance(patterns, list) and len(patterns) > 0:
                    for pattern in patterns[:3]:
                        logger.info("  * %s", pattern)

        except Exception as e:
            logger.error("Failed to generate or validate review: %s", str(e))


if __name__ == "__main__":
    main()
