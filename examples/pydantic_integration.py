#!/usr/bin/env python3
"""
Pydantic Integration Example for Sifaka.

This example demonstrates:
1. Using Pydantic for structured data validation with movie reviews
2. Integrating with LangChain for LLM generation
3. Pattern analysis with Sifaka's symmetry and repetition rules
4. Content improvement with LLM-powered critics
5. Error handling and structured validation

Usage:
    python pydantic_integration.py

Requirements:
    - Python environment with Sifaka installed
    - Pydantic, LangChain packages
    - OpenAI or Anthropic API key set in environment variables
"""

import json
import os
import sys
from typing import Any, Dict, List, Union

# Add parent directory to system path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    from langchain.output_parsers import PydanticOutputParser
    from langchain.prompts import PromptTemplate
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError:
    print("Missing required packages. Install with: pip install pydantic langchain dotenv")
    sys.exit(1)

from sifaka.critics.base import CriticMetadata
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.integrations.langchain import ChainConfig, wrap_chain
from sifaka.models import AnthropicProvider, OpenAIProvider
from sifaka.models.base import ModelConfig, ModelProvider
from sifaka.rules import RepetitionRule, SymmetryRule
from sifaka.rules.base import RuleConfig, RulePriority
from sifaka.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

class MovieReview(BaseModel):
    """Structured movie review model with validation constraints."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released", ge=1900, le=2024)
    rating: float = Field(..., description="Rating from 0.0 to 10.0", ge=0.0, le=10.0)
    genre: List[str] = Field(..., description="List of movie genres", min_items=1)
    strengths: List[str] = Field(..., description="List of movie strengths", min_items=1)
    weaknesses: List[str] = Field(..., description="List of movie weaknesses", min_items=1)
    summary: str = Field(
        ...,
        description="Brief summary of the movie",
        min_length=50,
        max_length=250,
    )
    recommended: bool = Field(..., description="Whether the movie is recommended")

class ReviewCritic:
    """Critic for analyzing and improving movie reviews."""

    def __init__(self, model_provider: ModelProvider):
        """Initialize the movie review critic."""
        # Critic for analyzing reviews
        self.critic = PromptCritic(
            model=model_provider,
            config=PromptCriticConfig(
                name="movie_review_critic",
                description="Analyzes movie reviews for quality and balance",
                system_prompt=(
                    "You are an expert film critic. Analyze movie reviews for quality, "
                    "accuracy, coherence, and balance between strengths and weaknesses."
                ),
                temperature=0.7,
                max_tokens=1000,
            ),
        )

        # Critic for improving reviews
        self.improver = PromptCritic(
            model=model_provider,
            config=PromptCriticConfig(
                name="review_improver",
                description="Improves movie reviews based on feedback",
                system_prompt=(
                    "You are an expert editor specializing in movie reviews. Improve reviews "
                    "while maintaining the core assessment. Fix issues identified while "
                    "ensuring output is valid JSON matching the MovieReview schema."
                ),
                temperature=0.5,
                max_tokens=1000,
            ),
        )

    def critique_review(self, review: Union[str, Dict[str, Any]]) -> CriticMetadata:
        """Analyze a movie review for quality."""
        if isinstance(review, dict):
            review_str = json.dumps(review, indent=2)
        else:
            review_str = review
        return self.critic.critique(review_str)

    def improve_review(
        self, review: Union[str, Dict[str, Any]], feedback: List[Dict[str, Any]]
    ) -> str:
        """Improve a movie review based on validation feedback."""
        if isinstance(review, dict):
            review_str = json.dumps(review, indent=2)
        else:
            review_str = review
        return self.improver.improve(review_str, feedback)

    def evaluate_review_quality(self, review: MovieReview) -> Dict[str, Any]:
        """Evaluate the quality of a movie review across multiple dimensions."""
        issues = []

        # Check for balance between strengths and weaknesses
        if len(review.strengths) > len(review.weaknesses) * 2:
            issues.append(
                {
                    "rule": "balance",
                    "message": "Review has significantly more strengths than weaknesses",
                    "metadata": {
                        "strengths_count": len(review.strengths),
                        "weaknesses_count": len(review.weaknesses),
                    },
                }
            )
        elif len(review.weaknesses) > len(review.strengths) * 2:
            issues.append(
                {
                    "rule": "balance",
                    "message": "Review has significantly more weaknesses than strengths",
                    "metadata": {
                        "strengths_count": len(review.strengths),
                        "weaknesses_count": len(review.weaknesses),
                    },
                }
            )

        # Check summary length
        if len(review.summary) < 100:
            issues.append(
                {
                    "rule": "summary_quality",
                    "message": "Summary is too short (under 100 characters)",
                    "metadata": {"current_length": len(review.summary)},
                }
            )

        # Check alignment between rating and recommendation
        if review.rating >= 7.0 and not review.recommended:
            issues.append(
                {
                    "rule": "consistency",
                    "message": "High rating (≥7.0) contradicts not recommending the movie",
                    "metadata": {"rating": review.rating, "recommended": review.recommended},
                }
            )
        elif review.rating <= 4.0 and review.recommended:
            issues.append(
                {
                    "rule": "consistency",
                    "message": "Low rating (≤4.0) contradicts recommending the movie",
                    "metadata": {"rating": review.rating, "recommended": review.recommended},
                }
            )

        return {
            "quality_score": 1.0 - (len(issues) * 0.2),  # Reduce score for each issue
            "issues": issues,
            "is_balanced": len(issues) == 0,
        }

def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text using Sifaka's pattern detection rules."""
    # Configure symmetry rule
    symmetry_rule = SymmetryRule(
        name="symmetry_check",
        description="Checks for text symmetry patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            metadata={
                "symmetry_threshold": 0.4,
                "ignore_punctuation": True,
            },
        ),
    )

    # Configure repetition rule
    repetition_rule = RepetitionRule(
        name="repetition_check",
        description="Detects repetitive patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            metadata={
                "pattern_length": 3,
                "case_sensitive": False,
            },
        ),
    )

    # Run analyses
    symmetry_result = symmetry_rule._validate_impl(text)
    repetition_result = repetition_rule._validate_impl(text)

    return {
        "symmetry": {
            "passed": symmetry_result.passed,
            "message": symmetry_result.message,
            "metadata": symmetry_result.metadata,
        },
        "repetition": {
            "passed": repetition_result.passed,
            "message": repetition_result.message,
            "metadata": repetition_result.metadata,
        },
    }

def setup_model():
    """Set up the LLM provider based on available API keys."""
    # Check for Anthropic API key first
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        config = ModelConfig(
            api_key=api_key,
            temperature=0.7,
            max_tokens=1500,
        )
        return AnthropicProvider(model_name="claude-3-haiku-20240307", config=config)

    # Fall back to OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        config = ModelConfig(
            api_key=api_key,
            temperature=0.7,
            max_tokens=1500,
        )
        return OpenAIProvider(model_name="gpt-3.5-turbo", config=config)

    # No API keys found
    logger.error("No API key found for Anthropic or OpenAI")
    sys.exit(1)

def setup_chain(model_provider):
    """Set up the LangChain pipeline with Pydantic validation."""
    # Create Pydantic parser
    parser = PydanticOutputParser(pydantic_object=MovieReview)

    # Create prompt template
    prompt = PromptTemplate(
        template="""
        Write a detailed movie review for the following movie: {movie_title}

        {format_instructions}

        CRITICAL SUMMARY LENGTH REQUIREMENT:
        The summary field MUST be between 100-250 characters.

        Follow these guidelines:
        1. Be objective in your assessment
        2. Provide specific examples for strengths and weaknesses
        3. Base the recommendation on clear criteria
        4. Ensure all text fields are properly formatted as valid JSON
        """,
        input_variables=["movie_title"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create LangChain pipeline
    llm = model_provider.get_langchain_llm()
    chain = (
        {"movie_title": lambda x: x["movie_title"]}  # Extract movie title
        | prompt  # Format prompt
        | llm  # Generate review
        | (lambda x: x.content if hasattr(x, "content") else x)  # Extract content
    )

    # Wrap with Sifaka for validation
    return wrap_chain(chain=chain, config=ChainConfig(output_parser=parser, critique=True))

def process_review(movie_title, chain, critic):
    """Generate and process a review for a movie."""
    logger.info("\n%s", "=" * 50)
    logger.info("Generating review for: %s", movie_title)
    logger.info("=" * 50)

    try:
        # Generate review
        result = chain.run({"movie_title": movie_title})

        # Parse the review
        try:
            review = MovieReview.model_validate(result)
            review_dict = review.model_dump()
            logger.info("Generated valid review on first attempt")
        except ValidationError as e:
            logger.warning("Initial validation failed: %s", str(e))

            # Try to fix with critic
            violations = [{"rule": "validation_error", "message": str(e)}]
            improved_text = critic.improve_review(result, violations)
            review = MovieReview.model_validate_json(improved_text)
            review_dict = review.model_dump()
            logger.info("Successfully fixed review with critic")

        # Display review details
        logger.info("\nValidated Review:")
        logger.info("Title: %s (%d)", review.title, review.year)
        logger.info(
            "Rating: %.1f/10 | Recommended: %s",
            review.rating,
            "Yes" if review.recommended else "No",
        )
        logger.info("Genre: %s", ", ".join(review.genre))
        logger.info("Summary: %s", review.summary)
        logger.info("Summary Length: %d characters", len(review.summary))
        logger.info("\nStrengths: %d items", len(review.strengths))
        logger.info("Weaknesses: %d items", len(review.weaknesses))

        # Analyze patterns
        pattern_results = analyze_text(review.summary)
        logger.info("\nSummary Pattern Analysis:")
        logger.info(
            "Symmetry Score: %.2f", pattern_results["symmetry"]["metadata"]["symmetry_score"]
        )

        if "patterns" in pattern_results["repetition"]["metadata"]:
            patterns = pattern_results["repetition"]["metadata"]["patterns"]
            if patterns:
                logger.info("Found repetitive patterns: %s", patterns[0])

        # Evaluate quality
        quality = critic.evaluate_review_quality(review)
        logger.info("\nQuality Score: %.1f/1.0", quality["quality_score"])

        if quality["issues"]:
            logger.info("Quality Issues:")
            for issue in quality["issues"]:
                logger.info("- %s: %s", issue["rule"], issue["message"])

            # Improve review if issues found
            logger.info("\nImproving review...")
            try:
                improved_text = critic.improve_review(review_dict, quality["issues"])
                improved = MovieReview.model_validate_json(improved_text)

                # Compare improvements
                logger.info("Original Summary: %s", review.summary)
                logger.info("Improved Summary: %s", improved.summary)
                logger.info(
                    "Original Balance: %d strengths, %d weaknesses",
                    len(review.strengths),
                    len(review.weaknesses),
                )
                logger.info(
                    "Improved Balance: %d strengths, %d weaknesses",
                    len(improved.strengths),
                    len(improved.weaknesses),
                )
            except Exception as e:
                logger.error("Failed to improve: %s", str(e))
        else:
            logger.info("Review is well-balanced and high-quality")

        return review_dict

    except Exception as e:
        logger.error("Failed to generate review: %s", str(e))
        return None

def main():
    """Run the Pydantic integration example."""
    # Load environment variables
    load_dotenv()

    # Set up model, chain and critic
    model = setup_model()
    chain = setup_chain(model)
    critic = ReviewCritic(model)

    # Movies to review
    movies = ["The Matrix", "Inception", "The Shawshank Redemption"]

    # Process each movie
    for movie in movies:
        process_review(movie, chain, critic)

    logger.info("\nPydantic integration example completed")

if __name__ == "__main__":
    main()
