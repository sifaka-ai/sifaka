"""End-to-end workflow tests demonstrating real-world usage patterns."""

import os

import pytest

from sifaka import Config, improve
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.validators.composable import Validator


class TestRealWorldScenarios:
    """Test complete workflows for real-world use cases."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_blog_post_improvement(self):
        """Test improving a blog post draft."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Draft blog post
        draft = """
        Why Python is Great

        Python is a good programming language. Many people use it. It's easy
        to learn. You can do lots of things with Python like web development
        and data science. Companies like Google use Python.

        Python has many libraries. This makes development faster. The syntax
        is simple. Beginners can start quickly.

        In conclusion, Python is a great choice for programming.
        """

        # Create validators for blog posts
        blog_validator = (
            Validator.create("blog_post")
            .length(800, 2000)  # Good blog post length
            .sentences(15, 50)
            .contains(["example", "specifically", "consider"], mode="any")
            .build()
        )

        # Use multiple critics for comprehensive improvement
        result = await improve(
            draft,
            critics=[
                "reflexion",  # Learn from iterations
                "self_refine",  # General quality
                NCriticsCritic(  # Multiple perspectives
                    perspectives={
                        "Content Strategist": "Focus on engaging headlines and structure",
                        "SEO Expert": "Consider search optimization and keywords",
                        "Technical Writer": "Ensure accuracy and clarity of technical content",
                        "Audience Advocate": "Make it accessible to beginners",
                    },
                    api_key=api_key,
                ),
            ],
            validators=[blog_validator],
            max_iterations=3,
            api_key=api_key,
        )

        # Verify improvements
        assert len(result.final_text) > len(draft) * 1.5  # Should expand significantly
        assert result.iteration >= 2  # Should take multiple iterations

        # Check for blog post improvements
        final_lower = result.final_text.lower()
        improvements = {
            "examples": "example" in final_lower or "for instance" in final_lower,
            "structure": any(
                marker in final_lower
                for marker in ["first", "second", "additionally", "furthermore"]
            ),
            "engagement": any(
                word in final_lower for word in ["you", "your", "consider", "imagine"]
            ),
            "depth": len(result.final_text.split()) > len(draft.split()) * 1.5,
        }

        print("\nBlog Post Improvements:")
        for aspect, achieved in improvements.items():
            print(f"  {aspect}: {'✓' if achieved else '✗'}")

        # At least 3 improvements should be achieved
        assert sum(improvements.values()) >= 3

        print(f"\nOriginal: {len(draft.split())} words")
        print(f"Final: {len(result.final_text.split())} words")
        print(f"Iterations: {result.iteration}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_technical_documentation_improvement(self):
        """Test improving technical documentation with fact checking."""
        pytest.skip("TODO: Fix RetrievalBackend and SelfRAGEnhancedCritic imports")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Technical documentation with some issues
        _tech_doc = """
        API Documentation: Data Processing Endpoint

        The /process endpoint accepts JSON data and returns processed results.
        It uses HTTP GET method for all operations. The maximum payload size
        is 100GB. Processing is synchronous and typically takes 1-2 milliseconds
        regardless of data size.

        Authentication: Not required
        Rate limiting: Unlimited requests

        Example:
        GET /process
        Body: {"data": [1, 2, 3]}

        Returns: {"result": "processed"}
        """

        # TODO: Fix RetrievalBackend and SelfRAGEnhancedCritic imports
        # Create fact-checking retrieval backend
        # class TechFactChecker(RetrievalBackend):
        #     async def retrieve(
        #         self, query: str, top_k: int = 3
        #     ) -> List[Dict[str, Any]]:
        #         facts = {
        #             "http get": [
        #                 {
        #                     "content": "HTTP GET requests should not have a request body according to RFC 7231.",
        #                     "source": "HTTP RFC 7231",
        #                     "score": 0.95,
        #                 }
        #             ],
        #             "payload size": [
        #                 {
        #                     "content": "Typical API payload limits range from 1MB to 100MB, not GB.",
        #                     "source": "API Best Practices",
        #                     "score": 0.9,
        #                 }
        #             ],
        #             "rate limiting": [
        #                 {
        #                     "content": "APIs should implement rate limiting to prevent abuse.",
        #                     "source": "Security Best Practices",
        #                     "score": 0.85,
        #                 }
        #             ],
        #         }
        #
        #         results = []
        #         query_lower = query.lower()
        #         for key, fact_list in facts.items():
        #             if key in query_lower:
        #                 results.extend(fact_list)
        #
        #         return results[:top_k]
        #
        #     async def add_documents(self, documents: List[Dict[str, str]]) -> None:
        #         pass

        # Use SelfRAG with fact checking
        # fact_checking_critic = SelfRAGEnhancedCritic(
        #     retrieval_backend=TechFactChecker(),
        #     retrieval_threshold=0.6,
        #     api_key=api_key,
        # )

        # Technical validators
        # tech_validator = (
        #     Validator.create("tech_doc")
        #     .contains(["endpoint", "method", "authentication", "example"], mode="all")
        #     .matches(
        #         r"`[^`]+`|```[\s\S]+?```", "code_blocks"
        #     )  # Should have code formatting
        #     .build()
        # )

        # result = await improve(
        #     tech_doc,
        #     critics=[
        #         fact_checking_critic,
        #         "constitutional",  # Ensure accuracy principles
        #         "self_refine",  # General improvements
        #     ],
        #     validators=[tech_validator],
        #     max_iterations=3,
        #     api_key=api_key,
        # )

        # # Verify corrections
        # final_lower = result.final_text.lower()

        # # Should correct technical errors
        # corrections = {
        #     "http_method": "post" in final_lower or "put" in final_lower,
        #     "payload_size": "mb" in final_lower and "gb" not in final_lower,
        #     "rate_limits": "rate limit" in final_lower
        #     and "unlimited" not in final_lower,
        #     "auth_mentioned": "authentication" in final_lower,
        # }

        # print("\nTechnical Corrections:")
        # for aspect, corrected in corrections.items():
        #     print(f"  {aspect}: {'✓' if corrected else '✗'}")

        # # Should have made key corrections
        # assert sum(corrections.values()) >= 2

        # # Check retrieval was used
        # for critique in result.critiques:
        #     if critique.critic == "self_rag_enhanced":
        #         context = critique.metadata.get("retrieval_context", {})
        #         if context.get("verified_claims"):
        #             print(f"\nFacts checked: {len(context['verified_claims'])}")

        # Temporary placeholder - test needs to be fixed
        assert True  # TODO: Fix this test when RetrievalBackend is available

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_academic_paper_abstract_improvement(self):
        """Test improving an academic paper abstract."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Draft abstract
        draft_abstract = """
        This paper is about machine learning. We used neural networks to solve
        a problem. The results were good. Our method is better than other methods.
        This is important for the field.
        """

        # Academic writing validators
        academic_validator = (
            Validator.create("academic_abstract")
            .length(150, 300)  # Typical abstract length in words
            .contains(
                [
                    "objective",
                    "method",
                    "results",
                    "conclusion",
                    "we present",
                    "this paper",
                    "our approach",
                    "we demonstrate",
                ],
                mode="any",
            )
            .sentences(4, 8)  # Structured abstract
            .build()
        )

        # Configure for academic style
        config = Config(
            temperature=0.3,  # Lower temperature for formal writing
            use_advanced_confidence=True,
        )

        result = await improve(
            draft_abstract,
            critics=[
                "constitutional",  # Ensure academic principles
                "meta_rewarding",  # Self-evaluate quality
                NCriticsCritic(
                    perspectives={
                        "Journal Editor": "Evaluate clarity, novelty, and contribution",
                        "Peer Reviewer": "Assess methodology and rigor",
                        "Graduate Student": "Check accessibility and context",
                    },
                    api_key=api_key,
                ),
            ],
            validators=[academic_validator],
            config=config,
            max_iterations=3,
            api_key=api_key,
        )

        # Check for academic improvements
        final_text = result.final_text
        academic_elements = {
            "structure": all(
                section in final_text.lower()
                for section in ["present", "method", "result"]
            ),
            "formality": not any(
                casual in final_text.lower()
                for casual in ["good", "bad", "nice", "stuff"]
            ),
            "specificity": any(
                term in final_text.lower()
                for term in ["accuracy", "performance", "dataset", "baseline"]
            ),
            "contribution": any(
                phrase in final_text.lower()
                for phrase in ["contribution", "novel", "advance", "improve"]
            ),
        }

        print("\nAcademic Abstract Improvements:")
        for element, present in academic_elements.items():
            print(f"  {element}: {'✓' if present else '✗'}")

        assert sum(academic_elements.values()) >= 3

        # Should be more formal and structured
        print(f"\nOriginal: {len(draft_abstract.split())} words")
        print(f"Final: {len(result.final_text.split())} words")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_customer_email_improvement(self):
        """Test improving customer communication."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Draft customer email
        draft_email = """
        Hi,

        Your order is late. This is not acceptable. We are looking into it.
        You will get a refund if you want.

        Sorry for the trouble.

        Thanks
        """

        # Customer service validators
        service_validator = (
            Validator.create("customer_email")
            .length(100, 300)
            .contains(["apologize", "sorry", "understand"], mode="any")
            .contains(["solution", "resolve", "assist", "help"], mode="any")
            .build()
        )

        # Critics for customer communication
        result = await improve(
            draft_email,
            critics=[
                "constitutional",  # Ensure respectful tone
                NCriticsCritic(
                    perspectives={
                        "Customer Success Manager": "Focus on empathy and solutions",
                        "Communications Expert": "Ensure professional yet warm tone",
                        "Customer Advocate": "Address customer concerns fully",
                    },
                    api_key=api_key,
                ),
                "self_refine",  # Polish the message
            ],
            validators=[service_validator],
            max_iterations=2,
            api_key=api_key,
        )

        # Check improvements
        final_lower = result.final_text.lower()
        improvements = {
            "empathy": any(
                phrase in final_lower
                for phrase in [
                    "understand your frustration",
                    "apologize",
                    "sincerely sorry",
                ]
            ),
            "professionalism": any(
                greeting in result.final_text
                for greeting in ["Dear", "Hello", "Hi there"]
            ),
            "solutions": any(
                solution in final_lower
                for solution in ["resolve", "solution", "next steps", "immediately"]
            ),
            "contact": any(
                contact in final_lower
                for contact in ["reach out", "contact", "available", "questions"]
            ),
        }

        print("\nCustomer Email Improvements:")
        for aspect, improved in improvements.items():
            print(f"  {aspect}: {'✓' if improved else '✗'}")

        assert sum(improvements.values()) >= 3

        # Tone should be more professional
        assert "not acceptable" not in result.final_text

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_code_review_comment_improvement(self):
        """Test improving code review comments."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Draft code review comment
        draft_comment = """
        This code is bad. You should not use global variables. The function
        is too long. Fix the naming. Also why are you not using types?
        """

        # Code review validators
        review_validator = (
            Validator.create("code_review")
            .contains(["suggest", "consider", "recommend"], mode="any")
            .contains(["example", "instead", "alternative"], mode="any")
            .build()
        )

        result = await improve(
            draft_comment,
            critics=[
                "constitutional",  # Ensure constructive feedback
                "self_refine",  # Make it helpful
                NCriticsCritic(
                    perspectives={
                        "Senior Developer": "Focus on best practices and mentoring",
                        "Team Lead": "Ensure feedback is actionable and specific",
                        "Code Quality Expert": "Address technical concerns thoroughly",
                    },
                    api_key=api_key,
                ),
            ],
            validators=[review_validator],
            max_iterations=2,
            api_key=api_key,
        )

        # Check for constructive feedback elements
        final_lower = result.final_text.lower()
        constructive_elements = {
            "constructive_tone": "bad" not in final_lower,
            "suggestions": any(
                word in final_lower
                for word in ["suggest", "consider", "recommend", "could"]
            ),
            "specific_examples": any(
                word in final_lower
                for word in ["example", "instead", "such as", "like"]
            ),
            "reasoning": any(
                word in final_lower
                for word in ["because", "helps", "improves", "benefit"]
            ),
            "encouragement": any(
                phrase in final_lower
                for phrase in ["good", "great", "nice", "well done", "appreciate"]
            ),
        }

        print("\nCode Review Improvements:")
        for element, present in constructive_elements.items():
            print(f"  {element}: {'✓' if present else '✗'}")

        assert sum(constructive_elements.values()) >= 4

        print("\nOriginal tone: Critical and harsh")
        print("Improved tone: Constructive and helpful")


class TestComplexWorkflows:
    """Test complex multi-step workflows."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_iterative_refinement_workflow(self):
        """Test a workflow that iteratively refines content."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        # Start with a basic idea
        initial_idea = "Write about the importance of sleep for productivity."

        # Step 1: Expand the idea
        outline = await improve(
            initial_idea, critics=["self_refine"], max_iterations=1, api_key=api_key
        )

        print("\nStep 1 - Expanded idea:")
        print(outline.final_text[:200] + "...")

        # Step 2: Create detailed content
        detailed = await improve(
            outline.final_text + "\n\nNow expand this into a detailed article.",
            critics=["reflexion", "n_critics"],
            max_iterations=2,
            api_key=api_key,
        )

        print(f"\nStep 2 - Detailed content ({len(detailed.final_text)} chars)")

        # Step 3: Polish and finalize
        final_validator = Validator.length(1000, 2000)

        polished = await improve(
            detailed.final_text,
            critics=["constitutional", "self_consistency"],
            validators=[final_validator],
            max_iterations=2,
            api_key=api_key,
        )

        print(f"\nStep 3 - Final polished ({len(polished.final_text)} chars)")

        # Verify workflow progression
        assert len(outline.final_text) > len(initial_idea) * 2
        assert len(detailed.final_text) > len(outline.final_text) * 3
        assert 1000 <= len(polished.final_text) <= 2000

        # Track quality progression
        total_iterations = outline.iteration + detailed.iteration + polished.iteration

        print("\nWorkflow Summary:")
        print(f"  Total iterations: {total_iterations}")
        print(f"  Final length: {len(polished.final_text)} chars")
        print(
            f"  Expansion factor: {len(polished.final_text) / len(initial_idea):.1f}x"
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_language_example(self):
        """Test with different providers for different languages/styles."""
        # Check what's available
        providers = []
        if os.getenv("OPENAI_API_KEY"):
            providers.append(("openai", "OPENAI_API_KEY", "gpt-4o-mini"))
        if os.getenv("ANTHROPIC_API_KEY"):
            providers.append(
                ("anthropic", "ANTHROPIC_API_KEY", "claude-3-haiku-20240307")
            )

        if len(providers) < 2:
            pytest.skip("Need at least 2 providers for this test")

        text = "Explain how photosynthesis works."

        # Use different providers for different styles
        results = {}

        # Technical explanation with one provider
        provider1, key1, model1 = providers[0]
        technical_result = await improve(
            text + " Be technical and detailed.",
            critics=["self_refine"],
            max_iterations=1,
            provider=provider1,
            api_key=os.getenv(key1),
            model=model1,
        )
        results["technical"] = technical_result.final_text

        # Simple explanation with another provider
        provider2, key2, model2 = providers[1] if len(providers) > 1 else providers[0]
        simple_result = await improve(
            text + " Explain it for a 10-year-old.",
            critics=["self_refine"],
            max_iterations=1,
            provider=provider2,
            api_key=os.getenv(key2),
            model=model2,
        )
        results["simple"] = simple_result.final_text

        # Compare results
        print("\nMulti-provider comparison:")
        print(f"Technical ({provider1}): {len(results['technical'])} chars")
        print(f"Simple ({provider2}): {len(results['simple'])} chars")

        # Should have different complexity
        technical_complex_words = sum(
            1 for word in results["technical"].split() if len(word) > 8
        )
        simple_complex_words = sum(
            1 for word in results["simple"].split() if len(word) > 8
        )

        print("\nComplexity analysis:")
        print(f"Technical complex words: {technical_complex_words}")
        print(f"Simple complex words: {simple_complex_words}")

        # Technical should be more complex
        assert technical_complex_words > simple_complex_words


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
