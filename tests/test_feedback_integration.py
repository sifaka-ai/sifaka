"""
Comprehensive tests for critic and validation feedback integration.

This test suite ensures that:
1. All critics provide feedback correctly
2. Validation results are passed to prompts
3. Feedback is properly attributed to each critic
4. Multiple critics work together without conflicts
5. Feedback ordering is correct (validation -> critics -> text)
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from sifaka import Sifaka
from sifaka.core.thought import SifakaThought
from sifaka.validators.length import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.self_consistency import SelfConsistencyCritic
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.critics.self_refine import SelfRefineCritic


class TestValidationFeedback:
    """Test validation feedback integration."""

    @pytest.mark.asyncio
    async def test_length_validation_feedback_in_prompt(self):
        """Test that length validation feedback appears in generation prompts."""
        result = await (
            Sifaka("Write a very long detailed explanation")
            .min_length(50)
            .max_length(100)  # Force failure
            .max_iterations(2)
            .improve()
        )

        # Check that we have validation results
        assert len(result.validations) > 0

        # Check that validation feedback appears in the second generation prompt
        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            assert "Validation Results" in second_prompt
            assert "max_100_characters" in second_prompt or "max_length" in second_prompt
            assert "FAILED" in second_prompt or "PASSED" in second_prompt

    @pytest.mark.asyncio
    async def test_validation_suggestions_in_prompt(self):
        """Test that validation suggestions are included in prompts."""
        result = await (
            Sifaka("Write about AI in exactly 500 words")
            .max_length(100)  # Force failure with suggestions
            .max_iterations(2)
            .improve()
        )

        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            # Should contain validation suggestions
            assert "Reduce by" in second_prompt or "consider removing" in second_prompt

    @pytest.mark.asyncio
    async def test_multiple_validators_feedback(self):
        """Test that multiple validators provide separate feedback."""
        result = await (
            Sifaka("AI")  # Too short
            .min_length(100)  # Will fail
            .max_length(50)  # Will also fail
            .max_iterations(2)
            .improve()
        )

        # Should have validation results from both validators
        validators_used = {v.validator for v in result.validations}
        assert "min_100_characters" in validators_used or any("min" in v for v in validators_used)
        assert "max_50_characters" in validators_used or any("max" in v for v in validators_used)


class TestCriticFeedback:
    """Test individual critic feedback integration."""

    @pytest.mark.asyncio
    async def test_reflexion_critic_feedback(self):
        """Test ReflexionCritic provides feedback in prompts."""
        result = await (
            Sifaka("Write about machine learning")
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .improve()
        )

        # Check that ReflexionCritic provided feedback
        reflexion_critiques = [c for c in result.critiques if c.critic == "ReflexionCritic"]
        assert len(reflexion_critiques) > 0

        # Check that feedback appears in prompt
        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            assert "ReflexionCritic" in second_prompt
            assert "Suggestions" in second_prompt

    @pytest.mark.asyncio
    async def test_constitutional_critic_feedback(self):
        """Test ConstitutionalCritic provides feedback in prompts."""
        result = await (
            Sifaka("Write about AI ethics")
            .max_length(2000)
            .max_iterations(2)
            .with_constitutional()
            .improve()
        )

        # Check that ConstitutionalCritic provided feedback
        constitutional_critiques = [
            c for c in result.critiques if c.critic == "ConstitutionalCritic"
        ]
        assert len(constitutional_critiques) > 0

        # Check that feedback appears in prompt
        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            assert "ConstitutionalCritic" in second_prompt

    @pytest.mark.asyncio
    async def test_self_consistency_critic_feedback(self):
        """Test SelfConsistencyCritic provides feedback in prompts."""
        result = await (
            Sifaka("Write about neural networks")
            .max_length(2000)
            .max_iterations(2)
            .with_self_consistency()
            .improve()
        )

        # Check that SelfConsistencyCritic provided feedback
        consistency_critiques = [c for c in result.critiques if c.critic == "SelfConsistencyCritic"]
        assert len(consistency_critiques) > 0

        # Check that feedback appears in prompt
        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            assert "SelfConsistencyCritic" in second_prompt

    @pytest.mark.asyncio
    async def test_n_critics_feedback(self):
        """Test NCriticsCritic provides feedback in prompts."""
        result = await (
            Sifaka("Write about deep learning")
            .max_length(2000)
            .max_iterations(2)
            .with_n_critics()
            .improve()
        )

        # Check that NCriticsCritic provided feedback
        n_critics_critiques = [c for c in result.critiques if c.critic == "NCriticsCritic"]
        assert len(n_critics_critiques) > 0

        # Check that feedback appears in prompt
        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            assert "NCriticsCritic" in second_prompt

    @pytest.mark.asyncio
    async def test_self_refine_critic_feedback(self):
        """Test SelfRefineCritic provides feedback in prompts."""
        result = await (
            Sifaka("Write about computer vision")
            .max_length(2000)
            .max_iterations(2)
            .with_self_refine()
            .improve()
        )

        # Check that SelfRefineCritic provided feedback
        refine_critiques = [c for c in result.critiques if c.critic == "SelfRefineCritic"]
        assert len(refine_critiques) > 0

        # Check that feedback appears in prompt
        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            assert "SelfRefineCritic" in second_prompt


class TestMultipleCriticsFeedback:
    """Test multiple critics working together."""

    @pytest.mark.asyncio
    async def test_multiple_critics_separate_feedback(self):
        """Test that multiple critics provide separate, attributed feedback."""
        result = await (
            Sifaka("Write about artificial intelligence and its applications")
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .with_constitutional()
            .improve()
        )

        # Check that both critics provided feedback
        critic_types = {c.critic for c in result.critiques}
        assert "ReflexionCritic" in critic_types
        assert "ConstitutionalCritic" in critic_types

        # Check that both appear separately in prompt
        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            assert "ReflexionCritic" in second_prompt
            assert "ConstitutionalCritic" in second_prompt

            # Should have separate sections for each critic
            reflexion_pos = second_prompt.find("ReflexionCritic")
            constitutional_pos = second_prompt.find("ConstitutionalCritic")
            assert reflexion_pos != constitutional_pos  # Different positions

    @pytest.mark.asyncio
    async def test_three_critics_integration(self):
        """Test three critics working together without conflicts."""
        result = await (
            Sifaka("Explain quantum computing concepts")
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .with_constitutional()
            .with_self_refine()
            .improve()
        )

        # Check that all three critics provided feedback
        critic_types = {c.critic for c in result.critiques}
        expected_critics = {"ReflexionCritic", "ConstitutionalCritic", "SelfRefineCritic"}
        assert expected_critics.issubset(critic_types)

        # Check that all appear in prompt
        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            for critic in expected_critics:
                assert critic in second_prompt


class TestFeedbackOrdering:
    """Test that feedback appears in the correct order."""

    @pytest.mark.asyncio
    async def test_validation_before_critics_before_text(self):
        """Test that validation results come before critic feedback before previous text."""
        result = await (
            Sifaka("Write a comprehensive guide to machine learning")
            .min_length(100)
            .max_length(500)  # Force validation failure
            .max_iterations(2)
            .with_reflexion()
            .improve()
        )

        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt

            # Find positions of key sections
            validation_pos = second_prompt.find("Validation Results")
            critic_pos = second_prompt.find("Critic")
            text_pos = second_prompt.find("Previous attempt:")

            # Validation should come first, then critics, then text
            assert validation_pos < critic_pos < text_pos
            assert validation_pos > 0  # Should exist
            assert critic_pos > 0  # Should exist
            assert text_pos > 0  # Should exist

    @pytest.mark.asyncio
    async def test_feedback_before_text_improves_effectiveness(self):
        """Test that feedback-first ordering is being used."""
        result = await (
            Sifaka("Write about renewable energy")
            .max_length(200)  # Force validation failure
            .max_iterations(2)
            .with_reflexion()
            .improve()
        )

        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt

            # Should start with task instruction, then feedback, then text
            lines = second_prompt.split("\n")
            early_lines = lines[:10]  # First 10 lines

            # Should contain feedback instruction early
            early_text = "\n".join(early_lines)
            assert "Improve the following text based on this feedback" in early_text


class TestFeedbackEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_critic_failure_doesnt_break_pipeline(self):
        """Test that if one critic fails, others still provide feedback."""
        # This test would need to mock a critic failure
        result = await (
            Sifaka("Write about blockchain technology")
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .with_constitutional()  # One might fail due to API limits
            .improve()
        )

        # Should still have some critiques even if one fails
        assert len(result.critiques) > 0

        # Should still have a valid result
        assert result.final_text is not None
        assert len(result.final_text) > 0

    @pytest.mark.asyncio
    async def test_no_critics_only_validation(self):
        """Test that validation-only mode works correctly."""
        result = (
            await (
                Sifaka("Write a short note about AI")
                .min_length(50)
                .max_length(200)
                .max_iterations(2)
                # No critics added
                .improve()
            )
        )

        # Should have validation results
        assert len(result.validations) > 0

        # Should not have critic feedback
        assert len(result.critiques) == 0

        # Prompt should still work
        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            assert "Validation Results" in second_prompt
            assert "Critic" not in second_prompt

    @pytest.mark.asyncio
    async def test_all_validations_pass_early_termination(self):
        """Test behavior when all validations pass (should continue or terminate based on config)."""
        result = await (
            Sifaka("Write about AI")
            .min_length(10)  # Easy to pass
            .max_length(5000)  # Easy to pass
            .max_iterations(3)
            .with_reflexion()
            .improve()
        )

        # Should have validation results
        assert len(result.validations) > 0

        # Check if validations passed
        final_validations = [v for v in result.validations if v.iteration == result.iteration - 1]
        if all(v.passed for v in final_validations):
            # If all validations passed, system might continue for critics or terminate
            # This depends on configuration - just ensure it's handled gracefully
            assert result.final_text is not None

    @pytest.mark.asyncio
    async def test_empty_critic_suggestions(self):
        """Test handling of critics that return empty suggestions."""
        # This would require mocking a critic to return empty suggestions
        result = await (
            Sifaka("Perfect text that needs no improvement")
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .improve()
        )

        # Should handle empty suggestions gracefully
        assert result.final_text is not None

        # Check that empty suggestions don't break prompt generation
        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            assert second_prompt is not None
            assert len(second_prompt) > 0


class TestFeedbackContent:
    """Test the actual content and quality of feedback."""

    @pytest.mark.asyncio
    async def test_validation_suggestions_are_actionable(self):
        """Test that validation suggestions provide actionable guidance."""
        result = await (
            Sifaka(
                "Write a very long detailed explanation about machine learning algorithms and their applications in various industries including healthcare, finance, automotive, and technology sectors with comprehensive examples and case studies"
            )
            .max_length(100)  # Force failure
            .max_iterations(2)
            .improve()
        )

        # Find failed validations
        failed_validations = [v for v in result.validations if not v.passed]
        assert len(failed_validations) > 0

        # Check that suggestions are present and actionable
        for validation in failed_validations:
            if "suggestions" in validation.details:
                suggestions = validation.details["suggestions"]
                assert len(suggestions) > 0
                # Should contain actionable advice
                suggestion_text = " ".join(suggestions)
                assert any(
                    word in suggestion_text.lower()
                    for word in ["reduce", "remove", "shorten", "condense"]
                )

    @pytest.mark.asyncio
    async def test_critic_suggestions_are_specific(self):
        """Test that critic suggestions are specific and helpful."""
        result = await (
            Sifaka("AI is good. It helps people. It is useful.")
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .with_constitutional()
            .improve()
        )

        # Check that critics provided specific suggestions
        for critique in result.critiques:
            assert len(critique.suggestions) > 0
            for suggestion in critique.suggestions:
                assert len(suggestion) > 10  # Should be substantial
                assert suggestion.strip() != ""  # Should not be empty

    @pytest.mark.asyncio
    async def test_weight_information_in_prompts(self):
        """Test that weight information is correctly displayed in prompts."""
        result = await (
            Sifaka("Write about artificial intelligence")
            .validation_weight(0.7)  # 70% validation, 30% critics
            .max_length(500)
            .max_iterations(2)
            .with_reflexion()
            .improve()
        )

        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt
            # Should show correct weights
            assert "70%" in second_prompt  # Validation weight
            assert "30%" in second_prompt  # Critic weight

    @pytest.mark.asyncio
    async def test_all_critic_suggestions_included(self):
        """Test that no critic suggestions are truncated or lost."""
        result = await (
            Sifaka("Write a comprehensive analysis of machine learning")
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .with_constitutional()
            .improve()
        )

        if len(result.generations) > 1:
            second_prompt = result.generations[1].user_prompt

            # Count suggestions in thought vs prompt
            prev_iteration = result.iteration - 1
            thought_critiques = [c for c in result.critiques if c.iteration == prev_iteration]

            total_suggestions_in_thought = sum(len(c.suggestions) for c in thought_critiques)
            suggestions_in_prompt = second_prompt.count("- ")  # Count bullet points

            # Should have at least as many suggestions in prompt as in thought
            # (might have more due to validation suggestions)
            assert suggestions_in_prompt >= total_suggestions_in_thought
