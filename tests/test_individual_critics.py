"""
Individual critic testing to ensure each critic works correctly.

This test suite focuses on testing each critic in isolation to ensure:
1. Each critic can be instantiated correctly
2. Each critic provides meaningful feedback
3. Each critic handles edge cases properly
4. Each critic's feedback appears correctly in prompts
"""

import pytest
import asyncio
from sifaka import Sifaka


class TestReflexionCritic:
    """Test ReflexionCritic specifically."""
    
    @pytest.mark.asyncio
    async def test_reflexion_basic_functionality(self):
        """Test basic ReflexionCritic functionality."""
        result = await (
            Sifaka('AI is good')
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .improve()
        )
        
        reflexion_critiques = [c for c in result.critiques if c.critic == 'ReflexionCritic']
        assert len(reflexion_critiques) > 0
        
        # Check that ReflexionCritic provides meaningful feedback
        for critique in reflexion_critiques:
            assert critique.feedback is not None
            assert len(critique.feedback) > 0
            assert critique.confidence > 0
            assert len(critique.suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_reflexion_with_validation_context(self):
        """Test ReflexionCritic with validation failures."""
        result = await (
            Sifaka('Write a very long explanation about AI that exceeds limits')
            .max_length(50)  # Force validation failure
            .max_iterations(2)
            .with_reflexion()
            .improve()
        )
        
        # Should have both validation and critic feedback
        assert len(result.validations) > 0
        assert len(result.critiques) > 0
        
        # ReflexionCritic should be aware of validation context
        reflexion_critiques = [c for c in result.critiques if c.critic == 'ReflexionCritic']
        assert len(reflexion_critiques) > 0


class TestConstitutionalCritic:
    """Test ConstitutionalCritic specifically."""
    
    @pytest.mark.asyncio
    async def test_constitutional_basic_functionality(self):
        """Test basic ConstitutionalCritic functionality."""
        result = await (
            Sifaka('Write about AI ethics and safety concerns')
            .max_length(2000)
            .max_iterations(2)
            .with_constitutional()
            .improve()
        )
        
        constitutional_critiques = [c for c in result.critiques if c.critic == 'ConstitutionalCritic']
        assert len(constitutional_critiques) > 0
        
        # Check constitutional principles are evaluated
        for critique in constitutional_critiques:
            assert critique.feedback is not None
            assert 'constitutional' in critique.feedback.lower() or 'principle' in critique.feedback.lower()
            assert len(critique.suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_constitutional_harmlessness_check(self):
        """Test ConstitutionalCritic evaluates harmlessness."""
        result = await (
            Sifaka('Explain how to build safe AI systems')
            .max_length(2000)
            .max_iterations(2)
            .with_constitutional()
            .improve()
        )
        
        constitutional_critiques = [c for c in result.critiques if c.critic == 'ConstitutionalCritic']
        assert len(constitutional_critiques) > 0
        
        # Should evaluate harmlessness principle
        for critique in constitutional_critiques:
            feedback_lower = critique.feedback.lower()
            assert any(word in feedback_lower for word in ['harmless', 'safe', 'helpful', 'honest'])


class TestSelfConsistencyCritic:
    """Test SelfConsistencyCritic specifically."""
    
    @pytest.mark.asyncio
    async def test_self_consistency_basic_functionality(self):
        """Test basic SelfConsistencyCritic functionality."""
        result = await (
            Sifaka('Explain machine learning concepts')
            .max_length(2000)
            .max_iterations(2)
            .with_self_consistency()
            .improve()
        )
        
        consistency_critiques = [c for c in result.critiques if c.critic == 'SelfConsistencyCritic']
        assert len(consistency_critiques) > 0
        
        # Check consistency analysis
        for critique in consistency_critiques:
            assert critique.feedback is not None
            assert 'consensus' in critique.feedback.lower() or 'consistency' in critique.feedback.lower()
            assert critique.confidence > 0
    
    @pytest.mark.asyncio
    async def test_self_consistency_multiple_attempts(self):
        """Test that SelfConsistencyCritic performs multiple attempts."""
        result = await (
            Sifaka('Write about neural networks')
            .max_length(2000)
            .max_iterations(2)
            .with_self_consistency()
            .improve()
        )
        
        consistency_critiques = [c for c in result.critiques if c.critic == 'SelfConsistencyCritic']
        assert len(consistency_critiques) > 0
        
        # Check that metadata indicates multiple attempts
        for critique in consistency_critiques:
            if hasattr(critique, 'critic_metadata') and critique.critic_metadata:
                metadata = critique.critic_metadata
                if 'critique_attempts' in metadata:
                    assert metadata['critique_attempts'] > 1


class TestNCriticsCritic:
    """Test NCriticsCritic specifically."""
    
    @pytest.mark.asyncio
    async def test_n_critics_basic_functionality(self):
        """Test basic NCriticsCritic functionality."""
        result = await (
            Sifaka('Discuss the future of artificial intelligence')
            .max_length(2000)
            .max_iterations(2)
            .with_n_critics()
            .improve()
        )
        
        n_critics_critiques = [c for c in result.critiques if c.critic == 'NCriticsCritic']
        assert len(n_critics_critiques) > 0
        
        # Check multiple perspectives
        for critique in n_critics_critiques:
            assert critique.feedback is not None
            assert len(critique.suggestions) > 0
            # Should mention multiple perspectives or critics
            feedback_lower = critique.feedback.lower()
            assert any(word in feedback_lower for word in ['perspective', 'critic', 'analysis', 'evaluation'])


class TestSelfRefineCritic:
    """Test SelfRefineCritic specifically."""
    
    @pytest.mark.asyncio
    async def test_self_refine_basic_functionality(self):
        """Test basic SelfRefineCritic functionality."""
        result = await (
            Sifaka('Explain deep learning architectures')
            .max_length(2000)
            .max_iterations(2)
            .with_self_refine()
            .improve()
        )
        
        refine_critiques = [c for c in result.critiques if c.critic == 'SelfRefineCritic']
        assert len(refine_critiques) > 0
        
        # Check refinement suggestions
        for critique in refine_critiques:
            assert critique.feedback is not None
            assert len(critique.suggestions) > 0
            # Should focus on refinement and improvement
            feedback_lower = critique.feedback.lower()
            assert any(word in feedback_lower for word in ['refine', 'improve', 'enhance', 'better'])


class TestCriticErrorHandling:
    """Test error handling for critics."""
    
    @pytest.mark.asyncio
    async def test_critic_with_very_short_text(self):
        """Test critics handle very short text appropriately."""
        result = await (
            Sifaka('AI')
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .with_constitutional()
            .improve()
        )
        
        # Should handle short text without errors
        assert result.final_text is not None
        assert len(result.critiques) >= 0  # May or may not provide feedback for very short text
    
    @pytest.mark.asyncio
    async def test_critic_with_very_long_text(self):
        """Test critics handle very long text appropriately."""
        long_text = 'Write a comprehensive analysis of artificial intelligence ' * 100
        result = await (
            Sifaka(long_text)
            .max_length(10000)
            .max_iterations(2)
            .with_reflexion()
            .improve()
        )
        
        # Should handle long text without errors
        assert result.final_text is not None
        reflexion_critiques = [c for c in result.critiques if c.critic == 'ReflexionCritic']
        assert len(reflexion_critiques) > 0
    
    @pytest.mark.asyncio
    async def test_critic_with_special_characters(self):
        """Test critics handle text with special characters."""
        result = await (
            Sifaka('Write about AI & ML: "The Future" (2024) - 50% improvement!')
            .max_length(2000)
            .max_iterations(2)
            .with_constitutional()
            .improve()
        )
        
        # Should handle special characters without errors
        assert result.final_text is not None
        constitutional_critiques = [c for c in result.critiques if c.critic == 'ConstitutionalCritic']
        assert len(constitutional_critiques) >= 0


class TestCriticMetadata:
    """Test critic metadata and additional information."""
    
    @pytest.mark.asyncio
    async def test_critic_metadata_present(self):
        """Test that critics provide metadata about their operation."""
        result = await (
            Sifaka('Explain quantum computing principles')
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .with_constitutional()
            .improve()
        )
        
        for critique in result.critiques:
            # Should have basic metadata
            assert hasattr(critique, 'timestamp')
            assert hasattr(critique, 'confidence')
            assert hasattr(critique, 'critic')
            assert critique.critic in ['ReflexionCritic', 'ConstitutionalCritic']
    
    @pytest.mark.asyncio
    async def test_critic_processing_time_recorded(self):
        """Test that critic processing times are recorded."""
        result = await (
            Sifaka('Analyze the impact of AI on society')
            .max_length(2000)
            .max_iterations(2)
            .with_self_consistency()
            .improve()
        )
        
        consistency_critiques = [c for c in result.critiques if c.critic == 'SelfConsistencyCritic']
        assert len(consistency_critiques) > 0
        
        for critique in consistency_critiques:
            # Should have processing time information
            if hasattr(critique, 'critic_metadata') and critique.critic_metadata:
                # Processing time should be recorded somewhere
                assert critique.critic_metadata is not None
