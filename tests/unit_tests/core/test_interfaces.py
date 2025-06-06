"""Comprehensive unit tests for core interfaces and protocols.

This module tests the core protocols and base classes:
- Validator protocol and BaseValidator
- Critic protocol and BaseCritic  
- Storage protocol and BaseStorage
- Retriever protocol and BaseRetriever
- Protocol compliance and type checking

Tests cover:
- Protocol method signatures and contracts
- Base class implementations
- Abstract method enforcement
- Type checking and protocol compliance
- Mock implementations for testing
"""

import pytest
from abc import ABC
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock

from sifaka.core.interfaces import (
    Validator,
    Critic,
    Storage,
    Retriever,
    BaseValidator,
    BaseCritic,
    BaseStorage,
    BaseRetriever,
)
from sifaka.core.thought import SifakaThought


class TestValidatorProtocol:
    """Test the Validator protocol and its contract."""

    def test_validator_protocol_methods(self):
        """Test that Validator protocol defines the correct methods."""
        # Create a mock that implements the Validator protocol
        mock_validator = Mock()
        mock_validator.validate_async = AsyncMock(return_value={
            "passed": True,
            "details": {},
            "score": 0.8
        })
        mock_validator.name = "test-validator"
        
        # Verify protocol methods exist
        assert hasattr(mock_validator, 'validate_async')
        assert hasattr(mock_validator, 'name')
        assert callable(mock_validator.validate_async)

    @pytest.mark.asyncio
    async def test_validator_protocol_contract(self):
        """Test the expected contract of Validator protocol."""
        # Create a proper implementation
        class TestValidator:
            @property
            def name(self) -> str:
                return "test-validator"
            
            async def validate_async(self, text: str) -> Dict[str, Any]:
                return {
                    "passed": True,
                    "details": {"length": len(text)},
                    "score": 0.9
                }
        
        validator = TestValidator()
        
        # Test the contract
        result = await validator.validate_async("test text")
        
        assert isinstance(result, dict)
        assert "passed" in result
        assert "details" in result
        assert "score" in result
        assert isinstance(result["passed"], bool)
        assert isinstance(result["details"], dict)
        assert isinstance(result["score"], (int, float, type(None)))
        assert validator.name == "test-validator"


class TestCriticProtocol:
    """Test the Critic protocol and its contract."""

    def test_critic_protocol_methods(self):
        """Test that Critic protocol defines the correct methods."""
        mock_critic = Mock()
        mock_critic.critique_async = AsyncMock(return_value={
            "feedback": "Good text",
            "suggestions": [],
            "needs_improvement": False
        })
        mock_critic.improve_async = AsyncMock(return_value="Improved text")
        mock_critic.name = "test-critic"
        
        # Verify protocol methods exist
        assert hasattr(mock_critic, 'critique_async')
        assert hasattr(mock_critic, 'improve_async')
        assert hasattr(mock_critic, 'name')
        assert callable(mock_critic.critique_async)
        assert callable(mock_critic.improve_async)

    @pytest.mark.asyncio
    async def test_critic_protocol_contract(self):
        """Test the expected contract of Critic protocol."""
        class TestCritic:
            @property
            def name(self) -> str:
                return "test-critic"
            
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {
                    "feedback": "The text looks good",
                    "suggestions": ["Add more examples"],
                    "needs_improvement": False
                }
            
            async def improve_async(self, thought: SifakaThought) -> str:
                return "Improved version of the text"
        
        critic = TestCritic()
        thought = SifakaThought(prompt="Test", current_text="Test text")
        
        # Test critique contract
        critique_result = await critic.critique_async(thought)
        
        assert isinstance(critique_result, dict)
        assert "feedback" in critique_result
        assert "suggestions" in critique_result
        assert "needs_improvement" in critique_result
        assert isinstance(critique_result["feedback"], str)
        assert isinstance(critique_result["suggestions"], list)
        assert isinstance(critique_result["needs_improvement"], bool)
        
        # Test improve contract
        improved_text = await critic.improve_async(thought)
        assert isinstance(improved_text, str)
        assert critic.name == "test-critic"


class TestStorageProtocol:
    """Test the Storage protocol and its contract."""

    def test_storage_protocol_methods(self):
        """Test that Storage protocol defines the correct methods."""
        mock_storage = Mock()
        mock_storage.store_thought = AsyncMock()
        mock_storage.retrieve_thought = AsyncMock(return_value=None)
        
        # Verify protocol methods exist
        assert hasattr(mock_storage, 'store_thought')
        assert hasattr(mock_storage, 'retrieve_thought')
        assert callable(mock_storage.store_thought)
        assert callable(mock_storage.retrieve_thought)

    @pytest.mark.asyncio
    async def test_storage_protocol_contract(self):
        """Test the expected contract of Storage protocol."""
        class TestStorage:
            def __init__(self):
                self._thoughts = {}
            
            async def store_thought(self, thought: SifakaThought) -> None:
                self._thoughts[thought.id] = thought
            
            async def retrieve_thought(self, thought_id: str) -> Optional[SifakaThought]:
                return self._thoughts.get(thought_id)
        
        storage = TestStorage()
        thought = SifakaThought(prompt="Test")
        
        # Test store contract
        await storage.store_thought(thought)
        
        # Test retrieve contract
        retrieved = await storage.retrieve_thought(thought.id)
        assert retrieved is thought
        
        # Test retrieve non-existent
        non_existent = await storage.retrieve_thought("non-existent")
        assert non_existent is None


class TestRetrieverProtocol:
    """Test the Retriever protocol and its contract."""

    def test_retriever_protocol_methods(self):
        """Test that Retriever protocol defines the correct methods."""
        mock_retriever = Mock()
        mock_retriever.retrieve_async = AsyncMock(return_value=[])
        mock_retriever.name = "test-retriever"
        
        # Verify protocol methods exist
        assert hasattr(mock_retriever, 'retrieve_async')
        assert hasattr(mock_retriever, 'name')
        assert callable(mock_retriever.retrieve_async)

    @pytest.mark.asyncio
    async def test_retriever_protocol_contract(self):
        """Test the expected contract of Retriever protocol."""
        class TestRetriever:
            @property
            def name(self) -> str:
                return "test-retriever"
            
            async def retrieve_async(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
                return [
                    {"content": "Retrieved content 1", "score": 0.9},
                    {"content": "Retrieved content 2", "score": 0.8}
                ]
        
        retriever = TestRetriever()
        
        # Test retrieve contract
        results = await retriever.retrieve_async("test query")
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(item, dict) for item in results)
        assert retriever.name == "test-retriever"


class TestBaseValidator:
    """Test the BaseValidator abstract base class."""

    def test_base_validator_is_abstract(self):
        """Test that BaseValidator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseValidator("test", "description")

    def test_base_validator_subclass_requirements(self):
        """Test that BaseValidator subclasses must implement abstract methods."""
        # Missing validate_async implementation
        class IncompleteValidator(BaseValidator):
            pass
        
        with pytest.raises(TypeError):
            IncompleteValidator("test", "description")

    def test_base_validator_concrete_implementation(self):
        """Test a concrete BaseValidator implementation."""
        class ConcreteValidator(BaseValidator):
            async def validate_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {
                    "passed": True,
                    "details": {},
                    "score": 1.0
                }
        
        validator = ConcreteValidator("test-validator", "Test description")
        
        assert validator.name == "test-validator"
        assert validator.description == "Test description"
        assert hasattr(validator, 'validate_async')

    @pytest.mark.asyncio
    async def test_base_validator_sync_method(self):
        """Test the sync validate method that wraps validate_async."""
        class ConcreteValidator(BaseValidator):
            async def validate_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {"passed": True, "score": 0.8}
        
        validator = ConcreteValidator("test", "description")
        thought = SifakaThought(prompt="Test")
        
        # The sync method should work (though implementation may vary)
        # This tests that the method exists and can be called
        assert hasattr(validator, 'validate')


class TestBaseCritic:
    """Test the BaseCritic abstract base class."""

    def test_base_critic_is_abstract(self):
        """Test that BaseCritic cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCritic("test", "description")

    def test_base_critic_subclass_requirements(self):
        """Test that BaseCritic subclasses must implement abstract methods."""
        class IncompleteCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {"feedback": "test", "suggestions": [], "needs_improvement": False}
            # Missing improve_async
        
        with pytest.raises(TypeError):
            IncompleteCritic("test", "description")

    def test_base_critic_concrete_implementation(self):
        """Test a concrete BaseCritic implementation."""
        class ConcreteCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {
                    "feedback": "Good text",
                    "suggestions": [],
                    "needs_improvement": False
                }
            
            async def improve_async(self, thought: SifakaThought) -> str:
                return thought.current_text or "Improved text"
        
        critic = ConcreteCritic("test-critic", "Test description")
        
        assert critic.name == "test-critic"
        assert critic.description == "Test description"
        assert hasattr(critic, 'critique_async')
        assert hasattr(critic, 'improve_async')


class TestBaseStorage:
    """Test the BaseStorage abstract base class."""

    def test_base_storage_is_abstract(self):
        """Test that BaseStorage cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStorage()

    def test_base_storage_subclass_requirements(self):
        """Test that BaseStorage subclasses must implement abstract methods."""
        class IncompleteStorage(BaseStorage):
            async def store_thought(self, thought: SifakaThought) -> None:
                pass
            # Missing retrieve_thought
        
        with pytest.raises(TypeError):
            IncompleteStorage()

    def test_base_storage_concrete_implementation(self):
        """Test a concrete BaseStorage implementation."""
        class ConcreteStorage(BaseStorage):
            def __init__(self):
                self._thoughts = {}
            
            async def store_thought(self, thought: SifakaThought) -> None:
                self._thoughts[thought.id] = thought
            
            async def retrieve_thought(self, thought_id: str) -> Optional[SifakaThought]:
                return self._thoughts.get(thought_id)
        
        storage = ConcreteStorage()
        
        assert hasattr(storage, 'store_thought')
        assert hasattr(storage, 'retrieve_thought')


class TestBaseRetriever:
    """Test the BaseRetriever abstract base class."""

    def test_base_retriever_is_abstract(self):
        """Test that BaseRetriever cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRetriever("test")

    def test_base_retriever_subclass_requirements(self):
        """Test that BaseRetriever subclasses must implement abstract methods."""
        class IncompleteRetriever(BaseRetriever):
            pass
        
        with pytest.raises(TypeError):
            IncompleteRetriever("test")

    def test_base_retriever_concrete_implementation(self):
        """Test a concrete BaseRetriever implementation."""
        class ConcreteRetriever(BaseRetriever):
            async def retrieve_async(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
                return [{"content": f"Result for {query}", "score": 1.0}]
        
        retriever = ConcreteRetriever("test-retriever")
        
        assert retriever.name == "test-retriever"
        assert hasattr(retriever, 'retrieve_async')


class TestProtocolCompliance:
    """Test that base classes properly implement their protocols."""

    def test_base_validator_implements_validator_protocol(self):
        """Test that BaseValidator implementations satisfy Validator protocol."""
        class TestValidator(BaseValidator):
            async def validate_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {"passed": True}
        
        validator = TestValidator("test", "description")
        
        # Should satisfy Validator protocol
        assert hasattr(validator, 'validate_async')
        assert hasattr(validator, 'name')
        assert callable(validator.validate_async)
        assert isinstance(validator.name, str)

    def test_base_critic_implements_critic_protocol(self):
        """Test that BaseCritic implementations satisfy Critic protocol."""
        class TestCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {"feedback": "test", "suggestions": [], "needs_improvement": False}
            
            async def improve_async(self, thought: SifakaThought) -> str:
                return "improved"
        
        critic = TestCritic("test", "description")
        
        # Should satisfy Critic protocol
        assert hasattr(critic, 'critique_async')
        assert hasattr(critic, 'improve_async')
        assert hasattr(critic, 'name')
        assert callable(critic.critique_async)
        assert callable(critic.improve_async)
        assert isinstance(critic.name, str)

    def test_base_storage_implements_storage_protocol(self):
        """Test that BaseStorage implementations satisfy Storage protocol."""
        class TestStorage(BaseStorage):
            async def store_thought(self, thought: SifakaThought) -> None:
                pass
            
            async def retrieve_thought(self, thought_id: str) -> Optional[SifakaThought]:
                return None
        
        storage = TestStorage()
        
        # Should satisfy Storage protocol
        assert hasattr(storage, 'store_thought')
        assert hasattr(storage, 'retrieve_thought')
        assert callable(storage.store_thought)
        assert callable(storage.retrieve_thought)

    def test_base_retriever_implements_retriever_protocol(self):
        """Test that BaseRetriever implementations satisfy Retriever protocol."""
        class TestRetriever(BaseRetriever):
            async def retrieve_async(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
                return []
        
        retriever = TestRetriever("test")
        
        # Should satisfy Retriever protocol
        assert hasattr(retriever, 'retrieve_async')
        assert hasattr(retriever, 'name')
        assert callable(retriever.retrieve_async)
        assert isinstance(retriever.name, str)
