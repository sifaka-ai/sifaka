"""Test PydanticAI integration."""

from sifaka.core.llm_client import LLMClient, Provider


def test_pydantic_ai_agent_creation():
    """Test that we can create a PydanticAI agent."""
    from sifaka.critics.core.base import CriticResponse

    client = LLMClient(Provider.OPENAI, "gpt-4o-mini")
    agent = client.create_agent(
        system_prompt="You are a helpful assistant.", result_type=CriticResponse
    )

    assert agent is not None
    assert hasattr(agent, "run")


def test_improvement_response():
    """Test ImprovementResponse model."""
    from sifaka.core.engine.generation import ImprovementResponse

    response = ImprovementResponse(
        improved_text="Better text",
        changes_made=["Fixed grammar", "Added clarity"],
        confidence=0.9,
    )

    assert response.improved_text == "Better text"
    assert len(response.changes_made) == 2
    assert response.confidence == 0.9


# Removed test_improve_with_pydantic_ai as it requires OpenAI API key
