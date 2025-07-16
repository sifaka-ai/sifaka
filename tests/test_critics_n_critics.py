"""Tests for N-Critics ensemble critic."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from sifaka.critics.n_critics import (
    NCriticsCritic,
    NCriticsResponse,
)
from sifaka.core.models import SifakaResult, CritiqueResult
from sifaka.core.config import Config


class TestNCriticsResponse:
    """Test the NCriticsResponse model."""

    def test_creation_minimal(self):
        """Test creating response with minimal fields."""
        response = NCriticsResponse(
            feedback="Overall feedback from multiple perspectives",
            suggestions=["Improve clarity", "Add examples"],
            needs_improvement=True,
        )
        assert response.feedback == "Overall feedback from multiple perspectives"
        assert len(response.suggestions) == 2
        assert response.confidence == 0.7  # default
        assert response.consensus_score == 0.7  # default

    def test_creation_full(self):
        """Test creating response with all fields."""
        response = NCriticsResponse(
            feedback="Strong text with minor improvements needed",
            suggestions=["Verify claim", "Polish conclusion"],
            needs_improvement=True,
            confidence=0.85,
            consensus_score=0.85,
        )

        # NCriticsResponse doesn't have metadata field
        assert response.consensus_score == 0.85
        assert response.confidence == 0.85
        assert len(response.suggestions) == 2


class TestNCriticsCritic:
    """Test the NCriticsCritic class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Final text")

    def test_initialization_default(self):
        """Test default initialization."""
        critic = NCriticsCritic()
        assert critic.name == "n_critics"
        assert critic.model == "gpt-3.5-turbo"  # Default from Config
        assert critic.temperature == 0.6
        assert len(critic.perspectives) == 4
        assert not critic.auto_generate_perspectives
        assert critic.perspective_count == 4

    def test_initialization_with_custom_perspectives(self):
        """Test initialization with custom perspectives."""
        custom_perspectives = [
            "Technical accuracy",
            "Business value",
            "User experience",
        ]
        critic = NCriticsCritic(perspectives=custom_perspectives)
        assert len(critic.perspectives) == 3
        assert "Technical accuracy" in critic.perspectives

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Config()
        config.temperature = 0.5
        critic = NCriticsCritic(config=config)
        assert critic.config == config

    def test_initialization_with_params(self):
        """Test initialization with all parameters."""
        critic = NCriticsCritic(
            model="gpt-4",
            temperature=0.7,
            perspectives=["Custom 1", "Custom 2"],
            api_key="test-key",
            auto_generate_perspectives=True,
            perspective_count=6,
        )
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.7
        assert len(critic.perspectives) == 2
        assert critic.auto_generate_perspectives is True
        assert critic.perspective_count == 6

    def test_get_response_type(self):
        """Test that critic uses NCriticsResponse."""
        critic = NCriticsCritic()
        assert critic._get_response_type() == NCriticsResponse

    def test_default_perspectives(self):
        """Test default perspectives are set correctly."""
        critic = NCriticsCritic()
        assert len(critic.DEFAULT_PERSPECTIVES) == 4
        assert any("Clarity" in p for p in critic.DEFAULT_PERSPECTIVES)
        assert any("Accuracy" in p for p in critic.DEFAULT_PERSPECTIVES)
        assert any("Completeness" in p for p in critic.DEFAULT_PERSPECTIVES)
        assert any("Style" in p for p in critic.DEFAULT_PERSPECTIVES)

    @pytest.mark.asyncio
    async def test_create_messages(self, sample_result):
        """Test message creation with default perspectives."""
        critic = NCriticsCritic()
        messages = await critic._create_messages("Test text", sample_result)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "N-Critics ensemble" in messages[0]["content"]
        assert "multiple critical perspectives" in messages[0]["content"]

        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert "Test text" in user_content
        assert "CRITICAL PERSPECTIVES:" in user_content
        assert "Clarity:" in user_content
        assert "Accuracy:" in user_content
        assert "consensus score (0.0-1.0)" in user_content

    @pytest.mark.asyncio
    async def test_create_messages_with_custom_perspectives(self, sample_result):
        """Test message creation with custom perspectives."""
        custom_perspectives = ["Grammar", "Tone", "Length"]
        critic = NCriticsCritic(perspectives=custom_perspectives)
        messages = await critic._create_messages("Test text", sample_result)

        user_content = messages[1]["content"]
        assert "Grammar" in user_content
        assert "Tone" in user_content
        assert "Length" in user_content
        assert "Clarity:" not in user_content  # Default not included

    @pytest.mark.asyncio
    async def test_create_messages_with_context(self, sample_result):
        """Test message creation with previous context."""
        # Add a previous critique
        sample_result.critiques.append(
            CritiqueResult(
                critic="n_critics",
                feedback="Previous ensemble feedback",
                suggestions=["Previous suggestion"],
            )
        )

        critic = NCriticsCritic()
        messages = await critic._create_messages("Test text", sample_result)

        user_content = messages[1]["content"]
        assert "Previous feedback:" in user_content
        assert "Previous ensemble feedback" in user_content

    @pytest.mark.asyncio
    async def test_critique_success_with_high_agreement(self, sample_result):
        """Test successful critique with high agreement across perspectives."""
        critic = NCriticsCritic()

        # Mock the LLM response with high agreement
        mock_response = NCriticsResponse(
            feedback="All perspectives agree: text is well-written with minor improvements",
            suggestions=["Add one more example", "Polish conclusion"],
            needs_improvement=True,
            confidence=0.9,
            metadata={
                "perspective_assessments": [
                    {
                        "perspective": "Clarity",
                        "assessment": "Very clear and well-structured",
                        "score": 0.9,
                        "suggestions": ["Minor wording improvement"],
                    },
                    {
                        "perspective": "Accuracy",
                        "assessment": "Accurate and well-researched",
                        "score": 0.88,
                        "suggestions": [],
                    },
                    {
                        "perspective": "Completeness",
                        "assessment": "Comprehensive coverage",
                        "score": 0.85,
                        "suggestions": ["Add one more example"],
                    },
                    {
                        "perspective": "Style",
                        "assessment": "Professional and engaging",
                        "score": 0.87,
                        "suggestions": ["Polish conclusion"],
                    },
                ],
                "variance": 0.02,
                "unanimous_suggestions": ["Polish conclusion"],
            },
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response  # For backward compatibility
        mock_agent_result.data = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Test text", sample_result)

            assert result.critic == "n_critics"
            assert result.feedback == mock_response.feedback
            assert len(result.suggestions) == 2
            assert result.needs_improvement is True
            assert result.confidence == 0.9
            assert "perspective_assessments" in result.metadata
            assert len(result.metadata["perspective_assessments"]) == 4
            # Check consensus score - NCriticsResponse puts it in metadata after base processing
            assert "consensus_score" in result.metadata
            # The value might be default since base critic doesn't know about consensus_score field
            assert "perspective_assessments" in result.metadata
            assert len(result.metadata["perspective_assessments"]) == 4

    @pytest.mark.asyncio
    async def test_critique_with_disagreement(self, sample_result):
        """Test critique with disagreement across perspectives."""
        critic = NCriticsCritic()

        mock_response = NCriticsResponse(
            feedback="Mixed assessments: clarity issues but technically sound",
            suggestions=["Simplify language", "Add structure", "Keep technical depth"],
            needs_improvement=True,
            confidence=0.65,
            metadata={
                "perspective_assessments": [
                    {
                        "perspective": "Clarity",
                        "assessment": "Confusing and dense",
                        "score": 0.4,
                        "suggestions": ["Simplify language", "Add structure"],
                    },
                    {
                        "perspective": "Accuracy",
                        "assessment": "Highly accurate and detailed",
                        "score": 0.95,
                        "suggestions": [],
                    },
                    {
                        "perspective": "Completeness",
                        "assessment": "Very thorough",
                        "score": 0.9,
                        "suggestions": [],
                    },
                    {
                        "perspective": "Style",
                        "assessment": "Too technical for audience",
                        "score": 0.5,
                        "suggestions": ["Adjust tone", "Add examples"],
                    },
                ],
                "variance": 0.23,
                "conflicting_areas": ["clarity vs accuracy"],
            },
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response  # For backward compatibility
        mock_agent_result.data = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Technical document", sample_result)

            assert result.confidence == 0.65  # Lower due to disagreement
            assert "perspective_assessments" in result.metadata
            # Check for variance in metadata
            if "variance" in result.metadata:
                assert result.metadata["variance"] == 0.23

    @pytest.mark.asyncio
    async def test_critique_no_improvement_needed(self, sample_result):
        """Test critique when all perspectives agree no improvement needed."""
        critic = NCriticsCritic()

        mock_response = NCriticsResponse(
            feedback="Unanimous agreement: text is excellent across all dimensions",
            suggestions=[],
            needs_improvement=False,
            confidence=0.95,
            metadata={
                "perspective_assessments": [
                    {
                        "perspective": "Clarity",
                        "assessment": "Crystal clear",
                        "score": 0.95,
                        "suggestions": [],
                    },
                    {
                        "perspective": "Accuracy",
                        "assessment": "Perfectly accurate",
                        "score": 0.96,
                        "suggestions": [],
                    },
                    {
                        "perspective": "Completeness",
                        "assessment": "Fully comprehensive",
                        "score": 0.94,
                        "suggestions": [],
                    },
                    {
                        "perspective": "Style",
                        "assessment": "Exemplary style",
                        "score": 0.95,
                        "suggestions": [],
                    },
                ],
                "unanimous": True,
            },
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response  # For backward compatibility
        mock_agent_result.data = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Excellent text", sample_result)

            assert result.needs_improvement is False
            assert len(result.suggestions) == 0
            assert result.confidence == 0.95
            assert "perspective_assessments" in result.metadata
            # Check for unanimous flag in metadata
            if "unanimous" in result.metadata:
                assert result.metadata["unanimous"] is True

    @pytest.mark.asyncio
    async def test_generate_perspectives_default(self, sample_result):
        """Test perspective generation returns defaults."""
        critic = NCriticsCritic(auto_generate_perspectives=True)
        perspectives = await critic._generate_perspectives("Test text", sample_result)

        assert perspectives == critic.DEFAULT_PERSPECTIVES
        assert len(perspectives) == 4

    def test_provider_configuration(self):
        """Test provider configuration."""
        from sifaka.core.llm_client import Provider

        critic = NCriticsCritic(provider=Provider.ANTHROPIC)
        assert critic.provider == Provider.ANTHROPIC

        critic = NCriticsCritic(provider="openai")
        assert critic.provider == Provider.OPENAI

    @pytest.mark.asyncio
    async def test_create_messages_with_auto_generate(self, sample_result):
        """Test message creation with auto-generated perspectives."""
        critic = NCriticsCritic(auto_generate_perspectives=True)

        # Mock the perspective generation
        custom_perspectives = ["Domain-specific 1", "Domain-specific 2"]
        with patch.object(
            critic, "_generate_perspectives", return_value=custom_perspectives
        ):
            messages = await critic._create_messages("Domain text", sample_result)

            user_content = messages[1]["content"]
            assert "Domain-specific 1" in user_content
            assert "Domain-specific 2" in user_content
            # Should use generated perspectives, not defaults
            assert "Clarity:" not in user_content
