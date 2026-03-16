import json
import pytest
from unittest.mock import MagicMock, patch

from src.config import Config
from src.types import Persona
from src.persona_manager import PersonaManager


def create_mock_moderator_response():
    """Create mock moderator persona JSON response."""
    return json.dumps({
        "name": "Sarah Host",
        "role": "moderator",
        "expertise": "Podcast hosting and facilitation",
        "personality_traits": ["engaging", "warm", "professional", "attentive"],
        "speaking_style": "Clear, warm, and inviting with smooth transitions",
        "system_prompt": "You are an excellent podcast moderator."
    })


def create_mock_experts_response():
    """Create mock expert personas JSON array response."""
    return json.dumps([
        {
            "name": "Dr. AI Smith",
            "role": "expert",
            "expertise": "AI Research and Development",
            "personality_traits": ["analytical", "enthusiastic", "detail-oriented"],
            "speaking_style": "Precise and technical but accessible",
            "system_prompt": "You are an AI research expert."
        },
        {
            "name": "Dr. Ethical Jones",
            "role": "expert",
            "expertise": "AI Ethics and Policy",
            "personality_traits": ["thoughtful", "skeptical", "principled"],
            "speaking_style": "Measured and questioning",
            "system_prompt": "You are an ethics expert."
        },
        {
            "name": "Dr. Future Chen",
            "role": "expert",
            "expertise": "Future Tech Applications",
            "personality_traits": ["visionary", "optimistic", "pragmatic"],
            "speaking_style": "Forward-thinking and inspiring",
            "system_prompt": "You are a futurist expert."
        }
    ])


class TestPersonaManager:
    
    @pytest.fixture
    def mock_config(self):
        return Config(
            OPENAI_BASE_URL="https://api.openai.com/v1",
            OPENAI_API_KEY="test-key",
            OPENAI_MODEL="gpt-4o"
        )
    
    @pytest.fixture
    def persona_manager(self, mock_config):
        mock_moderator_response = MagicMock()
        mock_moderator_response.choices = [
            MagicMock(message=MagicMock(content=create_mock_moderator_response()))
        ]
        
        mock_experts_response = MagicMock()
        mock_experts_response.choices = [
            MagicMock(message=MagicMock(content=create_mock_experts_response()))
        ]
        
        mock_client = MagicMock()
        call_count = [0]
        
        def mock_create(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_moderator_response
            return mock_experts_response
        
        mock_client.chat.completions.create = mock_create
        
        with patch("openai.OpenAI", return_value=mock_client):
            pm = PersonaManager(mock_config)
            pm.client = mock_client
            return pm
    
    def test_generate_personas_count(self, persona_manager):
        personas = persona_manager.generate_personas(
            topic="Future of AI",
            num_participants=4,
            language="english"
        )
        
        assert len(personas) == 4
    
    def test_generate_personas_has_moderator(self, persona_manager):
        personas = persona_manager.generate_personas(
            topic="Future of AI",
            num_participants=4,
            language="english"
        )
        
        moderators = [p for p in personas if p.role == "moderator"]
        assert len(moderators) == 1
    
    def test_generate_personas_has_experts(self, persona_manager):
        personas = persona_manager.generate_personas(
            topic="Future of AI",
            num_participants=4,
            language="english"
        )
        
        experts = [p for p in personas if p.role == "expert"]
        assert len(experts) == 3
    
    def test_personas_have_unique_names(self, persona_manager):
        personas = persona_manager.generate_personas(
            topic="Future of AI",
            num_participants=4,
            language="english"
        )
        
        names = [p.name for p in personas]
        assert len(names) == len(set(names))
    
    def test_personas_have_expertise(self, persona_manager):
        personas = persona_manager.generate_personas(
            topic="Future of AI",
            num_participants=4,
            language="english"
        )
        
        for persona in personas:
            assert persona.expertise is not None
            assert len(persona.expertise) > 0
    
    def test_personas_have_system_prompt(self, persona_manager):
        personas = persona_manager.generate_personas(
            topic="Future of AI",
            num_participants=4,
            language="english"
        )
        
        for persona in personas:
            assert persona.system_prompt is not None
            assert len(persona.system_prompt) > 0
    
    def test_personas_have_personality_traits(self, persona_manager):
        personas = persona_manager.generate_personas(
            topic="Future of AI",
            num_participants=4,
            language="english"
        )
        
        for persona in personas:
            assert persona.personality_traits is not None
            assert len(persona.personality_traits) >= 3
    
    def test_personas_have_speaking_style(self, persona_manager):
        personas = persona_manager.generate_personas(
            topic="Future of AI",
            num_participants=4,
            language="english"
        )
        
        for persona in personas:
            assert persona.speaking_style is not None
            assert len(persona.speaking_style) > 0
    
    def test_create_system_prompt_includes_language(self, persona_manager):
        persona = Persona(
            name="Test Persona",
            role="moderator",
            expertise="Test expertise",
            personality_traits=["test"],
            speaking_style="test style",
            system_prompt="You are a test persona."
        )
        
        prompt = persona_manager.create_system_prompt(persona, "Test topic", "spanish")
        assert "spanish" in prompt.lower()
