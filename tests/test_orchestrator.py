import pytest
from unittest.mock import MagicMock, patch

from src.config import Config
from src.types import ConversationState, Persona, Script, DialogueTurn, Segment
from src.orchestrator import ConversationOrchestrator


class TestConversationOrchestrator:
    
    @pytest.fixture
    def mock_config(self):
        config = Config(
            OPENAI_BASE_URL="https://api.openai.com/v1",
            OPENAI_API_KEY="test-key",
            OPENAI_MODEL="gpt-4o"
        )
        return config
    
    @pytest.fixture
    def orchestrator(self, mock_config):
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content="This is a test response."))
        ]
        
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value=mock_completion)
        
        with patch("openai.OpenAI", return_value=mock_client):
            orc = ConversationOrchestrator(mock_config)
            orc.client = mock_client
            return orc
    
    @pytest.fixture
    def sample_personas(self):
        return [
            Persona(
                name="Host Jane",
                role="moderator",
                expertise="Podcast Hosting",
                personality_traits=["engaging", "warm"],
                speaking_style="friendly",
                system_prompt="You are the host."
            ),
            Persona(
                name="Dr. Smith",
                role="expert",
                expertise="AI Research",
                personality_traits=["analytical"],
                speaking_style="precise",
                system_prompt="You are an AI expert."
            ),
            Persona(
                name="Dr. Jones",
                role="expert",
                expertise="Ethics",
                personality_traits=["reflective"],
                speaking_style="thoughtful",
                system_prompt="You are an ethics expert."
            )
        ]
    
    @pytest.fixture
    def sample_state(self):
        return ConversationState(
            current_segment_idx=0,
            current_turn_count=0,
            speakers_used={},
            last_speaker=None
        )
    
    def test_select_next_speaker_no_consecutive_same(self, orchestrator, sample_personas, sample_state):
        """Test that same speaker is never selected twice in a row."""
        sample_state.last_speaker = "Dr. Smith"
        
        next_speaker = orchestrator.select_next_speaker(sample_state, sample_personas)
        
        assert next_speaker.name != "Dr. Smith"
    
    def test_select_next_speaker_prefers_moderator_after_experts(self, orchestrator, sample_personas, sample_state):
        """Test that moderator is preferred after 2+ non-moderator turns."""
        sample_state.last_speaker = "Dr. Jones"
        sample_state.speakers_used = {"Dr. Smith": 1, "Dr. Jones": 2}
        orchestrator._last_speaker_count = 2
        
        next_speaker = orchestrator.select_next_speaker(sample_state, sample_personas)
        
        assert next_speaker.role == "moderator"
    
    def test_select_next_speaker_balances_speaking_time(self, orchestrator, sample_personas, sample_state):
        """Test that speaker with fewer turns is preferred."""
        sample_state.last_speaker = "Host Jane"
        sample_state.speakers_used = {"Host Jane": 5, "Dr. Smith": 2, "Dr. Jones": 3}
        
        next_speaker = orchestrator.select_next_speaker(sample_state, sample_personas)
        
        assert next_speaker.name == "Dr. Smith"
    
    def test_select_next_speaker_no_personas_raises_error(self, orchestrator, sample_state):
        """Test that empty personas raises ValueError."""
        with pytest.raises(ValueError, match="No personas available"):
            orchestrator.select_next_speaker(sample_state, [])
    
    def test_build_context_prompt_includes_persona_info(self, orchestrator, sample_personas):
        """Test that context prompt includes persona details."""
        script = Script(
            turns=[DialogueTurn(
                speaker_name="Host Jane",
                role="moderator",
                content="Welcome to the show",
                turn_number=1,
                word_count=4
            )],
            topic="AI Discussion",
            language="english",
            total_word_count=4
        )
        
        segment = Segment(
            topic="Introduction",
            key_points=["Welcome"],
            duration_mins=5,
            suggested_speakers=["Host Jane"]
        )
        
        state = ConversationState()
        
        messages = orchestrator.build_context_prompt(
            persona=sample_personas[0],
            script=script,
            current_segment=segment,
            state=state,
            language="english"
        )
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "AI Discussion" in messages[1]["content"]
    
    def test_invoke_speaker_creates_dialogue_turn(self, orchestrator, sample_personas):
        """Test that invoke_speaker creates valid DialogueTurn."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="This is a test response."))]
        orchestrator.client.chat.completions.create = MagicMock(return_value=mock_response)
        
        turn = orchestrator.invoke_speaker(
            persona=sample_personas[0],
            messages=[
                {"role": "system", "content": "You are host"},
                {"role": "user", "content": "Start conversation"}
            ],
            turn_number=1
        )
        
        assert isinstance(turn, DialogueTurn)
        assert turn.speaker_name == "Host Jane"
        assert turn.turn_number == 1
        assert len(turn.content) > 0
    
    def test_update_state_increments_counter(self, orchestrator, sample_personas):
        """Test that state update increments turn count."""
        state = ConversationState()
        turn = DialogueTurn(
            speaker_name="Dr. Smith",
            role="expert",
            content="Test response",
            turn_number=1,
            word_count=5
        )
        
        original_turn_count = state.current_turn_count
        new_state = orchestrator.update_state(state, sample_personas[1], turn)
        
        assert state.current_turn_count == original_turn_count + 1
        assert new_state.last_speaker == "Dr. Smith"
        assert sample_personas[1].name in new_state.speakers_used
    
    def test_assemble_script_from_turns(self, orchestrator):
        """Test that script is assembled correctly."""
        turns = [
            DialogueTurn(
                speaker_name="Host Jane",
                role="moderator",
                content="Welcome",
                turn_number=1,
                word_count=1
            ),
            DialogueTurn(
                speaker_name="Dr. Smith",
                role="expert",
                content="Hello there",
                turn_number=2,
                word_count=2
            )
        ]
        
        script = orchestrator.assemble_script(turns)
        
        assert len(script.turns) == 2
        assert script.total_word_count == 3
    
    def test_assemble_script_no_turns_raises_error(self, orchestrator):
        """Test that empty turns raises ValueError."""
        with pytest.raises(ValueError, match="no turns"):
            orchestrator.assemble_script([])
    
    def test_estimate_completion_returns_percentage(self, orchestrator):
        """Test that completion estimate returns 0-100."""
        script = Script(
            turns=[
                DialogueTurn(
                    speaker_name="Host",
                    role="moderator",
                    content="A",
                    turn_number=1,
                    word_count=1000
                )
            ],
            topic="Test",
            language="en",
            total_word_count=1000
        )
        
        completion = orchestrator.estimate_completion(script, target_words=10000)
        
        assert 0 <= completion <= 1.0
    
    def test_estimate_completion_over_target_caps_at_one(self, orchestrator):
        """Test that completion caps at 1.0 when over target."""
        script = Script(
            turns=[
                DialogueTurn(
                    speaker_name="Host",
                    role="moderator",
                    content="A",
                    turn_number=1,
                    word_count=15000
                )
            ],
            topic="Test",
            language="en",
            total_word_count=15000
        )
        
        completion = orchestrator.estimate_completion(script, target_words=10000)
        
        assert completion == 1.0
    
    def test_invoke_speaker_handles_empty_response(self, orchestrator, sample_personas):
        """Test that empty AI response raises error."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=""))]
        orchestrator.client.chat.completions.create = MagicMock(return_value=mock_response)
        
        with pytest.raises(RuntimeError, match="empty response"):
            orchestrator.invoke_speaker(
                persona=sample_personas[0],
                messages=[
                    {"role": "system", "content": "You are host"},
                    {"role": "user", "content": "Start"}
                ],
                turn_number=1
            )
