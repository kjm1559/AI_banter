import pytest

from src.types import (
    Persona,
    DialogueTurn,
    Segment,
    FlowGuide,
    Script,
    ConversationState
)


class TestPersona:
    
    def test_persona_creation(self):
        """Test creating a valid Persona."""
        persona = Persona(
            name="Dr. Smith",
            role="expert",
            expertise="AI Ethics",
            personality_traits=["thoughtful", "skeptical", "analytical"],
            speaking_style="measured and precise",
            system_prompt="You are an AI ethics expert."
        )
        
        assert persona.name == "Dr. Smith"
        assert persona.role == "expert"
        assert len(persona.personality_traits) == 3
    
    def test_persona_str(self):
        """Test Persona string representation."""
        persona = Persona(
            name="Dr. Smith",
            role="expert",
            expertise="AI Ethics",
            personality_traits=["thoughtful"],
            speaking_style="precise",
            system_prompt="You are an expert."
        )
        
        persona_str = str(persona)
        assert "Dr. Smith" in persona_str
        assert "expert" in persona_str
        assert "AI Ethics" in persona_str
    
    def test_persona_missing_required_fields(self):
        """Test that missing required fields raises error."""
        with pytest.raises(Exception):
            Persona(  # type: ignore
                role="expert"
            )


class TestDialogueTurn:
    
    def test_dialogue_turn_creation(self):
        """Test creating a valid DialogueTurn."""
        turn = DialogueTurn(
            speaker_name="Dr. Smith",
            role="expert",
            content="This is a test response.",
            turn_number=1,
            word_count=5
        )
        
        assert turn.speaker_name == "Dr. Smith"
        assert turn.turn_number == 1
        assert turn.word_count == 5
    
    def test_dialogue_turn_str(self):
        """Test DialogueTurn string representation."""
        turn = DialogueTurn(
            speaker_name="Dr. Smith",
            role="expert",
            content="Test response",
            turn_number=1,
            word_count=2
        )
        
        turn_str = str(turn)
        assert "Dr. Smith" in turn_str
        assert "1" in turn_str


class TestSegment:
    
    def test_segment_creation(self):
        """Test creating a valid Segment."""
        segment = Segment(
            topic="Introduction to AI",
            key_points=["What is AI", "History", "Current state"],
            duration_mins=10,
            suggested_speakers=["Dr. Smith", "Dr. Jones"]
        )
        
        assert segment.topic == "Introduction to AI"
        assert len(segment.key_points) == 3
        assert segment.duration_mins == 10
    
    def test_segment_str(self):
        """Test Segment string representation."""
        segment = Segment(
            topic="AI Discussion",
            key_points=["Point 1"],
            duration_mins=5,
            suggested_speakers=["Speaker 1"]
        )
        
        segment_str = str(segment)
        assert "AI Discussion" in segment_str


class TestFlowGuide:
    
    def test_flow_guide_creation(self):
        """Test creating a valid FlowGuide."""
        guide = FlowGuide(
            segments=[
                Segment(
                    topic="Intro",
                    key_points=["Welcome"],
                    duration_mins=5,
                    suggested_speakers=["Moderator"]
                )
            ],
            total_duration_mins=60
        )
        
        assert len(guide.segments) == 1
        assert guide.total_duration_mins == 60
    
    def test_flow_guide_str(self):
        """Test FlowGuide string representation."""
        guide = FlowGuide(
            segments=[
                Segment(
                    topic="Intro",
                    key_points=["Welcome"],
                    duration_mins=5,
                    suggested_speakers=["Moderator"]
                )
            ],
            total_duration_mins=60
        )
        
        guide_str = str(guide)
        assert "60" in guide_str


class TestScript:
    
    def test_script_creation(self):
        """Test creating a valid Script."""
        script = Script(
            turns=[
                DialogueTurn(
                    speaker_name="Dr. Smith",
                    role="expert",
                    content="Hello",
                    turn_number=1,
                    word_count=1
                )
            ],
            topic="AI Discussion",
            language="english",
            total_word_count=1
        )
        
        assert len(script.turns) == 1
        assert script.topic == "AI Discussion"
        assert script.language == "english"
    
    def test_script_str(self):
        """Test Script string representation."""
        script = Script(
            turns=[
                DialogueTurn(
                    speaker_name="Dr. Smith",
                    role="expert",
                    content="Hello",
                    turn_number=1,
                    word_count=1
                )
            ],
            topic="AI Discussion",
            language="english",
            total_word_count=1
        )
        
        script_str = str(script)
        assert "AI Discussion" in script_str


class TestConversationState:
    
    def test_conversation_state_creation(self):
        """Test creating a valid ConversationState."""
        state = ConversationState(
            current_segment_idx=0,
            current_turn_count=0,
            speakers_used={},
            last_speaker=None
        )
        
        assert state.current_segment_idx == 0
        assert state.current_turn_count == 0
        assert state.speakers_used == {}
    
    def test_conversation_state_default(self):
        """Test ConversationState with defaults."""
        state = ConversationState()  # type: ignore
        
        assert state.current_segment_idx == 0
        assert state.current_turn_count == 0


class TestEdgeCases:
    
    def test_persona_moderator_role(self):
        """Test creating moderator persona."""
        persona = Persona(
            name="Jane Doe",
            role="moderator",
            expertise="Host",
            personality_traits=["engaging", "attentive"],
            speaking_style="warm and welcoming",
            system_prompt="You are the host."
        )
        
        assert persona.role == "moderator"
    
    def test_script_empty_turns_raises_error(self):
        """Test Script validation rejects empty turns."""
        with pytest.raises(Exception, match="too_short"):
            Script(
                turns=[],
                topic="Empty",
                language="english",
                total_word_count=0
            )
