"""Data types for AI Banter."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Persona(BaseModel):
    """Represents a persona for dialogue participants."""

    name: str = Field(..., description="Name of the persona")
    role: str = Field(..., pattern="^(moderator|expert)$", description="Role: moderator or expert")
    expertise: str = Field(..., description="Area of expertise")
    personality_traits: List[str] = Field(default_factory=list, description="List of personality traits")
    speaking_style: str = Field(..., description="Speaking style description")
    system_prompt: str = Field(..., description="System prompt for this persona")

    def __str__(self) -> str:
        return f"Persona(name={self.name}, role={self.role}, expertise={self.expertise})"


class DialogueTurn(BaseModel):
    """Represents a single turn in a dialogue."""

    speaker_name: str = Field(..., description="Name of the speaker")
    role: str = Field(..., description="Role of the speaker (moderator/expert)")
    content: str = Field(..., description="The spoken content")
    turn_number: int = Field(..., ge=1, description="Sequence number of this turn")
    word_count: int = Field(default=0, ge=0, description="Number of words in content")

    @field_validator("word_count", mode="before")
    @classmethod
    def calculate_word_count(cls, v: Any, info: Any) -> int:
        if v is None or v == "" or v == 0:
            return 0
        return int(v)

    def __str__(self) -> str:
        return f"DialogueTurn(turn={self.turn_number}, speaker={self.speaker_name}, words={self.word_count})"


class Segment(BaseModel):
    """Represents a segment of conversation."""

    topic: str = Field(..., description="Topic of this segment")
    key_points: List[str] = Field(default_factory=list, description="Key discussion points")
    duration_mins: int = Field(..., gt=0, description="Planned duration in minutes")
    suggested_speakers: List[str] = Field(default_factory=list, description="Suggested speakers for this segment")

    def __str__(self) -> str:
        return f"Segment(topic={self.topic}, duration={self.duration_mins}min, points={len(self.key_points)})"


class FlowGuide(BaseModel):
    """Guide for conversation flow and structure."""

    segments: List[Segment] = Field(..., min_length=1, description="List of conversation segments")
    total_duration_mins: int = Field(default=0, ge=0, description="Total planned duration")

    def __str__(self) -> str:
        return f"FlowGuide(segments={len(self.segments)}, total_duration={self.total_duration_mins}min)"


class Script(BaseModel):
    """Complete dialogue script."""

    turns: List[DialogueTurn] = Field(..., min_length=1, description="List of dialogue turns")
    topic: str = Field(..., description="Main topic of the dialogue")
    language: str = Field(default="en", description="Language code")
    total_word_count: int = Field(default=0, ge=0, description="Total word count across all turns")

    def __str__(self) -> str:
        return f"Script(topic={self.topic}, turns={len(self.turns)}, words={self.total_word_count})"


class ConversationState(BaseModel):
    """Tracks the current state of an ongoing conversation."""

    current_segment_idx: int = Field(default=0, ge=0, description="Index of current segment")
    current_turn_count: int = Field(default=0, ge=0, description="Number of turns in current segment")
    speakers_used: Dict[str, int] = Field(default_factory=dict, description="Turn counts per speaker")
    last_speaker: Optional[str] = Field(default=None, description="Name of the last speaker")

    def __str__(self) -> str:
        return (
            f"ConversationState(segment={self.current_segment_idx}, "
            f"turns={self.current_turn_count}, last={self.last_speaker})"
        )
