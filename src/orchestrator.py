"""Conversation orchestrator module for AI Banter.

Handles speaker selection, context building, AI invocation,
and state management during conversation generation.
"""

from typing import Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from src.config import Config
from src.types import ConversationState, DialogueTurn, Persona, Script, Segment


class ConversationOrchestrator:
    """Orchestrates multi-persona conversations with AI invocation and turn-taking logic."""

    def __init__(self, config: Config):
        """Initialize the orchestrator with configuration.

        Args:
            config: Configuration object with OpenAI credentials.

        """
        self.config = config
        self.client = OpenAI(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY
        )
        self._last_speaker_count = 0

    def select_next_speaker(
        self,
        state: ConversationState,
        personas: List[Persona]
    ) -> Persona:
        """
        Select next speaker using algorithm:
        - Prefer moderator after 2+ consecutive non-moderator turns
        - Never same speaker twice in a row
        - Balance speaking time (prefer speakers with fewer turns)
        - Follow flow guide suggestions when available

        Args:
            state: Current conversation state.
            personas: List of available personas.

        Returns:
            Selected Persona for the next turn.

        """
        if not personas:
            raise ValueError("No personas available for speaker selection")

        moderator = next((p for p in personas if p.role == "moderator"), None)
        experts = [p for p in personas if p.role == "expert"]

        should_moderate = self._last_speaker_count >= 2 and moderator is not None

        last_speaker_name = state.last_speaker
        eligible = [p for p in personas if p.name != last_speaker_name]

        if not eligible:
            eligible = personas

        if should_moderate and moderator in eligible:
            return moderator

        speaker_turn_counts = {p.name: state.speakers_used.get(p.name, 0) for p in eligible}
        min_turns = min(speaker_turn_counts.values())
        balanced_candidates = [
            name for name, count in speaker_turn_counts.items() if count == min_turns
        ]

        if last_speaker_name and any(
            p.name == last_speaker_name and p.role == "moderator" for p in personas
        ):
            expert_candidates = [
                name for name in balanced_candidates
                if any(p.name == name and p.role == "expert" for p in personas)
            ]
            if expert_candidates:
                candidates = expert_candidates
            else:
                candidates = balanced_candidates
        else:
            candidates = balanced_candidates

        import random
        selected_name = random.choice(candidates)

        return next(p for p in personas if p.name == selected_name)

    def build_context_prompt(
        self,
        persona: Persona,
        script: Script,
        current_segment: Segment,
        state: ConversationState,
        language: str
    ) -> list[dict]:
        """
        Build prompt context:
        - System prompt: persona system_prompt
        - User prompt:
          * Current topic and segment
          * Conversation history (last 5-10 turns)
          * Instructions for this turn
          * Language instruction

        Args:
            persona: The persona who will speak.
            script: Current script with accumulated turns.
            current_segment: The current segment being discussed.
            state: Current conversation state.
            language: Output language code.

        Returns:
            List of message dicts in OpenAI chat format.

        """
        max_history_turns = 8
        recent_turns = script.turns[-max_history_turns:] if script.turns else []

        history_text = ""
        for turn in recent_turns:
            role_label = "MODERATOR" if turn.role == "moderator" else "EXPERT"
            history_text += f"[{role_label} - {turn.speaker_name}]: {turn.content}\n"

        segment_context = f"""CURRENT SEGMENT: {current_segment.topic}

Key Discussion Points:
"""
        for idx, point in enumerate(current_segment.key_points, 1):
            segment_context += f"{idx}. {point}\n"

        if current_segment.suggested_speakers:
            segment_context += f"\nSuggested Speakers: {', '.join(current_segment.suggested_speakers)}\n"

        if persona.role == "moderator":
            turn_instructions = """YOUR ROLE as Moderator:
- Guide the flow of conversation smoothly
- Make natural transitions between topics and speakers
- Ensure all participants get opportunities to contribute
- Ask probing questions to deepen the discussion
- Connect ideas between different speakers
- Keep the discussion engaging and on-track
- Summarize key points when transitioning segments
"""
        else:
            turn_instructions = """YOUR ROLE as Expert:
- Share insights and expertise from your specific perspective
- Engage thoughtfully with other speakers' ideas
- Build on or respectfully challenge others' points
- Keep contributions substantive (~150-300 words)
- Ask questions of other participants when appropriate
- Use natural conversational language
"""

        user_content = f"""You are participating in a podcast-style discussion.

TOPIC: {script.topic}
{segment_context}

CONVERSATION HISTORY:
{"No prior conversation yet." if not history_text else history_text}

{turn_instructions}

CURRENT CONTEXT:
- This is turn #{len(script.turns) + 1}
- You are: {persona.name} ({persona.expertise})
- Your personality traits: {', '.join(persona.personality_traits)}
- Your speaking style: {persona.speaking_style}

LANGUAGE: All content must be in {language}.

Provide your next contribution to the conversation. Keep it engaging, natural, and relevant to the current segment.
"""

        messages = [
            {
                "role": "system",
                "content": persona.system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        return messages

    def invoke_speaker(
        self,
        persona: Persona,
        messages: list[dict],
        turn_number: int
    ) -> DialogueTurn:
        """
        Call OpenAI API with messages.
        - Use persona's system prompt
        - Parse response
        - Create DialogueTurn
        - Handle errors

        Args:
            persona: The persona to invoke.
            messages: List of messages in OpenAI chat format.
            turn_number: The sequence number for this turn.

        Returns:
            DialogueTurn object with the generated response.

        Raises:
            RuntimeError: If API call fails or response is empty.

        """
        try:
            completion = self.client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.7,
                max_tokens=500
            )

            response_message: ChatCompletionMessage = completion.choices[0].message
            content = response_message.content

            if not content:
                raise RuntimeError("AI returned empty response")

            word_count = len(content.split())

            return DialogueTurn(
                speaker_name=persona.name,
                role=persona.role,
                content=content,
                turn_number=turn_number,
                word_count=word_count
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to invoke speaker {persona.name}: {str(e)}"
            )

    def update_state(
        self,
        state: ConversationState,
        persona: Persona,
        turn: DialogueTurn
    ) -> ConversationState:
        """
        Update conversation state after turn.

        Args:
            state: Current conversation state.
            persona: The persona who just spoke.
            turn: The completed dialogue turn.

        Returns:
            Updated ConversationState object.

        """
        state.speakers_used[persona.name] = state.speakers_used.get(persona.name, 0) + 1
        state.last_speaker = persona.name

        if persona.role == "moderator":
            self._last_speaker_count = 0
        else:
            self._last_speaker_count += 1

        state.current_turn_count += 1

        return state

    def assemble_script(self, turns: List[DialogueTurn]) -> Script:
        """
        Create final script from turns.

        Args:
            turns: List of completed dialogue turns.

        Returns:
            Complete Script object.

        """
        if not turns:
            raise ValueError("Cannot create script with no turns")

        topic = "AI Discussion"
        total_word_count = sum(turn.word_count for turn in turns)

        return Script(
            turns=turns,
            topic=topic,
            language="en",
            total_word_count=total_word_count
        )

    def estimate_completion(
        self,
        script: Script,
        target_words: int = 9000
    ) -> float:
        """
        Return completion percentage (0.0-1.0).

        Args:
            script: Current script with accumulated turns.
            target_words: Target word count (default: 9000 for ~1 hour).

        Returns:
            Completion percentage as float between 0.0 and 1.0.

        """
        if target_words <= 0:
            return 0.0

        completion = script.total_word_count / target_words
        return min(completion, 1.0)
