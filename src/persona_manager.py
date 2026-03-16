"""Persona generation and management module."""

import json
from typing import List

from openai import OpenAI

from src.config import Config
from src.types import Persona


class PersonaManager:
    """Manages generation and retrieval of AI personas for dialogue participants."""

    def __init__(self, config: Config):
        """Initialize PersonaManager with configuration.

        Args:
            config: Configuration object with OpenAI credentials.

        """
        self.config = config
        self.client = OpenAI(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY
        )

    def generate_personas(
        self,
        topic: str,
        num_participants: int,
        language: str
    ) -> List[Persona]:
        """
        Generate personas based on topic.
        - 1 moderator (flow control, engaging, connecting)
        - num_participants-1 experts (diverse expertise areas related to topic)
        - Each with unique name, role, expertise, personality, speaking style
        - Include language instruction in system_prompt

        Args:
            topic: The discussion topic.
            num_participants: Total number of participants including moderator.
            language: Output language for personas.

        Returns:
            List of generated Persona objects.

        """
        personas = []
        moderator = self.get_moderator_persona(topic, language)
        personas.append(moderator)
        num_experts = num_participants - 1
        experts = self.generate_expert_personas(topic, num_experts, language)
        personas.extend(experts)
        return personas

    def get_moderator_persona(self, topic: str, language: str) -> Persona:
        """
        Generate moderator with flow control and engagement focus.

        Args:
            topic: The discussion topic.
            language: Output language.

        Returns:
            Persona object for the moderator.

        """
        system_content = """You are a persona generator. Create a unique moderator persona for a podcast-style discussion. The moderator should be engaging, excellent at flow control, making smooth transitions, and keeping everyone involved. Return JSON only, no markdown formatting.

Required JSON fields:
- name: creative and memorable name
- role: "moderator"
- expertise: flow control, engagement, connecting ideas
- personality_traits: list of 4-6 traits
- speaking_style: description of how they speak
- system_prompt: detailed instruction for the AI persona"""

        user_content = f"""Create a moderator persona for a discussion on: {topic}

Language: {language}

Requirements:
1. Name should be unique and memorable
2. Expertise in guiding discussions and facilitation
3. Excellent at transitions and keeping engagement
4. Personality should be engaging, fair, and inclusive
5. Speaking style should be clear, warm, and professional
6. System prompt must include language instruction: {language}

Return ONLY valid JSON, no markdown, no code blocks."""

        completion = self.client.chat.completions.create(
            model=self.config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            response_format={"type": "json_object"}
        )

        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("Failed to generate moderator persona: empty response")
        persona_data = json.loads(content)

        return Persona(
            name=persona_data["name"],
            role=persona_data["role"],
            expertise=persona_data["expertise"],
            personality_traits=persona_data["personality_traits"],
            speaking_style=persona_data["speaking_style"],
            system_prompt=persona_data["system_prompt"]
        )

    def generate_expert_personas(
        self,
        topic: str,
        num_experts: int,
        language: str
    ) -> List[Persona]:
        """
        Generate diverse expert personas.
        - Each with different expertise area related to topic
        - Unique personality and speaking style
        - Topic-specific knowledge focus

        Args:
            topic: The discussion topic.
            num_experts: Number of expert personas to generate.
            language: Output language.

        Returns:
            List of Persona objects for experts.

        """
        system_content = """You are a persona generator. Create diverse expert personas for a podcast-style discussion. Each expert should have unique expertise areas related to the topic, distinct personalities, and complementary perspectives. Return JSON only, no markdown formatting.

Required JSON structure (array of personas):
[
  {
    "name": "name of the expert",
    "role": "expert",
    "expertise": "specific area of expertise",
    "personality_traits": ["trait1", "trait2", ...],
    "speaking_style": "description",
    "system_prompt": "detailed instruction"
  }
]"""

        user_content = f"""Create {num_experts} diverse expert personas for a discussion on: {topic}

Language: {language}

Requirements:
1. Each expert must have a DIFFERENT expertise area related to the topic
2. Each expert must have a unique name
3. Ensure diversity in perspectives (analytical, creative, practical, theoretical, etc.)
4. Each should have 4-6 personality traits
5. Speaking styles should be distinct between experts
6. System prompts must include language instruction: {language}
7. Expertise areas should be complementary, not overlapping

Important: NO DUPLICATE expertise areas. Each expert brings a unique angle.

Return ONLY valid JSON array, no markdown, no code blocks."""

        completion = self.client.chat.completions.create(
            model=self.config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            response_format={"type": "json_object"}
        )

        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError(f"Failed to generate {num_experts} expert personas: empty response")
        persona_data = json.loads(content)

        if isinstance(persona_data, list):
            experts_data = persona_data
        elif isinstance(persona_data, dict) and "personas" in persona_data:
            experts_data = persona_data["personas"]
        else:
            experts_data = [persona_data]

        return [
            Persona(
                name=data["name"],
                role=data["role"],
                expertise=data["expertise"],
                personality_traits=data["personality_traits"],
                speaking_style=data["speaking_style"],
                system_prompt=data["system_prompt"]
            )
            for data in experts_data
        ]

    def create_system_prompt(
        self, persona: Persona, conversation_context: str, language: str
    ) -> str:
        """
        Create role-specific system prompt with persona details and instructions.

        Args:
            persona: The persona object to generate system prompt for.
            conversation_context: Current conversation context/topic.
            language: Output language.

        Returns:
            System prompt string for the AI persona.

        """
        base_prompt = persona.system_prompt
        context_instruction = f"""Current conversation context: {conversation_context}
You are participating in a podcast-style discussion."""

        if persona.role == "moderator":
            role_instruction = """Your responsibilities:
- Guide the flow of conversation smoothly
- Make interesting transitions between topics
- Ensure all participants get opportunities to speak
- Keep the discussion engaging and on-track
- Ask probing questions to deepen the discussion
- Connect ideas between different speakers"""
        else:
            role_instruction = """Your responsibilities:
- Share expertise and insights from your perspective
- Engage with other speakers' ideas
- Build on or thoughtfully challenge others' points
- Keep contributions substantive and interesting
- Ask questions of other participants"""

        language_instruction = f"IMPORTANT: All your responses must be in {language}."

        full_prompt = f"""{base_prompt}

{context_instruction}
{role_instruction}
{language_instruction}"""

        return full_prompt
