"""Conversation flow guide generation module."""

import json
from typing import List, Optional

from openai import OpenAI

from src.config import Config
from src.types import FlowGuide, Segment


class FlowGuideGenerator:
    """Generates structured conversation flow guides for podcast discussions."""

    def __init__(self, config: Config):
        """Initialize FlowGuideGenerator with configuration.

        Args:
            config: Configuration object with OpenAI credentials.

        """
        self.config = config
        self.client = OpenAI(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY
        )

    def generate_flow_guide(self, topic: str, duration_mins: int = 60) -> FlowGuide:
        """
        Generate structured conversation flow for podcast discussion.
        
        Structure:
        1. Introduction (5-10 mins)
        2. Main Discussion Segments (40-50 mins)
        3. Deep Dive/Debate (10-15 mins)
        4. Conclusion (5 mins)
        
        Args:
            topic: The main discussion topic.
            duration_mins: Total duration in minutes (default: 60).
            
        Returns:
            FlowGuide object with structured segments.

        """
        intro_duration = max(5, int(duration_mins * 0.1))
        main_duration = duration_mins - intro_duration - 15
        deep_dive_duration = max(10, int(duration_mins * 0.15))
        conclusion_duration = 5

        subtopics = self._generate_subtopics(topic, duration_mins, main_duration)

        segments = []

        intro_segment = Segment(
            topic="Introduction: Welcome and Overview",
            key_points=[
                "Welcome listeners and introduce the podcast",
                f"Introduce the main topic: {topic}",
                "Introduce all speakers and their expertise areas",
                "Set the tone and expectations for the discussion",
                "Hook the audience with an engaging opening"
            ],
            duration_mins=intro_duration,
            suggested_speakers=["Moderator"]
        )
        segments.append(intro_segment)

        for subtopic_info in subtopics:
            main_segment = Segment(
                topic=subtopic_info["topic"],
                key_points=subtopic_info["key_points"],
                duration_mins=subtopic_info["duration"],
                suggested_speakers=subtopic_info["suggested_speakers"]
            )
            segments.append(main_segment)

        debate_segment = Segment(
            topic="Deep Dive: Controversial Perspectives and Complex Questions",
            key_points=[
                "Explore contentious aspects of the topic",
                "Challenge assumptions and present counterarguments",
                "Discuss emerging debates or unanswered questions",
                "Allow for passionate but respectful disagreement",
                "Synthesize diverse viewpoints"
            ],
            duration_mins=deep_dive_duration,
            suggested_speakers=["All participants"]
        )
        segments.append(debate_segment)

        conclusion_segment = Segment(
            topic="Conclusion: Summary and Final Thoughts",
            key_points=[
                "Summarize key insights from the discussion",
                "Highlight the most compelling arguments",
                "Provide final reflections from each speaker",
                "Suggest further reading or resources",
                "Thank listeners and sign off"
            ],
            duration_mins=conclusion_duration,
            suggested_speakers=["Moderator", "All participants"]
        )
        segments.append(conclusion_segment)

        total_duration = sum(seg.duration_mins for seg in segments)

        return FlowGuide(
            segments=segments,
            total_duration_mins=total_duration
        )

    def _generate_subtopics(
        self,
        topic: str,
        total_duration: int,
        main_duration: int
    ) -> List[dict]:
        """
        Generate topic-specific subtopics using OpenAI.

        Args:
            topic: The main discussion topic.
            total_duration: Total conversation duration in minutes.
            main_duration: Duration allocated for main discussion segments.

        Returns:
            List of subtopic dictionaries with topic, key_points, duration, suggested_speakers.

        """
        num_subtopics = min(5, max(3, total_duration // 15))
        
        system_content = """
        You are a conversation flow designer for podcast discussions. Generate engaging 
        subtopics and discussion points for a podcast episode. Return JSON only, no markdown.

        Required JSON structure (array):
        [{
          "topic": "subtopic title",
          "key_points": ["point1", "point2"],
          "duration": minutes_for_segment,
          "suggested_speakers": ["speaker1", "speaker2"]
        }]
        """.strip()

        user_content = f"""
        Generate {num_subtopics} subtopics for: "{topic}"
        Main time: {main_duration} minutes. Requirements:
        1. Each subtopic distinct aspect of main topic
        2. Progress from foundational to complex
        3. Each has 3-5 key discussion points/questions
        4. Distribute time evenly or by importance
        5. Speakers: "Moderator", "Expert 1", "Expert 2", etc.
        6. Vary angles (technical, social, ethical)
        7. Natural flow between subtopics

        Return ONLY JSON array.
        """.strip()

        completion = self.client.chat.completions.create(
            model=self.config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}
        )

        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned empty response")
        subtopics_data = json.loads(content)
        
        if isinstance(subtopics_data, list):
            return subtopics_data
        elif isinstance(subtopics_data, dict) and "subtopics" in subtopics_data:
            return subtopics_data["subtopics"]
        else:
            return [subtopics_data]

    def get_current_segment(
        self,
        flow_guide: FlowGuide,
        current_idx: int
    ) -> Optional[Segment]:
        """
        Return current segment or None if completed.

        Args:
            flow_guide: The conversation flow guide.
            current_idx: Current segment index.

        Returns:
            Current Segment object or None if all segments completed.

        """
        if current_idx < 0 or current_idx >= len(flow_guide.segments):
            return None
        return flow_guide.segments[current_idx]

    def is_conversation_complete(
        self,
        flow_guide: FlowGuide,
        current_idx: int
    ) -> bool:
        """
        Check if all segments covered.

        Args:
            flow_guide: The conversation flow guide.
            current_idx: Current segment index.

        Returns:
            True if all segments completed, False otherwise.

        """
        return current_idx >= len(flow_guide.segments)

    def get_remaining_segments(
        self,
        flow_guide: FlowGuide,
        current_idx: int
    ) -> List[Segment]:
        """
        Get list of segments yet to be discussed.

        Args:
            flow_guide: The conversation flow guide.
            current_idx: Current segment index.

        Returns:
            List of remaining Segment objects.

        """
        if current_idx < 0:
            return flow_guide.segments.copy()
        if current_idx >= len(flow_guide.segments):
            return []
        return flow_guide.segments[current_idx:]

    def get_elapsed_duration(
        self,
        flow_guide: FlowGuide,
        current_idx: int
    ) -> int:
        """
        Calculate total duration of completed segments.

        Args:
            flow_guide: The conversation flow guide.
            current_idx: Current segment index.

        Returns:
            Total minutes of completed segments.

        """
        if current_idx <= 0:
            return 0
        return sum(seg.duration_mins for seg in flow_guide.segments[:current_idx])

    def get_next_segment(
        self,
        flow_guide: FlowGuide,
        current_idx: int
    ) -> Optional[Segment]:
        """
        Get the next segment in the flow.

        Args:
            flow_guide: The conversation flow guide.
            current_idx: Current segment index.

        Returns:
            Next Segment object or None if no segments remain.

        """
        next_idx = current_idx + 1
        return self.get_current_segment(flow_guide, next_idx)

    def format_segment_summary(self, segment: Segment, include_speakers: bool = True) -> str:
        """
        Format a segment as a human-readable summary.

        Args:
            segment: The segment to format.
            include_speakers: Whether to include suggested speakers.

        Returns:
            Formatted string representation of the segment.

        """
        lines = [
            f"Topic: {segment.topic}",
            f"Duration: {segment.duration_mins} minutes",
            "Key Points:"
        ]

        for point in segment.key_points:
            lines.append(f"  - {point}")

        if include_speakers and segment.suggested_speakers:
            speakers_str = ", ".join(segment.suggested_speakers)
            lines.append(f"Suggested Speakers: {speakers_str}")

        return "\n".join(lines)

    def format_flow_guide(self, flow_guide: FlowGuide) -> str:
        """
        Format the entire flow guide as a human-readable guide.

        Args:
            flow_guide: The flow guide to format.

        Returns:
            Formatted string representation of the flow guide.

        """
        lines = [
            "=" * 60,
            "CONVERSATION FLOW GUIDE",
            "=" * 60,
            f"Total Duration: {flow_guide.total_duration_mins} minutes",
            f"Number of Segments: {len(flow_guide.segments)}",
            "",
            "-" * 60,
            ""
        ]

        for idx, segment in enumerate(flow_guide.segments, 1):
            lines.append(f"SEGMENT {idx}: {segment.topic}")
            lines.append(f"Duration: {segment.duration_mins} minutes")
            lines.append("")
            lines.append("Key Discussion Points:")
            for point in segment.key_points:
                lines.append(f"  - {point}")
            if segment.suggested_speakers:
                lines.append(f"\nSuggested Speakers: {', '.join(segment.suggested_speakers)}")
            lines.append("")
            lines.append("-" * 60)
            lines.append("")

        return "\n".join(lines)
