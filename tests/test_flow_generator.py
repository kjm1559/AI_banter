"""Comprehensive unit tests for flow_generator module."""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.config import Config
from src.types import FlowGuide, Segment
from src.flow_generator import FlowGuideGenerator


def create_mock_subtopics_response():
    """Create mock subtopics response as JSON."""
    subtopics = [
        {
            "topic": "AI Safety and Alignment",
            "key_points": ["What is AI alignment", "Safety challenges", "Current solutions"],
            "duration": 15,
            "suggested_speakers": ["Moderator", "Expert 1"]
        },
        {
            "topic": "Ethical Considerations",
            "key_points": ["Privacy concerns", "Bias in AI", "Regulatory frameworks"],
            "duration": 15,
            "suggested_speakers": ["Moderator", "Expert 2"]
        },
        {
            "topic": "Future Applications",
            "key_points": ["Healthcare applications", "Education impact", "Economic implications"],
            "duration": 15,
            "suggested_speakers": ["Moderator", "Expert 3"]
        }
    ]
    return json.dumps(subtopics)


class TestFlowGuideGenerator:
    
    @pytest.fixture
    def mock_config(self):
        return Config(
            OPENAI_BASE_URL="https://api.openai.com/v1",
            OPENAI_API_KEY="test-key",
            OPENAI_MODEL="gpt-4o"
        )
    
    @pytest.fixture
    def flow_generator(self, mock_config):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=create_mock_subtopics_response()))
        ]
        
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)
        
        with patch("openai.OpenAI", return_value=mock_client):
            fg = FlowGuideGenerator(mock_config)
            fg.client = mock_client
            return fg
    
    def test_generate_flow_guide_returns_flowguide(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        assert isinstance(guide, FlowGuide)
        assert len(guide.segments) >= 5
    
    def test_generate_flow_guide_total_duration(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        calculated_total = sum(seg.duration_mins for seg in guide.segments)
        assert guide.total_duration_mins == calculated_total
        assert guide.total_duration_mins > 0
    
    def test_segment_has_required_fields(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        for segment in guide.segments:
            assert segment.topic is not None and len(segment.topic) > 0
            assert segment.key_points is not None and isinstance(segment.key_points, list)
            assert segment.duration_mins > 0
            assert isinstance(segment.suggested_speakers, list)
    
    def test_intro_segment_has_moderator(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        intro_segment = guide.segments[0]
        assert "Moderator" in intro_segment.suggested_speakers
    
    def test_get_current_segment_valid_index(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        current = flow_generator.get_current_segment(guide, 0)
        assert isinstance(current, Segment)
    
    def test_get_current_segment_out_of_bounds(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        current = flow_generator.get_current_segment(guide, 99)
        assert current is None
    
    def test_get_current_segment_negative_index(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        current = flow_generator.get_current_segment(guide, -1)
        assert current is None
    
    def test_is_conversation_complete_true(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        complete = flow_generator.is_conversation_complete(guide, len(guide.segments))
        assert complete is True
    
    def test_is_conversation_complete_false(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        complete = flow_generator.is_conversation_complete(guide, 0)
        assert complete is False
    
    def test_get_remaining_segments_at_start(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        remaining = flow_generator.get_remaining_segments(guide, 0)
        assert len(remaining) == len(guide.segments)
    
    def test_get_remaining_segments_empty(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        remaining = flow_generator.get_remaining_segments(guide, len(guide.segments))
        assert len(remaining) == 0
    
    def test_format_segment_summary(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        summary = flow_generator.format_segment_summary(guide.segments[0])
        assert guide.segments[0].topic in summary
        assert "Key Points:" in summary
    
    def test_format_flow_guide(self, flow_generator):
        guide = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=60
        )
        
        formatted = flow_generator.format_flow_guide(guide)
        assert "CONVERSATION FLOW GUIDE" in formatted
    
    def test_different_durations_produce_different_results(self, flow_generator):
        guide_short = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=30
        )
        
        guide_long = flow_generator.generate_flow_guide(
            topic="Future of AI",
            duration_mins=90
        )
        
        assert guide_short.total_duration_mins < guide_long.total_duration_mins
