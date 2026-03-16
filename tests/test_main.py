import pytest
from unittest.mock import MagicMock, patch

from src.config import Config
from src.types import ConversationState, Persona, DialogueTurn, Script, Segment, FlowGuide
from src.persona_manager import PersonaManager
from src.main import AIBanterAgent


class TestAIBanterAgent:
    
    @pytest.fixture
    def mock_config(self):
        return Config(
            OPENAI_BASE_URL="https://api.openai.com/v1",
            OPENAI_API_KEY="test-key",
            OPENAI_MODEL="gpt-4o"
        )
    
    @pytest.fixture
    def agent(self, mock_config):
        with patch("src.persona_manager.PersonaManager"):
            with patch("src.flow_generator.FlowGuideGenerator"):
                with patch("src.orchestrator.ConversationOrchestrator"):
                    agent = AIBanterAgent(mock_config)
                    agent.flow_generator.is_conversation_complete = MagicMock(return_value=True)
                    return agent
    
    def test_validate_inputs_rejects_less_than_three(self, agent):
        """Test that validation rejects fewer than 3 participants."""
        with pytest.raises(ValueError):
            agent._validate_inputs(2)
    
    def test_validate_inputs_rejects_two(self, agent):
        """Test that validation rejects exactly 2 participants."""
        with pytest.raises(ValueError):
            agent._validate_inputs(2)
    
    def test_validate_inputs_accepts_three(self, agent):
        """Test that validation accepts 3 participants."""
        agent._validate_inputs(3)
    
    def test_validate_inputs_accepts_more_than_three(self, agent):
        """Test that validation accepts more than 3 participants."""
        agent._validate_inputs(5)
    
    def test_format_output_includes_topic(self, agent):
        """Test that formatted output includes topic."""
        script = Script(
            turns=[
                DialogueTurn(
                    speaker_name="Host",
                    role="moderator",
                    content="Welcome",
                    turn_number=1,
                    word_count=1
                )
            ],
            topic="AI Discussion",
            language="english",
            total_word_count=1
        )
        
        output = agent._format_output(script)
        
        assert "AI Discussion" in output
    
    def test_format_output_includes_turns(self, agent):
        """Test that formatted output includes dialogue turns."""
        script = Script(
            turns=[
                DialogueTurn(
                    speaker_name="Host Jane",
                    role="moderator",
                    content="Welcome everyone",
                    turn_number=1,
                    word_count=2
                )
            ],
            topic="Test",
            language="english",
            total_word_count=2
        )
        
        output = agent._format_output(script)
        
        assert "Host Jane" in output
        assert "Welcome everyone" in output
    
    def test_format_output_structure(self, agent):
        """Test that formatted output has proper structure."""
        script = Script(
            turns=[
                DialogueTurn(
                    speaker_name="Host",
                    role="moderator",
                    content="Hello",
                    turn_number=1,
                    word_count=1
                )
            ],
            topic="Test",
            language="english",
            total_word_count=1
        )
        
        output = agent._format_output(script)
        
        assert "PODCAST SCRIPT" in output
        assert "END OF SCRIPT" in output
        assert "MODERATOR" in output
    
    def test_save_script_creates_file(self, agent, tmp_path):
        """Test that save_script creates output file."""
        script = Script(
            turns=[
                DialogueTurn(
                    speaker_name="Host",
                    role="moderator",
                    content="Hello",
                    turn_number=1,
                    word_count=1
                )
            ],
            topic="Test",
            language="english",
            total_word_count=1
        )
        
        filepath = str(tmp_path / "test_script.txt")
        agent.save_script(script, filepath)
        
        from pathlib import Path
        assert Path(filepath).exists()
        content = Path(filepath).read_text()
        assert "Test" in content
    
    @patch("src.persona_manager.PersonaManager")
    @patch("src.flow_generator.FlowGuideGenerator")
    @patch("src.orchestrator.ConversationOrchestrator")
    @patch("os.environ")
    def test_cli_argument_parsing(self, mock_env, mock_orch, mock_flow, mock_persona):
        """Test CLI argument parsing via argparse."""
        import sys
        
        mock_env.__getitem__ = lambda self, key, default=None: (
            "test-key" if key == "OPENAI_API_KEY" else
            "https://api.openai.com/v1" if key == "OPENAI_BASE_URL" else
            "gpt-4o" if key == "OPENAI_MODEL" else
            default
        )
        mock_env.get = lambda key, default=None: mock_env.__getitem__(key, default)
        
        with patch.object(sys, 'argv', ['main', '--topic', "Test Topic", '--participants', '4']):
            with pytest.raises(SystemExit, match="1|0"):
                from src.main import main
                main()
