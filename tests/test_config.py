import pytest
import os
from unittest.mock import patch, MagicMock

from src.config import Config, load_config


class TestConfig:
    
    def test_load_config_with_valid_env(self):
        """Test loading config with valid environment variables."""
        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_KEY": "test-key-123",
            "OPENAI_MODEL": "gpt-4o"
        }):
            config = load_config()
            
            assert config.OPENAI_BASE_URL == "https://api.openai.com/v1"
            assert config.OPENAI_API_KEY == "test-key-123"
            assert config.OPENAI_MODEL == "gpt-4o"
    
    def test_load_config_without_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_MODEL": "gpt-4o"
        }):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                load_config()
    
    def test_load_config_default_model(self):
        """Test that OPENAI_MODEL defaults to gpt-4o."""
        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_KEY": "test-key-123"
        }):
            config = load_config()
            assert config.OPENAI_MODEL == "gpt-4o"
    
    def test_load_config_custom_model(self):
        """Test loading config with custom model."""
        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_KEY": "test-key-123",
            "OPENAI_MODEL": "gpt-3.5-turbo"
        }):
            config = load_config()
            assert config.OPENAI_MODEL == "gpt-3.5-turbo"
    
    def test_load_config_empty_api_key_raises_error(self):
        """Test that empty API key raises ValueError."""
        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_KEY": "",
            "OPENAI_MODEL": "gpt-4o"
        }):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                load_config()
