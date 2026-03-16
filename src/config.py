"""Configuration module for AI Banter."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration settings for AI Banter."""

    OPENAI_BASE_URL: str
    """Base URL for OpenAI API."""

    OPENAI_API_KEY: str
    """API key for OpenAI API."""

    OPENAI_MODEL: str
    """Model name to use (default: gpt-4o)."""


def load_config() -> Config:
    """Load configuration from environment variables.

    Returns:
        Config: Configuration object with loaded values.

    Raises:
        ValueError: If OPENAI_API_KEY is missing or empty.

    """
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required. "
            "Please set it in your .env file or environment variables."
        )

    return Config(
        OPENAI_BASE_URL=base_url,
        OPENAI_API_KEY=api_key,
        OPENAI_MODEL=model,
    )


