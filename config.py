from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE, override=True)

DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BASE_URL = "https://aicredits.in/v1"
DEFAULT_QDRANT_URL = "http://localhost:6333"


def get_setting(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if value is not None:
        value = value.strip().strip("\"'")
    if required and (value is None or str(value).strip() == ""):
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_app_settings() -> dict[str, str]:
    base_url = (
        get_setting("OPENAI_BASE_URL")
        or get_setting("OPENAI_API_BASE")
        or get_setting("BASE_URL")
        or DEFAULT_OPENAI_BASE_URL
    )
    return {
        "OPENAI_API_KEY": get_setting("OPENAI_API_KEY", "") or "",
        "OPENAI_BASE_URL": base_url,
        "OPENAI_CHAT_MODEL": get_setting("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        "OPENAI_EMBEDDING_MODEL": get_setting("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "QDRANT_URL": get_setting("QDRANT_URL", DEFAULT_QDRANT_URL),
    }


def validate_openai_settings() -> dict[str, str]:
    settings = get_app_settings()
    if not settings["OPENAI_API_KEY"] or settings["OPENAI_API_KEY"].lower() in {
        "your_api_key_here",
        "your-api-key-here",
        "replace_me",
    }:
        raise ValueError(
            "The API key is missing or still set to a placeholder. "
            "Add your provider key to the .env file as OPENAI_API_KEY."
        )
    if not settings["OPENAI_BASE_URL"].startswith(("http://", "https://")):
        raise ValueError("OPENAI_BASE_URL must be a complete http:// or https:// URL.")
    settings["ENV_FILE"] = str(ENV_FILE)
    return settings
