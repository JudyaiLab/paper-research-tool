"""
Config Manager — Load config.yaml or environment variables
"""

import copy
import os
from pathlib import Path

import yaml


FREE_PAPER_LIMIT = 3
MIN_NARRATIVE_PAPERS = 2

SUPPORTED_LANGUAGES = {
    "zh-TW": "繁體中文（Traditional Chinese）",
    "en": "English",
    "ko": "한국어（Korean）",
}


def get_lang_instruction(lang: str) -> str:
    """Return a prompt instruction for the AI to respond in the given language."""
    label = SUPPORTED_LANGUAGES.get(lang, lang)
    if lang == "zh-TW":
        return ""  # Prompts are already in Chinese, no extra instruction needed
    return f"\n\n⚠️ IMPORTANT: You MUST write your ENTIRE response in {label}. Do NOT use any other language.\n\n"

DEFAULT_CONFIG = {
    "ai": {
        "provider": "openai",  # openai | anthropic | gemini | ollama
        "openai": {
            "api_key": "",
            "model": "gpt-4o-mini",
        },
        "anthropic": {
            "api_key": "",
            "model": "claude-sonnet-4-20250514",
        },
        "gemini": {
            "api_key": "",
            "model": "gemini-2.0-flash",
        },
        "ollama": {
            "model": "llama3",
            "base_url": "http://localhost:11434",
        },
    },
    "defaults": {
        "output_dir": "./output",
        "language": "zh-TW",
    },
}


def load_config() -> dict:
    """
    Load config with priority: config.yaml > environment variables > defaults.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Try loading config.yaml
    config_paths = [
        Path("config.yaml"),
        Path("config.yml"),
        Path.home() / ".paper-research-tool" / "config.yaml",
    ]

    for path in config_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, user_config)
            break

    # Environment variables override (highest priority)
    env_mappings = {
        "OPENAI_API_KEY": ("ai", "openai", "api_key"),
        "ANTHROPIC_API_KEY": ("ai", "anthropic", "api_key"),
        "GEMINI_API_KEY": ("ai", "gemini", "api_key"),
        "OPENAI_MODEL": ("ai", "openai", "model"),
        "ANTHROPIC_MODEL": ("ai", "anthropic", "model"),
        "GEMINI_MODEL": ("ai", "gemini", "model"),
        "OLLAMA_MODEL": ("ai", "ollama", "model"),
        "OLLAMA_BASE_URL": ("ai", "ollama", "base_url"),
        "AI_PROVIDER": ("ai", "provider"),
    }

    for env_key, config_path in env_mappings.items():
        val = os.environ.get(env_key)
        if val:
            _set_nested(config, config_path, val)

    # Auto-detect provider if not explicitly set
    if not os.environ.get("AI_PROVIDER"):
        config["ai"]["provider"] = _auto_detect_provider(config)

    return config


def _auto_detect_provider(config: dict) -> str:
    """Auto-detect which AI provider to use based on available keys."""
    if config["ai"]["openai"].get("api_key"):
        return "openai"
    if config["ai"]["anthropic"].get("api_key"):
        return "anthropic"
    if config["ai"]["gemini"].get("api_key"):
        return "gemini"
    # Check if ollama is running with a usable model
    try:
        import urllib.request
        import json as _json
        url = config["ai"]["ollama"].get("base_url", "http://localhost:11434")
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as resp:
            data = _json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            # Filter out embedding-only models
            chat_models = [m for m in models if "embed" not in m.lower()]
            if chat_models:
                return "ollama"
    except Exception:
        pass
    return "none"


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(d: dict, keys: tuple, value):
    """Set a nested dict value by key path."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def get_ai_provider_info(config: dict) -> str:
    """Return a human-readable string of current AI config."""
    provider = config["ai"]["provider"]
    if provider == "openai":
        model = config["ai"]["openai"]["model"]
        has_key = bool(config["ai"]["openai"]["api_key"])
        return f"OpenAI ({model}) — API Key: {'✓' if has_key else '✗ Not configured'}"
    elif provider == "anthropic":
        model = config["ai"]["anthropic"]["model"]
        has_key = bool(config["ai"]["anthropic"]["api_key"])
        return f"Anthropic ({model}) — API Key: {'✓' if has_key else '✗ Not configured'}"
    elif provider == "gemini":
        model = config["ai"]["gemini"]["model"]
        has_key = bool(config["ai"]["gemini"]["api_key"])
        return f"Google Gemini ({model}) — API Key: {'✓' if has_key else '✗ Not configured'}"
    elif provider == "ollama":
        model = config["ai"]["ollama"]["model"]
        url = config["ai"]["ollama"]["base_url"]
        return f"Ollama ({model}) @ {url}"
    else:
        return "No AI configured — please set up config.yaml or environment variables"
