"""Tests for tekijin.config."""

from __future__ import annotations

import os
from pathlib import Path

from tekijin.config import Settings, get_settings


def test_defaults(monkeypatch) -> None:
    # Isolate from ambient TEKIJIN_* env (the Makefile exports .env) and any
    # local .env file so this asserts the real code defaults, not the dev's env.
    for key in list(os.environ):
        if key.startswith("TEKIJIN_"):
            monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.database_url == "postgresql+psycopg://tekijin:tekijin@localhost:5432/tekijin"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_model == "qwen/qwen3.6-35b-a3b"
    assert settings.llm_api_key == "dummy"
    assert settings.llm_thinking_control == "reasoning"
    assert settings.llm_provider_routing is None
    assert settings.llm_structured_output_method == "function_calling"
    assert settings.embedding_model == "nvidia/Nemotron-3-Embed-1B-BF16"
    assert settings.embedding_trust_remote_code is True
    assert settings.bm25_weight == 0.2
    # System-1 knowledge surfacing enabled by default after the task3 DGX floor
    # sweep (#413 additive / #433 daily), both routing-safe (person recall 1.000).
    assert settings.additive_self_answer_enabled is True
    assert settings.additive_self_answer_floor == 0.20
    assert settings.daily_knowledge_enabled is True
    assert settings.cors_origins == ("http://localhost:3000",)
    assert isinstance(settings.fixtures_dir, Path)
    assert settings.fixtures_dir.name == "synthetic"
    assert settings.fixtures_dir.parent.name == "fixtures"


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TEKIJIN_LLM_MODEL", "custom-model")
    monkeypatch.setenv("TEKIJIN_APP_ENV", "production")

    settings = Settings()

    assert settings.llm_model == "custom-model"
    assert settings.app_env == "production"


def test_max_concurrent_runs_default_and_override(monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("TEKIJIN_"):
            monkeypatch.delenv(key, raising=False)
    assert Settings(_env_file=None).max_concurrent_runs == 8  # matches vLLM max-num-seqs
    monkeypatch.setenv("TEKIJIN_MAX_CONCURRENT_RUNS", "3")
    assert Settings(_env_file=None).max_concurrent_runs == 3


def test_durability_enforced_resolution() -> None:
    # None derives from app_env; an explicit bool overrides it either way (#180).
    assert Settings(_env_file=None, app_env="development").durability_enforced() is False
    assert Settings(_env_file=None, app_env="production").durability_enforced() is True
    assert (
        Settings(
            _env_file=None, app_env="development", strict_durability=True
        ).durability_enforced()
        is True
    )
    assert (
        Settings(
            _env_file=None, app_env="production", strict_durability=False
        ).durability_enforced()
        is False
    )


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second

    get_settings.cache_clear()
