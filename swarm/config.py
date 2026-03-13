"""Configuration loading and the SwarmConfig dataclass.

This module is always the first thing imported — it reads .env and builds
the CONFIG singleton before any other module touches os.environ.

Environment variables (all optional, all have safe defaults):
    LLM_PROVIDER            'ollama' or 'cerebras'  (default: ollama)
    MODEL                   Ollama model name        (default: llama3.2)
    CEREBRAS_MODEL          Cerebras model name      (default: llama3.1-8b)
    CEREBRAS_API_KEY        Required for LLM_PROVIDER=cerebras
    OLLAMA_URL              Ollama chat endpoint     (default: localhost:11434)
    SEARCH_MODE             'simulated' or 'live'   (default: simulated)
    TAVILY_API_KEY          Required for SEARCH_MODE=live
    REQUEST_TIMEOUT_SECONDS Seconds to wait per LLM call (default: 120)
    MAX_AGENT_STEPS         Tool-call iterations per agent (default: 8)
    MAX_OUTPUT_CHARS        Character cap on agent output  (default: 20000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# ── .env loader ───────────────────────────────────────────────────────────────
# Reads variables from a local .env file so users don't need to manually
# `source .env` before running the script.

def load_env_file(env_path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file into process env.

    - Only sets variables that are NOT already present in the shell environment.
    - Supports bare KEY=VALUE and `export KEY=VALUE` formats.
    - Silently skips missing or malformed lines.
    """
    path = Path(env_path)
    if not path.exists() or not path.is_file():
        return  # .env is optional; silently skip if absent

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):  # skip blanks and comments
            continue

        if line.startswith("export "):  # support shell-style `export KEY=VALUE`
            line = line[len("export "):].strip()

        if "=" not in line:  # skip lines without an assignment operator
            continue

        key, value = line.split("=", 1)  # split only on the first `=`
        key = key.strip()
        value = value.strip().strip('"').strip("'")  # remove surrounding quotes
        if key:
            # setdefault keeps any value already exported in the shell env
            os.environ.setdefault(key, value)


# ── Config helpers ────────────────────────────────────────────────────────────

def _to_int(value: str, default: int, minimum: int) -> int:
    """Safely parse an integer from a string env var.

    Falls back to `default` if parsing fails and clamps to `minimum`.
    This prevents bad env values (e.g. empty strings) from crashing startup.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, minimum)  # enforce a lower bound


def _normalize_search_mode(value: str) -> str:
    """Validate SEARCH_MODE and fall back to 'simulated' for unknown values."""
    mode = (value or "").strip().lower()
    if mode in {"live", "simulated"}:
        return mode
    return "simulated"  # safe default — works fully offline


def _normalize_llm_provider(value: str) -> str:
    """Validate LLM_PROVIDER and fall back to 'ollama' for unknown values."""
    provider = (value or "").strip().lower()
    if provider in {"ollama", "cerebras"}:
        return provider
    return "ollama"  # safe default — works without cloud credentials


# ── Swarm configuration ───────────────────────────────────────────────────────
# All runtime settings live in this single frozen dataclass.
# `frozen=True` makes it immutable after creation — no accidental mutation.

@dataclass(frozen=True)
class SwarmConfig:
    llm_provider: str             # 'ollama' (local) or 'cerebras' (cloud)
    model: str                    # Active model name (resolved per provider)
    cerebras_api_key: str         # Required when llm_provider == 'cerebras'
    ollama_url: str               # Ollama REST endpoint (used when llm_provider == 'ollama')
    search_mode: str              # 'simulated' (offline) or 'live' (Tavily)
    tavily_api_key: str           # Required only when search_mode == 'live'
    request_timeout_seconds: int  # Max wait for a single LLM API call
    max_agent_steps: int          # Max tool-call iterations per agent
    max_output_chars: int         # Character cap on agent output

    @classmethod
    def from_env(cls) -> "SwarmConfig":
        """Build a SwarmConfig by reading from environment variables.

        Every variable has a safe default so the script works out of the
        box without any configuration.  For local inference, install Ollama
        and run `ollama serve`.  For cloud inference, set LLM_PROVIDER=cerebras
        and supply a CEREBRAS_API_KEY in .env.
        """
        provider = _normalize_llm_provider(os.getenv("LLM_PROVIDER", "ollama"))
        # Each provider has its own default model name
        if provider == "cerebras":
            model = os.getenv("CEREBRAS_MODEL", "llama3.1-8b").strip()
        else:
            model = os.getenv("MODEL", "llama3.2").strip()

        return cls(
            llm_provider=provider,
            model=model,
            cerebras_api_key=os.getenv("CEREBRAS_API_KEY", "").strip(),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat").strip(),
            search_mode=_normalize_search_mode(os.getenv("SEARCH_MODE", "simulated")),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            request_timeout_seconds=_to_int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"), 120, 10),
            max_agent_steps=_to_int(os.getenv("MAX_AGENT_STEPS", "8"), 8, 1),
            max_output_chars=_to_int(os.getenv("MAX_OUTPUT_CHARS", "20000"), 20000, 1000),
        )


# Load .env before reading env vars — order matters.
load_env_file()
# Build the global config singleton once at import time.
CONFIG: SwarmConfig = SwarmConfig.from_env()
