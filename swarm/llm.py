"""LLM provider clients and the unified chat dispatcher.

Each provider function shares the same signature so llm_chat can route
between them transparently.  Adding a new provider requires:
  1. A new function here following the (messages, system, config) signature.
  2. A new branch in llm_chat.
  3. A new value in config._normalize_llm_provider.
"""

from __future__ import annotations

import requests

from swarm.config import SwarmConfig


# ── Ollama API client ─────────────────────────────────────────────────────────

def ollama_chat(messages: list[dict[str, str]], system: str, config: SwarmConfig) -> str:
    """Send a message list to Ollama and return the model's reply as plain text.

    The system prompt is prepended to messages on every call so each
    request is stateless from Ollama's perspective — the full conversation
    history is carried in `messages`.
    """
    # Build the request body in Ollama's expected format
    payload = {
        "model": config.model,
        "stream": False,   # wait for the full response rather than streaming tokens
        "messages": [{"role": "system", "content": system}] + messages,
    }

    try:
        response = requests.post(
            config.ollama_url,
            json=payload,
            timeout=config.request_timeout_seconds,
        )
        response.raise_for_status()  # raises on 4xx/5xx HTTP errors
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("Cannot connect to Ollama. Start it with: ollama serve") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    try:
        data = response.json()
        return str(data["message"]["content"]).strip()
    except (ValueError, KeyError, TypeError) as exc:
        # Raise instead of returning empty string so callers know something broke
        raise RuntimeError(f"Unexpected Ollama response format: {exc}") from exc


# ── Cerebras Cloud API client ─────────────────────────────────────────────────

def cerebras_chat(messages: list[dict[str, str]], system: str, config: SwarmConfig) -> str:
    """Send a message list to Cerebras Cloud and return the model's reply.

    The cerebras-cloud-sdk is imported lazily so users who only run Ollama
    do not need to install it.  Any import or API error is surfaced as a
    RuntimeError so run_agent can propagate it cleanly.
    """
    try:
        from cerebras.cloud.sdk import Cerebras  # optional dependency
    except ImportError:
        raise RuntimeError(
            "cerebras-cloud-sdk not installed. Run: pip install cerebras-cloud-sdk"
        )

    if not config.cerebras_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=cerebras requires CEREBRAS_API_KEY to be set in .env."
        )

    client = Cerebras(api_key=config.cerebras_api_key)
    all_messages: list[dict[str, str]] = [{"role": "system", "content": system}] + messages

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=all_messages,
        )
        return str(response.choices[0].message.content).strip()
    except Exception as exc:
        raise RuntimeError(f"Cerebras request failed: {exc}") from exc


# ── LLM dispatcher ────────────────────────────────────────────────────────────

def llm_chat(messages: list[dict[str, str]], system: str, config: SwarmConfig) -> str:
    """Route to the configured LLM provider."""
    if config.llm_provider == "cerebras":
        return cerebras_chat(messages, system, config)
    return ollama_chat(messages, system, config)
