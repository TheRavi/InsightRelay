"""InsightRelay - CLI entry point.

All logic lives in the swarm/ package.  This file only wires CLI arguments
to the package and starts the process.

LLM Providers (set LLM_PROVIDER in .env):
  ollama    - local inference via Ollama (default)
  cerebras  - cloud inference via Cerebras Cloud

Defaults:
  ollama provider with llama3.2 model
  Simulated search mode (offline-friendly)

Optional:
  Live Tavily search: SEARCH_MODE=live and TAVILY_API_KEY
  Cerebras cloud:     LLM_PROVIDER=cerebras and CEREBRAS_API_KEY

Run:
    python agentsSwarm.py --topic "AI in healthcare"
"""

from __future__ import annotations

import argparse

from swarm.config import CONFIG, SwarmConfig
from swarm.orchestrator import run_swarm


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments.

    --topic      lets callers pass the topic non-interactively (useful for
                 shell scripts, cron jobs, or CI pipelines).
    --max-steps  lets you tune iteration depth per run without editing .env.
    """
    parser = argparse.ArgumentParser(description="Run the InsightRelay 3-agent research pipeline.")
    parser.add_argument("--topic", type=str, help="Research topic")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps per agent for this run",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point - wires up CLI args, config, and the swarm.

    Returns 0 on success, 1 on any error (standard Unix convention).
    This lets shell scripts check $? to detect failures.
    """
    args = parse_args()

    print("InsightRelay")
    print(f"Provider: {CONFIG.llm_provider}")
    print(f"Model: {CONFIG.model}")
    print(f"Search mode: {CONFIG.search_mode}")
    print()

    # Accept topic from CLI flag; fall back to interactive prompt.
    topic = (args.topic or "").strip()
    if not topic:
        topic = input("What topic should InsightRelay research?\n> ").strip()

    # If --max-steps was supplied, create a one-off config that overrides
    # only max_agent_steps while preserving all other .env settings.
    if args.max_steps is not None:
        override_steps = max(args.max_steps, 1)  # clamp to at least 1
        runtime_config = SwarmConfig(
            llm_provider=CONFIG.llm_provider,
            model=CONFIG.model,
            cerebras_api_key=CONFIG.cerebras_api_key,
            ollama_url=CONFIG.ollama_url,
            search_mode=CONFIG.search_mode,
            tavily_api_key=CONFIG.tavily_api_key,
            request_timeout_seconds=CONFIG.request_timeout_seconds,
            max_agent_steps=override_steps,
            max_output_chars=CONFIG.max_output_chars,
        )
    else:
        runtime_config = CONFIG  # use the global config as-is

    try:
        run_swarm(topic=topic, config=runtime_config)
    except Exception as exc:
        print(f"\nError: {exc}")
        return 1  # non-zero exit code signals failure to the shell
    return 0


# Standard guard: run main() only when executed directly, not when imported.
if __name__ == "__main__":
    raise SystemExit(main())
