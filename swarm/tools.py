"""Tool definitions, tool-call parsing, and tool execution.

Tools are implemented as plain functions that the model can request by
emitting a JSON object.  We use a JSON-in-system-prompt convention instead
of each provider's native function-calling API so the mechanism works
identically across Ollama, Cerebras, and any future provider.

Two tool-instruction strings are exported:
  TOOL_INSTRUCTIONS           — includes search_web; injected for the Researcher only.
  TOOL_INSTRUCTIONS_SAVE_ONLY — omits search_web; injected for Analyst and Writer.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from swarm.config import SwarmConfig


# ── Tool instructions for the LLM ─────────────────────────────────────────────
# Injected into each agent's system prompt so the model knows how to call tools.

# Full set — given only to the Researcher, who is the sole search-capable agent.
TOOL_INSTRUCTIONS = """
You can use these tools by returning ONLY a JSON object:

{"tool": "search_web", "query": "your search query"}
{"tool": "save_output", "output": "your complete final output"}

Rules:
- Return ONLY valid JSON when calling a tool.
- Use search_web as needed, then call save_output when your task is complete.
- For search_web, include a non-empty string query.
"""

# Restricted set — given to Analyst and Writer, who must not search the web.
# Advertising only save_output prevents the model from attempting a web search.
TOOL_INSTRUCTIONS_SAVE_ONLY = """
You have one tool available. Call it by returning ONLY a JSON object:

{"tool": "save_output", "output": "your complete final output"}

Rules:
- Return ONLY valid JSON when calling save_output.
- Do NOT search the web — work only from the information already provided to you.
- Call save_output once your task is fully complete.
"""


# ── Tool implementations ──────────────────────────────────────────────────────

def simulate_search_results(query: str) -> str:
    """Return deterministic local placeholder search results.

    Used when SEARCH_MODE=simulated (the default).  Keeps the script
    fully offline — useful for testing prompts without a Tavily API key.
    """
    return (
        f"[Simulated results for '{query}']\n"
        "- Finding 1: Interest and investment have increased year over year.\n"
        "- Finding 2: Adoption barriers include policy, cost, and talent.\n"
        "- Finding 3: Case studies show measurable outcomes in pilot programs.\n"
        "- Finding 4: Tooling maturity is improving rapidly across vendors.\n"
        "Tip: Set SEARCH_MODE=live and TAVILY_API_KEY for live web search."
    )


def execute_tool(tool_call: dict[str, Any], config: SwarmConfig) -> str:
    """Execute a model-requested tool call and return the result as plain text.

    Returns a plain-text result that gets fed back into the conversation,
    or the sentinel string 'OUTPUT_SAVED' to signal the agent is done.
    """
    tool = tool_call.get("tool")

    if tool == "search_web":
        query = str(tool_call.get("query", "")).strip()
        if not query:
            # Return an error string so the agent can self-correct on next turn
            return "Tool error: search_web requires a non-empty 'query'."

        print(f"    Searching: {query}")

        if config.search_mode == "live":
            # Live mode uses the Tavily API for real web search results
            if not config.tavily_api_key:
                return "Tool error: SEARCH_MODE=live requires TAVILY_API_KEY."

            try:
                from tavily import TavilyClient  # optional dependency
            except ImportError:
                return "Tool error: tavily-python not installed. Run: pip install tavily-python"

            try:
                client = TavilyClient(api_key=config.tavily_api_key)
                results = client.search(query)
                # Serialize results so the model receives consistent text
                return json.dumps(results, ensure_ascii=False)
            except Exception as exc:
                # Surface the error as a tool result; the agent can retry
                return f"Tool error: live web search failed ({exc})."

        # Simulated mode — return canned results for offline/testing use
        return simulate_search_results(query)

    if tool == "save_output":
        # Sentinel value; run_agent checks for this to know the agent finished
        return "OUTPUT_SAVED"

    # Guard against hallucinated tool names from the model
    return f"Tool error: unknown tool '{tool}'."


# ── Tool-call parser ──────────────────────────────────────────────────────────

def parse_tool_call(text: str) -> Optional[dict[str, Any]]:
    """Scan model output for the first valid, supported JSON tool call.

    The model is instructed to return ONLY JSON when it wants to call a tool,
    but it sometimes adds surrounding prose.  We scan character-by-character
    for the first opening brace, attempt a JSON parse from that position,
    and accept only objects that match a known tool schema.

    Returns None if no valid tool call is found (the response is a final answer).
    """
    source = text.strip()
    decoder = json.JSONDecoder()

    for index, char in enumerate(source):
        if char != "{":  # fast skip for non-JSON characters
            continue
        try:
            # raw_decode parses the JSON object starting at `index` and
            # returns (parsed_obj, end_position) — we only need the object
            obj, _ = decoder.raw_decode(source[index:])
        except json.JSONDecodeError:
            continue  # not valid JSON at this position; keep scanning

        if not isinstance(obj, dict):
            continue  # ignore JSON arrays or scalars

        tool = obj.get("tool")
        if tool not in {"search_web", "save_output"}:
            continue  # reject unknown / hallucinated tool names

        # Validate required fields per tool to catch malformed calls early
        if tool == "search_web" and not str(obj.get("query", "")).strip():
            continue  # empty query would produce useless search results

        if tool == "save_output" and "output" not in obj:
            continue  # save_output without content has no value

        return obj  # first valid tool call wins

    return None  # model responded directly — treat as final output


# ── Output utilities ──────────────────────────────────────────────────────────

def coerce_output(output: Any, max_chars: int) -> str:
    """Normalise agent output to a string and enforce the character cap.

    Models can emit dicts/lists via save_output — we serialise those to JSON
    so every caller always receives a plain string.  Large outputs are
    truncated to avoid bloating downstream prompts and report files.
    """
    if isinstance(output, (dict, list)):
        text = json.dumps(output, indent=2, ensure_ascii=False)
    else:
        text = str(output)

    if len(text) <= max_chars:
        return text

    # Append a notice so the reader knows the content was cut
    return text[:max_chars] + "\n\n[Output truncated due to size limit.]"
