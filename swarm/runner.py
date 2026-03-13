"""Core agent execution loop shared by all three agents.

All specialised agents (Researcher, Analyst, Writer) call run_agent,
which handles the LLM call, tool-call parsing, search blocking, and
output coercion.  The agents themselves only supply role-specific prompts.

Loop behaviour:
  1. Call the LLM provider and get a response.
  2. Scan the response for a JSON tool call.
  3. If search_web is detected and allow_search=False, reject it and loop.
  4a. If a valid tool call is found — execute it, append the result, loop again.
  4b. If no tool call — treat the response as the agent's final answer and return.

The loop stops early when the agent calls save_output (task complete)
or when max_steps is reached (prevents infinite loops).
"""

from __future__ import annotations

from typing import Optional

from swarm.config import SwarmConfig
from swarm.llm import llm_chat
from swarm.tools import (
    TOOL_INSTRUCTIONS,
    TOOL_INSTRUCTIONS_SAVE_ONLY,
    coerce_output,
    execute_tool,
    parse_tool_call,
)


def run_agent(
    *,
    name: str,
    system_prompt: str,
    user_message: str,
    config: SwarmConfig,
    max_steps: Optional[int] = None,
    allow_search: bool = True,
) -> str:
    """Run an agent with a tool execution loop and return its final output.

    Args:
        name:          Human-readable agent name shown in progress output.
        system_prompt: Role description and task instructions for the model.
        user_message:  Initial user turn that starts the agent's work.
        config:        Swarm-wide settings (model, search mode, limits).
        max_steps:     Override the config limit for this specific run.
        allow_search:  When False, search_web is removed from the tool prompt
                       and hard-blocked at the loop level.  Only the Researcher
                       should pass True (the default).
    """
    print("\n" + "-" * 60)
    print(f"Agent: {name}")
    print("-" * 60)

    step_limit = max_steps or config.max_agent_steps
    # Select tool instructions based on whether this agent is allowed to search
    tool_instructions = TOOL_INSTRUCTIONS if allow_search else TOOL_INSTRUCTIONS_SAVE_ONLY
    # Combine the agent's role prompt with the appropriate tool instructions
    full_system = f"{system_prompt}\n\n{tool_instructions}"
    # Maintain the full conversation so the model has context on each call
    messages: list[dict[str, str]] = [{"role": "user", "content": user_message}]

    for _ in range(step_limit):
        # --- Call the model ---
        response_text = llm_chat(messages=messages, system=full_system, config=config)
        messages.append({"role": "assistant", "content": response_text})

        # --- Decide what to do with the response ---
        tool_call = parse_tool_call(response_text)

        # Hard-block search_web for agents that are not permitted to search.
        # TOOL_INSTRUCTIONS_SAVE_ONLY tells the model not to search, but this
        # guard ensures it cannot slip through even if the model ignores that.
        if tool_call and tool_call.get("tool") == "search_web" and not allow_search:
            messages.append({
                "role": "user",
                "content": "Tool error: search_web is not available to this agent. "
                           "Work from the information already provided and call save_output when done.",
            })
            continue

        if not tool_call:
            # No tool call detected — the model produced a final answer directly
            print(f"Completed: {name}")
            return coerce_output(response_text, config.max_output_chars)

        # --- Execute the requested tool ---
        result = execute_tool(tool_call, config)

        if result == "OUTPUT_SAVED":
            # The model called save_output — its work is done
            saved = tool_call.get("output", "")
            print(f"Completed: {name}")
            return coerce_output(saved, config.max_output_chars)

        # Feed the tool result back so the model can continue its task
        messages.append({
            "role": "user",
            "content": f"Tool result: {result}\n\nContinue working toward your goal.",
        })

    # Step limit reached — return whatever the last message was as a best-effort
    fallback = messages[-1]["content"] if messages else ""
    return coerce_output(fallback, config.max_output_chars)
