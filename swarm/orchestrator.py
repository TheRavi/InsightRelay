"""Swarm orchestrator and report persistence.

run_swarm strings the three agents together in a linear pipeline:

    agent_researcher → agent_analyst → agent_writer

Each stage fails fast with a RuntimeError if an agent produces no output,
preventing silent bad reports from being saved.

save_report writes the final output to a timestamped .txt file so each
run produces a unique, non-overwriting artefact.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from swarm.agents import agent_analyst, agent_researcher, agent_writer
from swarm.config import SwarmConfig


def save_report(topic: str, report_text: str, config: SwarmConfig) -> Path:
    """Write the final report to a timestamped .txt file.

    A single datetime snapshot is captured once so the filename and the
    header always show the same timestamp, even if the call crosses a
    second boundary.

    Returns the Path so callers can display the filename in the terminal.
    """
    now = datetime.now()  # capture once — keeps filename and header in sync
    filename = Path(f"report_{now.strftime('%Y%m%d_%H%M%S')}.txt")
    header = (
        f"RESEARCH REPORT: {topic}\n"
        f"Generated: {now.strftime('%Y-%m-%d %H:%M')} | Model: {config.model}\n"
        + "=" * 55
        + "\n\n"
    )
    filename.write_text(header + report_text, encoding="utf-8")
    return filename


def run_swarm(topic: str, config: SwarmConfig) -> str:
    """Run the full 3-agent pipeline and return the final report as a string.

    Stages:
        1. Researcher — collects raw information about the topic.
        2. Analyst    — extracts insights from the research notes.
        3. Writer     — turns the analysis into a formatted report.

    The report is also written to a timestamped .txt file on disk.
    """
    cleaned_topic = topic.strip()
    if not cleaned_topic:
        raise ValueError("Topic cannot be empty.")

    print("\n" + "=" * 55)
    print("RESEARCH SWARM STARTING")
    print(f"Topic: {cleaned_topic}")
    print(f"Provider: {config.llm_provider}")
    print(f"Model: {config.model}")
    print(f"Search mode: {config.search_mode}")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 55)

    # Stage 1 — research
    print("\n[1/3] Launching Researcher")
    research_notes = agent_researcher(cleaned_topic, config)
    if not research_notes.strip():
        raise RuntimeError("Researcher produced no output.")

    # Stage 2 — analysis (receives the notes from stage 1)
    print("\n[2/3] Launching Analyst")
    analysis = agent_analyst(cleaned_topic, research_notes, config)
    if not analysis.strip():
        raise RuntimeError("Analyst produced no output.")

    # Stage 3 — write final report (receives the analysis from stage 2)
    print("\n[3/3] Launching Writer")
    final_report = agent_writer(cleaned_topic, analysis, config)
    if not final_report.strip():
        raise RuntimeError("Writer produced no output.")

    report_path = save_report(cleaned_topic, final_report, config)

    print("\n" + "=" * 55)
    print("SWARM COMPLETE")
    print(f"Report saved to: {report_path}")
    print("=" * 55 + "\n")
    print(final_report)

    return final_report
