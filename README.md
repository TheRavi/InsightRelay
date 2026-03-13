# InsightRelay

Discover, analyze, deliver.

InsightRelay is a modular Python research pipeline that runs three LLM-driven agents in sequence to turn a raw topic into a structured, decision-ready report:

1. Researcher gathers source material using simulated or live web search.
2. Analyst turns the raw notes into ranked findings and conclusions.
3. Writer produces a final report and saves it to a timestamped text file.

The code is split into a small CLI entry point and a reusable `swarm/` package so the workflow is easier to maintain, extend, and publish.

## Features

- Multi-agent workflow with a clear research-to-report handoff.
- Supports `ollama` for local inference.
- Supports `cerebras` for cloud inference.
- Uses simulated search by default for offline runs.
- Restricts web search to the Researcher only.
- Writes UTF-8 reports with timestamped filenames.
- Keeps runtime configuration in `.env`.

## How It Works

InsightRelay runs a three-stage pipeline:

1. `Researcher` collects source material from simulated or live search.
2. `Analyst` ranks findings, surfaces patterns, and flags gaps.
3. `Writer` turns the analysis into a polished final report.

This separation keeps each role narrow and makes the outputs easier to reason about and improve.

## Project Structure

```text
.
├── agentsSwarm.py
├── swarm/
│   ├── __init__.py
│   ├── agents.py
│   ├── config.py
│   ├── llm.py
│   ├── orchestrator.py
│   ├── runner.py
│   └── tools.py
├── .env.example
├── .gitignore
└── requirements.txt
```

## Requirements

- Python 3.11+
- An LLM provider:
  - Ollama running locally, or
  - a Cerebras API key
- Tavily API key only if you want live web search

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` with the provider and search settings you want.

## Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` or `cerebras` | `ollama` |
| `MODEL` | Ollama model name | `llama3.2` |
| `CEREBRAS_MODEL` | Cerebras model name | `llama3.1-8b` |
| `CEREBRAS_API_KEY` | Required for Cerebras runs | empty |
| `OLLAMA_URL` | Ollama chat endpoint | `http://localhost:11434/api/chat` |
| `SEARCH_MODE` | `simulated` or `live` | `simulated` |
| `TAVILY_API_KEY` | Required for live search | empty |
| `REQUEST_TIMEOUT_SECONDS` | Per-request timeout | `120` |
| `MAX_AGENT_STEPS` | Tool-loop step cap per agent | `8` |
| `MAX_OUTPUT_CHARS` | Output truncation limit | `20000` |

## Running

Run interactively:

```bash
python3 agentsSwarm.py
```

Run non-interactively:

```bash
python3 agentsSwarm.py --topic "AI in healthcare"
python3 agentsSwarm.py --topic "Semiconductor supply chains" --max-steps 10
```

Each run writes a report like `report_YYYYMMDD_HHMMSS.txt` in the project root.

## Provider Notes

### Ollama

```bash
ollama serve
ollama pull llama3.2
```

Set in `.env`:

```env
LLM_PROVIDER=ollama
MODEL=llama3.2
```

### Cerebras

Set in `.env`:

```env
LLM_PROVIDER=cerebras
CEREBRAS_API_KEY=your_api_key_here
CEREBRAS_MODEL=llama3.1-8b
```

## GitHub Push Checklist

- Keep `.env` out of version control.
- Commit `.env.example` instead of real secrets.
- Review generated `report_*.txt` files before committing anything.
- Validate the code before pushing:

```bash
python3 -m py_compile swarm/config.py swarm/llm.py swarm/tools.py swarm/runner.py swarm/agents.py swarm/orchestrator.py swarm/__init__.py agentsSwarm.py
```

## Optional Next Steps

- Add tests for config loading and tool-call parsing.
- Add a GitHub Actions workflow for linting and compile checks.
- Add structured logging if you want to run this non-interactively in automation.