"""Research Agent Swarm — public package API.

Importing from this package is enough for external scripts and tests:

    from swarm import run_swarm, SwarmConfig, CONFIG
"""

from swarm.config import CONFIG, SwarmConfig
from swarm.orchestrator import run_swarm

__all__ = ["CONFIG", "SwarmConfig", "run_swarm"]
