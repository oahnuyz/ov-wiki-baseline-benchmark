"""TencentDB Agent Memory benchmark backend.

The backend intentionally talks to MemoryKnowledge over its documented HTTP API
and keeps the Agent loop in this repository.  It does not require the TencentDB
monorepo to be imported into the benchmark process.
"""

from .config import TencentDBConfig
from .runner import PreparedTencentExperiment, TencentDBRunner

__all__ = ["PreparedTencentExperiment", "TencentDBConfig", "TencentDBRunner"]
