from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import subprocess

__version__ = "0.2.0"


@lru_cache(maxsize=1)
def resolve_git_commit(*, short: bool = True) -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return None
    command = ["git", "rev-parse"]
    if short:
        command.append("--short")
    command.append("HEAD")
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    token = completed.stdout.strip()
    return token or None


__all__ = ["__version__", "resolve_git_commit"]
