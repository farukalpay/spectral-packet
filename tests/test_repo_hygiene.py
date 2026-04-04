from __future__ import annotations

from pathlib import Path
import re


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_bytecode_artifacts_are_ignored_in_repo_hygiene_policy() -> None:
    gitignore = (_repo_root() / ".gitignore").read_text(encoding="utf-8")

    assert "__pycache__/" in gitignore
    assert "*.pyc" in gitignore


def test_examples_surface_is_curated_for_public_use() -> None:
    examples_dir = _repo_root() / "examples"
    public_scripts = sorted(path.name for path in examples_dir.glob("*.py"))

    assert public_scripts == [
        "api_workflow.py",
        "core_engine_workflow.py",
        "modal_surrogate_workflow.py",
        "profile_table_workflow.py",
    ]
    assert (examples_dir / "README.md").exists()
    assert (examples_dir / "reference" / "README.md").exists()
    assert (examples_dir / "experimental" / "README.md").exists()


def test_examples_do_not_mutate_sys_path() -> None:
    for path in (_repo_root() / "examples").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sys.path.insert" not in text, path.as_posix()


def test_repo_does_not_reference_openrouter_or_embedding_stack() -> None:
    root = _repo_root()
    banned_patterns = (
        re.compile(r"\bopenrouter\b", re.IGNORECASE),
        re.compile(r"\bopenai\b", re.IGNORECASE),
        re.compile(r"\banthropic\b", re.IGNORECASE),
        re.compile(r"\bcohere\b", re.IGNORECASE),
        re.compile(r"\bembedding(s)?\b", re.IGNORECASE),
        re.compile(r"\bllm(s)?\b", re.IGNORECASE),
    )
    checked_files = [
        root / "README.md",
        root / "pyproject.toml",
        root / "AGENTS.md",
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "tests").rglob("*.py")),
        *sorted((root / "docs").rglob("*.md")),
        *sorted((root / "examples").rglob("*.py")),
        *sorted((root / "examples").rglob("*.md")),
    ]
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for pattern in banned_patterns:
            assert pattern.search(text) is None, f"{path.as_posix()} matched {pattern.pattern}"
