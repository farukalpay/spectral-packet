from __future__ import annotations

from pathlib import Path
import tomllib

from spectral_packet_engine.version import __version__


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_pyproject_version_matches_package_version() -> None:
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == __version__


def test_cli_entrypoint_targets_package_main() -> None:
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["spectral-packet-engine"] == "spectral_packet_engine.cli:main"


def test_dev_extra_contains_release_build_tools() -> None:
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    dev_dependencies = set(pyproject["project"]["optional-dependencies"]["dev"])
    assert any(dependency.startswith("build>=") for dependency in dev_dependencies)
    assert any(dependency.startswith("twine>=") for dependency in dev_dependencies)


def test_repository_facing_markdown_does_not_leak_local_machine_paths() -> None:
    root = _repo_root()
    markdown_files = [
        root / "README.md",
        root / "AGENTS.md",
        *sorted((root / "docs").rglob("*.md")),
    ]
    forbidden_tokens = (
        "/Users/farukalpay",
        "file://",
        "vscode://",
    )
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in forbidden_tokens), path.as_posix()
