#!/usr/bin/env python3
"""Run the shared MCP self-probe suite and write reproducible artifacts.

This wrapper intentionally delegates to the library-owned MCP probe workflow
instead of reimplementing direct function calls. The probe hits the real MCP
stdio surface with safe adversarial checks:

- startup and tool discovery
- runtime policy inspection
- malformed-input handling
- scratch-path containment
- SQLite query guard behavior
- tunneling-pipeline numerical stability
- unsafe execute_python capability gating
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spectral Packet Engine MCP self-probe")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mcp_probe"))
    parser.add_argument("--working-directory", type=Path, default=Path.cwd())
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--max-concurrent-tasks", type=int, default=1)
    parser.add_argument("--slot-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="warning")
    parser.add_argument("--allow-unsafe-python", action="store_true")
    parser.add_argument("--profile", choices=["stress", "audit"], default="stress")
    parser.add_argument("--source-checkout", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.working_directory).resolve()
    sys.path.insert(0, str(repo_root / "src"))

    from spectral_packet_engine.mcp_probe import (
        build_local_probe_server_spec,
        run_mcp_probe_suite,
        write_mcp_probe_artifacts,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_mcp_probe_suite(
        build_local_probe_server_spec(
            working_directory=repo_root,
            python_executable=args.python_executable,
            max_concurrent_tasks=args.max_concurrent_tasks,
            slot_timeout_seconds=args.slot_timeout_seconds,
            log_level=args.log_level,
            log_file=output_dir / "server.log",
            allow_unsafe_python=args.allow_unsafe_python,
            source_checkout=args.source_checkout,
        ),
        expect_unsafe_python_enabled=args.allow_unsafe_python,
        profile=args.profile,
    )
    write_mcp_probe_artifacts(report, output_dir)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
