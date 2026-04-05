from __future__ import annotations

import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import sys
from threading import Thread


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_profile_table_example_runs_and_writes_report_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "profile_report"
    result = subprocess.run(
        [
            sys.executable,
            "examples/profile_table_workflow.py",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=True,
    )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2
    overview = json.loads(lines[0])
    artifact_report = json.loads(lines[1])

    assert overview["analyze_num_modes"] == 16
    assert overview["compress_num_modes"] == 8
    assert artifact_report["complete"] is True
    assert "analysis/artifacts.json" in artifact_report["files"]
    assert "compression/artifacts.json" in artifact_report["files"]


def test_api_example_posts_profile_report_workflow() -> None:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            requests.append({"path": self.path, "payload": payload})
            response = {
                "overview": {
                    "analyze_num_modes": payload["analyze_num_modes"],
                    "compress_num_modes": payload["compress_num_modes"],
                    "dominant_modes": [0, 1],
                }
            }
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "examples/api_workflow.py",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
            ],
            cwd=_repo_root(),
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert requests
    request = requests[0]
    assert request["path"] == "/profiles/report"
    payload = request["payload"]
    assert payload["analyze_num_modes"] == 16
    assert payload["compress_num_modes"] == 8
    assert payload["output_dir"] == "artifacts/profile_report"
    assert payload["table"]["source"] == "examples/data/synthetic_profiles.csv"

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines[0] == "Profile report overview:"
    overview = json.loads(lines[1])
    assert overview["analyze_num_modes"] == 16
    assert overview["compress_num_modes"] == 8


def test_inverse_physics_example_runs_and_writes_vertical_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "inverse_physics"
    result = subprocess.run(
        [
            sys.executable,
            "examples/inverse_physics_workflow.py",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=True,
    )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2
    inference = json.loads(lines[0])
    artifact_report = json.loads(lines[1])

    assert inference["best_family"] == "harmonic"
    assert artifact_report["complete"] is True
    assert "family_inference/artifacts.json" in artifact_report["files"]
    assert "observed_transitions.csv" in artifact_report["files"]


def test_reduced_model_example_runs_and_writes_structured_2d_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "reduced_model"
    result = subprocess.run(
        [
            sys.executable,
            "examples/reduced_model_workflow.py",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=True,
    )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) >= 3
    overview = json.loads(lines[0])
    artifact_report = json.loads(lines[1])

    assert overview["example_name"] == "box-plus-box"
    assert artifact_report["complete"] is True
    assert "separable_2d_report.json" in artifact_report["files"]
    assert "eigenvalues.csv" in artifact_report["files"]
    assert "mode_budget.json" in artifact_report["files"]
