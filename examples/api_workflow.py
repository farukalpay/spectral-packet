from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen

from spectral_packet_engine import (
    DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
    DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
    DEFAULT_PROFILE_REPORT_OUTPUT_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call the optional Spectral Packet Engine API for the hero profile-table report workflow."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the API service.")
    parser.add_argument(
        "--table-path",
        type=Path,
        default=Path("examples/data/synthetic_profiles.csv"),
        help="Profile table CSV used to build the API request payload.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_PROFILE_REPORT_OUTPUT_DIR,
        help="Artifact directory to request from the API report workflow.",
    )
    return parser.parse_args()


def _load_profile_table_payload(table_path: Path) -> dict[str, object]:
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) < 2 or header[0] != "time":
            raise ValueError("expected a profile-table CSV with a 'time' column followed by position columns")
        position_grid = [float(value) for value in header[1:]]
        sample_times: list[float] = []
        profiles: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            sample_times.append(float(row[0]))
            profiles.append([float(value) for value in row[1:]])
    return {
        "position_grid": position_grid,
        "sample_times": sample_times,
        "profiles": profiles,
        "source": str(table_path),
    }


def main() -> int:
    args = parse_args()
    payload = {
        "table": _load_profile_table_payload(args.table_path),
        "analyze_num_modes": DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
        "compress_num_modes": DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
        "device": "cpu",
        "output_dir": args.output_dir,
    }
    request = Request(
        f"{args.base_url}/profiles/report",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        body = json.loads(response.read().decode("utf-8"))

    print("Profile report overview:")
    print(json.dumps(body["overview"], sort_keys=True))
    print("Artifacts:")
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "expected_root_files": [
                    "artifacts.json",
                    "profile_table_report.json",
                    "profile_table_summary.json",
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
