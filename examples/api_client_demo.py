from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the optional Spectral Packet Engine API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the API service.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "packet": {
            "center": 0.30,
            "width": 0.07,
            "wavenumber": 25.0,
            "phase": 0.0,
        },
        "times": [0.0, 1e-3, 5e-3],
        "num_modes": 64,
        "quadrature_points": 2048,
        "grid_points": 128,
        "device": "cpu",
    }
    request = Request(
        f"{args.base_url}/forward",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        body = response.read().decode("utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
