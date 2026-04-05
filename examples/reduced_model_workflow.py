from __future__ import annotations

import argparse
import json

from spectral_packet_engine import analyze_coupled_channel_surfaces, analyze_separable_tensor_product_spectrum, to_serializable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two reduced-model workflows: separable tensor-product spectrum and coupled-channel surfaces."
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    separable = analyze_separable_tensor_product_spectrum(
        family_x="harmonic",
        parameters_x={"omega": 8.0},
        family_y="harmonic",
        parameters_y={"omega": 6.0},
        device=args.device,
    )
    coupled = analyze_coupled_channel_surfaces(device=args.device)
    print(json.dumps(to_serializable(separable), sort_keys=True))
    print(json.dumps(to_serializable(coupled), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
