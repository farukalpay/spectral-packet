#!/usr/bin/env python3
"""Complete quantum tunneling experiment using the spectral packet engine.

This script demonstrates the full tunneling analysis pipeline:

1. Transfer-matrix scattering: compute T(E) and R(E) across an energy range
2. WKB semiclassical: estimate tunneling probability and compare with exact result
3. Split-operator propagation: evolve a Gaussian wavepacket toward the barrier
4. Wigner function: compute phase-space distribution and non-classicality witness

Usage:
    python examples/tunneling_experiment.py
    python examples/tunneling_experiment.py --barrier-height 100 --grid-points 512
    python examples/tunneling_experiment.py --device cuda  # if GPU available

This same experiment can be run via MCP:
    MCP tool: tunneling_experiment
    MCP tool: execute_python (with this script's logic)
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantum tunneling experiment")
    parser.add_argument("--barrier-height", type=float, default=50.0)
    parser.add_argument("--barrier-width", type=float, default=0.03, help="Gaussian sigma")
    parser.add_argument("--grid-points", type=int, default=256)
    parser.add_argument("--num-energies", type=int, default=300)
    parser.add_argument("--propagation-steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--packet-center", type=float, default=0.25)
    parser.add_argument("--packet-width", type=float, default=0.04)
    parser.add_argument("--packet-wavenumber", type=float, default=40.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    # Check device availability
    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print(f"WARNING: CUDA not available, falling back to CPU", file=sys.stderr)
        device = "cpu"
    elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print(f"WARNING: MPS not available, falling back to CPU", file=sys.stderr)
        device = "cpu"

    if device != "cpu":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'Apple Metal'})", file=sys.stderr)
    else:
        print(f"Using device: CPU", file=sys.stderr)

    from spectral_packet_engine.pipelines import analyze_tunneling

    print("Running tunneling experiment...", file=sys.stderr)
    report = analyze_tunneling(
        barrier_height=args.barrier_height,
        barrier_width_sigma=args.barrier_width,
        grid_points=args.grid_points,
        num_energies=args.num_energies,
        propagation_steps=args.propagation_steps,
        dt=args.dt,
        packet_center=args.packet_center,
        packet_width=args.packet_width,
        packet_wavenumber=args.packet_wavenumber,
        device=device,
    )

    if args.json:
        result = {
            "barrier": {"height": report.barrier_height, "width_fwhm": report.barrier_width},
            "scattering": {
                "transmission_at_half_barrier": report.transmission_at_half_barrier,
                "num_resonances": report.num_resonances,
                "resonance_energies": report.resonance_energies,
            },
            "wkb": {
                "transmission_at_half_barrier": report.wkb_transmission_at_half_barrier,
                "wkb_exact_ratio": report.wkb_exact_ratio,
            },
            "propagation": {
                "norm_drift": report.propagation_norm_drift,
                "energy_drift": report.propagation_energy_drift,
                "steps": report.propagation_steps,
            },
            "wigner": {"negativity": report.wigner_negativity},
            "device": report.device,
        }
        print(json.dumps(result, indent=2))
    else:
        print()
        print("=" * 60)
        print("  QUANTUM TUNNELING EXPERIMENT — SPECTRAL PACKET ENGINE")
        print("=" * 60)
        print()
        print(f"  Barrier: V0 = {report.barrier_height:.1f}, FWHM = {report.barrier_width:.4f}")
        print(f"  Energy range: {report.energy_range[0]:.1f} to {report.energy_range[1]:.1f}")
        print(f"  Device: {report.device}")
        print()
        print("--- Transfer-Matrix Scattering ---")
        print(f"  T(E = V0/2) = {report.transmission_at_half_barrier:.6e}")
        print(f"  Resonances detected: {report.num_resonances}")
        if report.resonance_energies:
            for i, (e, w) in enumerate(zip(report.resonance_energies, report.resonance_widths)):
                print(f"    Resonance {i+1}: E = {e:.4f}, FWHM = {w:.4f}")
        print()
        print("--- WKB Semiclassical Comparison ---")
        print(f"  T_WKB(E = V0/2) = {report.wkb_transmission_at_half_barrier:.6e}")
        print(f"  WKB / exact ratio = {report.wkb_exact_ratio:.4f}")
        print()
        print("--- Split-Operator Propagation ---")
        print(f"  Steps: {report.propagation_steps}")
        print(f"  Norm drift:   {report.propagation_norm_drift:.2e}")
        print(f"  Energy drift: {report.propagation_energy_drift:.2e}")
        print()
        print("--- Wigner Phase-Space Analysis ---")
        print(f"  Negativity: {report.wigner_negativity:.6f}")
        print(f"  (>0 indicates non-classical state)")
        print()
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
