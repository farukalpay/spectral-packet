from __future__ import annotations

import argparse
from pathlib import Path

from spectral_packet_engine import (
    ProfileTable,
    analyze_profile_table_spectra,
    compare_profile_tables,
    compress_profile_table,
    fit_gaussian_packet_to_profile_table,
    load_profile_table,
    write_compression_artifacts,
    write_inverse_artifacts,
    write_profile_comparison_artifacts,
    write_spectral_analysis_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spectral analysis, compression, inverse fitting, and comparison for a profile table."
    )
    parser.add_argument(
        "--table-path",
        "--csv-path",
        dest="table_path",
        type=Path,
        default=Path("examples/data/synthetic_profiles.csv"),
        help="Profile table to load: CSV, TSV, JSON, or optionally XLSX.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/csv_profile_workflow"),
        help="Directory for written artifacts.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device: auto, cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = load_profile_table(args.table_path)

    analysis = analyze_profile_table_spectra(table, num_modes=16, device=args.device)
    compression = compress_profile_table(table, num_modes=8, device=args.device)
    inverse = fit_gaussian_packet_to_profile_table(
        table,
        initial_guess={
            "center": 0.40,
            "width": 0.12,
            "wavenumber": 18.0,
            "phase": 0.0,
        },
        num_modes=64,
        quadrature_points=1024,
        steps=120,
        learning_rate=0.05,
        device=args.device,
    )
    comparison = compare_profile_tables(
        table,
        ProfileTable(
            position_grid=inverse.observation_grid.detach().cpu().numpy(),
            sample_times=inverse.times.detach().cpu().numpy(),
            profiles=inverse.predicted_density.detach().cpu().numpy(),
            source="inverse_fit_prediction",
        ),
        device=args.device,
    )

    write_spectral_analysis_artifacts(args.output_dir / "analysis", analysis)
    write_compression_artifacts(args.output_dir / "compression", compression)
    write_inverse_artifacts(args.output_dir / "inverse", inverse)
    write_profile_comparison_artifacts(args.output_dir / "comparison", comparison)

    print(f"Wrote artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
