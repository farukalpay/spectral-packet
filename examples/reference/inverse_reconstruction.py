from __future__ import annotations

import argparse

from reference_case import build_reference_problem, reference_times

from spectral_packet_engine import (
    GaussianPacketEstimator,
    GaussianPacketParameters,
    inspect_torch_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover Gaussian packet parameters from density observations.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = inspect_torch_runtime(args.device)
    problem = build_reference_problem(
        num_modes=128,
        quadrature_points=2048,
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )
    estimator = GaussianPacketEstimator(problem.domain, basis=problem.basis)

    observation_grid = problem.domain.grid(96)
    times = reference_times(dtype=problem.domain.real_dtype, device=problem.domain.device)
    truth = GaussianPacketParameters.single(
        center=0.30,
        width=0.07,
        wavenumber=25.0,
        dtype=problem.domain.real_dtype,
        device=problem.domain.device,
    )
    target_density = estimator.predict(
        truth,
        observation_grid=observation_grid,
        times=times,
        observation_mode="density",
    ).detach()

    initial_guess = GaussianPacketParameters.single(
        center=0.36,
        width=0.11,
        wavenumber=22.0,
        dtype=problem.domain.real_dtype,
        device=problem.domain.device,
    )
    result = estimator.fit(
        observation_grid=observation_grid,
        times=times,
        target=target_density,
        initial_guess=initial_guess,
        observation_mode="density",
        steps=180,
        learning_rate=0.05,
    )

    estimated = result.parameters
    print("Inverse reconstruction")
    print(f"torch backend: {runtime.backend}")
    print(f"torch device: {runtime.device}")
    print(f"final loss: {result.final_loss:.6e}")
    print(f"center      true=0.300000  estimated={estimated.center[0].item():.6f}")
    print(f"width       true=0.070000  estimated={estimated.width[0].item():.6f}")
    print(f"wavenumber  true=25.000000 estimated={estimated.wavenumber[0].item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
