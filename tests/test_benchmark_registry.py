from __future__ import annotations

from spectral_packet_engine import (
    list_official_benchmarks,
    official_benchmark_registry,
    run_benchmark_registry,
)


def test_official_benchmark_registry_declares_product_suite() -> None:
    expected = {
        "harmonic-oscillator",
        "double-well",
        "barrier-scattering",
        "anharmonic-inverse-fit",
        "noisy-reconstruction",
        "reduced-model-tradeoff",
    }

    assert set(list_official_benchmarks()) == expected
    definitions = official_benchmark_registry()

    assert len(definitions) == len(expected)
    assert all(definition.default_mode_budget for definition in definitions)
    assert all("identifiability" in " ".join(definition.reported_metrics) for definition in definitions)
    assert all(definition.honest_limits for definition in definitions)


def test_benchmark_registry_reports_common_measurement_axes() -> None:
    report = run_benchmark_registry(
        case_ids=("harmonic-oscillator", "barrier-scattering"),
        device="cpu",
    )

    assert report.summary["case_count"] == 2
    assert report.summary["failed"] == 0
    assert report.runtime.backend == "cpu"

    for result in report.case_results:
        assert result.status == "passed"
        assert result.metrics is not None
        assert result.metrics.score >= 0.0
        assert result.metrics.error
        assert "total_elapsed_seconds" in result.metrics.timing
        assert "python_peak_bytes" in result.metrics.memory
        assert result.metrics.mode_budget
        assert result.metrics.identifiability
        assert result.metrics.backend["backend"] == "cpu"
        assert result.metrics.backend_comparison["cpu"]["status"] == "primary_backend"


def test_benchmark_registry_writes_artifact_bundle(tmp_path) -> None:
    report = run_benchmark_registry(case_ids=("harmonic-oscillator",), device="cpu")
    directory = report.write_artifacts(tmp_path / "benchmark_registry")

    assert directory.complete is True
    assert "benchmark_registry.json" in directory.files
    assert "benchmark_cases.csv" in directory.files
    assert directory.metadata["workflow"] == "benchmark-registry"
