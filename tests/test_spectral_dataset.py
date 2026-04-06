from __future__ import annotations

import json

import numpy as np

from spectral_packet_engine import (
    ArtifactLineage,
    ProfileTable,
    SpectralDatasetSplit,
    SpectralUncertainty,
    inspect_artifact_directory,
    spectral_dataset_from_profile_table,
    write_spectral_dataset_artifacts,
)


def test_profile_table_can_materialize_spectral_dataset_contract() -> None:
    grid = np.linspace(0.0, 1.0, 5)
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    profiles = np.asarray(
        [
            [0.1, 0.4, 0.8, 0.4, 0.1],
            [0.2, 0.5, 0.7, 0.3, 0.1],
            [0.3, 0.5, 0.6, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    table = ProfileTable(position_grid=grid, sample_times=times, profiles=profiles, source="unit-test")

    dataset = spectral_dataset_from_profile_table(
        table,
        position_units="meter",
        time_units="second",
        uncertainty=SpectralUncertainty(model="independent-gaussian", scale=0.05),
        splits=(
            SpectralDatasetSplit("train", (0, 1), regime="in-distribution"),
            SpectralDatasetSplit("test", (2,), regime="ood-width"),
        ),
        lineage=ArtifactLineage(source="unit-test", transform="normalized profile ingest"),
    )

    payload = dataset.to_dict()
    assert dataset.shape == (3, 5)
    assert len(dataset.content_hash) == 64
    assert payload["grids"][0]["axis_name"] == "time"
    assert payload["grids"][1]["units"] == "meter"
    assert payload["uncertainty"]["model"] == "independent-gaussian"
    assert payload["splits"][1]["regime"] == "ood-width"
    assert dataset.uncertainty_values().shape == dataset.values.shape


def test_spectral_dataset_artifact_bundle_records_lineage(tmp_path) -> None:
    table = ProfileTable(
        position_grid=np.linspace(0.0, 1.0, 4),
        sample_times=np.asarray([0.0, 0.1], dtype=np.float64),
        profiles=np.asarray([[0.2, 0.4, 0.4, 0.2], [0.1, 0.3, 0.5, 0.3]], dtype=np.float64),
        source="artifact-test",
    )
    dataset = spectral_dataset_from_profile_table(table)

    output_dir = tmp_path / "spectral_dataset"
    write_spectral_dataset_artifacts(output_dir, dataset)
    report = inspect_artifact_directory(output_dir)
    payload = json.loads((output_dir / "spectral_dataset.json").read_text(encoding="utf-8"))

    assert report.complete is True
    assert "spectral_dataset.json" in report.files
    assert "spectral_dataset_values.json" in report.files
    assert report.metadata["workflow"] == "spectral-dataset"
    assert report.metadata["content_hash"] == dataset.content_hash
    assert payload["lineage"]["source"] == "artifact-test"
