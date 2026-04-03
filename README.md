# Spectral Packet Engine

Spectral Packet Engine is a bounded-domain physics and ML library for people who need more than a small Schrodinger notebook.

It combines four jobs under one interface:

1. exact spectral simulation of localized wave packets in a 1D box,
2. inverse recovery of packet parameters from observations,
3. compression and analysis of real bounded density profiles from a published experiment,
4. a TensorFlow surrogate that learns `density profile + time -> modal coefficients + moments`.

The repository is meant to be useful as a library, not only as a reference exercise. It gives you one codebase for analytic models, real measurements, and repeated inference workloads.

## Why use it

Most repositories in this area stop at "here is a solver". That leaves three practical questions unanswered:

- How do you fit a compact packet model back to observations?
- How do you prove the basis is useful on real detector data, not only on synthetic states?
- How do you replace repeated profile-to-mode projection with a trainable surrogate?

This library answers all three.

Use it when you want to:

- simulate bounded packet dynamics without hiding the spectral truncation machinery,
- estimate packet parameters from density data with differentiable optimization,
- compress real experimental line profiles into a compact bounded basis,
- benchmark how modal resolution trades off against reconstruction error,
- export a trained TensorFlow model for repeated inference.

## What the library contains

### Physics layer

`src/spectral_packet_engine/domain.py`, `basis.py`, `state.py`, `projector.py`, `dynamics.py`, and `simulation.py` implement:

- a 1D infinite-well domain,
- packet and spectral state representations,
- packet-to-basis projection,
- exact spectral time evolution,
- structured simulation outputs and observable queries.

### Inverse layer

`src/spectral_packet_engine/inference.py` provides differentiable packet reconstruction with `torch.autograd`.

### Real-data layer

`src/spectral_packet_engine/datasets.py` and `profiles.py` implement:

- reproducible download of published transport data,
- explicit preprocessing controls,
- bounded-profile moments,
- modal projection and reconstruction,
- compression summaries on real measurements.

### TensorFlow layer

`src/spectral_packet_engine/tf_surrogate.py` provides:

- host/runtime inspection for TensorFlow workloads,
- a time-conditioned multi-output regressor,
- optional XLA and mixed-precision runtime setup,
- `tf.data` batching, caching, and prefetching,
- SavedModel export after training.

## Why the real-data path matters

The repository includes a benchmark on the published quantum-gas transport dataset from Zenodo:

- Dataset: *Replication Data for: Characterising transport in a quantum gas by measuring Drude weights*
- DOI: [10.5281/zenodo.16701012](https://doi.org/10.5281/zenodo.16701012)
- Example scan: `scan11879_56`
- Measured grid used here: 126 positions, 31 time points, 75 shots per time point
- Reported temperature in file: `39.0 +/- 3.27 nK`

On the shot-averaged normalized experimental profiles, sine-basis compression gives:

| Modes | Mean relative L2 error | Max relative L2 error |
|---|---:|---:|
| 8 | 0.09565 | 0.10489 |
| 16 | 0.03648 | 0.04852 |
| 32 | 0.01644 | 0.02251 |
| 64 | 0.00965 | 0.01334 |

That is the practical value of the package: the bounded basis is not justified by theory alone, it is validated on real transport profiles.

## Installation

The core package requires Python 3.11 or newer. The TensorFlow path in this repository was verified on Python 3.11 because the default Python 3.14 environment on this machine does not currently provide TensorFlow wheels.

### Core physics, inverse modeling, and real-data analysis

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,examples]"
```

The first real-data run downloads the published scan through `pooch` and stores it in the platform cache directory for `spectral-packet-engine`.

### macOS / Apple Silicon

For the TensorFlow benchmark path:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,examples,ml]"
```

On Darwin, the `ml` extra includes `tensorflow-metal` through a platform marker. The physics side can still target Apple MPS independently through `inspect_torch_runtime("auto")`, so modal target generation can use the Mac GPU even when TensorFlow exposes no GPU device.

### Windows

Native Windows CPU workflow:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[data,examples,ml]"
```

Native Windows GPU is not the recommended TensorFlow path anymore. For GPU execution, use WSL2 and run the Linux-style setup inside that environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,examples]"
python -m pip install "tensorflow[and-cuda]>=2.17,<2.22"
```

### Linux / WSL2 CUDA

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,examples]"
python -m pip install "tensorflow[and-cuda]>=2.17,<2.22"
```

## Quick start

### Reference bounded-packet workflows

```bash
python examples/forward_reference.py
python examples/packet_to_spectral.py
python examples/inverse_reconstruction.py --device auto
```

### Real experimental profile benchmark

```bash
python examples/experimental_transport_analysis.py --plot-dir artifacts/transport
```

This command downloads the published scan, projects the averaged detector profiles onto the bounded basis, reports compression error, and writes figures.

### TensorFlow benchmark with exported artifacts

```bash
python examples/experimental_tf_regression.py \
  --modes 16 \
  --benchmark-modes 8 16 \
  --epochs 10 \
  --batch-size 128 \
  --output-dir artifacts/tf_benchmark
```

The benchmark command writes a reproducible result bundle:

- `artifacts/tf_benchmark/benchmark_summary.json`
- `artifacts/tf_benchmark/mode_16/report.json`
- `artifacts/tf_benchmark/mode_16/history.csv`
- `artifacts/tf_benchmark/mode_16/history.png`
- `artifacts/tf_benchmark/mode_16/validation_reconstruction.png`
- `artifacts/tf_benchmark/mode_sweep.png`

If you also want an exported model:

```bash
python examples/experimental_tf_regression.py \
  --modes 16 \
  --epochs 10 \
  --batch-size 128 \
  --output-dir artifacts/tf_benchmark \
  --export-dir artifacts/tf_benchmark/saved_model
```

## Verified benchmark numbers

### Real-data modal compression

The 32-mode reconstruction test used in the test suite satisfies:

- mean relative L2 error `< 0.02`
- max relative L2 error `< 0.03`

### TensorFlow surrogate verification on this machine

Verified in a separate Python 3.11 environment with TensorFlow `2.21.0`:

- primary configuration: 16 modes, 10 epochs, batch size 128
- samples: 1741 shot-level profiles
- train samples: 1393
- validation samples: 348
- parameter count: 807090
- training throughput: `1397.47` profiles/s
- validation inference throughput: `467.83` profiles/s
- validation coefficient MSE: `8.49e-01`
- validation moment MAE: `8.66e-02`
- validation profile relative L2: `9.54e-02`

Platform result from that run:

- TensorFlow visible GPUs: `0`
- TensorFlow device types: `CPU`
- Torch target backend for modal target generation: `mps`

The current code therefore supports a split acceleration strategy on MacBook hardware: Torch can use MPS for spectral targets while TensorFlow follows whatever devices it actually exposes.

## Repository layout

- `src/spectral_packet_engine/`: reusable library code
- `examples/`: command-line workflows and benchmark entry points
- `tests/`: reference and regression tests
- `pyproject.toml`: packaging and extras

## Design choices

### Explicit preprocessing, not hidden cleanup

NaN filling, negative-value clipping, profile normalization, and dropping non-positive-mass shots are explicit controls in `DensityPreprocessingConfig`. The library does not silently invent a cleaned dataset.

### Profile analysis is separate from wavefunction analysis

Real detector density profiles are treated as bounded scalar fields, not as if they were full quantum states. That separation lives in `profiles.py` and avoids mixing experimental profile compression with state-vector projection logic.

### The ML layer is optional

The simulation and inverse stack stay usable without TensorFlow. The surrogate is an opt-in layer for repeated inference workloads, not a hidden runtime dependency of the physics engine.

### Runtime decisions are inspectable

Torch device selection and TensorFlow host/runtime selection are explicit. The code does not bury device choice behind opaque defaults.

## Testing

Run the standard suite with:

```bash
python -m pytest -q
```

Current result in the default workspace environment:

- `17 passed, 1 skipped`

The skipped test is the optional TensorFlow test when TensorFlow is not installed in the active interpreter.

## References

- Experimental dataset: [Zenodo record 16701012](https://doi.org/10.5281/zenodo.16701012)
- Associated paper: [Characterising transport in a quantum gas by measuring Drude weights](https://arxiv.org/abs/2508.17279)
- TensorFlow installation guide: [Install TensorFlow with pip](https://www.tensorflow.org/install/pip)
- Apple Metal plugin guide: [Get started with tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/)
