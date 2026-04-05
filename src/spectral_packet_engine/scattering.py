"""Transfer matrix method for 1D quantum scattering through piecewise-constant potentials.

The transfer matrix formalism connects plane-wave amplitudes across a
sequence of constant-potential segments.  In each region j with potential
V_j the wavefunction takes the form:

    psi_j(x) = A_j exp(ik_j x) + B_j exp(-ik_j x)   for E > V_j
    psi_j(x) = A_j exp(kappa_j x) + B_j exp(-kappa_j x)  for E < V_j

where k_j = sqrt(2m(E - V_j)) / hbar  and  kappa_j = sqrt(2m(V_j - E)) / hbar.

At each interface we impose continuity of psi and psi'.  The interface
matrix connecting amplitudes across a boundary from region j to region j+1
and the propagation matrix across a region combine to give the segment
transfer matrix.

The total transfer matrix M relates the amplitudes in the first and last
regions:

    [A_1]     [A_N]
    [   ] = M [   ]
    [B_1]     [B_N]

For a right-incident wave (A_N=1, B_1=0 after solving) the transmission
and reflection coefficients are:

    t = 1 / M_11          T = |t|^2 (times k_N/k_1 for different k)
    r = M_21 / M_11       R = |r|^2

The S-matrix provides an equivalent unitary description.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor

_CDTYPE = torch.complex128
_RDTYPE = torch.float64
_THRESHOLD = 1e-12


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PotentialSegment:
    """A single constant-potential region.

    Attributes
    ----------
    left : float
        Left boundary of the segment.
    right : float
        Right boundary of the segment.
    height : float
        Potential energy V in this region.
    """

    left: float
    right: float
    height: float


@dataclass(frozen=True, slots=True)
class TransferMatrixResult:
    """Result of a full transfer matrix calculation.

    Attributes
    ----------
    M : Tensor
        Shape ``(2, 2)`` -- complex total transfer matrix relating
        plane-wave amplitudes in the first region to those in the last.
    segment_matrices : list[Tensor]
        Individual 2x2 transfer matrices for each segment.
    transmission : Tensor
        Scalar -- transmission coefficient T = (k_N / k_1) / |M_11|^2.
    reflection : Tensor
        Scalar -- reflection coefficient R = |M_21 / M_11|^2.
    unitarity_error : Tensor
        Scalar -- |R + T - 1|, a measure of numerical accuracy.
    """

    M: Tensor
    segment_matrices: list[Tensor]
    transmission: Tensor
    reflection: Tensor
    unitarity_error: Tensor


@dataclass(frozen=True, slots=True)
class ScatteringResult:
    """Energy-resolved scattering spectrum.

    Attributes
    ----------
    energies : Tensor
        Shape ``(num_E,)`` -- energy grid.
    transmission : Tensor
        Shape ``(num_E,)`` -- T(E).
    reflection : Tensor
        Shape ``(num_E,)`` -- R(E).
    resonance_energies : Tensor
        Energies where T has local maxima close to 1.
    resonance_widths : Tensor
        Full width at half maximum (FWHM) of each resonance peak.
    """

    energies: Tensor
    transmission: Tensor
    reflection: Tensor
    resonance_energies: Tensor
    resonance_widths: Tensor


@dataclass(frozen=True, slots=True)
class SMatrixResult:
    """S-matrix from a transfer matrix.

    Attributes
    ----------
    S : Tensor
        Shape ``(2, 2)`` -- complex S-matrix.
    eigenphases : Tensor
        Eigenvalues of S (should lie on the unit circle).
    unitarity_error : Tensor
        Scalar -- ||S^dag S - I||_F, a measure of unitarity.
    """

    S: Tensor
    eigenphases: Tensor
    unitarity_error: Tensor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wavevector(energy: float, V: float, mass: float, hbar: float) -> complex:
    """Compute k = sqrt(2m(E-V)) / hbar.  Complex for E < V."""
    diff = energy - V
    prefactor = 2.0 * mass / (hbar * hbar)
    if diff >= 0:
        return cmath.sqrt(prefactor * diff)
    return 1j * cmath.sqrt(prefactor * (-diff)).real


def _interface_matrix(k_left: complex, k_right: complex) -> Tensor:
    """Interface transfer matrix from region with k_left to region with k_right.

    Continuity of psi and psi' at a boundary gives:

        D_left [A_L, B_L]^T = D_right [A_R, B_R]^T

    where D_k = [[1, 1], [ik, -ik]], so the interface matrix is:

        M_interface = D_left^{-1} D_right
                    = (1/2) [[1+k_R/k_L, 1-k_R/k_L],
                             [1-k_R/k_L, 1+k_R/k_L]]

    Parameters
    ----------
    k_left : complex
        Wavevector in the left region.
    k_right : complex
        Wavevector in the right region.

    Returns
    -------
    Tensor
        Shape ``(2, 2)`` -- complex interface matrix.
    """
    if abs(k_left) < _THRESHOLD:
        # Degenerate case: treat as identity
        return torch.eye(2, dtype=_CDTYPE)

    ratio = k_right / k_left
    M = torch.zeros(2, 2, dtype=_CDTYPE)
    M[0, 0] = (1.0 + ratio) / 2.0
    M[0, 1] = (1.0 - ratio) / 2.0
    M[1, 0] = (1.0 - ratio) / 2.0
    M[1, 1] = (1.0 + ratio) / 2.0
    return M


def _propagation_matrix(k: complex, d: float) -> Tensor:
    """Free propagation matrix across a region of width d with wavevector k.

    P = [[exp(ikd), 0], [0, exp(-ikd)]]

    Parameters
    ----------
    k : complex
        Wavevector in the region.
    d : float
        Width of the region.

    Returns
    -------
    Tensor
        Shape ``(2, 2)`` -- complex propagation matrix.
    """
    phase = cmath.exp(1j * k * d)
    M = torch.zeros(2, 2, dtype=_CDTYPE)
    M[0, 0] = phase
    M[1, 1] = 1.0 / phase  # exp(-ikd)
    return M


# ---------------------------------------------------------------------------
# Single-segment transfer matrix
# ---------------------------------------------------------------------------

def segment_transfer_matrix(
    energy: float,
    segment: PotentialSegment,
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Tensor:
    """Transfer matrix for a single constant-potential segment.

    Combines the propagation across the segment with the interface
    matching at both boundaries.  In the context of the full potential
    structure, the interfaces are handled at the boundaries between
    adjacent segments.

    For a standalone segment embedded in free space (V=0), this function
    returns the propagation matrix P = diag(exp(ikd), exp(-ikd)) where
    k depends on the energy and potential height.

    Parameters
    ----------
    energy : float
        Particle kinetic energy E.
    segment : PotentialSegment
        The potential region.
    mass : float
        Particle mass (default 1.0, natural units).
    hbar : float
        Reduced Planck constant (default 1.0, natural units).

    Returns
    -------
    Tensor
        Shape ``(2, 2)`` -- complex transfer matrix (propagation only).
    """
    d = segment.right - segment.left
    if d < 0:
        raise ValueError("segment width must be non-negative (right >= left)")

    k = _wavevector(energy, segment.height, mass, hbar)
    return _propagation_matrix(k, d)


# ---------------------------------------------------------------------------
# Total transfer matrix
# ---------------------------------------------------------------------------

def total_transfer_matrix(
    energy: float,
    segments: list[PotentialSegment],
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> TransferMatrixResult:
    """Compute the total transfer matrix for a sequence of segments.

    The total matrix is built by alternating interface matrices (matching
    wavevectors at each boundary) with propagation matrices (phase
    accumulation across each region):

        M_total = I_01 * P_0 * I_12 * P_1 * ... * I_{N-1,N} * P_N

    Here I_{j,j+1} is the interface matrix from segment j to j+1 and P_j
    is the propagation matrix across segment j.  Since we have a sequence
    of segments with potentially different potentials, every boundary
    between adjacent segments produces an interface matrix.

    The convention is M_total maps amplitudes in the last segment to
    amplitudes in the first segment:
        [A_first, B_first]^T = M_total [A_last, B_last]^T

    For right-to-left incidence (particle coming from the right with
    A_last = 1) and no reflection on the left (B_first = 0 after solving):
        t = 1/M_11,  r = M_21/M_11

    Transmission probability:
        T = (k_last / k_first) * |t|^2 = (k_last / k_first) / |M_11|^2

    Parameters
    ----------
    energy : float
        Particle kinetic energy.
    segments : list[PotentialSegment]
        Ordered sequence of constant-potential regions.
    mass : float
        Particle mass.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    TransferMatrixResult
    """
    if not segments:
        raise ValueError("at least one segment is required")

    segment_matrices: list[Tensor] = []
    wavevectors: list[complex] = []
    for seg in segments:
        wavevectors.append(_wavevector(energy, seg.height, mass, hbar))

    M_total = torch.eye(2, dtype=_CDTYPE)

    for j in range(len(segments)):
        # Interface matrix from previous segment to this one
        if j > 0:
            I_mat = _interface_matrix(wavevectors[j - 1], wavevectors[j])
            M_total = M_total @ I_mat

        # Propagation matrix across this segment
        d_j = segments[j].right - segments[j].left
        P_j = _propagation_matrix(wavevectors[j], d_j)
        segment_matrices.append(P_j)
        M_total = M_total @ P_j

    M11 = M_total[0, 0]
    M21 = M_total[1, 0]

    M11_abs2 = (M11 * M11.conj()).real

    # Transmission: T = (k_last / k_first) / |M_11|^2
    # For segments starting and ending in free space, k_first = k_last
    k_first = wavevectors[0]
    k_last = wavevectors[-1]

    # Current ratio (real for propagating waves in asymptotic regions)
    if abs(k_first) > _THRESHOLD:
        k_ratio = (k_last / k_first).real
    else:
        k_ratio = 1.0

    safe_M11_abs2 = M11_abs2 if M11_abs2.item() > 0 else torch.tensor(1.0, dtype=_RDTYPE)
    T = torch.tensor(abs(k_ratio), dtype=_RDTYPE) / safe_M11_abs2
    R = (M21 * M21.conj()).real / safe_M11_abs2
    unitarity_err = torch.abs(R + T - 1.0)

    return TransferMatrixResult(
        M=M_total,
        segment_matrices=segment_matrices,
        transmission=T,
        reflection=R,
        unitarity_error=unitarity_err,
    )


# ---------------------------------------------------------------------------
# Scattering spectrum
# ---------------------------------------------------------------------------

def scattering_spectrum(
    segments: list[PotentialSegment],
    *,
    energy_min: float,
    energy_max: float,
    num_energies: int = 500,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ScatteringResult:
    """Compute T(E) and R(E) over an energy range and detect resonances.

    Resonances are identified as local maxima of T(E) where T > 0.5.
    The FWHM of each resonance is estimated by interpolation.

    Parameters
    ----------
    segments : list[PotentialSegment]
        The potential structure.
    energy_min : float
        Lower bound of the energy scan (must be positive).
    energy_max : float
        Upper bound of the energy scan.
    num_energies : int
        Number of energy sample points.
    mass : float
        Particle mass.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    ScatteringResult
    """
    if energy_min <= 0:
        raise ValueError("energy_min must be positive")
    if energy_max <= energy_min:
        raise ValueError("energy_max must exceed energy_min")
    if num_energies < 3:
        raise ValueError("num_energies must be at least 3")

    energies = torch.linspace(energy_min, energy_max, num_energies, dtype=_RDTYPE)
    T_vals = torch.zeros(num_energies, dtype=_RDTYPE)
    R_vals = torch.zeros(num_energies, dtype=_RDTYPE)

    for i in range(num_energies):
        E_i = energies[i].item()
        result = total_transfer_matrix(E_i, segments, mass=mass, hbar=hbar)
        T_vals[i] = result.transmission
        R_vals[i] = result.reflection

    # Clamp to physical range [0, 1]
    T_vals = T_vals.clamp(0.0, 1.0)
    R_vals = R_vals.clamp(0.0, 1.0)

    # Detect resonances: local maxima of T where T > 0.5
    resonance_indices: list[int] = []
    for i in range(1, num_energies - 1):
        if (
            T_vals[i] > T_vals[i - 1]
            and T_vals[i] > T_vals[i + 1]
            and T_vals[i].item() > 0.5
        ):
            resonance_indices.append(i)

    if resonance_indices:
        res_energies = energies[resonance_indices]
        # Estimate FWHM for each resonance
        dE = (energy_max - energy_min) / (num_energies - 1)
        widths_list: list[float] = []
        for idx in resonance_indices:
            T_peak = T_vals[idx].item()
            half_max = T_peak / 2.0
            # Search left
            left_idx = idx
            while left_idx > 0 and T_vals[left_idx].item() > half_max:
                left_idx -= 1
            # Search right
            right_idx = idx
            while right_idx < num_energies - 1 and T_vals[right_idx].item() > half_max:
                right_idx += 1
            width = (right_idx - left_idx) * dE
            widths_list.append(width)
        res_widths = torch.tensor(widths_list, dtype=_RDTYPE)
    else:
        res_energies = torch.zeros(0, dtype=_RDTYPE)
        res_widths = torch.zeros(0, dtype=_RDTYPE)

    return ScatteringResult(
        energies=energies,
        transmission=T_vals,
        reflection=R_vals,
        resonance_energies=res_energies,
        resonance_widths=res_widths,
    )


# ---------------------------------------------------------------------------
# S-matrix
# ---------------------------------------------------------------------------

def s_matrix_from_transfer(M: Tensor) -> SMatrixResult:
    """Convert a 2x2 transfer matrix to the scattering S-matrix.

    The transfer matrix maps right-region amplitudes to left-region
    amplitudes:

        [A_L]     [A_R]
        [   ] = M [   ]
        [B_L]     [B_R]

    The S-matrix relates incoming to outgoing amplitudes:

        [B_L]     [A_L]       [r,  t']   [A_L]
        [   ] = S [   ]   S = [        ] [    ]
        [A_R]     [B_R]       [t,  r']   [B_R]

    Conversion:
        t  = 1 / M_11
        r  = M_21 / M_11
        t' = det(M) / M_11
        r' = -M_12 / M_11

    Parameters
    ----------
    M : Tensor
        Shape ``(2, 2)`` -- complex transfer matrix.

    Returns
    -------
    SMatrixResult
    """
    M = coerce_tensor(M, dtype=_CDTYPE)
    if M.shape != (2, 2):
        raise ValueError("transfer matrix must be (2, 2)")

    M11 = M[0, 0]
    M12 = M[0, 1]
    M21 = M[1, 0]
    M22 = M[1, 1]
    det_M = M11 * M22 - M12 * M21

    t = 1.0 / M11
    r = M21 / M11
    t_prime = det_M / M11
    r_prime = -M12 / M11

    S = torch.zeros(2, 2, dtype=_CDTYPE)
    S[0, 0] = r
    S[0, 1] = t_prime
    S[1, 0] = t
    S[1, 1] = r_prime

    # Eigenvalues of S (should lie on the unit circle for unitary S)
    eigenphases = torch.linalg.eigvals(S)

    # Unitarity check: ||S^dag S - I||_F
    S_dag_S = S.conj().T @ S
    identity = torch.eye(2, dtype=_CDTYPE)
    unitarity_err = torch.linalg.norm(S_dag_S - identity, ord="fro")

    return SMatrixResult(
        S=S,
        eigenphases=eigenphases,
        unitarity_error=unitarity_err.real,
    )


# ---------------------------------------------------------------------------
# Convenience potential builders
# ---------------------------------------------------------------------------

def rectangular_barrier(
    height: float,
    width: float,
    *,
    center: float = 0.5,
) -> list[PotentialSegment]:
    """Create a rectangular potential barrier.

    Parameters
    ----------
    height : float
        Barrier height V.
    width : float
        Barrier width.
    center : float
        Centre position of the barrier.

    Returns
    -------
    list[PotentialSegment]
        Three segments: free | barrier | free, each extending 5*width from
        the barrier edges to provide enough asymptotic region.
    """
    half = width / 2.0
    pad = max(5.0 * width, 1.0)
    return [
        PotentialSegment(left=center - half - pad, right=center - half, height=0.0),
        PotentialSegment(left=center - half, right=center + half, height=height),
        PotentialSegment(left=center + half, right=center + half + pad, height=0.0),
    ]


def double_barrier(
    height: float,
    width: float,
    separation: float,
    *,
    center: float = 0.5,
) -> list[PotentialSegment]:
    """Create a double-barrier (Fabry-Perot resonator) structure.

    Layout:  free | barrier | well | barrier | free

    Parameters
    ----------
    height : float
        Barrier height V.
    width : float
        Width of each individual barrier.
    separation : float
        Distance between the inner edges of the two barriers (well width).
    center : float
        Centre position of the overall structure.

    Returns
    -------
    list[PotentialSegment]
    """
    half_total = (2.0 * width + separation) / 2.0
    left_barrier_left = center - half_total
    left_barrier_right = left_barrier_left + width
    well_right = left_barrier_right + separation
    right_barrier_right = well_right + width
    pad = max(5.0 * (2.0 * width + separation), 1.0)

    return [
        PotentialSegment(
            left=left_barrier_left - pad,
            right=left_barrier_left,
            height=0.0,
        ),
        PotentialSegment(
            left=left_barrier_left,
            right=left_barrier_right,
            height=height,
        ),
        PotentialSegment(
            left=left_barrier_right,
            right=well_right,
            height=0.0,
        ),
        PotentialSegment(
            left=well_right,
            right=right_barrier_right,
            height=height,
        ),
        PotentialSegment(
            left=right_barrier_right,
            right=right_barrier_right + pad,
            height=0.0,
        ),
    ]


def potential_step(
    height: float,
    *,
    position: float = 0.5,
) -> list[PotentialSegment]:
    """Create a potential step.

    Parameters
    ----------
    height : float
        Step height V.
    position : float
        Position of the step discontinuity.

    Returns
    -------
    list[PotentialSegment]
        Two segments: free region (V=0) on the left, raised region (V=height)
        on the right.
    """
    pad = 5.0
    return [
        PotentialSegment(left=position - pad, right=position, height=0.0),
        PotentialSegment(left=position, right=position + pad, height=height),
    ]


__all__ = [
    "PotentialSegment",
    "TransferMatrixResult",
    "ScatteringResult",
    "SMatrixResult",
    "segment_transfer_matrix",
    "total_transfer_matrix",
    "scattering_spectrum",
    "s_matrix_from_transfer",
    "rectangular_barrier",
    "double_barrier",
    "potential_step",
]
