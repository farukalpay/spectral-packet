"""Spectral Green's function and related propagators.

The retarded Green's function encodes the full single-particle response
of a quantum system.  In the spectral (Lehmann) representation it reads

    G_R(x, x'; E) = sum_n  psi_n(x) psi_n^*(x') / (E - E_n + i epsilon)

where {psi_n, E_n} are the eigenstates and eigenenergies and epsilon > 0
is an infinitesimal broadening that enforces retarded boundary conditions.

From G_R one obtains:

* The spectral function  A(x, E) = -(1/pi) Im G_R(x, x; E)
* The local density of states  LDOS(x, E) = A(x, E)
* The total density of states  DOS(E) = integral A(x, E) dx
* The time-domain propagator  K(x, x'; t) = sum_n psi_n(x) psi_n^*(x') exp(-i E_n t / hbar)

With finite broadening epsilon the Lorentzian

    delta_epsilon(E) = (epsilon / pi) / (E^2 + epsilon^2)

replaces the Dirac delta, yielding smooth, numerically stable spectra
whose width is controlled by the user.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import coerce_tensor, complex_dtype_for

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GreensResult:
    """Container for a full Green's function analysis.

    Attributes
    ----------
    energy_grid : Tensor
        Shape ``(num_E,)`` -- the energy sampling points.
    G_diagonal : Tensor
        Shape ``(num_x, num_E)`` -- complex diagonal Green's function
        G_R(x, x; E) evaluated on the spatial and energy grids.
    ldos : Tensor
        Shape ``(num_x, num_E)`` -- local density of states.
    dos : Tensor
        Shape ``(num_E,)`` -- total density of states.
    spectral_function : Tensor
        Shape ``(num_x, num_E)`` -- spectral function A(x, x; E).
    """
    energy_grid: Tensor
    G_diagonal: Tensor
    ldos: Tensor
    dos: Tensor
    spectral_function: Tensor


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def retarded_greens_function(
    x: Tensor,
    x_prime: Tensor,
    energy: Tensor,
    eigenstates: Tensor,
    eigenvalues: Tensor,
    *,
    broadening: float = 0.01,
    grid: Tensor | None = None,
) -> Tensor:
    r"""Retarded Green's function in the spectral representation.

    G_R(x, x'; E) = sum_n psi_n(x) psi_n^*(x') / (E - E_n + i epsilon)

    If *eigenstates* are provided as wavefunctions on a spatial grid
    (shape ``(num_states, num_grid_points)``), the values at x and x' are
    obtained by nearest-neighbour lookup on *grid*.  If *eigenstates*
    are expansion coefficients (no grid needed), set grid=None and pass
    the coefficient-space representations directly.

    Parameters
    ----------
    x : Tensor
        Shape ``(num_x,)`` or scalar -- position(s).
    x_prime : Tensor
        Shape ``(num_xp,)`` or scalar -- position(s).
    energy : Tensor
        Shape ``(num_E,)`` -- energy values at which to evaluate G.
    eigenstates : Tensor
        Shape ``(num_states, num_grid_points)`` -- wavefunctions on grid.
    eigenvalues : Tensor
        Shape ``(num_states,)`` -- eigenenergies E_n.
    broadening : float
        Lorentzian half-width epsilon.
    grid : Tensor or None
        Shape ``(num_grid_points,)`` -- spatial grid on which eigenstates
        are defined.  Required when eigenstates are grid-based.

    Returns
    -------
    Tensor
        Shape ``(num_x, num_xp, num_E)`` -- complex Green's function.
    """
    x = coerce_tensor(x, dtype=torch.float64)
    x_prime = coerce_tensor(x_prime, dtype=torch.float64)
    energy = coerce_tensor(energy, dtype=torch.float64)
    eigenstates = coerce_tensor(eigenstates, dtype=torch.float64)
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)

    if x.ndim == 0:
        x = x.unsqueeze(0)
    if x_prime.ndim == 0:
        x_prime = x_prime.unsqueeze(0)
    if energy.ndim == 0:
        energy = energy.unsqueeze(0)

    cdtype = complex_dtype_for(x.dtype)
    num_states = eigenvalues.shape[0]

    # Look up eigenstate values at x and x_prime via nearest-neighbour
    if grid is not None:
        grid = coerce_tensor(grid, dtype=torch.float64)
        # Nearest-neighbour index for each query point
        x_idx = torch.argmin(torch.abs(x.unsqueeze(1) - grid.unsqueeze(0)), dim=1)
        xp_idx = torch.argmin(torch.abs(x_prime.unsqueeze(1) - grid.unsqueeze(0)), dim=1)
        psi_x = eigenstates[:, x_idx]      # (num_states, num_x)
        psi_xp = eigenstates[:, xp_idx]    # (num_states, num_xp)
    else:
        # Assume eigenstates columns correspond to x and x_prime directly
        psi_x = eigenstates
        psi_xp = eigenstates

    # G_R(x, x'; E) = sum_n psi_n(x) psi_n*(x') / (E - E_n + i*eps)
    # Shape gymnastics: (num_x, num_xp, num_E)
    # psi_x:  (num_states, num_x) -> (num_states, num_x, 1, 1)
    # psi_xp: (num_states, num_xp) -> (num_states, 1, num_xp, 1)
    # denom:  (num_states, 1, 1, num_E)
    psi_x_c = psi_x.to(cdtype).unsqueeze(2).unsqueeze(3)      # (S, Nx, 1, 1)
    psi_xp_c = psi_xp.to(cdtype).unsqueeze(1).unsqueeze(3)    # (S, 1, Nxp, 1)
    E = energy.to(cdtype)
    En = eigenvalues.to(cdtype)
    denom = (E.unsqueeze(0) - En.unsqueeze(1) + 1j * broadening)  # (S, NE)
    denom = denom.unsqueeze(1).unsqueeze(2)                        # (S, 1, 1, NE)

    numerator = psi_x_c * psi_xp_c.conj()  # (S, Nx, Nxp, 1)
    G = torch.sum(numerator / denom, dim=0)  # (Nx, Nxp, NE)

    return G


def local_density_of_states(
    x_grid: Tensor,
    energy_grid: Tensor,
    eigenstates: Tensor,
    eigenvalues: Tensor,
    *,
    broadening: float = 0.01,
) -> Tensor:
    r"""Local density of states via Lorentzian-broadened delta functions.

    LDOS(x, E) = -(1/pi) Im[G_R(x, x; E + i epsilon)]
               = sum_n |psi_n(x)|^2 * (epsilon / pi) / ((E - E_n)^2 + epsilon^2)

    Parameters
    ----------
    x_grid : Tensor
        Shape ``(num_x,)`` -- spatial positions.
    energy_grid : Tensor
        Shape ``(num_E,)`` -- energy values.
    eigenstates : Tensor
        Shape ``(num_states, num_x)`` -- wavefunctions evaluated at *x_grid*.
    eigenvalues : Tensor
        Shape ``(num_states,)`` -- eigenenergies.
    broadening : float
        Lorentzian half-width epsilon.

    Returns
    -------
    Tensor
        Shape ``(num_x, num_E)`` -- LDOS.
    """
    x_grid = coerce_tensor(x_grid, dtype=torch.float64)
    energy_grid = coerce_tensor(energy_grid, dtype=torch.float64)
    eigenstates = coerce_tensor(eigenstates, dtype=torch.float64)
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)

    # |psi_n(x)|^2: shape (num_states, num_x)
    psi_sq = eigenstates ** 2

    # Lorentzian: eps/pi / ((E - E_n)^2 + eps^2)
    # energy_grid: (num_E,),  eigenvalues: (num_states,)
    # diff: (num_states, num_E)
    diff = energy_grid.unsqueeze(0) - eigenvalues.unsqueeze(1)
    lorentz = (broadening / torch.pi) / (diff ** 2 + broadening ** 2)

    # LDOS(x, E) = sum_n |psi_n(x)|^2 * lorentz_n(E)
    # psi_sq: (S, Nx),  lorentz: (S, NE)  =>  (S, Nx, 1) * (S, 1, NE)
    ldos = torch.sum(psi_sq.unsqueeze(2) * lorentz.unsqueeze(1), dim=0)
    return ldos


def spectral_function(
    x_grid: Tensor,
    energy_grid: Tensor,
    eigenstates: Tensor,
    eigenvalues: Tensor,
    *,
    broadening: float = 0.01,
) -> Tensor:
    r"""Spectral function A(x, E).

    Identical to the LDOS for systems with real-valued eigenstates:

    A(x, E) = sum_n |psi_n(x)|^2 * (epsilon / pi) / ((E - E_n)^2 + epsilon^2)

    Provided as a distinct API point for conceptual clarity: A is the
    imaginary part of the diagonal Green's function divided by -pi.
    """
    return local_density_of_states(
        x_grid, energy_grid, eigenstates, eigenvalues, broadening=broadening,
    )


def density_of_states(
    energy_grid: Tensor,
    eigenvalues: Tensor,
    *,
    broadening: float = 0.01,
) -> Tensor:
    r"""Total density of states.

    DOS(E) = sum_n delta_epsilon(E - E_n)
           = sum_n (epsilon / pi) / ((E - E_n)^2 + epsilon^2)

    Parameters
    ----------
    energy_grid : Tensor
        Shape ``(num_E,)`` -- energies at which to evaluate the DOS.
    eigenvalues : Tensor
        Shape ``(num_states,)`` -- eigenenergies.
    broadening : float
        Lorentzian half-width.

    Returns
    -------
    Tensor
        Shape ``(num_E,)`` -- DOS.
    """
    energy_grid = coerce_tensor(energy_grid, dtype=torch.float64)
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)

    diff = energy_grid.unsqueeze(0) - eigenvalues.unsqueeze(1)  # (S, NE)
    lorentz = (broadening / torch.pi) / (diff ** 2 + broadening ** 2)
    return lorentz.sum(dim=0)


def free_propagator(
    x: Tensor,
    x_prime: Tensor,
    times: Tensor,
    eigenstates: Tensor,
    eigenvalues: Tensor,
    *,
    hbar: float = 1.0,
    grid: Tensor | None = None,
) -> Tensor:
    r"""Time-domain propagator (kernel) in the spectral representation.

    K(x, x'; t) = sum_n psi_n(x) psi_n^*(x') exp(-i E_n t / hbar)

    Parameters
    ----------
    x : Tensor
        Shape ``(num_x,)`` or scalar.
    x_prime : Tensor
        Shape ``(num_xp,)`` or scalar.
    times : Tensor
        Shape ``(num_t,)`` -- time values.
    eigenstates : Tensor
        Shape ``(num_states, num_grid_points)`` -- wavefunctions on grid.
    eigenvalues : Tensor
        Shape ``(num_states,)`` -- eigenenergies.
    hbar : float
        Reduced Planck constant.
    grid : Tensor or None
        Spatial grid for nearest-neighbour lookup.

    Returns
    -------
    Tensor
        Shape ``(num_x, num_xp, num_t)`` -- complex propagator.
    """
    x = coerce_tensor(x, dtype=torch.float64)
    x_prime = coerce_tensor(x_prime, dtype=torch.float64)
    times = coerce_tensor(times, dtype=torch.float64)
    eigenstates = coerce_tensor(eigenstates, dtype=torch.float64)
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)
    cdtype = complex_dtype_for(x.dtype)

    if x.ndim == 0:
        x = x.unsqueeze(0)
    if x_prime.ndim == 0:
        x_prime = x_prime.unsqueeze(0)
    if times.ndim == 0:
        times = times.unsqueeze(0)

    # Lookup eigenstate values at query positions
    if grid is not None:
        grid = coerce_tensor(grid, dtype=torch.float64)
        x_idx = torch.argmin(torch.abs(x.unsqueeze(1) - grid.unsqueeze(0)), dim=1)
        xp_idx = torch.argmin(torch.abs(x_prime.unsqueeze(1) - grid.unsqueeze(0)), dim=1)
        psi_x = eigenstates[:, x_idx]    # (S, Nx)
        psi_xp = eigenstates[:, xp_idx]  # (S, Nxp)
    else:
        psi_x = eigenstates
        psi_xp = eigenstates

    # Phase factors: exp(-i E_n t / hbar)
    # eigenvalues: (S,),  times: (Nt,) -> phase: (S, Nt)
    phase = torch.exp(-1j * eigenvalues.to(cdtype).unsqueeze(1) * times.to(cdtype).unsqueeze(0) / hbar)

    # K(x, x'; t) = sum_n psi_n(x) psi_n*(x') exp(-i E_n t / hbar)
    psi_x_c = psi_x.to(cdtype)    # (S, Nx)
    psi_xp_c = psi_xp.to(cdtype)  # (S, Nxp)

    # Outer product: (S, Nx, 1, 1) * (S, 1, Nxp, 1) * (S, 1, 1, Nt)
    K = torch.sum(
        psi_x_c.unsqueeze(2).unsqueeze(3)
        * psi_xp_c.conj().unsqueeze(1).unsqueeze(3)
        * phase.unsqueeze(1).unsqueeze(2),
        dim=0,
    )  # (Nx, Nxp, Nt)

    return K


# ---------------------------------------------------------------------------
# Convenience analyser
# ---------------------------------------------------------------------------

def analyze_greens_function(
    basis: InfiniteWellBasis,
    *,
    num_x_points: int = 64,
    num_energy_points: int = 200,
    energy_range: tuple[float, float] | None = None,
    broadening: float = 0.01,
) -> GreensResult:
    """Full Green's function analysis for an infinite-well basis.

    Constructs the analytic eigenstates and energies from the basis,
    evaluates the LDOS, spectral function, and DOS on regular grids,
    and packages everything into a :class:`GreensResult`.

    Parameters
    ----------
    basis : InfiniteWellBasis
        Provides eigenenergies and sine-basis wavefunctions.
    num_x_points : int
        Spatial resolution.
    num_energy_points : int
        Energy resolution.
    energy_range : tuple of float or None
        ``(E_min, E_max)`` for the energy grid.  If None, spans from
        slightly below E_1 to slightly above E_{num_modes}.
    broadening : float
        Lorentzian half-width epsilon.

    Returns
    -------
    GreensResult
    """
    dtype = basis.domain.real_dtype
    device = basis.domain.device
    cdtype = complex_dtype_for(dtype)

    # Spatial grid
    x_grid = basis.domain.grid(num_x_points)

    # Eigenstates on grid: (num_modes, num_x_points)
    # basis.evaluate returns (num_x_points, num_modes), transpose it
    eigenstates = basis.evaluate(x_grid).T  # (num_modes, num_x)
    eigenvalues = basis.energies             # (num_modes,)

    # Energy grid
    if energy_range is None:
        E_min = eigenvalues[0].item() - 2.0 * broadening
        E_max = eigenvalues[-1].item() + 5.0 * broadening
    else:
        E_min, E_max = energy_range
    energy_grid = torch.linspace(E_min, E_max, num_energy_points, dtype=dtype, device=device)

    # Compute LDOS, spectral function, DOS
    ldos = local_density_of_states(x_grid, energy_grid, eigenstates, eigenvalues, broadening=broadening)
    A = spectral_function(x_grid, energy_grid, eigenstates, eigenvalues, broadening=broadening)
    dos = density_of_states(energy_grid, eigenvalues, broadening=broadening)

    # Diagonal Green's function G(x,x;E) from LDOS:
    # LDOS = -(1/pi) Im G  =>  Im G = -pi * LDOS
    # Re G comes from Kramers-Kronig, but we can compute it directly:
    # G(x,x;E) = sum_n |psi_n(x)|^2 / (E - E_n + i*eps)
    psi_sq = eigenstates ** 2  # (S, Nx)
    diff = energy_grid.unsqueeze(0) - eigenvalues.unsqueeze(1)  # (S, NE)
    denom = (diff + 1j * broadening).to(cdtype)  # (S, NE)

    # (S, Nx, 1) * (S, 1, NE)^{-1} => sum over S
    G_diag = torch.sum(
        psi_sq.to(cdtype).unsqueeze(2) / denom.unsqueeze(1),
        dim=0,
    )  # (Nx, NE)

    return GreensResult(
        energy_grid=energy_grid,
        G_diagonal=G_diag,
        ldos=ldos,
        dos=dos,
        spectral_function=A,
    )


__all__ = [
    "GreensResult",
    "analyze_greens_function",
    "density_of_states",
    "free_propagator",
    "local_density_of_states",
    "retarded_greens_function",
    "spectral_function",
]
