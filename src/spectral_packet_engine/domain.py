from __future__ import annotations

from dataclasses import dataclass, field

import torch

Tensor = torch.Tensor


def coerce_tensor(
    value,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def coerce_scalar_tensor(
    value,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    if isinstance(value, torch.Tensor):
        inferred_dtype = value.dtype if dtype is None else dtype
        inferred_device = value.device if device is None else device
        tensor = coerce_tensor(value, dtype=inferred_dtype, device=inferred_device)
    else:
        tensor = coerce_tensor(value, dtype=dtype or torch.float64, device=device)
    if tensor.numel() != 1:
        raise ValueError("expected a scalar tensor-compatible value")
    return tensor.reshape(())


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise TypeError(f"unsupported real dtype: {real_dtype}")


@dataclass(frozen=True, slots=True)
class InfiniteWell1D:
    left: Tensor
    right: Tensor
    mass: Tensor = field(default_factory=lambda: torch.tensor(1.0, dtype=torch.float64))
    hbar: Tensor = field(default_factory=lambda: torch.tensor(1.0, dtype=torch.float64))

    def __post_init__(self) -> None:
        left = coerce_scalar_tensor(self.left)
        right = coerce_scalar_tensor(self.right, dtype=left.dtype, device=left.device)
        mass = coerce_scalar_tensor(self.mass, dtype=left.dtype, device=left.device)
        hbar = coerce_scalar_tensor(self.hbar, dtype=left.dtype, device=left.device)

        if torch.is_complex(left) or torch.is_complex(right):
            raise TypeError("domain boundaries must be real-valued")
        if not torch.isfinite(left).item() or not torch.isfinite(right).item():
            raise ValueError("domain boundaries must be finite")
        if not torch.isfinite(mass).item() or not torch.isfinite(hbar).item():
            raise ValueError("physical constants must be finite")
        if not (right > left).item():
            raise ValueError("domain requires right > left")
        if not (mass > 0).item():
            raise ValueError("mass must be positive")
        if not (hbar > 0).item():
            raise ValueError("hbar must be positive")

        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "hbar", hbar)

    @property
    def device(self) -> torch.device:
        return self.left.device

    @property
    def real_dtype(self) -> torch.dtype:
        return self.left.dtype

    @property
    def complex_dtype(self) -> torch.dtype:
        return complex_dtype_for(self.real_dtype)

    @property
    def length(self) -> Tensor:
        return self.right - self.left

    @property
    def midpoint(self) -> Tensor:
        return (self.left + self.right) / 2

    def contains(self, x) -> Tensor:
        values = coerce_tensor(x, dtype=self.real_dtype, device=self.device)
        return (values >= self.left) & (values <= self.right)

    def grid(
        self,
        num_points: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if num_points < 2:
            raise ValueError("grid requires at least two points")
        return torch.linspace(
            self.left.item(),
            self.right.item(),
            steps=num_points,
            dtype=dtype or self.real_dtype,
            device=device or self.device,
        )

    @classmethod
    def from_length(
        cls,
        length,
        *,
        left=0.0,
        mass=1.0,
        hbar=1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "InfiniteWell1D":
        left_tensor = coerce_scalar_tensor(left, dtype=dtype, device=device)
        length_tensor = coerce_scalar_tensor(length, dtype=dtype, device=device)
        right_tensor = left_tensor + length_tensor
        return cls(
            left=left_tensor,
            right=right_tensor,
            mass=coerce_scalar_tensor(mass, dtype=dtype, device=device),
            hbar=coerce_scalar_tensor(hbar, dtype=dtype, device=device),
        )


InfiniteWellDomain = InfiniteWell1D


__all__ = [
    "InfiniteWell1D",
    "InfiniteWellDomain",
    "coerce_scalar_tensor",
    "coerce_tensor",
    "complex_dtype_for",
]
