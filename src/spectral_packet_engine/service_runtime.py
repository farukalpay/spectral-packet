from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import warnings


@dataclass(frozen=True, slots=True)
class APIStackRuntime:
    installed: bool
    compatible: bool
    fastapi_version: str | None
    starlette_version: str | None
    pydantic_version: str | None
    uvicorn_version: str | None
    error: str | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def inspect_api_stack() -> APIStackRuntime:
    fastapi_installed = importlib.util.find_spec("fastapi") is not None
    uvicorn_installed = importlib.util.find_spec("uvicorn") is not None
    installed = fastapi_installed and uvicorn_installed

    fastapi_version = _package_version("fastapi")
    starlette_version = _package_version("starlette")
    pydantic_version = _package_version("pydantic")
    uvicorn_version = _package_version("uvicorn")

    notes: list[str] = []
    if not fastapi_installed:
        notes.append("Install the 'api' extra to add the HTTP API surface.")
    if fastapi_installed and not uvicorn_installed:
        notes.append("FastAPI is installed, but Uvicorn is required to serve the API.")
    if not installed:
        return APIStackRuntime(
            installed=False,
            compatible=False,
            fastapi_version=fastapi_version,
            starlette_version=starlette_version,
            pydantic_version=pydantic_version,
            uvicorn_version=uvicorn_version,
            error=None,
            notes=tuple(notes),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from fastapi import FastAPI
        except ModuleNotFoundError:
            return APIStackRuntime(
                installed=False,
                compatible=False,
                fastapi_version=fastapi_version,
                starlette_version=starlette_version,
                pydantic_version=pydantic_version,
                uvicorn_version=uvicorn_version,
                error="FastAPI could not be imported.",
                notes=("Reinstall the 'api' extra in a clean environment.",),
            )
        try:
            FastAPI()
        except Exception as exc:
            return APIStackRuntime(
                installed=True,
                compatible=False,
                fastapi_version=fastapi_version,
                starlette_version=starlette_version,
                pydantic_version=pydantic_version,
                uvicorn_version=uvicorn_version,
                error=str(exc),
                notes=(
                    "FastAPI is installed, but the local FastAPI/Starlette/Pydantic stack is incompatible.",
                    "Reinstall the 'api' extra in a clean environment before serving the HTTP API.",
                ),
            )

    return APIStackRuntime(
        installed=True,
        compatible=True,
        fastapi_version=fastapi_version,
        starlette_version=starlette_version,
        pydantic_version=pydantic_version,
        uvicorn_version=uvicorn_version,
        error=None,
        notes=("The HTTP API stack is installed and compatible.",),
    )


__all__ = [
    "APIStackRuntime",
    "inspect_api_stack",
]
