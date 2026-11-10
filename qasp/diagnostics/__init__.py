"""Trainability and Fourier expressivity diagnostics."""

from .barren_plateau import barren_plateau_diagnoser

__all__ = [
    "barren_plateau_diagnoser",
    "expressivity_analyzer",
    "ExpressivitySweep",
    "FourierSpectrum",
]


def __getattr__(name):
    if name in {"expressivity_analyzer", "ExpressivitySweep", "FourierSpectrum"}:
        from . import expressivity

        return getattr(expressivity, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
