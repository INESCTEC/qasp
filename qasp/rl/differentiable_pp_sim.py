"""Compatibility import; the shared engine now lives in qasp.simulators."""
from qasp.simulators._pauli_propagation import DifferentiablePauliProp

__all__ = ["DifferentiablePauliProp"]
