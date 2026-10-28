"""Differentiable PennyLane NumPy losses on full computational-basis probability vectors."""

import pennylane.numpy as np


def nll_loss(probs: np.ndarray, batch_idx: np.ndarray) -> float:
    """Return differentiable mean negative log-likelihood for observed basis indices.

    probs has shape (2**n,); batch_idx contains integer basis indices and may
    repeat entries. The implementation adds 1e-12 inside the logarithm; it does
    not normalize or validate probs. Returns a scalar compatible with PennyLane
    NumPy/Autograd, not necessarily a detached Python float."""
    # Add small epsilon for numerical stability
    eps = 1e-12
    selected_probs = probs[batch_idx]
    return -np.mean(np.log(selected_probs + eps))


def kl_divergence_loss(probs: np.ndarray, target_probs: np.ndarray) -> float:
    """Return the differentiable smoothed sum for KL(target_probs || probs).

    Both arrays must have the same probability-vector shape. Adds 1e-12 to
    logarithm arguments and performs no normalization. Unlike nll_loss, this
    function expects a target distribution, not sample indices; adapt its
    signature before using it as a BornMachine training callback."""
    eps = 1e-12
    return np.sum(target_probs * (np.log(target_probs + eps) - np.log(probs + eps)))

# TODO: Add more losses (e.g., moment matching, MMD, etc.)
