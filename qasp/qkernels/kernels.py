"""Fixed-parameter Fisher score kernels over computational-basis outcomes.

The probability Jacobian enumerates all outcomes: storage grows exponentially
with the number of measured wires. This is not a continuous-feature kernel.
"""
import numpy as np
import pennylane as qml


def _array(value):
    """Detach a numerical snapshot from its autodiff interface."""
    return np.asarray(qml.math.toarray(value), dtype=float)


class QFKernel:
    """Score kernel s(x).T @ (F + lambda_reg I)^-1 @ s(y).

    circuit(params) must return one analytic probability vector. Parameters
    may be Torch tensors or PennyLane NumPy arrays. Classical Fisher uses the
    measured distribution; quantum Fisher uses PennyLane's metric transform,
    requiring a compatible pure-state circuit/device and sometimes an auxiliary
    wire. No simulator is silently replaced.

    This is a fixed numerical snapshot; rebuild after changing parameters.
    Gram matrices do not retain a training graph. Zero-probability outcomes
    have no defined score and cannot be queried.
    """
    def __init__(self, circuit: qml.QNode, params, *, fisher='quantum', lambda_reg=1e-2):
        if fisher not in {'classical', 'quantum'}:
            raise ValueError("fisher must be 'classical' or 'quantum'.")
        if not np.isfinite(lambda_reg) or lambda_reg < 0:
            raise ValueError('lambda_reg must be finite and non-negative.')
        if circuit.device.shots:
            raise ValueError('QFKernel requires analytic probabilities (shots=None).')
        if qml.math.get_interface(params) != 'torch':
            # Plain NumPy arrays/lists do not carry Autograd's trainable flag.
            params = qml.numpy.array(params, requires_grad=True)
        self.circuit, self.fisher = circuit, fisher
        self.params = params.detach().clone() if qml.math.get_interface(params) == 'torch' else np.array(params, copy=True)
        self.probabilities = _array(circuit(params))
        p = self.probabilities
        if (p.ndim != 1 or len(p) < 2 or len(p) & (len(p)-1)
                or not np.isfinite(p).all() or (p < -1e-12).any()
                or not np.isclose(p.sum(), 1, atol=1e-8)):
            raise ValueError('circuit must return one finite, normalized probability vector.')
        self.n_bits = len(p).bit_length() - 1
        if qml.math.get_interface(params) == 'torch':
            import torch
            jac = torch.autograd.functional.jacobian(circuit, params)
        else:
            jac = qml.jacobian(circuit)(params)
        jac = _array(jac).reshape(len(p), -1)
        if jac.shape[1] == 0 or not np.isfinite(jac).all():
            raise ValueError('The circuit must have finite derivatives with respect to params.')
        self.scores = np.divide(jac, p[:, None], out=np.zeros_like(jac), where=p[:, None] > 0)
        if fisher == 'classical':
            fim = jac.T @ self.scores
        else:
            # Expand templates before the metric transform while retaining
            # classical preprocessing derivatives, including input scales.
            metric_node = qml.transforms.decompose(circuit, gate_set={
                'RX', 'RY', 'RZ', 'CNOT', 'CZ', 'Hadamard', 'PauliX', 'PauliY', 'PauliZ', 'Identity',
            })
            fim = _array(qml.gradients.quantum_fisher(metric_node)(params))
            fim = fim.reshape(jac.shape[1], jac.shape[1])
        self.Freg = (fim + fim.T)/2 + float(lambda_reg)*np.eye(jac.shape[1])
        if not np.isfinite(self.Freg).all():
            raise ValueError('Fisher information contains non-finite values.')
        try:
            self._dual_scores = np.linalg.solve(self.Freg, self.scores.T)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Fisher information is singular; increase lambda_reg.') from exc

    def _indices(self, values):
        bits = np.asarray(values)
        if bits.ndim == 1:
            bits = bits.reshape(1, -1)
        if bits.ndim != 2 or bits.shape[1] != self.n_bits or not np.isin(bits, [0, 1]).all():
            raise ValueError(f'Kernel inputs must be binary rows with {self.n_bits} bits.')
        indices = bits.astype(np.int64) @ (1 << np.arange(self.n_bits-1, -1, -1))
        if np.any(self.probabilities[indices] <= 0):
            raise ValueError('The score is undefined for a zero-probability outcome.')
        return indices

    def gram(self, X, Y=None, normalize: bool = True):
        """Return the outcome Gram matrix; normalize=True scales nonzero diagonals to one.

        Zero-score rows/columns remain zero. Use normalize=False for raw Fisher
        inner products. Older releases ignored the normalization argument.
        """
        ix = self._indices(X)
        iy = ix if Y is None else self._indices(Y)
        kernel = self.scores[ix] @ self._dual_scores[:, iy]
        if normalize:
            nx = np.einsum('ij,ji->i', self.scores[ix], self._dual_scores[:, ix])
            ny = np.einsum('ij,ji->i', self.scores[iy], self._dual_scores[:, iy])
            scale = np.sqrt(np.maximum(nx, 0)[:, None] * np.maximum(ny, 0)[None, :])
            kernel = np.divide(kernel, scale, out=np.zeros_like(kernel), where=scale > 0)
        return kernel

    __call__ = gram

    def draw(self, style='pennylane', decimals=2):
        """Draw the probability circuit at the kernel's reference parameters."""
        return qml.draw_mpl(self.circuit, style=style, decimals=decimals)(self.params)
