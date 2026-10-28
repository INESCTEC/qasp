"""Born-distribution learning with PennyLane NumPy optimizers and Fisher updates.

This experimental module predates the Torch model APIs. See BornMachine.train
for current limitations of its default-loss and natural-gradient branches.
Sampling returns basis indices rather than arrays of bits."""

import pennylane as qml
from pennylane import numpy as np
from pennylane import metric_tensor
from pennylane.gradients import classical_fisher, quantum_fisher

def _bits_to_int(bits: np.ndarray) -> int:
    """Legacy decimal-digit conversion of a binary-looking array to an integer.

    This helper does not use base two: [1, 0] yields 10. BornMachine's actual
    training-index conversion uses base two and does not call this helper."""
    return int("".join(str(int(b)) for b in bits))


class BornMachine:
    """Experimental optimizer for a parameterized computational-basis distribution.

    Accepts a QNode or CircuitBuilder returning a probability vector over
    2**n_qubits outcomes, initial PennyLane NumPy parameters, and binary data
    of shape (N, n_qubits). Classical optimizer updates are available with an
    explicit loss_fn. Natural-gradient code is retained but has a known
    keyword mismatch in post-update loss evaluation; see train. This module
    should not be described as a fully validated end-to-end trainer."""

    def __init__(
        self,
        circuit: None,
        init_params: np.ndarray,
        data,
        *,
        optimizer=None,
        lr: float = 0.1,
        batch_size: int = None,
        natural_grad: bool = False,
        commuting: bool = False,
        fisher: str = "quantum",
        lambda_reg: float = 1e-2,
        n_qubits: int = None,
        device: str = "default.qubit",
        diff_method: str = "best",
        mpi: bool = False,
        device_kwargs: dict = None,
    ):
        """Store binary data, parameters and classical or Fisher-update settings.

        ``circuit`` is required: a QNode is reused; otherwise its compile method
        receives device/diff_method/mpi/device_kwargs. Existing QNodes cannot
        receive new device_kwargs. Training indices treat the leftmost bit as
        most significant. ``batch_size=None`` means full-batch; lr defaults to
        0.1. The classical default is PennyLane Adam.

        For natural_grad, fisher must be quantum or classical. lambda_reg adds
        a diagonal regularizer. commuting=True computes the Fisher at initialization
        and reuses it; the caller is responsible for that approximation. n_qubits
        is needed for entropy calculations. Parameter and data objects are retained."""
        if circuit is None:
            raise ValueError("circuit must be provided.")

        elif type(circuit) is qml.QNode:   
            if mpi:
                from qasp.devices import create_device

                create_device(circuit.device, wires=circuit.device.wires, mpi=True)
            if device_kwargs:
                raise ValueError("device_kwargs cannot configure an existing QNode.")
            self.circuit = circuit
        else:    
            self.circuit = circuit.compile(
                device=device,
                diff_method=diff_method,
                mpi=mpi,
                device_kwargs=device_kwargs,
            )
        self.params = init_params
        self.lr = lr
        self.batch_size = batch_size
        self.natural_grad = natural_grad
        self.commuting = commuting
        self.fisher = fisher
        self.lambda_reg = lambda_reg
        self.data = data
        self.n_qubits = n_qubits
        # prepare training indices
        self.indices = np.array([int("".join(map(str, x)), 2) for x in self.data])

        # set up classical optimizer (used if not natural_grad)
        if not natural_grad:
            self.opt = optimizer or qml.AdamOptimizer(stepsize=lr)

        # if using natural gradient, optionally precompute Fisher
        if natural_grad:
            if fisher not in {"quantum","classical"}:
                raise ValueError("fisher must be 'quantum' or 'classical'.")
            if commuting:
                if fisher == "quantum":
                    F = quantum_fisher(self.circuit)(self.params)
                    self.Freg = F + lambda_reg * np.eye(F.shape[0])
                    #self._chol = np.linalg.cholesky(Freg)
                else:
                    F_cl = classical_fisher(self.circuit)(self.params)
                    self.Freg = F_cl + lambda_reg * np.eye(F_cl.shape[0])
                    #self._chol = np.linalg.cholesky(Freg)


    def train(self, steps: int, loss_fn=None, print_every: int = 0):
        """Update parameters for steps iterations and return (params, loss_history).

        ``loss_fn(probs, batch_indices)`` must return a differentiable scalar.
        Batches are sampled without replacement; None or a size >= data length
        uses all rows. print_every enables periodic console output.

        Limitations:
            Supply an explicit loss, e.g. qasp.genlearn.losses.nll_loss: the default
            branch uses an unqualified legacy import. The natural-gradient branch
            currently calls its local loss with an incompatible keyword after its
            first update and is not a working training path without a code fix.
            Classical step_and_cost records the pre-update loss. This documentation
            records those implementation boundaries rather than changing behavior."""
        if loss_fn is None:
            
            from genlearn.losses import nll_loss
            loss_fn = nll_loss

        losses = []
        N = len(self.indices)

        for s in range(steps):
            # select batch
            if self.batch_size is None or self.batch_size >= N:
                batch = self.indices
            else:
                idx = np.random.choice(N, self.batch_size, replace=False)
                batch = self.indices[idx]

            # classical optimizer
            if not self.natural_grad:
                def cost_fn(p):
                    """Evaluate the supplied probability loss on the current batch for a candidate parameter vector."""
                    probs_p = self.circuit(params=p)
                    return loss_fn(probs_p, batch)

                self.params, loss_val = self.opt.step_and_cost(
                    cost_fn, self.params
                )

            # natural gradient optimizer
            else:
                def loss_callable(p):
                    """Evaluate the natural-gradient objective for positional parameter argument p."""
                    probs_p = self.circuit(params=p)
                    return loss_fn(probs_p, batch)

                # gradient of loss
                grad_fn = qml.grad(lambda p: loss_callable(p))
                grad = grad_fn(self.params).flatten()

                # compute or reuse Fisher factorization
                if not self.commuting:
                    if self.fisher == "quantum":
                        F = quantum_fisher(self.circuit)(params=self.params)
                    else:
                        F = classical_fisher(self.circuit)(params=self.params)
                    self.Freg = F + self.lambda_reg * np.eye(F.shape[0])
                    #chol = np.linalg.cholesky(Freg)
                #else:
                    #chol = self._chol

                # nat grad step: solve Freg @ v = grad
                v = np.linalg.solve(self.Freg, grad)
                #v = np.linalg.solve(chol.T, y)
                self.params = self.params - self.lr * v.reshape(self.params.shape)
                loss_val = float(loss_callable(params=self.params))

            losses.append(float(loss_val))

            if print_every and (s + 1) % print_every == 0:
                print(f"Step {s + 1}/{steps}: loss = {loss_val:.4f}")


        return self.params, losses

    def get_probs(self) -> np.ndarray:
        """Return the current output probabilities of the Born machine."""
        return self.circuit(params=self.params)

    def sample(self, shots: int = 1000) -> np.ndarray:
        """Draw shots computational-basis indices from the current distribution.

        Returns an integer array of shape (shots,), not a bit-string matrix.
        Uses PennyLane NumPy's random generator and does not execute shot sampling
        on the quantum device."""
        probs = self.get_probs()
        states = np.arange(len(probs))
        sampled = np.random.choice(states, size=shots, p=probs)
        n_qubits = int(np.log2(len(probs)))
        return sampled
    

    def shannon_entropy(self) -> tuple[float, float]:
        """
        Compute the Shannon entropy (in nats) of the training data and of the current model distribution.

        Returns:
            (data_entropy, model_entropy)
        """
        # data entropy
        counts = np.bincount(self.indices, minlength=2**self.n_qubits)
        p_data = counts / counts.sum()
        p_data = np.clip(p_data, 1e-12, None)
        H_data = -np.sum(p_data * np.log(p_data))

        # model entropy
        probs_model = self.get_probs()
        p_mod = np.clip(probs_model, 1e-12, None)
        H_model = -np.sum(p_mod * np.log(p_mod))

        return float(H_data), float(H_model)

