# Fourier expressivity analyzer

`qasp.expressivity_analyzer` computes the Fourier series of an encoded circuit at its current weights. Pass an RL policy, Q-function, classifier, fidelity kernel, or projected kernel using QASP’s `AngleEmbedding` or `NeuralEmbedding` with `AngleAnsatz`, `IQPAnsatz`, or an encoding-only kernel.

```python
from qasp import expressivity_analyzer

s = expressivity_analyzer(model, observables=["X 0", "Z 0"])
print(s.output_labels, s.n_frequencies, s.active_counts)
for omega, c, magnitude in zip(s.frequencies, s.coefficients, s.magnitudes):
    print(omega, c, magnitude)
expectations = s.evaluate(x)  # one value per observable, also accepts batches
```

For output $a$, QASP uses $f_a(x)=\sum_{\omega\in\Omega} c_{\omega,a}\exp(-i\omega\cdot x)$. Each row represents a signed frequency vector in feature-map coordinates. The zero vector gives the constant term. For a real expectation, opposite frequencies have complex-conjugate coefficients. A one-qubit $R_Y(sx)$, for example, has $Z$ expectation $\cos(sx)$ with coefficients $1/2$ at $\pm s$. Measuring $X$ gives $\sin(sx)$, with coefficient $+i/2$ at $+s$.

**Allowed and active frequencies.** Each Pauli rotation through $w\cdot x$ contributes possible generator gaps $0,+w,-w$. The analyzer sums these choices over the rotations actually applied, including repeated uploads. Axis encodings supply scaled coordinate vectors. Neural encodings supply the full learned projection vectors. Zero scales collapse and coincident frequencies share one complex coefficient. `s.n_frequencies` counts distinct allowed vectors, including DC. `s.active_counts[a]` counts rows with $|c_{\omega,a}|$ above `atol` (default $10^{-10}$). This threshold affects reporting only. For example, the identity observable has just one active coefficient even on a rich encoding. Re-run after training to capture the new weights and scales. The generator construction follows [Sahebi et al., Eq. (7) and Appendix B.3](https://arxiv.org/abs/2505.15902v1).

The [cross-module notebook](../tutorials/tools/expressivity_analyzer.ipynb) checks each model family against direct outputs and plots complex coefficients across qubit widths.

## Model readouts

With explicit `observables`, each output is a separate expectation on the configured circuit. Accepts QASP Pauli strings or fixed Hermitian PennyLane operators. Omitting the argument selects the following readouts.

| Model | Fourier outputs |
| --- | --- |
| `PQCSoftmax` | Forward logits including output scaling and the affine head, before softmax |
| `PQCBornVariance` (factory name `PQCBorn`) | Decoded action probabilities before optional softmax. Local Pauli sampling mode returns the measured expectations. |
| Legacy `PQCBorn` / adaptive `LearnablePQCBorn` | The legacy sampler exposes its basis probabilities or expectations. The adaptive policy requires explicit fixed observables. |
| `PQDQN` | Q-values including the current affine head and bias. With the default input activation, reconstruct as `s.evaluate(np.tanh(state))`. |
| `PQCClassifier` | Grouped, scaled expectation scores or decoded Born probabilities before temperature/scale calibration |
| `FidelityKernel` | The mean of its configured global or local overlap projectors |
| `ProjectedKernel` | The labeled observable features before the Gaussian kernel |

The finite series applies to the ideal circuit and its affine readouts. Sampling noise, truncated simulation and nonlinear classical activations can change the predictions. The legacy Fisher score kernel uses discrete bit strings and has no continuous input embedding.

For a fidelity kernel, the default expansion varies both $x$ and $y$. A frequency row contains $(\omega_x,\omega_y)$ multiplying $\exp[-i(\omega_x\cdot x+\omega_y\cdot y)]$. For the paper’s convention $\exp[i(\omega\cdot x-\nu\cdot y)]$, that row corresponds to $\omega=-\omega_x$ and $\nu=\omega_y$. Set `reference=y0` to obtain a smaller single-input spectrum for a fixed comparison point.

```python
joint = expressivity_analyzer(fidelity_kernel)
k_xy = joint.evaluate(x, y)[0]

slice_at_y = expressivity_analyzer(fidelity_kernel, reference=y)
k_xy_from_slice = slice_at_y.evaluate(x)[0]

# Reconstruct the full projected kernel from the finite feature spectra.
import numpy as np
features = expressivity_analyzer(projected_kernel)
delta = features.evaluate(x) - features.evaluate(y)
k_xy_projected = np.exp(-projected_kernel.gamma * np.sum(projected_kernel.weights * delta**2))
```

## Width and observable sweeps

Supply a factory that builds or loads the intended model at each width. To isolate the effect of width, keep the input features and weight preparation comparable. An observable factory can choose the corresponding operator on each register.

```python
sweep = expressivity_analyzer(
    model_factory=make_classifier,
    qubit_list=[2, 4, 6, 8],
    observables=lambda n: ["Z 0", "Z" * n],
)
print(sweep.frequency_counts)       # aligned with qubit_list
print(sweep.summary())             # counts, peak magnitude and non-DC power per readout
s4 = sweep.results_by_qubit[4]     # retains every frequency, coefficient and magnitude
```

The [runnable plotting example](../examples/expressivity_sweep.py) encodes one scalar as $R_Y(x)$ on every qubit. The local $Z$ signal stays $\cos x$ at every width. The full parity signal becomes $\cos^n x$. At eight qubits the encoding allows 17 signed frequencies, local $Z$ activates two, and parity activates nine. The parity coefficient at $\omega=8$ is $1/256$.

## Computational cost

The analyzer traces the encoding with the current scale tensors, then evaluates an ideal statevector circuit on a grid of independent *gate phases*. A direction used $r$ times needs $2r+1$ grid points. An inverse discrete Fourier transform yields its harmonics, which are mapped back to input-frequency vectors and combined. This construction works for noninteger and incommensurate weights without assuming a period in the physical input.

The product of the grid sizes can grow exponentially. `n_evaluations` reports the grid size used. The default `max_evaluations=65_536` applies per width and can be raised explicitly. The implementation also bounds statevector batches, total circuit wires (21, including ancillas) and the output array (four million phase-grid/output entries). A fixed reference can reduce the cost of a fidelity spectrum.

## Independent theoretical validation

`tools/validate_expressivity.py` compares complete complex spectra with a separate spectral-projector density-matrix calculation. It covers unscaled/scaled AngleEmbedding with one or n features and one/two uploads, plus NeuralEmbedding with one/two features on one/all wires. Both commuting and entangled circuits are tested through classifier scores, policy logits, and classifier/policy parity and strict-majority probabilities. The commuting cases are also checked against closed-form coefficient convolutions.

Run from this source checkout:

```sh
python tools/validate_expressivity.py --source . --output /tmp/qasp-fourier-validation
```

The default matrix uses widths 2, 3, 4 and seeds 17, 29, 43. It writes full spectra, actual parameters, direct evaluations, and summary tables. The regression wrapper in `tests/test_expressivity_reference.py` runs the same matrix at widths 2 and 3 with one seed. These are finite configuration checks, not a proof covering every circuit the library can construct.

## Before and after training

The [executed training workflow](../tutorials/tools/expressivity_training.ipynb)
compares a classifier, a softmax policy, and a Born policy. Save the initial
spectrum, train the same model, and recompute:

```python
initial = expressivity_analyzer(model)
# Run your existing classifier or policy training loop here.
final = expressivity_analyzer(model)
print(initial.active_counts, final.active_counts)
```

Each result is a snapshot. It does not change when the model is trained later.
The analyzer preserves model weights, existing gradients, and training mode.
For fixed unscaled encodings the allowed frequencies stay fixed, although their
coefficients can change. Learned input scales and neural projections can change
frequency locations too. Compare predictions, coefficient magnitudes and non-DC
power as well as active counts; a larger count is not a guarantee of better
learning. Softmax policy spectra reconstruct logits, not the nonlinear softmax;
apply softmax to reconstructed logits if action probabilities are needed.

The regression matrix trains binary classifiers (expectation and probability
readouts), softmax and Born policies, with Angle/Neural embeddings, fixed/learned
scales, Angle/IQP ansätze, two initializations, and multiple qubit widths. Policy
tests optimize exact expected reward on a small contextual bandit. These are
short correctness experiments, not evidence of convergence on a benchmark.
Each snapshot is checked against direct outputs at independent inputs, with
complex conjugate symmetry and snapshot/state-preservation checks. A structured
width sweep through 16 qubits additionally checks every coefficient against
cos(x) and cos(x)^n. Nine CPU execution settings are compared where supported;
VN uses exact enumeration in this regression matrix.

Analysis itself uses an ideal statevector circuit, independently of the model's
execution backend. Consequently it does not inherit a simulator's large-qubit
scaling or model noise, sampling error, or Pauli truncation. General independent
input directions can make the Fourier grid exponential, even when a structured
shared-input sweep is inexpensive. Guardrails still apply.

To rerun the training matrix and retain the complex spectra:

```sh
PYTHONPATH=.:tests QASP_EXPRESSIVITY_RESULTS=validation/expressivity/training \
  python -m pytest tests/test_expressivity_training.py -q
```

Each training case writes before/after frequency vectors and complex coefficients
in an `.npz` file, together with inputs, reconstructed predictions and losses.
Adjacent JSON files summarize active counts and non-DC power. For a combined
installed-wheel audit, build/install the candidate first and run
`QASP_VALIDATION_OUTPUT=validation/expressivity/installed python tools/validate_installed_configuration.py`.
