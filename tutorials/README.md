# QASP tutorials

Start with the circuit builder, then choose the model you want to train. Each notebook runs on CPU using the published QASP 0.1.2 API. The application notebooks use small datasets or short training runs so you can inspect the inputs, weights, model output and circuit together.

| Topic | Notebook | What you will do |
| --- | --- | --- |
| Building blocks | [Build, execute and train a circuit](building_blocks/circuit_builder.ipynb) | Pass features, input scales and angles to a QNode, then initialize a TorchLayer and optimize a basis probability. |
| Classification | [Binary classification](classifiers/binary_classification.ipynb) | Fit a neural-embedding classifier with minibatches and inspect probabilities, hinge loss and a confusion matrix. |
| Quantum kernels | [Train a fidelity kernel](qkernels/fidelity_kernel.ipynb) | Optimize centered target alignment, then compute training and test Gram matrices. |
| Reinforcement learning | [Quantum policy gradients](rl/policy_gradient.ipynb) | Train a CartPole policy with REINFORCE and inspect its action distribution. |
| Reinforcement learning | [Quantum Q-learning](rl/quantum_q_learning.ipynb) | Train an online Q-function with replay and a target network, then compare action values. |
| Generative learning | [Fit a Born machine](genlearn/born_machine.ipynb) | Learn a bitstring distribution with MMD and compare initial, fitted and target probabilities. |
| Optimization | [MaxCut with QAOA](optimization/maxcut_qaoa.ipynb) | Initialize cost and mixer angles, optimize the expected energy and check a candidate against the exact discrete optimum. |
| Tools | [Barren plateau diagnoser](tools/barren_plateau_diagnoser.ipynb) | Sweep seven compiled circuit widths from 4 to 16 qubits and compare measured parity variance with the analytic curve. |
| Simulators | [Differentiable simulation](simulators/differentiable_simulation.ipynb) | Compare IQP tensor contraction, sampling and auto selection with exact Pauli propagation and a full-state reference. Check gradients and train with both exact backends. |

## Run a tutorial

From this repository, create an environment with Python 3.12 or 3.13 and launch JupyterLab:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "qasp==0.1.2" jupyterlab
python -m jupyterlab tutorials
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. Open a notebook in JupyterLab and choose **Run → Run All Cells**. In VS Code, select the same environment as the notebook kernel. Run each notebook from its own directory.

## Change the starting weights

Edit the initializer objects directly in a model's Python configuration, then rerun its construction and training cells. For example, `"init": Normal(0, 0.1)` draws ansatz angles with standard deviation 0.1. In an encoded model, `"input_scaling_init": Uniform(0.8, 1.2)` initializes feature scales or neural projection weights. For a scaled AngleEmbedding, enable `input_scaling=True`. The low-level builder uses `initialization=` in `create_torchlayer()`. QAOA uses only angle initialization.

## View and export a model

Each notebook displays the Matplotlib circuit first. Pass standalone Quantikz source to `show_tikz(tex)` for an inline preview. Rendering requires pdfLaTeX with the `standalone` and `quantikz` packages plus Ghostscript. The helper compiles in a temporary directory and deletes the build files automatically. The inline image is part of the Jupyter cell output; clear that output before saving if you want a smaller notebook.

To keep a figure, use `builder.save_tikz(..., standalone=True)` with the same inputs and weights, or write a module's returned source explicitly with `Path("circuit.tex").write_text(tex, encoding="utf-8")`. Compile that file with `pdflatex circuit.tex` for a vector PDF.

The repository tracks notebooks with cleared outputs. **Run All Cells** regenerates the printed results and figures. For more configurations and API signatures, see the [interactive documentation](https://andre-sequeira10.github.io/QASP/) and the [package guide](../docs/package_guide.md).
