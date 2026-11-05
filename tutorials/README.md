# QASP tutorials

Start with the circuit builder, then choose the model you want to train. Each notebook runs on CPU. The inline TikZ examples use QASP 0.1.3. The application notebooks use small datasets or short training runs so you can inspect the inputs, weights, model output and circuit together.

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
python -m pip install "qasp[preview]==0.1.3" jupyterlab
python -m jupyterlab tutorials
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. Open a notebook in JupyterLab and choose **Run → Run All Cells**. In VS Code, select the same environment as the notebook kernel. Run each notebook from its own directory.

## Change the starting weights

Edit the initializer objects directly in a model's Python configuration, then rerun its construction and training cells. For example, `"init": Normal(0, 0.1)` draws ansatz angles with standard deviation 0.1. In an encoded model, `"input_scaling_init": Uniform(0.8, 1.2)` initializes feature scales or neural projection weights. For a scaled AngleEmbedding, enable `input_scaling=True`. The low-level builder uses `initialization=` in `create_torchlayer()`. QAOA uses only angle initialization.

## View and export a model

Each notebook displays the Matplotlib circuit first. For a model, request an inline TikZ figure directly from its drawing helper:

```python
draw_classifier(model, inputs=X_test[0], output="tikz", return_tex=False)
```

To display the same figure and also keep its LaTeX in memory, use `tex = draw_classifier(model, inputs=X_test[0], output="tikz")`. With a low-level circuit builder, call `tex = builder.to_tikz(..., standalone=True)` and pass it to `show_tikz(tex)` imported from `qasp`.

Inline previews require pdfLaTeX with the `standalone` and `quantikz` packages plus Ghostscript. QASP renders the image in a temporary directory and cleans up afterward. Jupyter stores the displayed image in the cell output when you save the notebook. To keep a paper-ready file separately, call `builder.save_tikz(..., standalone=True)` or write the module’s standalone source to a `.tex` file and compile it with pdfLaTeX.

The repository tracks notebooks with cleared outputs. **Run All Cells** regenerates the printed results and figures. For more configurations and API signatures, see the [interactive documentation](https://andre-sequeira10.github.io/QASP/) and the [package guide](../docs/package_guide.md).
