"""Worked Pauli propagation, Torch training, and approximation checks.

Run from the repository root: python examples/pauli_propagation_basics.py
Each marked section is also used in the documentation guide.
"""

# BEGIN builder
import torch
from qasp import AngleAnsatz, CircuitBuilder

ansatz = AngleAnsatz(
    2, n_layers=1, rotations=["Y 0"],
    include_imprimitive=True, imprimitive="CNOT",
    imprimitive_pattern=[(0, 1)], include_final_imprimitive=True,
)
builder = CircuitBuilder(2)
builder.set_ansatz(ansatz)
builder.set_measurement("expectation", observables=["Z 1"])
builder.build()
node = builder.compile(
    device="PauliPropagationSimulator",
    diff_method="backprop", shots=None,
)
theta = torch.tensor([0.37], dtype=torch.float64, requires_grad=True)
value = node(params=theta)[0]
gradient = torch.autograd.grad(value, theta)[0]
torch.testing.assert_close(value, torch.cos(theta[0]))
torch.testing.assert_close(gradient, -torch.sin(theta))
print("Exact value and gradient:", value.item(), gradient.item())
# END builder

# BEGIN cutoff
truncated = builder.compile(
    device="PauliPropagationSimulator",
    diff_method="backprop", shots=None,
    device_kwargs={"k": 1, "prune": False},
)
approximate = truncated(params=theta)[0]
approximate_gradient = torch.autograd.grad(approximate, theta)[0]
print("Weight-one result:", approximate.item(), approximate_gradient.item())
# Both are zero: the backward CNOT produces Z0 Z1, which k=1 removes.
torch.testing.assert_close(approximate, torch.zeros_like(approximate))
torch.testing.assert_close(approximate_gradient, torch.zeros_like(theta))
# END cutoff

# BEGIN policy
from qasp.initialization import Normal
from qasp.rl import create_policy

execution = {
    "device": "PauliPropagationSimulator",
    "device_kwargs": {"k": None, "prune": False},
    "diff_method": "backprop", "shots": None,
}
config = {
    "n_actions": 2, "observables": ["Z 0", "ZZ 0 1"], "fc_layer": None,
    "feature_map": "NeuralEmbedding",
    "feature_map_config": {"features": 2, "rotations": ["Y 0", "YY 0 1"]},
    "ansatz": "AngleAnsatz",
    "ansatz_config": {
        "n_layers": 2, "rotations": "ZY", "reuploading": True,
        "include_imprimitive": True, "imprimitive": "CNOT",
        "imprimitive_pattern": "chain",
    },
    "init": Normal(0.0, 0.3), "input_scaling_init": Normal(0.0, 0.5),
    **execution,
}
policy = create_policy("PQCSoftmax", config, n_qubits=2).double()
x = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
optimizer.zero_grad()
loss = -torch.log_softmax(policy(x), dim=-1)[1]
loss.backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all()
           for p in policy.parameters())
optimizer.step()
print("One action-loss update:", loss.item())
# END policy

# BEGIN comparison
from copy import deepcopy

def make_reference(device, options):
    cfg = deepcopy(config)
    cfg.update(device=device, device_kwargs=options)
    model = create_policy("PQCSoftmax", cfg, n_qubits=2).double()
    model.load_state_dict(policy.state_dict())  # Identical trained parameters.
    return model

def evaluate_with_gradients(model):
    features = x.detach().clone().requires_grad_(True)
    logits = model(features)
    loss = -torch.log_softmax(logits, dim=-1)[1]
    grads = torch.autograd.grad(loss, (*model.parameters(), features))
    return logits.detach(), torch.cat([g.detach().reshape(-1) for g in grads])

reference = make_reference("default.qubit", {})
ref_value, ref_grad = evaluate_with_gradients(reference)
settings = {
    "exact": {"k": None, "prune": False},
    "weight 1": {"k": 1, "prune": False},
    "weight 2": {"k": 2, "prune": False},
    "pruned": {"prune": True, "tau": 1e-4, "topM": 8, "prune_every": 1},
}
for name, options in settings.items():
    model = make_reference("PauliPropagationSimulator", options)
    result, grads = evaluate_with_gradients(model)
    value_error = (result - ref_value).abs().max().item()
    gradient_error = (grads - ref_grad).abs().max().item()
    print(f"{name:>8}: max output error={value_error:.3e}, "
          f"max loss-gradient error={gradient_error:.3e}")
    if name in {"exact", "weight 2"}:
        torch.testing.assert_close(result, ref_value, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(grads, ref_grad, atol=1e-10, rtol=1e-10)
# END comparison
