"""Inference caches must never specialize parameters or alter the training graph."""
import numpy as np
import pennylane as qml
import pytest
import torch

from qasp import IQPTNSimulator, IQPTNOptions, DecoderSpec, action_probs

GRAPHS = {
    "chain": [(i, i+1) for i in range(5)],
    "star": [(0, i) for i in range(1, 6)],
    "cycle": [(i, (i+1) % 6) for i in range(6)],
    "cubic": [(i, j) for i in range(3) for j in range(3, 6)],
    "hypergraph": [(0, 1, 2), (2, 3, 4), (1, 3, 4, 5)],
}
DECODERS = [DecoderSpec(), DecoderSpec("majority-modulo", 3, ((0, 1, 2), (2, 3, 4))),
            DecoderSpec("parity", 3), DecoderSpec("contiguous", 3)]


def program(t, x, edges, readout):
    for w in range(6):
        qml.Hadamard(w)
        qml.RZ(t[w] + x[..., w % 2] * t[-1], w)
    for j, edge in enumerate(edges):
        qml.MultiRZ(t[6+j], edge)
    for w in range(6):
        qml.Hadamard(w)
    return readout()


def make_node(device, edges, readout):
    return qml.QNode(lambda t, x: program(t, x, edges, readout), device,
                     interface="torch", diff_method="backprop")


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("decoder", DECODERS)
@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_graph_decoder_values_updates_and_gradients(graph, decoder, precision):
    dtype = torch.float32 if precision == "complex64" else torch.float64
    tol = 5e-6 if dtype == torch.float32 else 3e-11
    gen = torch.Generator().manual_seed(902)
    t = torch.zeros(16, dtype=dtype, requires_grad=True)
    x = torch.randn(4, dtype=dtype, generator=gen)[::2].requires_grad_()
    wires = [4, 1, 5, 0, 2]
    readout = lambda: action_probs(wires, decoder)
    fast = make_node(IQPTNSimulator(6, precision=precision, inference_execution="torchscript"), GRAPHS[graph], readout)
    eager = make_node(IQPTNSimulator(6, precision=precision), GRAPHS[graph], readout)
    ref = make_node(qml.device("default.qubit", wires=6), GRAPHS[graph], lambda: qml.probs(wires=wires))
    # Tracing at zero must not delete gates or freeze zero tensor values.
    with torch.no_grad():
        assert torch.equal(fast(t, x), eager(t, x))
        traces = fast.device.inference_cache_info()["traces"]
        assert traces > 0
        for _ in range(2):
            t.copy_(torch.randn(t.shape, dtype=dtype, generator=gen))
            x.add_(.17)
            actual = fast(t, x)
            assert torch.equal(actual, eager(t, x))
            torch.testing.assert_close(actual, decoder.aggregate(ref(t, x)).to(actual.dtype), atol=tol, rtol=tol)
        assert fast.device.inference_cache_info()["traces"] == traces
    info = fast.device.inference_cache_info()
    a, b = fast(t, x), eager(t, x)
    assert torch.equal(a, b)
    weights = torch.arange(decoder.n_actions, dtype=dtype)
    ga = torch.autograd.grad((a * weights).sum(), (t, x))
    gb = torch.autograd.grad((b * weights).sum(), (t, x))
    gr = torch.autograd.grad((decoder.aggregate(ref(t, x)) * weights).sum(), (t, x))
    assert all(torch.equal(u, v) for u, v in zip(ga, gb))
    for u, v in zip(ga, gr):
        torch.testing.assert_close(u, v, atol=tol, rtol=tol)
    assert fast.device.inference_cache_info() == info


def test_broadcast_noncontiguous_inputs_and_cache_modes():
    t = torch.linspace(-.5, .8, 16, dtype=torch.float64, requires_grad=True)
    d = IQPTNSimulator(6, inference_execution="torchscript")
    f = make_node(d, GRAPHS["chain"], lambda: action_probs([5, 2, 0]))
    ref = make_node(IQPTNSimulator(6), GRAPHS["chain"], lambda: action_probs([5, 2, 0]))
    with torch.no_grad():
        for shape in [(4,), (1, 4), (3, 4), (7, 4)]:
            x = torch.linspace(-.6, .7, int(np.prod(shape)), dtype=torch.float64).reshape(shape)[..., ::2]
            assert torch.equal(f(t, x), ref(t, x))
        assert d.inference_cache_info()["traces"] == 1
    with torch.inference_mode():
        assert torch.equal(f(t, x), ref(t, x))
    assert d.inference_cache_info()["traces"] == 2
    with torch.no_grad():
        assert torch.equal(f(t, x), ref(t, x))
    a, b = f(t, x), ref(t, x)
    assert torch.equal(torch.autograd.grad(a[:, 1].sum(), t)[0], torch.autograd.grad(b[:, 1].sum(), t)[0])


def test_checkpoint_buffers_optimizer_and_second_derivatives():
    class Policy(torch.nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.4, -.2], dtype=torch.float64))
            self.register_buffer("unitary", torch.eye(2, dtype=torch.float64))
            self.device = IQPTNSimulator(2, inference_execution=mode)
            def circuit(x):
                qml.QubitUnitary(self.unitary, 0)
                qml.RY(self.weight[0]*x[0], 0)
                qml.Hadamard(1)
                qml.IsingZZ(self.weight[1]*x[1], [0, 1])
                qml.RY(self.weight[0], 1)
                return action_probs([0, 1])
            self.node = qml.QNode(circuit, self.device, interface="torch", diff_method="backprop")
        def forward(self, x):
            return self.node(x)
    fast, eager = Policy("torchscript"), Policy("eager")
    x = torch.tensor([.6, -.8], dtype=torch.float64, requires_grad=True)
    opts = [torch.optim.Adam(m.parameters(), lr=.03) for m in (fast, eager)]
    for _ in range(4):
        with torch.no_grad():
            assert torch.equal(fast(x), eager(x))
        for m, opt in zip((fast, eager), opts):
            opt.zero_grad()
            m(x)[1].backward()
            opt.step()
        assert torch.equal(fast.weight, eager.weight)
    with torch.no_grad():
        eager.unitary.copy_(torch.tensor([[0., 1.], [1., 0.]]))
        eager.weight.add_(.2)
    fast.load_state_dict(eager.state_dict())
    with torch.no_grad():
        assert torch.equal(fast(x), eager(x))
    assert set(fast.state_dict()) == {"weight", "unitary"}
    hessians = []
    for m in (fast, eager):
        g = torch.autograd.grad(m(x)[1], x, create_graph=True)[0]
        hessians.append(torch.autograd.grad(g.sum(), x)[0])
    assert torch.equal(*hessians)


def test_measurement_structure_coefficients_and_precision_changes():
    d = IQPTNSimulator(["a", "b", "c"], inference_execution="torchscript")
    eager = IQPTNSimulator(["a", "b", "c"])
    for precision in ["complex128", "complex64", "complex128"]:
        d.options = IQPTNOptions(precision=precision, inference_execution="torchscript")
        eager.options = IQPTNOptions(precision=precision)
        for wires in [["a", "b"], ["b", "c"]]:
            for angle in [0., .7, -.3]:
                t = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
                ops = [qml.Hadamard(w) for w in d.wires] + [qml.IsingZZ(t, wires)] + [qml.RY(t, "a")]
                ms = [qml.probs(wires=["c", "a"]), qml.expval(t*qml.X("a") + .3*qml.Y("b")), qml.var(qml.X("a") + t*qml.Z("c"))]
                tape = qml.tape.QuantumScript(ops, ms)
                with torch.no_grad():
                    a, b = d.execute(tape), eager.execute(tape)
                    assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert d.inference_cache_info()["traces"] > 1
    d.clear_inference_cache()
    assert d.inference_cache_info()["kernel_entries"] == d.inference_cache_info()["constant_bytes"] == 0


@pytest.mark.parametrize("kind", ["samples", "counts", "action"])
def test_shots_keep_random_stream_and_dynamic_prefixes(kind):
    def circuit(t):
        qml.RY(t, 0); qml.RY(.9, 1); qml.CZ([0, 1]); qml.Hadamard(0)
        if kind == "samples": return qml.sample(wires=[1, 0])
        if kind == "counts": return qml.counts(wires=[1, 0])
        return action_probs([1, 0])
    nodes = [qml.QNode(circuit, IQPTNSimulator(2, shots=40, seed=11, inference_execution=m),
                       interface="torch", diff_method=None) for m in ["torchscript", "eager"]]
    with torch.no_grad():
        for t in [torch.tensor(.3), torch.tensor(1.3)]:
            a, b = [node(t) for node in nodes]
            assert a == b if kind == "counts" else torch.equal(a, b)


def test_eager_fallback_on_compiler_failure_and_bounded_cache(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("test compiler unavailable")
    monkeypatch.setattr(torch.jit, "trace", fail)
    t = torch.ones(16, dtype=torch.float64)
    x = torch.ones(2, dtype=torch.float64)
    d = IQPTNSimulator(6, inference_execution="torchscript", inference_cache_size=1, inference_constant_bytes=64)
    for edges in [GRAPHS["chain"], GRAPHS["star"]]:
        f = make_node(d, edges, lambda: action_probs(range(6)))
        ref = make_node(IQPTNSimulator(6), edges, lambda: action_probs(range(6)))
        with torch.no_grad(), pytest.warns(RuntimeWarning, match="test compiler unavailable"):
            assert torch.equal(f(t, x), ref(t, x))
        with torch.no_grad():
            assert torch.equal(f(t+.2, x), ref(t+.2, x))
        assert d.inference_cache_info()["kernel_entries"] == 1
        assert d.inference_cache_info()["constant_bytes"] <= 64
    assert d.inference_cache_info()["traces"] == 0


@pytest.mark.parametrize("mode", ["cached", "torchscript"])
def test_guards_and_grad_mode_bypass_caching(mode):
    t = torch.ones(16, dtype=torch.float64)
    x = torch.ones(2, dtype=torch.float64)
    for kwargs in [{"mode": "auto", "auto_max_flops": 0}, {"hard_max_flops": 0}]:
        d = IQPTNSimulator(6, inference_execution=mode, **kwargs)
        f = make_node(d, GRAPHS["chain"], lambda: action_probs(range(6)))
        with torch.no_grad():
            if "hard_max_flops" in kwargs:
                with pytest.raises(MemoryError): f(t, x)
            else:
                f(t, x)
                assert d.last_report.selected == "default.qubit"
        assert d.inference_cache_info()["traces"] == d.inference_cache_info()["constant_bytes"] == 0
    d = IQPTNSimulator(6, inference_execution=mode)
    f = make_node(d, GRAPHS["chain"], lambda: action_probs(range(6)))
    # Even untrainable inputs use eager execution while global grad mode is on.
    f(t, x)
    assert d.inference_cache_info()["constant_entries"] == 0
    with torch.no_grad():
        f(t, x)
    assert d.inference_cache_info()["constant_entries"] > 0
    assert d.inference_cache_info()["traces"] == (1 if mode == "torchscript" else 0)


def test_operand_shape_stride_and_conjugation_guards():
    import opt_einsum as oe
    from qasp.simulators._inference import InferenceCache
    cache = InferenceCache()
    cache.configure(2, 0)
    with torch.no_grad():
        for n in [2, 3]:
            a = torch.complex(torch.arange(n*n, dtype=torch.float64).reshape(n, n), torch.ones(n, n, dtype=torch.float64))
            expr = oe.contract_expression("ab,bc->ac", (n, n), (n, n))
            structure = (((n, n), ("a", "b")), ((n, n), ("b", "c")))
            for operand in [a, a.T, a.conj()]:
                expected = expr(operand, a, backend="torch").real
                for _ in range(2):
                    actual = cache.contract(expr, (operand, a), structure, ("a", "c"), "greedy")
                    assert torch.equal(actual, expected)
        assert cache.info()["traces"] == 6
        assert cache.info()["kernel_entries"] == 2


@pytest.mark.parametrize("decoder", ["majority-modulo", "parity-like", "contiguous-like"])
def test_public_policy_cache_checkpoint_and_input_gradients(decoder):
    from qasp import IQPAnsatz
    from qasp.feature_maps import NeuralEmbedding
    from qasp.rl.policies import PQCBornVariance
    kwargs = dict(n_qubits=5, n_actions=3, policy=decoder,
        majority_blocks=[[0, 1], [2, 3, 4]] if decoder == "majority-modulo" else None,
        ansatz=IQPAnsatz, ansatz_config={"n_qubits":5, "n_layers":1, "imprimitive_pattern":"chain"},
        feature_map=NeuralEmbedding, feature_map_config={"n_qubits":5, "features":2, "rotations":"Z"})
    torch.manual_seed(82)
    a = PQCBornVariance(**kwargs, device=IQPTNSimulator(5, inference_execution="torchscript"))
    b = PQCBornVariance(**kwargs, device=IQPTNSimulator(5))
    b.load_state_dict(a.state_dict())
    x = torch.tensor([.2, -.4], requires_grad=True)
    with torch.no_grad():
        assert torch.equal(a.action_probs(x), b.action_probs(x))
    for update in range(2):
        ya, yb = a.action_probs(x), b.action_probs(x)
        ga = torch.autograd.grad(ya[..., 1].sum(), (*a.qlayer.parameters(), x))
        gb = torch.autograd.grad(yb[..., 1].sum(), (*b.qlayer.parameters(), x))
        assert all(torch.equal(u, v) for u, v in zip(ga, gb))
        with torch.no_grad():
            for p in b.parameters(): p.add_(.03)
        a.load_state_dict(b.state_dict())
        with torch.no_grad():
            assert torch.equal(a.action_probs(x), b.action_probs(x))
    assert a.circuit.device.inference_cache_info()["kernel_hits"] > 0


def test_autocast_and_forward_mode_ad_use_eager_contractions():
    t = torch.linspace(-.3, .6, 16, dtype=torch.float64)
    x = torch.tensor([.3, -.7], dtype=torch.float64)
    fast = make_node(IQPTNSimulator(6, inference_execution="torchscript"), GRAPHS["chain"], lambda: action_probs(range(6)))
    eager = make_node(IQPTNSimulator(6), GRAPHS["chain"], lambda: action_probs(range(6)))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(fast(t, x), eager(t, x))
    assert fast.device.inference_cache_info()["traces"] == 0
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(t, torch.ones_like(t))
        a = torch.autograd.forward_ad.unpack_dual(fast(dual, x))
        b = torch.autograd.forward_ad.unpack_dual(eager(dual, x))
        assert torch.equal(a.primal, b.primal)
        assert a.tangent is not None and torch.equal(a.tangent, b.tangent)
    assert fast.device.inference_cache_info()["traces"] == 0


def test_runtime_compilation_failure_retains_live_values():
    t = torch.linspace(-.3, .6, 16, dtype=torch.float64)
    x = torch.tensor([.3, -.7], dtype=torch.float64)
    d = IQPTNSimulator(6, inference_execution="torchscript")
    f = make_node(d, GRAPHS["chain"], lambda: action_probs(range(6)))
    eager = make_node(IQPTNSimulator(6), GRAPHS["chain"], lambda: action_probs(range(6)))
    with torch.no_grad():
        f(t, x)
        def fail(*operands): raise RuntimeError("test cached executor unavailable")
        key = next(iter(d._inference_cache.kernels))
        d._inference_cache.kernels[key] = fail
        with pytest.warns(RuntimeWarning, match="cached executor unavailable"):
            assert torch.equal(f(t+.1, x), eager(t+.1, x))
        assert torch.equal(f(t-.2, x), eager(t-.2, x))
        assert d.inference_cache_info()["traces"] == 1
        assert d.inference_cache_info()["fallbacks"] == 2


def test_kernel_does_not_retain_example_tensors_or_tensor_constants():
    import gc
    import weakref
    import opt_einsum as oe
    from qasp.simulators._inference import InferenceCache
    cache = InferenceCache()
    cache.configure(2, 0)
    expr = oe.contract_expression("ab,bc->ac", (2, 2), (2, 2))
    structure = (((2, 2), ("a", "b")), ((2, 2), ("b", "c")))
    def execute():
        a = torch.randn(2, 2, dtype=torch.complex128)
        b = torch.randn(2, 2, dtype=torch.complex128)
        refs = [weakref.ref(a), weakref.ref(b)]
        cache.contract(expr, (a, b), structure, ("a", "c"), "greedy")
        return refs
    with torch.no_grad():
        refs = execute()
    gc.collect()
    assert all(ref() is None for ref in refs)
    kernel = next(iter(cache.kernels.values()))
    assert kernel is not None
    assert not any(node.kind() == "prim::Constant" and any(out.type().kind() == "TensorType" for out in node.outputs())
                   for node in kernel.graph.nodes())
