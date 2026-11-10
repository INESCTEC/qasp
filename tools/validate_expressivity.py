"""Independent, full-coefficient audit. No QASP tape or Fourier internals in oracle.
Run: python code/validate.py --source /path/to/qasp/source --output data
"""

import argparse
import csv
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

IDENTITY_2 = np.eye(2, dtype=complex)
P = {
    "X": np.array([[0, 1], [1, 0]], complex),
    "Y": np.array([[0, -1j], [1j, 0]], complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def local(p, wire, n):
    out = np.ones((1, 1), complex)
    for j in range(n):
        out = np.kron(out, p if j == wire else IDENTITY_2)
    return out


def cnot(control, target, n):
    out = np.zeros((2**n, 2**n), complex)
    for b in range(2**n):
        dest = b ^ (1 << (n - 1 - target)) if (b >> (n - 1 - control)) & 1 else b
        out[dest, b] = 1
    return out


def cases(n):
    for scale, upload in product([False, True], repeat=2):
        group = 1 + int(upload) + 2 * int(scale)
        for k in [1, n]:
            yield dict(
                case=f"{group}.{1 if k == 1 else 2}",
                fm="AngleEmbedding",
                k=k,
                wires=list(range(k)),
                scaled=scale,
                R=2 if upload else 1,
            )
    for k, full in product([1, 2], [False, True]):
        yield dict(
            case=f"5.{k}-" + ("all" if full else "one"),
            fm="NeuralEmbedding",
            k=k,
            wires=list(range(n)) if full else [0],
            scaled=True,
            R=1,
        )


def directions(case, scales):
    k, wires, R = case["k"], case["wires"], case["R"]
    out = []
    pos = 0
    for layer in range(R):
        rows = []
        for j in wires:
            if case["fm"] == "AngleEmbedding":
                a = [F(0)] * k
                a[j] = F(float(scales[pos])) if case["scaled"] else F(1)
                pos += int(case["scaled"])
            else:
                a = [F(float(s)) for s in scales[pos : pos + k]]
                pos += k
            rows.append(tuple(a))
        out.append(rows)
    return out


def oracle(case, n, profile, theta, scales):
    """Propagate Fourier density matrices by projector channels, never sampling x."""
    k = case["k"]
    dim = 2**n
    zero = tuple([F(0)] * k)
    keys = [zero]
    rho = np.zeros((1, dim, dim), complex)
    rho[0, 0, 0] = 1
    dirs = directions(case, scales)
    idx = 0
    for layer in range(case["R"]):
        for wire, a in zip(case["wires"], dirs[layer]):
            py = local(P["Y"], wire, n)
            qp = (np.eye(dim) + py) / 2
            qm = (np.eye(dim) - py) / 2
            channels = [qp @ rho @ qm, qp @ rho @ qp + qm @ rho @ qm, qm @ rho @ qp]
            dest = {}
            inds = []
            for sign in [1, 0, -1]:
                inds.extend(
                    dest.setdefault(
                        tuple(x + sign * y for x, y in zip(key, a)), len(dest)
                    )
                    for key in keys
                )
            expanded = np.zeros((len(dest), dim, dim), complex)
            np.add.at(expanded, np.asarray(inds), np.concatenate(channels))
            keys = list(dest)
            rho = expanded
        for axis in "Y" if profile == "commuting" else "YX":
            for wire in range(n):
                angle = theta[idx]
                idx += 1
                u = np.cos(angle / 2) * np.eye(dim) - 1j * np.sin(angle / 2) * local(
                    P[axis], wire, n
                )
                rho = u @ rho @ u.conj().T
        if profile == "entangled":
            for wire in range(n - 1):
                u = cnot(wire, wire + 1, n)
                rho = u @ rho @ u.T
    order = sorted(range(len(keys)), key=lambda j: keys[j])
    keys = [keys[j] for j in order]
    rho = rho[order]
    freq = np.array([[float(x) for x in key] for key in keys])
    bits = np.array([[(b >> (n - 1 - j)) & 1 for j in range(n)] for b in range(dim)])
    parity = bits.sum(axis=1) % 2
    majority = (bits.sum(axis=1) > n / 2).astype(int)
    obs = [
        local(P["Z"], 0, n),
        np.diag((-1.0) ** bits.sum(axis=1)),
        local(P["Y"], 0, n),
    ]
    obs.extend(np.diag((parity == a).astype(float)) for a in [0, 1])
    obs.extend(np.diag((majority == a).astype(float)) for a in [0, 1])
    coeff = np.stack([np.einsum("ij,bji->b", o, rho) for o in obs], axis=1)
    return keys, freq, coeff


def closed_form(case, n, theta, scales, keys):
    """Product-state formula for local Z, parity, local Y, parity/majority probabilities."""
    k = case["k"]
    zero = tuple([F(0)] * k)
    dirs = directions(case, scales)
    b = [zero] * n
    for row in dirs:
        for wire, a in zip(case["wires"], row):
            b[wire] = tuple(x + y for x, y in zip(b[wire], a))
    beta = np.asarray(theta).reshape(case["R"], n).sum(axis=0)
    q = []
    for wire in range(n):
        d = {}
        for sign in [-1, 1]:
            key = tuple(sign * x for x in b[wire])
            d[key] = d.get(key, 0) + np.exp(-1j * sign * beta[wire]) / 2
        q.append(d)

    def mul(a, b):
        out = {}
        for ka, va in a.items():
            for kb, vb in b.items():
                key = tuple(x + y for x, y in zip(ka, kb))
                out[key] = out.get(key, 0) + va * vb
        return out

    def scaleadd(a, scale, base):
        out = {key: scale * value for key, value in a.items()}
        out[zero] = out.get(zero, 0) + base
        return out

    parity = {zero: 1}
    for d in q:
        parity = mul(parity, d)
    maj = {}
    for bits in product([0, 1], repeat=n):
        if sum(bits) <= n / 2:
            continue
        prob = {zero: 1}
        for j, bit in enumerate(bits):
            prob = mul(prob, scaleadd(q[j], (-1) ** bit / 2, 0.5))
        for key, value in prob.items():
            maj[key] = maj.get(key, 0) + value
    maps = [
        q[0],
        parity,
        {},
        scaleadd(parity, 0.5, 0.5),
        scaleadd(parity, -0.5, 0.5),
        scaleadd(maj, -1, 1),
        maj,
    ]
    return np.array([[d.get(key, 0) for d in maps] for key in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default="data")
    ap.add_argument("--widths", default="2,3,4")
    ap.add_argument("--seeds", default="17,29,43")
    args = ap.parse_args()
    sys.path.insert(0, args.source)
    import pennylane as qml
    import torch
    from qasp import (
        Uniform,
        Normal,
        create_classifier,
        create_policy,
        expressivity_analyzer,
    )

    torch.set_default_dtype(torch.float64)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spectra").mkdir(exist_ok=True)
    rows = []
    parameter_records = []
    start = time.perf_counter()
    modules = [
        "classifier_scores",
        "classifier_parity",
        "classifier_majority",
        "policy_logits",
        "policy_parity",
        "policy_majority",
    ]
    for n, seed, profile in product(
        map(int, args.widths.split(",")),
        map(int, args.seeds.split(",")),
        ["commuting", "entangled"],
    ):
        for case in cases(n):
            fm = {
                "features": case["k"],
                "rotations": "Y"
                if len(case["wires"]) > 1 or case["fm"] == "AngleEmbedding"
                else ["Y 0"],
            }
            if case["fm"] == "AngleEmbedding":
                fm["input_scaling"] = case["scaled"]
            cfg = dict(
                feature_map=case["fm"],
                feature_map_config=fm,
                ansatz="AngleAnsatz",
                ansatz_config=dict(
                    n_layers=case["R"],
                    rotations="Y" if profile == "commuting" else "YX",
                    reuploading=case["R"] > 1,
                    include_imprimitive=profile == "entangled",
                    imprimitive="CNOT",
                    imprimitive_pattern="chain",
                ),
                init=Uniform(-np.pi, np.pi),
                device="default.qubit",
            )
            if case["scaled"]:
                cfg["input_scaling_init"] = Normal(0, 1)
            labels = ["Z 0", "Z" * n + " " + " ".join(map(str, range(n))), "Y 0"]
            cached = None
            for module in modules:
                torch.manual_seed(seed)
                config = dict(cfg)
                if module == "classifier_scores":
                    config.update(readout="expectation", observables=labels)
                    model = create_classifier(config, n_qubits=n, n_classes=3)
                    cols = [0, 1, 2]
                elif module.startswith("classifier"):
                    dec = module.split("_")[1]
                    config.update(
                        readout="probabilities",
                        decoder=dec,
                        measurement_qubits=list(range(n)),
                    )
                    if dec == "majority":
                        config["majority_blocks"] = [list(range(n))]
                    model = create_classifier(config, n_qubits=n, n_classes=2)
                    cols = [3, 4] if dec == "parity" else [5, 6]
                elif module == "policy_logits":
                    config.update(n_actions=3, observables=labels)
                    model = create_policy("PQCSoftmax", config, n_qubits=n)
                    cols = [0, 1, 2]
                else:
                    dec = module.split("_")[1]
                    config.update(
                        n_actions=2,
                        policy="parity-like" if dec == "parity" else "majority-modulo",
                        training_method="probs",
                        parity_cnot_decomposition=False,
                        measurement_qubits=list(range(n)),
                    )
                    if dec == "majority":
                        config["majority_blocks"] = [list(range(n))]
                    model = create_policy("PQCBorn", config, n_qubits=n)
                    cols = [3, 4] if dec == "parity" else [5, 6]
                theta = model.qlayer.params.detach().numpy().copy()
                scales = (
                    model.qlayer.input_scaling_params.detach().numpy().copy()
                    if case["scaled"]
                    else np.array([])
                )
                signature = (theta.tobytes(), scales.tobytes())
                if cached is None or signature != cached[0]:
                    keys, freq, expected = oracle(case, n, profile, theta, scales)
                    closed_error = 0.0
                    if profile == "commuting":
                        cf = closed_form(case, n, theta, scales, keys)
                        closed_error = float(abs(cf - expected).max())
                        assert closed_error < 2e-11
                    cached = (signature, keys, freq, expected, closed_error)
                else:
                    _, keys, freq, expected, closed_error = cached
                before = {key: v.clone() for key, v in model.state_dict().items()}
                t = time.perf_counter()
                s = expressivity_analyzer(model)
                seconds = time.perf_counter() - t
                for key, v in before.items():
                    assert torch.equal(v, model.state_dict()[key]), key
                assert s.frequencies.shape == freq.shape, (
                    case,
                    module,
                    n,
                    s.frequencies.shape,
                    freq.shape,
                )
                freq_error = float(abs(s.frequencies - freq).max())
                assert freq_error < 1e-12
                theoretical = expected[:, cols]
                error = float(abs(s.coefficients - theoretical).max())
                assert error < 2e-10, (case, module, n, error)
                active = (abs(theoretical) > s.atol).sum(axis=0)
                assert np.array_equal(active, s.active_counts)
                rng = np.random.default_rng(seed + 1000 * n)
                points = rng.uniform(-2, 2, (7, case["k"]))
                with torch.no_grad():
                    direct = np.stack(
                        [model(torch.tensor(x)).numpy().reshape(-1) for x in points]
                    )
                pred = (np.exp(-1j * points @ freq.T) @ theoretical).real
                direct_error = float(abs(pred - direct).max())
                assert direct_error < 2e-10, (case, module, n, direct_error)
                reconstruction = float(abs(s.evaluate(points) - direct).max())
                assert reconstruction < 2e-10
                lookup = {key: i for i, key in enumerate(keys)}
                symmetry = max(
                    float(
                        abs(
                            theoretical[lookup[tuple(-x for x in key)]]
                            - theoretical[i].conj()
                        ).max()
                    )
                    for i, key in enumerate(keys)
                )
                assert symmetry < 2e-10
                if len(cols) == 2:
                    target = np.array([float(all(x == 0 for x in key)) for key in keys])
                    assert abs(theoretical.sum(axis=1) - target).max() < 2e-10
                tag = f"{case['case']}_n{n}_s{seed}_{profile}_{module}"
                np.savez_compressed(
                    out / "spectra" / f"{tag}.npz",
                    frequencies=freq,
                    theory=theoretical,
                    analyzer=s.coefficients,
                    theta=theta,
                    scales=scales,
                    points=points,
                    direct=direct,
                )
                parameter_records.append(
                    dict(
                        id=tag,
                        case=case,
                        n=n,
                        seed=seed,
                        profile=profile,
                        module=module,
                        theta=theta.tolist(),
                        scales=scales.tolist(),
                    )
                )
                for col, c in enumerate(cols):
                    rows.append(
                        dict(
                            case=case["case"],
                            n=n,
                            features=case["k"],
                            seed=seed,
                            profile=profile,
                            module=module,
                            output=[
                                "local_Z",
                                "parity_Z",
                                "local_Y",
                                "parity_p0",
                                "parity_p1",
                                "majority_p0",
                                "majority_p1",
                            ][c],
                            allowed=len(freq),
                            active=int(active[col]),
                            n_evaluations=s.n_evaluations,
                            coefficient_error=error,
                            frequency_error=freq_error,
                            direct_error=direct_error,
                            reconstruction_error=reconstruction,
                            closed_form_error=closed_error,
                            symmetry_error=symmetry,
                            seconds=seconds,
                            max_magnitude=float(abs(theoretical[:, col]).max()),
                            non_dc_power=float(
                                np.sum(
                                    abs(theoretical[np.any(freq != 0, axis=1), col])
                                    ** 2
                                )
                            ),
                        )
                    )
            print(f"n={n} seed={seed} {profile} {case['case']}: passed", flush=True)
    with (out / "results.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "parameters.json").write_text(json.dumps(parameter_records, indent=2))
    summary = dict(
        models=len(parameter_records),
        output_spectra=len(rows),
        coefficient_entries=sum(r["allowed"] for r in rows),
        max_coefficient_error=max(r["coefficient_error"] for r in rows),
        max_direct_error=max(r["direct_error"] for r in rows),
        max_reconstruction_error=max(r["reconstruction_error"] for r in rows),
        max_closed_form_error=max(r["closed_form_error"] for r in rows),
        elapsed_seconds=time.perf_counter() - start,
        python=sys.version,
        pennylane=qml.__version__,
        torch=torch.__version__,
        numpy=np.__version__,
        platform=platform.platform(),
        widths=args.widths,
        seeds=args.seeds,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
