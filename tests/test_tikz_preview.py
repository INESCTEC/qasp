"""Preview stays optional and removes all intermediate render files."""
from pathlib import Path
import sys
import tempfile

import pytest

from qasp import show_tikz
from qasp.circuits import preview

FRAGMENT = r"\begin{quantikz} & \gate{H} & \qw \end{quantikz}"


def test_missing_dependencies_are_actionable_and_skip_rendering(monkeypatch):
    monkeypatch.setattr(preview.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    with pytest.warns(UserWarning, match="pdfLaTeX.*Ghostscript.*qasp.*preview"):
        assert show_tikz(FRAGMENT) is None


def test_fragment_and_standalone_source():
    document = preview._standalone(FRAGMENT)
    assert document.startswith(r"\documentclass")
    assert FRAGMENT in document
    assert preview._standalone(document) == document
    with pytest.raises(ValueError, match="Quantikz"):
        preview._standalone("hello")


@pytest.mark.parametrize("missing", ["quantikz.sty", "standalone.cls"])
def test_missing_tex_package_gives_install_hint_and_cleans_up(monkeypatch, tmp_path, missing):
    from types import SimpleNamespace, ModuleType
    fake_display = ModuleType("IPython.display")
    fake_display.Image = fake_display.display = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(preview.shutil, "which", lambda name: name)
    def fail_latex(*args, **kwargs):
        assert (Path(kwargs["cwd"]) / "circuit.tex").is_file()
        return SimpleNamespace(returncode=1, stdout=f"! LaTeX Error: File `{missing}' not found.")
    monkeypatch.setattr(preview.subprocess, "run", fail_latex)
    with pytest.warns(UserWarning, match=f"Install the {missing.split('.')[0]} package"):
        show_tikz(FRAGMENT)
    assert not list(tmp_path.iterdir())


def test_actual_preview_renders_once_without_saving(monkeypatch, tmp_path):
    ipy = pytest.importorskip("IPython.display")
    if not preview.shutil.which("pdflatex") or not (preview.shutil.which("gs") or preview.shutil.which("gswin64c")):
        pytest.skip("Optional TeX and Ghostscript executables required")
    from shutil import which
    if not which("kpsewhich") or any(preview.subprocess.run(
        ["kpsewhich", file], capture_output=True).returncode for file in ("standalone.cls", "quantikz.sty")
    ):
        pytest.skip("Optional TeX packages required")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    images = []
    monkeypatch.setattr(ipy, "display", images.append)
    show_tikz(FRAGMENT, "Hadamard")
    assert len(images) == 1 and images[0].data.startswith(b"\x89PNG")
    assert images[0].alt == "Hadamard"
    width_px = int.from_bytes(images[0].data[16:20], "big")
    assert images[0].width == round(width_px * 120 / 300)
    assert images[0].height is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["classifier", "policy", "qfunction", "fidelity", "projected", "born", "qaoa"])
def test_model_tikz_previews_source_and_returns_it_on_request(kind, monkeypatch):
    import torch
    from qasp import QAOA, QUBO, create_classifier, create_kernel, create_policy, create_qcbm, create_qfunction

    config = {"feature_map": "AngleEmbedding", "feature_map_config": {"features": 2, "rotations": "Y"},
              "ansatz": "AngleAnsatz", "ansatz_config": {"n_layers": 1, "rotations": "Y"}, "device": "default.qubit"}
    x = torch.tensor([0.3, 0.7])
    kwargs = {"inputs": x}
    if kind == "classifier":
        model = create_classifier(config, n_qubits=2, n_classes=2)
    elif kind == "policy":
        model = create_policy("PQCSoftmax", {**config, "observables": ["Z 0", "Z 1"], "n_actions": 2}, n_qubits=2)
    elif kind == "qfunction":
        model = create_qfunction({**config, "observables": ["Z 0", "Z 1"]}, n_qubits=2, n_actions=2)
    elif kind in ("fidelity", "projected"):
        model = create_kernel({**config, "kernel_type": kind}, n_qubits=2)
        kwargs = {"x": x, "y": -x} if kind == "fidelity" else {"x": x}
    elif kind == "born":
        model = create_qcbm({**config, "inputs": x.tolist()}, n_qubits=2, data=[[0, 0], [1, 1]])
        kwargs = {}
    else:
        model = QAOA(QUBO.maxcut(2, [(0, 1)]))
        kwargs = {"parameters": model.initial_parameters()}
    from qasp import draw_classifier, draw_kernel, draw_model, draw_qcbm, draw_optimization
    figures = []
    monkeypatch.setattr(preview, "_in_notebook", lambda: True)
    monkeypatch.setattr(preview, "show_tikz", figures.append)
    if kind in ("policy", "qfunction"):
        draw, kwargs = draw_model, dict(kwargs, model=model)
    elif kind in ("fidelity", "projected"):
        draw, kwargs = draw_kernel, dict(kwargs, kernel=model)
    elif kind == "classifier":
        draw, kwargs = draw_classifier, dict(kwargs, classifier=model)
    elif kind == "born":
        draw, kwargs = draw_qcbm, dict(kwargs, model=model)
    else:
        draw, kwargs = draw_optimization, dict(kwargs, model=model)
    kwargs.update(output="tikz", tikz_options={"wire_labels": {0: "testwire"},
                                               "symbolic_parameters": True})
    tex = draw(**kwargs)
    assert r"\begin{quantikz}" in tex and "testwire" in tex
    assert r"\documentclass" not in tex  # Preserve the requested fragment format.
    assert figures == [tex]
    assert draw(**kwargs, return_tex=False) is None
    assert figures == [tex, tex]


def test_script_exports_source_without_running_a_renderer(monkeypatch):
    monkeypatch.setattr(preview, "_in_notebook", lambda: False)
    monkeypatch.setattr(preview, "show_tikz", lambda *a: pytest.fail("unexpected render"))
    assert preview._tikz_result(FRAGMENT, True) == FRAGMENT


def test_render_failure_preserves_returned_source(monkeypatch):
    def fail(source):
        raise RuntimeError("invalid LaTeX")
    monkeypatch.setattr(preview, "_in_notebook", lambda: True)
    monkeypatch.setattr(preview, "show_tikz", fail)
    with pytest.warns(UserWarning, match="TikZ preview failed"):
        assert preview._tikz_result(FRAGMENT, True) == FRAGMENT
