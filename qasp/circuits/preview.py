"""Optional inline previews for QASP model drawings and Quantikz source."""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import warnings


def _source(target, args, draw_kwargs, tikz_options):
    if isinstance(target, str):
        if args or draw_kwargs or tikz_options:
            raise TypeError("Drawing arguments apply to circuits and models, not rendered source.")
        if r"\documentclass" in target:
            return target
        if r"\begin{quantikz}" not in target:
            raise ValueError("Expected standalone LaTeX or a Quantikz fragment.")
        return "\n".join((r"\documentclass[tikz,border=4pt]{standalone}",
                          r"\usepackage{xcolor,quantikz}", r"\usetikzlibrary{fit}",
                          r"\begin{document}", target, r"\end{document}"))

    from pennylane import QNode
    from pennylane.tape import QuantumScript
    from .constructor import CircuitBuilder
    from .tikz import circuit_to_tikz

    options = dict(tikz_options or {}, standalone=True)
    if isinstance(target, (CircuitBuilder, QNode, QuantumScript)):
        return circuit_to_tikz(target, *args, **{**draw_kwargs, **options})
    if args:
        raise TypeError("Pass model drawing arguments by keyword, e.g. inputs=x or x=x, y=y.")

    from ..qkernels import FidelityKernel, ProjectedKernel, QFKernel, draw_kernel
    if isinstance(target, QFKernel):
        if draw_kwargs:
            raise TypeError("A Fisher kernel uses its stored circuit parameters; pass only tikz_options.")
        if not hasattr(target, "circuit"):
            raise ValueError("Pass the original QNode and parameters for this older Fisher kernel.")
        return circuit_to_tikz(target.circuit, target.params, **options)
    if isinstance(target, (FidelityKernel, ProjectedKernel)):
        return draw_kernel(target, output="tikz", tikz_options=options, **draw_kwargs)
    # Classifiers, Q-functions, Born machines and optimization models expose
    # a draw() method that delegates to the appropriate public drawing helper.
    if callable(getattr(target, "draw", None)):
        return target.draw(output="tikz", tikz_options=options, **draw_kwargs)

    from ..rl.policies import LearnablePQCBorn, PQCBorn, PQCBornVariance, PQCSoftmax
    if isinstance(target, (PQCBorn, PQCBornVariance, PQCSoftmax, LearnablePQCBorn)):
        from ..rl.model import draw_model
        if draw_kwargs.get("inputs") is None:
            fm = getattr(target.circuit_builder.ansatz, "feature_map", None)
            if fm is not None:
                draw_kwargs["inputs"] = target.qlayer.params.new_zeros(fm.features)
        return draw_model(model=target, output="tikz", tikz_options=options, **draw_kwargs)
    raise TypeError("Pass a drawable QASP model, CircuitBuilder, QNode, tape, or Quantikz source.")


def show_tikz(target, *args, name="Quantum circuit", tikz_options=None, **draw_kwargs):
    """Preview a model, builder, QNode, tape or Quantikz string in a notebook.

    Model arguments follow their drawing helper (e.g. inputs=x, x=x and y=y
    for fidelity kernels, or parameters=result for QAOA). Builder/QNode
    arguments follow the circuit signature. Pass export styles in tikz_options.
    name sets the image description. With source text it can also be positional.

    Uses IPython, pdfLaTeX with standalone and Quantikz, and Ghostscript only
    when called. A missing preview dependency emits a warning with setup
    instructions. Build files are temporary. Render trusted LaTeX source only.
    """
    if isinstance(target, str) and len(args) == 1 and isinstance(args[0], str):
        name, args = args[0], ()

    pdflatex = shutil.which("pdflatex")
    gs = shutil.which("gs") or shutil.which("gswin64c")
    missing = []
    if not pdflatex:
        missing.append("Install a TeX distribution (TeX Live, MacTeX or MiKTeX) with "
                       "pdfLaTeX, standalone and quantikz; put pdflatex on PATH. See https://tug.org/texlive/.")
    if not gs:
        missing.append("Install Ghostscript (https://ghostscript.com/releases/gsdnld.html) "
                       "and put gs or gswin64c on PATH.")
    try:
        from IPython.display import Image, display
    except ImportError:
        missing.append('Install IPython in this kernel with %pip install "qasp[preview]".')
    if missing:
        warnings.warn("TikZ preview unavailable. " + " ".join(missing), stacklevel=2)
        return

    source = _source(target, args, draw_kwargs, tikz_options)
    with TemporaryDirectory() as folder:
        (Path(folder) / "circuit.tex").write_text(source, encoding="utf-8")
        latex = subprocess.run(
            [pdflatex, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
             "circuit.tex"], cwd=folder, capture_output=True, text=True, errors="replace",
             timeout=120,
        )
        if latex.returncode:
            for filename in ("quantikz.sty", "standalone.cls"):
                if f"{filename}' not found" in latex.stdout:
                    warnings.warn(f"TikZ preview requires {filename}. Install the "
                                  f"{filename.split('.')[0]} package with your TeX "
                                  "distribution's package manager.", stacklevel=2)
                    return
            raise RuntimeError("pdfLaTeX could not render the circuit:\n" + latex.stdout[-2000:])
        png = subprocess.check_output(
            [gs, "-q", "-dSAFER", "-sDEVICE=png16m", "-r120", "-o", "-", "circuit.pdf"],
            cwd=folder, timeout=120,
        )
    display(Image(data=png, alt=name))
