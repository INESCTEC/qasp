"""Optional inline previews for QASP model drawings and Quantikz source."""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import warnings


def _standalone(source):
    if not isinstance(source, str):
        raise TypeError("show_tikz expects LaTeX source from a circuit drawing helper.")
    if r"\documentclass" in source:
        return source
    if r"\begin{quantikz}" not in source:
        raise ValueError("Expected standalone LaTeX or a Quantikz fragment.")
    return "\n".join((r"\documentclass[tikz,border=4pt]{standalone}",
                      r"\usepackage{xcolor,quantikz}", r"\usetikzlibrary{fit}",
                      r"\begin{document}", source, r"\end{document}"))


def _in_notebook():
    # A running IPython kernel has already imported IPython. Plain export
    # scripts can return source without importing the optional display stack.
    import sys
    ipython = sys.modules.get("IPython")
    return ipython is not None and getattr(ipython.get_ipython(), "kernel", None) is not None


def _tikz_result(source, return_tex):
    """Preview the exported source once in Jupyter, returning it on request."""
    if _in_notebook():
        try:
            show_tikz(source)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            warnings.warn(f"TikZ preview failed: {exc}", stacklevel=3)
    return source if return_tex else None


def show_tikz(source: str, name: str = "Quantum circuit") -> None:
    """Display trusted Quantikz or standalone LaTeX in Jupyter using temporary files.

    Missing IPython, TeX or Ghostscript produces optional installation guidance.
    The name is the image description. Module drawing helpers call this function
    automatically in Jupyter when the selected output format is TikZ.
    """
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

    source = _standalone(source)
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
