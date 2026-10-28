"""Regenerate the README's circuit figures from its actual Python examples.

Run from a checkout with QASP installed::

    python examples/render_readme_circuits.py
    python examples/render_readme_circuits.py --source-only

Only Python fences immediately following ``<!-- readme-circuit: NAME -->``
are executed. Fences with the same NAME share a namespace. Training and
dataset-dependent examples are unmarked and therefore are not executed.
The README is executable repository code; review changes before running it.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = re.compile(
    r"<!-- readme-circuit: ([a-z_]+) -->\s*```python\n(.*?)\n```", re.DOTALL
)


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def run(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command[0]}\n{result.stdout[-8000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets/circuits")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if not args.source_only:
        for tool in ("pdflatex", "pdftoppm"):
            if shutil.which(tool) is None:
                parser.error(f"{tool} must be on PATH, or use --source-only")

    examples = EXAMPLE.findall((ROOT / "README.md").read_text(encoding="utf-8"))
    if not examples:
        parser.error("No marked circuit examples found in README.md")

    # Execute the documented construction/export calls in isolated group scopes.
    # Scratch space keeps TeX logs and auxiliary files out of the repository.
    namespaces = {}
    with tempfile.TemporaryDirectory(prefix="qasp-readme-circuits-") as directory:
        scratch = Path(directory)
        with working_directory(scratch):
            for group, source in examples:
                namespace = namespaces.setdefault(group, {"__name__": "__readme__"})
                exec(compile(source, f"README.md:{group}", "exec"), namespace)
                print(f"Executed {group}", flush=True)

            sources = sorted(scratch.glob("*.tex"))
            if not sources:
                raise RuntimeError("The README examples did not produce any .tex files")
            for tex in sources:
                if not args.source_only:
                    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name])
                    run(["pdftoppm", "-singlefile", "-png", "-r", "200",
                         str(tex.with_suffix(".pdf")), str(tex.with_suffix(""))])
                print(f"Rendered {tex.stem}", flush=True)

        destination.mkdir(parents=True, exist_ok=True)
        suffixes = (".tex",) if args.source_only else (".tex", ".pdf", ".png")
        for tex in sources:
            for suffix in suffixes:
                artifact = tex.with_suffix(suffix)
                shutil.copy2(artifact, destination / artifact.name)
    print(f"Saved {len(sources)} circuit figures to {destination}")


if __name__ == "__main__":
    main()
