"""Execute the CartPole notebook with this environment; --full enables training."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--full", action="store_true")
args = parser.parse_args()
path = root / "examples/cartpole_rllib.ipynb"
output = root / "output/rllib"
output.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="qasp-kernel-") as temp:
    spec = Path(temp) / "qasp-validation"
    spec.mkdir()
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-I", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (qasp-validation)", "language": "python",
        "env": {"QASP_NOTEBOOK_SMOKE": "0" if args.full else "1", "MPLCONFIGDIR": "/tmp/qasp-rllib-mpl",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "RAY_USAGE_STATS_ENABLED": "0"},
    }))
    manager = KernelManager(kernel_name="qasp-validation",
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp]))
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, km=manager, timeout=None if args.full else 600,
                            resources={"metadata": {"path": str(output)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    deprecated = "The 'id' argument is deprecated"
    for cell in notebook.cells:
        for item in cell.get("outputs", []):
            assert deprecated not in str(item), "Deprecated PennyLane gate ID in notebook output"
    print(f"Executed {path}")
