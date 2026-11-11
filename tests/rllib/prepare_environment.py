"""Stage only the unreleased optional module beside PyPI QASP for validation.

Run with the qasp-validation interpreter and -I. No published model, trainer,
__init__.py or package metadata is modified. This is a development/testing step,
not a separate package or a change to the user's model-definition workflow.
"""
from hashlib import sha256
from importlib.metadata import distribution
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[2]
package = distribution("qasp")
if package.version != "0.3.0":
    raise RuntimeError("Validation expects the current PyPI release qasp==0.3.0.")
installed = Path(package.locate_file("qasp")).resolve()
files = [p for p in installed.rglob("*.py") if p.name != "rllib.py"]
before = {str(p): sha256(p.read_bytes()).hexdigest() for p in files}
shutil.copyfile(root / "qasp/rl/rllib.py", installed / "rl/rllib.py")
assert before == {str(p): sha256(p.read_bytes()).hexdigest() for p in files}
print(f"Staged optional qasp.rl.rllib in {installed}")
print(f"All {len(files)} existing PyPI Python source files remain unchanged.")
