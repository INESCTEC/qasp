"""Completed experiments survive a requested simulator switch without config drift."""

import json
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import pytest


def test_backend_switch_retains_completed_specs_and_rejects_training_changes(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    runner = import_module("pqdqn_cartpole_experiment")
    previous = {
        "specs": [
            {
                "readout": "shared",
                "seed": seed,
                "episodes": 400,
                "model_config": {
                    "device": "default.qubit",
                    "diff_method": "backprop",
                    "n_layers": 4,
                },
            }
            for seed in (7, 8)
        ]
    }
    completed = previous["specs"][0]
    marker = tmp_path / "shared" / "seed_7" / "summary.json"
    marker.parent.mkdir(parents=True)
    original = json.dumps(
        {"fingerprint": runner._fingerprint(completed), "spec": completed}
    )
    marker.write_text(original)
    requested = deepcopy(previous["specs"])
    for spec in requested:
        spec["model_config"].update(device="lightning.qubit", diff_method="adjoint")
    selected = runner._retain_completed_specs(requested, previous, tmp_path)
    assert selected[0] == completed
    assert selected[1] == requested[1]
    assert marker.read_text() == original
    assert requested[0]["model_config"]["device"] == "lightning.qubit"
    requested[1]["episodes"] = 600
    with pytest.raises(ValueError, match="Only device and diff_method"):
        runner._retain_completed_specs(requested, previous, tmp_path)
