import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from partd.capabilities import capabilities
from partd.candidates import MODEL_NAMES, build_candidate_pipeline


def test_registry_covers_all_candidates_and_filters_without_shared_mutation():
    original = capabilities()
    assert {m["name"] for m in original["models"]} == set(MODEL_NAMES) | {"frozen_vision"}
    assert {m["name"] for m in capabilities("image")["models"]} == {"frozen_vision"}
    assert not {"catboost", "hist_gradient_boosting"} & {m["name"] for m in capabilities("text")["models"]}
    original["models"].clear()
    assert capabilities()["models"]


def test_discovery_does_not_import_optional_backends():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    code = "import sys; from partd.capabilities import capabilities; capabilities(); assert 'torch' not in sys.modules and 'catboost' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)


@pytest.mark.parametrize("name", ["catboost", "hist_gradient_boosting"])
@pytest.mark.parametrize("params", [{"feature_scaling": "standard"}, {"feature_scaling": "maxabs"}, {"svd_components": 2}])
def test_native_boosting_rejects_transforms_that_change_category_codes(name, params):
    with pytest.raises(ValueError, match="preserve categorical codes"):
        build_candidate_pipeline({"name": name, "params": params}, 42, "passthrough")
