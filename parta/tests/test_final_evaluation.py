from types import SimpleNamespace

import numpy as np
import pytest

from parta.agent import evaluate_final, write_json


def setup_result(path, prediction_id="e1"):
    write_json(path / "selection.json", {"experiment_id": "e1"})
    write_json(path / "prediction_result.json", {"status": "success", "partd_result": {"source_experiment_id": prediction_id}})
    (path / "predictions.csv").write_text("sample_id,prediction\na,1\nb,0\n", encoding="utf-8")


def test_accuracy_only_after_selection_and_prediction(tmp_path):
    setup_result(tmp_path)
    accesses = []
    def labels(*, final_evaluation):
        accesses.append(final_evaluation)
        return SimpleNamespace(sample_ids=np.array(["a", "b"]), y=np.array([1, 1]))
    loader = SimpleNamespace(config={"metric": "accuracy"}, load_final_evaluation=labels)
    result = evaluate_final(loader, tmp_path)
    assert result["value"] == 0.5
    assert result["used_for_selection"] is False
    assert accesses == [True]
    with pytest.raises(FileExistsError):
        evaluate_final(loader, tmp_path)
    assert accesses == [True]


def test_wrong_candidate_cannot_unlock_test_labels(tmp_path):
    setup_result(tmp_path, prediction_id="another_candidate")
    loader = SimpleNamespace(config={"metric": "accuracy"},
        load_final_evaluation=lambda **kwargs: pytest.fail("Test labels must remain unopened"))
    with pytest.raises(ValueError, match="locked candidate"):
        evaluate_final(loader, tmp_path)


def test_misaligned_predictions_rejected(tmp_path):
    setup_result(tmp_path)
    loader = SimpleNamespace(config={"metric": "accuracy"}, load_final_evaluation=lambda **kwargs:
        SimpleNamespace(sample_ids=np.array(["b", "a"]), y=np.array([1, 0])))
    with pytest.raises(ValueError, match="sample IDs"):
        evaluate_final(loader, tmp_path)
