"""Regression checks for leakage, row alignment, paired text and image inputs."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from partb import DataModule, DataSplit, Preprocessor


def split(name, values, columns, labels=None):
    frame = pd.DataFrame(values, columns=columns)
    y = np.array(labels) if labels is not None else None
    return DataSplit(name, frame, y, np.array([f"{name}:{i}" for i in range(len(frame))]))


class PipelineTests(unittest.TestCase):
    def test_unknown_preprocessing_options_are_rejected(self):
        for modality in ("tabular", "text", "image"):
            with self.subTest(modality=modality), self.assertRaisesRegex(ValueError, "Unknown preprocessing"):
                Preprocessor(modality, {"misspelled_parameter": True})

    def test_preprocessing_config_is_independent_copy(self):
        config = {"numeric_columns": ["n"]}
        processor = Preprocessor("tabular", config)
        config["numeric_columns"].clear()
        self.assertEqual(processor.config["numeric_columns"], ["n"])

    def test_invalid_task_configuration_is_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = {"modality": "tabular", "format": "csv", "feature_columns": ["x"],
                    "metric": "accuracy", "label_column": "label",
                    "splits": {"train": {"path": "train.csv"}, "val": {"path": "val.csv"}}}
            cases = [dict(base, typo=True), dict(base, id_column="x"),
                     dict(base, splits={"train": {"path": "train.csv"},
                                        "val": {"path": "./train.csv"}})]
            path = root / "config.json"
            for config in cases:
                path.write_text(json.dumps(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    DataModule(path, root)

    def test_batch_transform_preserves_shuffled_nonconsecutive_index(self):
        from scipy import sparse
        for modality, values in [("tabular", [["red"], ["blue"], ["green"]]),
                                 ("text", [["red apple"], ["blue sky"], ["green grass"]])]:
            with self.subTest(modality=modality):
                train = split("train", values, ["feature"], [0, 1, 0])
                processor = Preprocessor(modality).fit(train)
                frame = train.x.iloc[[2, 0, 1]].copy()
                frame.index = [99, 4, 21]
                data = DataSplit("val", frame, np.array([0, 0, 1]), np.array(["c03", "c01", "c02"]))
                whole = processor.transform(frame)
                offset = 0
                for matrix, labels, ids in processor.iter_batches(data, batch_size=2):
                    for i in range(len(ids)):
                        expected = processor.transform(frame.iloc[[offset + i]])
                        actual = matrix[i:i + 1]
                        if sparse.issparse(expected):
                            self.assertEqual((actual != expected).nnz, 0)
                            self.assertEqual((whole[offset+i:offset+i+1] != expected).nnz, 0)
                        else:
                            np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(ids[i], data.sample_ids[offset + i])
                        self.assertEqual(labels[i], data.y[offset + i])
                    offset += len(ids)

    def test_validation_cannot_fit(self):
        with self.assertRaises(ValueError):
            Preprocessor("tabular").fit(split("val", [["a"]], ["c"]))

    def test_numeric_imputation_uses_training_median(self):
        train = split("train", [[1], [3]], ["n"])
        processor = Preprocessor("tabular", {"numeric_columns": ["n"]}).fit(train)
        actual = processor.transform(pd.DataFrame({"n": [None, 100]}))
        np.testing.assert_allclose(actual[:, 0], [0, 98])

    def test_missing_and_unknown_categories(self):
        train = split("train", [["a", None], ["b", None]], ["c", "empty"])
        processor = Preprocessor("tabular").fit(train)
        actual = processor.transform(pd.DataFrame([["new", None]], columns=train.x.columns))
        self.assertEqual(actual.shape, (1, 3))
        self.assertTrue(np.isfinite(actual.toarray()).all())
        with self.assertRaises(ValueError):
            processor.transform(train.x[["empty", "c"]])

    def test_text_vocabulary_excludes_validation_and_order_matters(self):
        train = split("train", [["alpha headline", "beta title"], ["beta", "alpha"]], ["a", "b"])
        processor = Preprocessor("text", {"combine": "difference"}).fit(train)
        probe = pd.DataFrame([["alpha", "beta"]], columns=["a", "b"])
        original = processor.transform(probe)
        swapped = processor.transform(pd.DataFrame([["beta", "alpha"]], columns=["a", "b"]))
        self.assertEqual((original + swapped).nnz, 0)
        processor.transform(pd.DataFrame([["validationonly", "beta"]], columns=["a", "b"]))
        self.assertNotIn("validationonly", processor.transformer.vocabulary_)

    def test_save_reload_and_single_fit(self):
        train = split("train", [["hello world"], ["other title"]], ["text"])
        processor = Preprocessor("text").fit(train)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.joblib"
            processor.save(path)
            restored = Preprocessor.load(path)
            self.assertEqual((processor.transform(train.x) != restored.transform(train.x)).nnz, 0)
        with self.assertRaises(ValueError):
            processor.fit(train)

    def test_generated_split_is_reproducible_and_test_guarded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame({"x": list(range(20)), "label": [0, 1] * 10}).to_csv(root / "train.csv", index=False)
            cfg = {"modality": "tabular", "format": "csv", "feature_columns": ["x"],
                   "label_column": "label", "metric": "accuracy", "splits": {"train": {"path": "train.csv"}},
                   "validation_split": {"fraction": 0.2, "seed": 42}}
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            module = DataModule(path, root)
            train, val = module.load_development()
            self.assertFalse(set(train.sample_ids) & set(val.sample_ids))
            again = DataModule(path, root).load_development()[1]
            np.testing.assert_array_equal(val.sample_ids, again.sample_ids)
            module.audit()  # No test file exists, so this also verifies development isolation.
            with self.assertRaises(PermissionError):
                module.load_final_evaluation()

    def test_prediction_ignores_invalid_test_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame({"a": ["text"], "b": ["other"], "target": ["DO_NOT_USE"]}).to_csv(root / "test.csv", index=False)
            cfg = {"modality": "text", "format": "csv", "feature_columns": ["a", "b"],
                   "label_column": "target", "allowed_labels": [0, 1], "metric": "accuracy",
                   "splits": {"test": {"path": "test.csv"}}}
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            data = DataModule(path, root).load_prediction()
            self.assertIsNone(data.y)
            self.assertEqual(list(data.x.columns), ["a", "b"])

    def test_image_folder_rgb_batches_and_corrupt_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["train", "val"]:
                for label, mode in [("cat", "L"), ("dog", "RGBA")]:
                    directory = root / name / label
                    directory.mkdir(parents=True)
                    Image.new(mode, (13, 17)).save(directory / "sample.png")
            config = Path(__file__).parents[1] / "configs/image_folder.example.json"
            module = DataModule(config, root)
            train, val = module.load_development()
            processor = Preprocessor("image", {"size": [8, 10]}).fit(train)
            batches = list(processor.iter_batches(val, batch_size=1))
            self.assertEqual(batches[0][0].shape, (1, 3, 8, 10))
            self.assertEqual(batches[0][0].dtype, np.float32)
            self.assertEqual(len(batches), 2)
            Path(val.x.iloc[0, 0]).write_bytes(b"broken image")
            with self.assertRaisesRegex(ValueError, "Cannot decode image"):
                processor.transform(val.x)

    def test_image_manifest_path_and_row_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in range(2):
                Image.new("RGB", (8, 8)).save(root / f"{i}.png")
            for name in ["train", "val"]:
                pd.DataFrame({"image_path": ["1.png", "0.png"], "label": [1, 0],
                              "sample_id": ["second", "first"]}).to_csv(root / f"{name}.csv", index=False)
            config = Path(__file__).parents[1] / "configs/image_manifest.example.json"
            train, _ = DataModule(config, root).load_development()
            self.assertEqual(train.sample_ids.tolist(), ["second", "first"])
            self.assertEqual(train.y.tolist(), [1, 0])
            self.assertTrue(Path(train.x.iloc[0, 0]).is_absolute())

    def test_csv_preserves_leading_zero_ids_and_categories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "test.csv").write_text("id,code\n001,002\n003,004\n")
            cfg = {"modality": "tabular", "format": "csv", "feature_columns": ["code"],
                   "id_column": "id", "metric": "accuracy", "splits": {"test": {"path": "test.csv"}}}
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            data = DataModule(path, root).load_prediction()
            self.assertEqual(data.sample_ids.tolist(), ["001", "003"])
            self.assertEqual(data.x["code"].tolist(), ["002", "004"])


if __name__ == "__main__":
    unittest.main()
