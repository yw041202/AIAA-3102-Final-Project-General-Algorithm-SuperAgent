"""Replaceable B/D boundary: numeric CSV with explicit labels and row IDs."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray | None
    ids: list[str]
    features: list[str]


def load_csv(path, *, target_column="label", id_column="sample_id", features=None, labeled=True):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("CSV requires unique column names")
        required = [id_column] + ([target_column] if labeled else [])
        missing = set(required) - set(columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        selected = list(features) if features is not None else [c for c in columns if c not in {target_column, id_column}]
        if not selected or len(set(selected)) != len(selected) or set(selected) & {target_column, id_column}:
            raise ValueError("feature columns must be unique and exclude the target and sample ID")
        if set(selected) - set(columns):
            raise ValueError(f"missing feature columns: {sorted(set(selected) - set(columns))}")
        matrix, labels, ids = [], [], []
        for index, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV row {index} has an inconsistent number of cells")
            sample_id = row[id_column]
            if not sample_id.strip():
                raise ValueError(f"empty sample ID at row {index}")
            ids.append(sample_id)
            values = []
            for column in selected:
                text = row[column].strip()
                try:
                    value = float(text) if text else float("nan")
                except ValueError as exc:
                    raise ValueError(f"feature {column!r} at row {index} is not numeric") from exc
                if np.isinf(value):
                    raise ValueError(f"infinite feature at row {index}, column {column!r}")
                values.append(value)
            matrix.append(values)
            if labeled:
                if not row[target_column].strip():
                    raise ValueError(f"empty label at row {index}")
                labels.append(row[target_column])
    if not ids:
        raise ValueError("CSV has no samples")
    if len(set(ids)) != len(ids):
        raise ValueError("sample IDs must be unique within each input")
    return Dataset(np.asarray(matrix, dtype=float), np.asarray(labels, dtype=str) if labeled else None, ids, selected)


def write_predictions(path, ids, predictions, probabilities=None, truth=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id"] + (["true_label"] if truth is not None else []) + ["prediction"]
    if probabilities is not None:
        fields += [f"probability_{i}" for i in range(probabilities.shape[1])]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for i, sample_id in enumerate(ids):
            row = [sample_id] + ([str(truth[i])] if truth is not None else []) + [str(predictions[i])]
            if probabilities is not None:
                row += [float(p) for p in probabilities[i]]
            writer.writerow(row)
