"""One image candidate per request; A owns search, stopping and final selection."""
from __future__ import annotations

from pathlib import Path
import time
import uuid

from .common import error_result, read_json, resolve_path, sha256, source_fingerprint, write_json
from .partb_adapter import code_fingerprint, import_partb
from .vision import (BACKBONES, development_description, extract_features, fit_head,
                     json_digest, read_feature_cache, request_context, vision_versions)


def validate_candidate(params):
    if not isinstance(params, dict) or set(params) - {"backbone", "head", "feature_scaling", "preprocessing"}:
        raise ValueError("frozen_vision params accepts backbone/head/feature_scaling/preprocessing")
    candidate = {"backbone": params.get("backbone", "resnet18"),
        "head": params.get("head", {"name": "logistic_regression", "params": {"C": 1.0}}),
        "feature_scaling": params.get("feature_scaling", "l2"),
        "preprocessing": params.get("preprocessing", "official")}
    if candidate["backbone"] not in BACKBONES:
        raise ValueError("unsupported frozen backbone")
    if candidate["feature_scaling"] not in {"standard", "l2", "none"}:
        raise ValueError("feature_scaling must be standard, l2 or none")
    if candidate["preprocessing"] not in {"official", "resize224"}:
        raise ValueError("preprocessing must be official or resize224")
    head = candidate["head"]
    if not isinstance(head, dict) or set(head) - {"name", "params"}:
        raise ValueError("image head accepts name and params")
    if head.get("name") not in {"logistic_regression", "linear_svm"}:
        raise ValueError("image head must be logistic_regression or linear_svm")
    options = head.get("params", {})
    if not isinstance(options, dict) or set(options) - {"C", "max_iter", "class_weight", "tol", "calibrate"}:
        raise ValueError("image head params accepts C/max_iter/class_weight/tol/calibrate")
    for key in ("C", "tol"):
        if key in options and (type(options[key]) not in {float, int} or not 0 < options[key] < float("inf")):
            raise ValueError(f"head {key} must be finite and positive")
    if "max_iter" in options and (type(options["max_iter"]) is not int or options["max_iter"] < 1):
        raise ValueError("head max_iter must be positive")
    if "class_weight" in options and options["class_weight"] not in (None, "balanced"):
        raise ValueError("image class_weight must be null or balanced")
    if "calibrate" in options and (type(options["calibrate"]) is not bool or head["name"] != "linear_svm"):
        raise ValueError("calibrate is a boolean option for linear_svm only")
    candidate["head"] = {"name": head["name"], "params": {"C": 1.0, **options}}
    return candidate


def train_image(request_path):
    started = time.perf_counter()
    context, directory, candidate, created = {}, None, None, False
    stage = "request"
    try:
        request_path = Path(request_path).resolve()
        base = request_path.parent
        request = read_json(request_path)
        context = request_context(request)
        if set(request) - {"schema_version", "task_id", "run_id", "experiment_id", "data", "model", "metrics",
                           "seed", "preprocessing", "output_dir", "vision"}:
            raise ValueError("unknown image train request fields")
        directory = resolve_path(request["output_dir"], base).joinpath(*context.values())
        directory.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(directory / "request.json", request)
        stage = "model_configuration"
        if set(request["model"]) - {"name", "params"} or request["model"]["name"] != "frozen_vision":
            raise ValueError("image candidate requires model.name=frozen_vision")
        candidate = validate_candidate(request["model"].get("params", {}))
        if request.get("preprocessing", "partb") != "partb":
            raise ValueError("image requests require preprocessing=partb")
        seed = request.get("seed", 42)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("invalid seed")
        options = request["vision"]
        if not isinstance(options, dict) or set(options) - {"weights_root", "cache_dir", "batch_size", "threads"}:
            raise ValueError("vision accepts weights_root/cache_dir/batch_size/threads")
        batch, threads = options.get("batch_size", 16), options.get("threads", 4)
        if any(type(v) is not int or v < 1 for v in (batch, threads)):
            raise ValueError("vision batch_size and threads must be positive")
        weight = resolve_path(options["weights_root"], base) / BACKBONES[candidate["backbone"]]["filename"]
        if sha256(weight) != BACKBONES[candidate["backbone"]]["sha256"]:
            raise ValueError("local pretrained weight checksum mismatch")
        stage = "data_preparation"
        data = request["data"]
        if set(data) - {"source", "config_path", "data_root", "partb_root", "preprocess_options"} or data.get("source") != "partb":
            raise ValueError("image data requires source=partb and the standard B path fields")
        if data.get("preprocess_options", {}):
            raise ValueError("image preprocessing belongs in model.params; do not apply B's 64px preprocessor")
        b_root = str(resolve_path(data["partb_root"], base)) if data.get("partb_root") else None
        module = import_partb(b_root)
        config_path, data_root = resolve_path(data["config_path"], base), resolve_path(data["data_root"], base)
        loader = module.DataModule(config_path, data_root)
        if loader.modality != "image":
            raise ValueError("frozen_vision requires image data")
        train, val = loader.load_development()
        description = development_description(train, val)
        identity = {"task_id": context["task_id"], "run_id": context["run_id"],
            "datasets": description, "config_sha256": sha256(config_path),
            "backbone": candidate["backbone"], "weight_sha256": sha256(weight),
            "preprocessing": candidate["preprocessing"], "batch_size": batch, "threads": threads,
            "versions": vision_versions(), "source_sha256": source_fingerprint(), "partb_sha256": code_fingerprint(module)}
        key = json_digest(identity)
        cache_root = resolve_path(options["cache_dir"], base) / context["task_id"] / context["run_id"]
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_dir = cache_root / key
        hit = cache_dir.exists()
        extraction_begin = time.perf_counter()
        stage = "feature_cache"
        if not hit:
            # Only complete directories become cache hits. C-killed partial jobs remain auditable.
            staging = cache_root / f".building-{key[:12]}-{uuid.uuid4().hex}"
            extraction_request = {"schema_version": 1, **context, "config_path": str(config_path),
                "data_root": str(data_root), "partb_root": b_root,
                "encoder": {"name": candidate["backbone"], "weight_path": str(weight), "preprocessing": candidate["preprocessing"]},
                "batch_size": batch, "threads": threads, "output_dir": str(staging)}
            path = directory / "extract_request.json"
            write_json(path, extraction_request)
            extracted = extract_features(path)["metadata"]
            observed = {name: {k: v for k, v in values.items() if k not in {"shape", "encoding_seconds"}}
                        for name, values in extracted["datasets"].items()}
            if observed != description:
                raise ValueError("development inputs changed during feature extraction")
            write_json(staging / "identity.json", identity)
            # The low-level extraction result originally named the staging directory.
            extracted_result = read_json(staging / "result.json")
            extracted_result["feature_directory"] = str(cache_dir)
            write_json(staging / "result.json", extracted_result)
            staging.rename(cache_dir)  # A serializes candidates within one task/run.
        if read_json(cache_dir / "identity.json") != identity:
            raise ValueError("feature cache identity mismatch; use a fresh cache directory")
        metadata = read_feature_cache(cache_dir)  # Always verify actual feature/label/ID bytes.
        cache_seconds = time.perf_counter() - extraction_begin
        stage = "training"
        head_request = {"schema_version": 1, **context, "feature_directory": str(cache_dir),
            "head": candidate["head"], "feature_scaling": candidate["feature_scaling"], "seed": seed,
            "metrics": request.get("metrics", {"primary": loader.config["metric"], "additional": ["f1_macro"]}),
            "output_dir": str(directory / "model")}
        path = directory / "head_request.json"
        write_json(path, head_request)
        result = fit_head(path)
        result.update(command="train", input_source="partb_image", candidate=candidate,
            cache={"hit": hit, "directory": str(cache_dir), "identity_sha256": key,
                   "feature_cache_key": metadata["cache_key"], "seconds": cache_seconds},
            data=metadata["datasets"], versions=metadata["versions"],
            partd_source_sha256=source_fingerprint(), data_warnings=[])
        result["artifact"]["kind"] = "frozen_vision_classifier"
    except Exception as exc:
        result = error_result("train", exc, stage=stage, **context)
        # Never overwrite another experiment's records on duplicate IDs.
        if not created:
            directory = None
        if candidate is not None:
            result["candidate"] = candidate
    result["elapsed_seconds"] = time.perf_counter() - started
    if directory is not None:
        result["result_path"] = str(directory / "result.json")
        write_json(directory / "result.json", result)
    return result
