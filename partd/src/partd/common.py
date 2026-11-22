from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy
import sklearn


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def versions(include_partb=False):
    result = {"python": platform.python_version(), "numpy": numpy.__version__,
              "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    if include_partb:
        import pandas
        import scipy
        import PIL
        result.update({"pandas": pandas.__version__, "scipy": scipy.__version__, "Pillow": PIL.__version__})
    return result


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(value, name):
    if not isinstance(value, str) or not value or len(value) > 100:
        raise ValueError(f"{name} must be a non-empty string of at most 100 characters")
    if value in {".", ".."} or any(c in value for c in '/\\:*?"<>|') or value.endswith((".", " ")):
        raise ValueError(f"unsafe {name}: {value!r}")
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError(f"reserved {name}: {value!r}")
    return value


def resolve_path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("file and directory paths must be non-empty strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


def error_result(command, exc, **context):
    return {"event": "partd.result", "schema_version": 1, "command": command,
            "status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}, **context}
