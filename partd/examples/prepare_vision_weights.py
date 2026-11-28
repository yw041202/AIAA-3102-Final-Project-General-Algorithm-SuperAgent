"""Verify pinned weights; download official files only with explicit --download."""
import argparse
from pathlib import Path
import sys
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from partd.common import sha256
from partd.vision import BACKBONES


def prepare(root, download=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for name, spec in BACKBONES.items():
        path = root / spec["filename"]
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"missing {path.name}; provision it before training")
            temporary = root / (spec["filename"] + "." + uuid.uuid4().hex + ".partial")
            urllib.request.urlretrieve("https://download.pytorch.org/models/" + spec["filename"], temporary)
            if sha256(temporary) != spec["sha256"]:
                raise ValueError(f"download checksum mismatch: {temporary.name}")
            temporary.rename(path)
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"weight checksum mismatch: {path.name}; existing file was not overwritten")
        results.append({"backbone": name, "file": path.name, "sha256": spec["sha256"], "status": "verified"})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(args.weights_root, args.download)))
