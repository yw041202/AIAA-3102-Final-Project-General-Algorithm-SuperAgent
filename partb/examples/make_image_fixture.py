"""Generate tiny synthetic RGB/grayscale/RGBA inputs for interface validation."""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    for split in ("train", "val"):
        records = []
        for label in ("dark", "light"):
            folder = root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for i, mode in enumerate(("L", "RGB", "RGBA")):
                path = folder / f"{i}.png"
                value = 20 if label == "dark" else 230
                color = value if mode == "L" else tuple([value] * 3 + ([255] if mode == "RGBA" else []))
                Image.new(mode, (12 + i, 18 + i), color).save(path)
                records.append({"image_path": path.relative_to(root).as_posix(),
                                "label": label, "sample_id": f"{split}-{label}-{i}"})
        pd.DataFrame(records).to_csv(root / f"{split}.csv", index=False)
    print(f"Synthetic interface fixture written to {root}")


if __name__ == "__main__":
    main()
