from __future__ import annotations

import argparse
import json

from .common import error_result
from .prediction import predict
from .training import train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Part D classification tools (no model selection)")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--request", required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--artifact", required=True)
    prediction.add_argument("--input", help="numeric CSV input; omit for Part B artifacts")
    prediction.add_argument("--data-root", help="Part B raw prediction data root")
    prediction.add_argument("--config", help="optional Part B config override (otherwise use saved config)")
    prediction.add_argument("--split", default="test", help="Part B prediction split; default test")
    prediction.add_argument("--partb-root", help="Part B package or repository root if not installed")
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--include-probabilities", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            result = train(args.request)
        else:
            result = predict(args.artifact, args.input, args.output, include_probabilities=args.include_probabilities,
                             data_root=args.data_root, config_path=args.config, split=args.split, partb_root=args.partb_root)
    except Exception as exc:
        result = error_result(args.command, exc)
    # ASCII JSON survives Windows redirected streams; JSON readers restore Unicode.
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    return 0 if result["status"] == "success" else 2
