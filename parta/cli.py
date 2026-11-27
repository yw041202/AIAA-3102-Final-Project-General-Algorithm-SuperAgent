import argparse
import json
from pathlib import Path

from .agent import load_task, run
from .demo import OfflinePlanner, make_fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description="A+B+C+D autonomous classification agent")
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("run")
    live.add_argument("--config", required=True)
    live.add_argument("--env-file", help="Local secret file; never committed")
    demo = commands.add_parser("demo")
    demo.add_argument("--output", required=True, help="New synthetic demo directory")
    demo.add_argument("--modality", choices=["tabular", "text"], default="tabular")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            config = load_task(make_fixture(args.output, args.modality))
            result = run(config, planner=OfflinePlanner(), offline=True)
        else:
            result = run(load_task(args.config), env_file=args.env_file)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0 if result["status"] == "success" else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True))
        return 2
