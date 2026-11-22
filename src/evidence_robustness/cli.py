"""Command-line entry points. Stage 0 deliberately exposes only `smoke`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .evaluate import evaluate_baselines, fit_baselines
from .pilot_data import build_pilot_perturbations, prepare_hotpot_pilot
from .smoke import run_smoke


def _read_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evidence_robustness.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke", help="run deterministic offline Stage-0 checks")
    smoke_parser.add_argument("--config", required=True, help="path to smoke YAML configuration")
    for command, help_text in (
        ("prepare-data", "download and normalise the pilot HotpotQA sample"),
        ("build-perturbations", "build auditable pilot variants"),
        ("fit-baselines", "fit B0/B1/B3/B4 diagnostic baselines"),
        ("evaluate", "evaluate diagnostic baselines on the held-out split"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("--config", required=True, help="path to pilot YAML configuration")
        command_parser.add_argument("--workdir", default="data", help="local artifact directory (default: data)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        config = _read_config(args.config)
        result = run_smoke(count=int(config.get("examples", 10)), seed=int(config.get("seed", 20261122)))
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    config = _read_config(args.config)
    workdir = Path(args.workdir)
    if args.command == "prepare-data":
        result = prepare_hotpot_pilot(config, workdir / "processed")
    elif args.command == "build-perturbations":
        result = build_pilot_perturbations(config, workdir / "processed", workdir / "perturbations")
    elif args.command == "fit-baselines":
        result = fit_baselines(workdir)
    elif args.command == "evaluate":
        result = evaluate_baselines(workdir)
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
