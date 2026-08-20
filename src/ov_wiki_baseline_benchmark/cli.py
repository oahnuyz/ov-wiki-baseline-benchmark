"""Command-line entry point for the fixed OV-Wiki dataset benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import prepare_experiment, verify_prepared
from .specs import load_specs, repository_root


def _data_dir(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else repository_root() / "data"


def _selected_ids(args: argparse.Namespace, available: list[str]) -> list[str]:
    if getattr(args, "all", False):
        return available
    values = list(getattr(args, "experiments", []) or [])
    if not values:
        raise ValueError("Select at least one experiment or pass --all")
    unknown = sorted(set(values) - set(available))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ov-wiki-data",
        description="Download and prepare the fixed OV-Wiki baseline datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List the thirteen experiment variants")

    prepare = subparsers.add_parser(
        "prepare", help="Download and prepare one or more experiment variants"
    )
    prepare.add_argument("experiments", nargs="*")
    prepare.add_argument("--all", action="store_true")
    prepare.add_argument("--data-dir")
    prepare.add_argument("--skip-download", action="store_true")
    prepare.add_argument("--force-download", action="store_true")

    verify = subparsers.add_parser(
        "verify", help="Verify already prepared canonical data"
    )
    verify.add_argument("experiments", nargs="*")
    verify.add_argument("--all", action="store_true")
    verify.add_argument("--data-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        specs = load_specs()
        available = sorted(specs)
        if args.command == "list":
            for experiment_id in available:
                spec = specs[experiment_id]
                print(
                    f"{experiment_id}\t{spec.expected_qas} QA\t"
                    f"{spec.expected_documents} documents"
                )
            return 0

        selected = _selected_ids(args, available)
        data_dir = _data_dir(args.data_dir)
        if args.command == "prepare":
            results = []
            for experiment_id in selected:
                print(f"\n=== Preparing {experiment_id} ===")
                results.append(
                    prepare_experiment(
                        specs[experiment_id],
                        data_dir,
                        force_download=args.force_download,
                        skip_download=args.skip_download,
                    )
                )
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 0

        for experiment_id in selected:
            verify_prepared(specs[experiment_id], data_dir)
            print(f"✓ {experiment_id}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
