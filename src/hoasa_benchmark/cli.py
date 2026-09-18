"""`hoasa` command line interface.

Commands:
  hoasa prepare [--data-dir DIR]
  hoasa preflight CONFIG
  hoasa majority-baseline --run-dir DIR [--data-dir DIR]
  hoasa baseline CONFIG --run-dir DIR
  hoasa train CONFIG --run-dir DIR
  hoasa evaluate CONFIG --checkpoint DIR/adapter --run-dir DIR
  hoasa compare --run-dir DIR

Heavy ML imports happen lazily inside each command so `--help`,
`hoasa prepare` and `hoasa majority-baseline` stay lightweight.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .runtime import (
    cache_dir_for,
    load_config,
    prepared_dir_for,
    resolve_path,
)


def cmd_prepare(args) -> int:
    from .data import prepare

    data_dir = resolve_path(args.data_dir, Path.cwd())
    prepare(data_dir)
    return 0


def cmd_preflight(args) -> int:
    from .models import preflight

    config = load_config(args.config)
    prepared_dir = prepared_dir_for(config)
    output = Path(config["paths"]["data_dir"]) / "preflight" / f"{config.get('name', 'config')}.json"
    report = preflight(config, prepared_dir, cache_dir_for(config), output)
    print(f"\nPreflight written to {output}")
    for split, data in report["splits"].items():
        print(
            f"  {split:5s}: rows={data['rows']:5d} "
            f"user<={data['user_tokens']['max']} "
            f"prompt<={data['prompt_tokens']['max']} "
            f"full<={data['full_tokens']['max']} "
            f"truncated={data['truncated_reviews']} "
            f"violations={data['full_length_violations']}"
        )
    print(f"  cache weight files: {report['cache_weight_files'] or 'none'}")
    return 0


def cmd_majority_baseline(args) -> int:
    from .evaluate import run_majority_phase

    data_dir = resolve_path(args.data_dir, Path.cwd())
    run_majority_phase(args.run_dir, data_dir=data_dir)
    return 0


def cmd_baseline(args) -> int:
    from .evaluate import run_model_phase

    config = load_config(args.config)
    run_model_phase(config, args.run_dir, "base", "baseline")
    return 0


def cmd_train(args) -> int:
    from .train import train

    config = load_config(args.config)
    train(config, args.run_dir)
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import run_model_phase

    config = load_config(args.config)
    run_model_phase(config, args.run_dir, args.checkpoint, "finetuned")
    return 0


def cmd_compare(args) -> int:
    from .compare import compare

    compare(args.run_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hoasa",
        description=(
            "IndoNLU HoASA Indonesian hotel-review ABSA benchmark: prepare, "
            "metadata preflight, majority baseline, BF16 LoRA SFT, evaluation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="download/verify the pinned CSVs and write prepared JSONL + stats"
    )
    prepare.add_argument(
        "--data-dir", default="data", help="data directory (raw/ and prepared/)"
    )
    prepare.set_defaults(func=cmd_prepare)

    preflight = subparsers.add_parser(
        "preflight",
        help="metadata-only check (AutoConfig + tokenizer/processor): lengths, truncation, cache weights",
    )
    preflight.add_argument("config", help="path to the YAML config")
    preflight.set_defaults(func=cmd_preflight)

    majority = subparsers.add_parser(
        "majority-baseline",
        help="train-majority per aspect, serialized and scored through the shared evaluator",
    )
    majority.add_argument("--run-dir", required=True, help="run directory to write into")
    majority.add_argument(
        "--data-dir", default="data", help="data directory holding prepared/"
    )
    majority.set_defaults(func=cmd_majority_baseline)

    baseline = subparsers.add_parser(
        "baseline", help="evaluate the pinned base model on the frozen test split"
    )
    baseline.add_argument("config", help="path to the YAML config")
    baseline.add_argument("--run-dir", required=True, help="run directory")
    baseline.set_defaults(func=cmd_baseline)

    train = subparsers.add_parser(
        "train", help="Unsloth BF16 LoRA SFT; writes adapter/, trainer/, train_metrics.json"
    )
    train.add_argument("config", help="path to the YAML config")
    train.add_argument("--run-dir", required=True, help="run directory")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate an adapter (or base) with the shared evaluator"
    )
    evaluate.add_argument("config", help="path to the YAML config")
    evaluate.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "the run-local adapter directory produced by `hoasa train`, i.e. "
            "training.adapter_subdir under --run-dir (default DIR/adapter); "
            "'base' is refused for fine-tuned evaluation"
        ),
    )
    evaluate.add_argument("--run-dir", required=True, help="run directory")
    evaluate.set_defaults(func=cmd_evaluate)

    compare = subparsers.add_parser(
        "compare", help="baseline vs fine-tuned report.md + comparison.csv"
    )
    compare.add_argument("--run-dir", required=True, help="run directory")
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
