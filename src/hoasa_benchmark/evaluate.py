"""Shared evaluator for the base model, LoRA adapters and the majority baseline.

Test rows, prompt, truncation, decoding and parser are identical for baseline
and fine-tuned evaluation; only the model identity differs. The majority
baseline serializes its per-aspect train-majority answer as canonical JSON and
runs it through the exact same parser and metric code.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from . import models as models_mod
from .data import load_jsonl, majority_labels
from .parsing import compute_metrics, parse_output
from .prompting import build_generation_input
from .runtime import (
    cuda_metadata,
    data_dir_for,
    ensure_workspace,
    prepared_dir_for,
    sha256_file,
)
from .spec import ASPECTS, SYSTEM_PROMPT, canonical_json

try:  # optional progress bar; correctness does not depend on it
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def resolve_checkpoint(checkpoint: str) -> tuple[str, str | None]:
    """`base` -> base model; a directory with adapter_config.json -> adapter."""
    value = str(checkpoint).strip()
    if not value or value.lower() == "base":
        return "base", None
    candidate = Path(value).expanduser()
    if candidate.is_dir() and (candidate / "adapter_config.json").is_file():
        return "adapter", str(candidate.resolve())
    raise FileNotFoundError(
        f"--checkpoint {checkpoint!r} is neither 'base' nor a directory containing "
        "adapter_config.json"
    )


def resolve_run_local_adapter(run_dir: str | Path, config: dict, checkpoint: str) -> Path:
    """Fine-tuned phases: the configured run-local adapter only.

    `--checkpoint` must resolve exactly to `<run_dir>/<training.adapter_subdir>`
    (itself confined to the run directory) and contain `adapter_config.json`.
    `base`, model ids, and arbitrary adapter directories are refused so that
    `finetuned_metrics.json` can only describe the adapter this run trained.
    """
    run_root = Path(run_dir).expanduser().resolve()
    subdir = str((config.get("training") or {}).get("adapter_subdir", "adapter"))
    expected = (run_root / subdir).resolve()
    if expected != run_root and run_root not in expected.parents:
        raise RuntimeError(
            f"training.adapter_subdir {subdir!r} escapes the run directory {run_root}"
        )
    value = str(checkpoint).strip()
    if not value or value.lower() == "base":
        raise RuntimeError(
            "fine-tuned evaluation requires the run-local LoRA adapter produced "
            "by `hoasa train`, never 'base'"
        )
    supplied = Path(value).expanduser().resolve()
    if not supplied.is_dir() or not (supplied / "adapter_config.json").is_file():
        raise RuntimeError(
            f"--checkpoint {checkpoint!r} is not a directory containing adapter_config.json"
        )
    if supplied != expected:
        raise RuntimeError(
            f"--checkpoint {supplied} is not the configured run-local adapter {expected}; "
            "fine-tuned metrics are only written for the adapter trained in this run directory"
        )
    return expected


def verify_finetuned_provenance(
    run_dir: str | Path, config: dict, manifest: dict, adapter: Path
) -> dict:
    """Cross-check the adapter against run-local train_metrics.json and manifest."""
    run_dir = Path(run_dir)
    train_metrics_path = run_dir / "train_metrics.json"
    if not train_metrics_path.is_file():
        raise FileNotFoundError(
            f"missing {train_metrics_path}; run `hoasa train` in this run directory "
            "before fine-tuned evaluation"
        )
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]

    for label, recorded, expected in (
        ("config hash", train_metrics.get("config_sha256"), manifest.get("config_sha256")),
        (
            "system prompt hash",
            train_metrics.get("system_prompt_sha256"),
            manifest.get("system_prompt_sha256"),
        ),
        ("dataset hashes", train_metrics.get("dataset"), manifest.get("dataset")),
        ("base model", train_metrics.get("base_model"), model_cfg.get("base_model")),
        (
            "requested revision",
            train_metrics.get("requested_revision"),
            model_cfg.get("revision"),
        ),
    ):
        if recorded != expected:
            raise RuntimeError(
                f"adapter provenance mismatch on {label}: "
                f"train_metrics={recorded!r} expected={expected!r}"
            )

    resolved = train_metrics.get("resolved_revision")
    pinned = (manifest.get("model") or {}).get("revision")
    if resolved and pinned and resolved != pinned:
        raise RuntimeError(
            "adapter provenance mismatch on resolved revision: "
            f"train_metrics={resolved!r} manifest={pinned!r}"
        )
    recorded_adapter = train_metrics.get("adapter_dir")
    if recorded_adapter and Path(recorded_adapter).expanduser().resolve() != adapter:
        raise RuntimeError(
            "adapter provenance mismatch: train_metrics adapter_dir="
            f"{recorded_adapter!r} != {adapter}"
        )

    return {
        "train_metrics_sha256": sha256_file(train_metrics_path),
        "adapter": str(adapter),
        "config_sha256": train_metrics.get("config_sha256"),
        "system_prompt_sha256": train_metrics.get("system_prompt_sha256"),
        "dataset": train_metrics.get("dataset"),
        "base_model": train_metrics.get("base_model"),
        "requested_revision": train_metrics.get("requested_revision"),
        "resolved_revision": resolved,
    }


def _tokenize(loaded, prompt: str) -> dict:
    if loaded.processor is not None:
        return loaded.processor(text=prompt, return_tensors="pt")
    return loaded.tokenizer(prompt, return_tensors="pt")


def _cuda_peaks_gib() -> tuple[float, float]:
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated() / 1024**3,
        torch.cuda.max_memory_reserved() / 1024**3,
    )


def _write_predictions(path: Path, records, parsed, raw_outputs, input_tokens, output_tokens, latencies) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record, item, raw, in_tokens, out_tokens, latency in zip(
            records, parsed, raw_outputs, input_tokens, output_tokens, latencies
        ):
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "gold": record["target"],
                        "parsed": item.parsed_payload(),
                        "raw_output": raw,
                        "validity": item.validity(),
                        "input_tokens": in_tokens,
                        "output_tokens": out_tokens,
                        "latency_ms": latency,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _base_metrics_record(manifest, config, run_dir, phase) -> dict:
    evaluation_cfg = (config or {}).get("evaluation") or {}
    model_cfg = (config or {}).get("model") or {}
    test = dict(manifest["dataset"]["test"])
    test["path"] = str(Path(run_dir) / "test.jsonl")
    return {
        "phase": phase,
        "config_name": manifest.get("config_name"),
        "base_model": model_cfg.get("base_model"),
        "kind": model_cfg.get("kind"),
        "system_prompt_sha256": manifest["system_prompt_sha256"],
        "config_sha256": manifest.get("config_sha256"),
        "dataset": test,
        "decoding": {
            "split": evaluation_cfg.get("split", "test"),
            "max_user_tokens": int(evaluation_cfg.get("max_user_tokens", 1536)),
            "max_new_tokens": int(evaluation_cfg.get("max_new_tokens", 256)),
            "do_sample": bool(evaluation_cfg.get("do_sample", False)),
            "max_seq_length": int(model_cfg.get("max_seq_length", 2048)),
            "chat_template_kwargs": model_cfg.get("chat_template_kwargs") or {},
        },
    }


def run_model_phase(config: dict, run_dir: str | Path, checkpoint: str, phase: str) -> dict:
    """Baseline (`--checkpoint base`) or adapter evaluation on the test split."""
    import torch

    run_dir = Path(run_dir)
    manifest = ensure_workspace(
        run_dir,
        config=config,
        config_path=config.get("_config_path"),
        prepared_dir=prepared_dir_for(config),
        phase=phase,
    )
    evaluation_cfg = config.get("evaluation") or {}
    model_cfg = config["model"]
    split = evaluation_cfg.get("split", "test")
    if split != "test":
        raise ValueError(f"only the frozen test split is evaluable, got {split!r}")
    records = load_jsonl(run_dir / "test.jsonl")

    if phase == "baseline":
        mode, adapter = resolve_checkpoint(checkpoint)
        if mode != "base":
            raise RuntimeError(
                "`hoasa baseline` evaluates the pinned base model only; use "
                "`hoasa evaluate` for adapters"
            )
        provenance = None
    elif phase == "finetuned":
        adapter_path = resolve_run_local_adapter(run_dir, config, checkpoint)
        mode, adapter = "adapter", str(adapter_path)
        provenance = verify_finetuned_provenance(run_dir, config, manifest, adapter_path)
    else:
        raise ValueError(f"unknown phase {phase!r}")

    loaded = models_mod.load_for_inference(config, adapter=adapter)
    model = loaded.model
    tokenizer = loaded.tokenizer

    load_peak_allocated, load_peak_reserved = _cuda_peaks_gib()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    max_user_tokens = int(evaluation_cfg.get("max_user_tokens", 1536))
    max_new_tokens = int(evaluation_cfg.get("max_new_tokens", 256))
    max_seq_length = int(model_cfg.get("max_seq_length", 2048))
    do_sample = bool(evaluation_cfg.get("do_sample", False))
    chat_template_kwargs = model_cfg.get("chat_template_kwargs") or {}

    parsed, raw_outputs, input_tokens, output_tokens, latencies = [], [], [], [], []
    total_generate_seconds = 0.0

    for record in tqdm(records):
        built = build_generation_input(
            loaded.prompt_source,
            record["input"],
            SYSTEM_PROMPT,
            max_user_tokens=max_user_tokens,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
        )
        inputs = _tokenize(loaded, built["prompt"])
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[1])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        total_generate_seconds += elapsed

        generated_ids = output[0][input_length:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        parsed.append(parse_output(raw_output))
        raw_outputs.append(raw_output)
        input_tokens.append(input_length)
        output_tokens.append(len(generated_ids))
        latencies.append(elapsed * 1000.0)

    metrics = compute_metrics(records, parsed)
    peak_allocated, peak_reserved = _cuda_peaks_gib()

    payload = {
        **_base_metrics_record(manifest, config, run_dir, phase),
        "mode": mode,
        "checkpoint": checkpoint,
        "adapter": adapter,
        "provenance": provenance,
        "system_prompt_sha256": manifest["system_prompt_sha256"],
        **loaded.revision_info,
        "precision": loaded.precision,
        "chat_template_kwargs": chat_template_kwargs,
        "cuda": cuda_metadata(torch),
        **metrics,
        "resource": {
            "measured": True,
            "peak_allocated_vram_gib": peak_allocated,
            "peak_reserved_vram_gib": peak_reserved,
            "load_peak_allocated_vram_gib": load_peak_allocated,
            "load_peak_reserved_vram_gib": load_peak_reserved,
            "counters_reset_after_model_load": bool(torch.cuda.is_available()),
            "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
            "reviews_per_second": (
                len(records) / total_generate_seconds if total_generate_seconds > 0 else 0.0
            ),
            "output_tokens_per_second": (
                sum(output_tokens) / total_generate_seconds
                if total_generate_seconds > 0
                else 0.0
            ),
            "output_tokens_per_second_includes_prefill": True,
            "total_generated_tokens": sum(output_tokens),
            "total_generate_seconds": total_generate_seconds,
            "mean_input_tokens": (
                statistics.mean(input_tokens) if input_tokens else 0.0
            ),
            "median_input_tokens": (
                statistics.median(input_tokens) if input_tokens else 0.0
            ),
        },
    }
    from .runtime import library_versions

    payload["library_versions"] = library_versions()

    metrics_path = run_dir / f"{phase}_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    _write_predictions(
        run_dir / f"{phase}_predictions.jsonl",
        records,
        parsed,
        raw_outputs,
        input_tokens,
        output_tokens,
        latencies,
    )

    print()
    print("=" * 60)
    print(f"{phase.upper()} RESULTS ({mode})")
    print("=" * 60)
    print(f"Base model              : {model_cfg['base_model']}")
    print(f"Adapter                 : {adapter or '-'}")
    print(f"Examples                : {metrics['examples']}")
    print(f"mean_aspect_macro_f1    : {metrics['mean_aspect_macro_f1']:.6f}")
    print(f"overall_aspect_accuracy : {metrics['overall_aspect_accuracy']:.6f}")
    print(f"whole_review_exact      : {metrics['whole_review_exact_accuracy']:.6f}")
    print(f"syntax/schema/output    : {metrics['syntax_valid_rate']:.4f} / "
          f"{metrics['schema_valid_rate']:.4f} / {metrics['output_valid_rate']:.4f}")
    print(f"Peak VRAM (alloc/res)   : "
          f"{payload['resource']['peak_allocated_vram_gib']:.2f} / "
          f"{payload['resource']['peak_reserved_vram_gib']:.2f} GiB")
    print(f"Saved: {metrics_path.name}, {phase}_predictions.jsonl")
    return payload


def run_majority_phase(run_dir: str | Path, *, data_dir: str | Path | None = None) -> dict:
    """Train-majority per aspect, serialized and scored through the same path."""
    run_dir = Path(run_dir)
    data_dir = Path(data_dir) if data_dir is not None else data_dir_for(None)
    manifest = ensure_workspace(
        run_dir,
        config=None,
        config_path=None,
        prepared_dir=Path(data_dir) / "prepared",
        phase="majority-baseline",
    )
    train_records = load_jsonl(run_dir / "train.jsonl")
    records = load_jsonl(run_dir / "test.jsonl")
    majority = majority_labels(train_records)
    raw_output = canonical_json(majority)
    parsed = [parse_output(raw_output) for _ in records]

    metrics = compute_metrics(records, parsed)
    payload = {
        **_base_metrics_record(manifest, None, run_dir, "majority"),
        "mode": "majority",
        "checkpoint": "majority",
        "adapter": None,
        "backend": "majority",
        "majority_labels": majority,
        "system_prompt_sha256": manifest["system_prompt_sha256"],
        **metrics,
        "resource": {
            "measured": False,
            "peak_allocated_vram_gib": 0.0,
            "peak_reserved_vram_gib": 0.0,
            "mean_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "reviews_per_second": 0.0,
            "output_tokens_per_second": 0.0,
            "output_tokens_per_second_includes_prefill": False,
            "total_generated_tokens": 0,
            "mean_input_tokens": 0.0,
            "median_input_tokens": 0.0,
        },
    }

    metrics_path = run_dir / "majority_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    _write_predictions(
        run_dir / "majority_predictions.jsonl",
        records,
        parsed,
        [raw_output] * len(records),
        [0] * len(records),
        [0] * len(records),
        [0.0] * len(records),
    )

    print()
    print("=" * 60)
    print("MAJORITY BASELINE RESULTS")
    print("=" * 60)
    print(f"Train-majority labels   : {majority}")
    print(f"Examples                : {metrics['examples']}")
    print(f"mean_aspect_macro_f1    : {metrics['mean_aspect_macro_f1']!r}")
    print(f"overall_aspect_accuracy : {metrics['overall_aspect_accuracy']!r}")
    print(f"whole_review_exact      : {metrics['whole_review_exact_accuracy']!r}")
    print(f"syntax/schema/output    : {metrics['syntax_valid_rate']!r} / "
          f"{metrics['schema_valid_rate']!r} / {metrics['output_valid_rate']!r}")
    print(f"Saved: {metrics_path.name}, majority_predictions.jsonl")
    return payload
