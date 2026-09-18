"""Baseline vs fine-tuned comparison: Markdown report + CSV.

Only runs after real `baseline_metrics.json` and `finetuned_metrics.json` exist
in one run directory; identity checks (dataset hash, prompt, base model,
revision, decoding, precision) must match before any comparison is written.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

SCALAR_METRICS = (
    ("mean_aspect_macro_f1", "Mean aspect macro-F1"),
    ("overall_aspect_accuracy", "Overall aspect accuracy"),
    ("whole_review_exact_accuracy", "Whole-review exact accuracy (schema-valid)"),
    ("syntax_valid_rate", "Syntax-valid rate"),
    ("schema_valid_rate", "Schema-valid rate"),
    ("output_valid_rate", "Output-valid rate"),
)

RESOURCE_METRICS = (
    ("peak_allocated_vram_gib", "Peak allocated VRAM (GiB)"),
    ("peak_reserved_vram_gib", "Peak reserved VRAM (GiB)"),
    ("mean_latency_ms", "Mean generate latency (ms/review)"),
    ("median_latency_ms", "Median generate latency (ms/review)"),
    ("reviews_per_second", "Reviews/second"),
    ("output_tokens_per_second", "Output tokens/second (includes prefill)"),
)

IDENTITY_FIELDS = (
    "base_model",
    "kind",
    "requested_revision",
    "resolved_revision",
    "system_prompt_sha256",
)

# ML library versions that must match between baseline and fine-tuned runs for
# the comparison to be controlled (missing in both = equal).
PARITY_LIBRARIES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate")


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run `hoasa baseline` and `hoasa evaluate` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def compare(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    baseline = _load(run_dir / "baseline_metrics.json")
    finetuned = _load(run_dir / "finetuned_metrics.json")

    # Phase provenance: baseline is the pinned base model, fine-tuned is the
    # run-local adapter verified against train_metrics.json at evaluation time.
    if baseline.get("mode") != "base":
        raise RuntimeError(
            f"baseline_metrics.json must come from the base model (mode='base'), "
            f"found mode={baseline.get('mode')!r}"
        )
    if baseline.get("adapter") is not None:
        raise RuntimeError("baseline_metrics.json must not reference an adapter")
    if finetuned.get("mode") != "adapter":
        raise RuntimeError(
            "finetuned_metrics.json must come from the run-local LoRA adapter "
            f"(mode='adapter'), found mode={finetuned.get('mode')!r}"
        )
    if not finetuned.get("adapter"):
        raise RuntimeError(
            "finetuned_metrics.json records no adapter path; refusing an "
            "unattributable fine-tuned evaluation"
        )
    provenance = finetuned.get("provenance") or {}
    for field in ("config_sha256", "system_prompt_sha256"):
        if provenance.get(field) is None or provenance.get(field) != baseline.get(field):
            raise RuntimeError(
                f"baseline/fine-tuned provenance mismatch on {field}: "
                f"{baseline.get(field)!r} != {provenance.get(field)!r}"
            )
    # provenance.dataset is the train_metrics/manifest all-split map
    # ({'train': ..., 'val': ..., 'test': ...}); only the test-split identity
    # (sha256 + rows) must match the baseline test record. The dataset path is
    # evaluation-local and not part of the identity.
    provenance_dataset = provenance.get("dataset")
    provenance_test = (
        provenance_dataset.get("test")
        if isinstance(provenance_dataset, dict)
        else None
    )
    baseline_dataset = baseline.get("dataset")
    for label, record in (
        ("baseline", baseline_dataset),
        ("fine-tuned provenance", provenance_test),
    ):
        if (
            not isinstance(record, dict)
            or record.get("sha256") is None
            or record.get("rows") is None
        ):
            raise RuntimeError(
                f"{label} test dataset identity lacks sha256/rows: {record!r}"
            )
    if (
        provenance_test["sha256"] != baseline_dataset["sha256"]
        or provenance_test["rows"] != baseline_dataset["rows"]
    ):
        raise RuntimeError(
            "baseline/fine-tuned provenance mismatch on the test dataset "
            f"(sha256/rows): baseline={baseline_dataset!r} "
            f"finetuned provenance test={provenance_test!r}"
        )

    if baseline["dataset"] != finetuned["dataset"]:
        raise RuntimeError(
            "baseline and fine-tuned runs used different test data: "
            f"{baseline['dataset']} != {finetuned['dataset']}"
        )
    for field in IDENTITY_FIELDS:
        if baseline.get(field) != finetuned.get(field):
            raise RuntimeError(
                f"baseline and fine-tuned runs disagree on {field}: "
                f"{baseline.get(field)!r} != {finetuned.get(field)!r}"
            )
    if baseline.get("decoding") != finetuned.get("decoding"):
        raise RuntimeError(
            "baseline and fine-tuned runs used different decoding settings: "
            f"{baseline.get('decoding')} != {finetuned.get('decoding')}"
        )
    if (baseline.get("precision") or {}).get("dtype") != (finetuned.get("precision") or {}).get("dtype"):
        raise RuntimeError("baseline and fine-tuned runs used different precision")

    # GPU/CUDA and ML library parity: fail closed instead of presenting one
    # run's resource numbers next to another machine's accuracy.
    baseline_cuda, finetuned_cuda = baseline.get("cuda"), finetuned.get("cuda")
    if not isinstance(baseline_cuda, dict) or not isinstance(finetuned_cuda, dict):
        raise RuntimeError(
            "cuda metadata missing from baseline/fine-tuned metrics; GPU parity "
            "cannot be verified, so resource comparison is unavailable"
        )
    if baseline_cuda != finetuned_cuda:
        raise RuntimeError(
            "baseline and fine-tuned runs ran on different CUDA/GPU environments: "
            f"{baseline_cuda} != {finetuned_cuda}"
        )
    baseline_libs = baseline.get("library_versions") or {}
    finetuned_libs = finetuned.get("library_versions") or {}
    library_mismatches = [
        (name, baseline_libs.get(name), finetuned_libs.get(name))
        for name in PARITY_LIBRARIES
        if baseline_libs.get(name) != finetuned_libs.get(name)
    ]
    if library_mismatches:
        raise RuntimeError(
            "baseline and fine-tuned runs used different ML library versions "
            f"(name, baseline, finetuned): {library_mismatches}"
        )

    rows = []
    for key, label in SCALAR_METRICS:
        left, right = baseline.get(key), finetuned.get(key)
        delta = (
            right - left
            if isinstance(left, (int, float)) and isinstance(right, (int, float))
            else ""
        )
        rows.append({"metric": key, "label": label, "baseline": left, "finetuned": right, "delta": delta})
    for aspect in baseline.get("aspects") or []:
        left = baseline["per_aspect"][aspect]["macro_f1"]
        right = finetuned["per_aspect"][aspect]["macro_f1"]
        rows.append(
            {
                "metric": f"macro_f1/{aspect}",
                "label": f"Macro-F1 {aspect}",
                "baseline": left,
                "finetuned": right,
                "delta": right - left,
            }
        )
    for key, label in RESOURCE_METRICS:
        left = (baseline.get("resource") or {})
        right = (finetuned.get("resource") or {})
        rows.append(
            {
                "metric": key,
                "label": label,
                "baseline": left.get(key),
                "finetuned": right.get(key),
                "delta": "",
            }
        )

    csv_path = run_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "label", "baseline", "finetuned", "delta"])
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return "n/a" if value is None else str(value)

    lines = [
        "# Baseline vs fine-tuned comparison",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Config: `{baseline.get('config_name') or finetuned.get('config_name')}`",
        f"- Base model: `{baseline.get('base_model')}`",
        f"- Requested revision: `{baseline.get('requested_revision')}` "
        f"(resolved `{baseline.get('resolved_revision')}`)",
        f"- Test dataset sha256: `{baseline['dataset']['sha256']}` "
        f"({baseline['dataset']['rows']} rows)",
        f"- Adapter: `{finetuned.get('adapter')}` "
        f"(train_metrics sha256 `{provenance.get('train_metrics_sha256')}`)",
        f"- GPUs: `{baseline_cuda.get('devices')}` (CUDA `{baseline_cuda.get('cuda_version')}`, "
        f"torch `{baseline_cuda.get('torch_version')}`) — identical for both runs",
        f"- Decoding: `{baseline.get('decoding')}`",
        "- Policy: final-epoch adapter (epoch 3), BF16 base, no quantization; "
        "no best-checkpoint selection.",
        "",
        "| Metric | Baseline | Fine-tuned | Delta |",
        "|---|---|---|---|",
    ]
    for row in rows[: len(SCALAR_METRICS)]:
        lines.append(
            f"| {row['label']} | {fmt(row['baseline'])} | {fmt(row['finetuned'])} | "
            f"{fmt(row['delta'])} |"
        )
    lines += [
        "",
        "## Per-aspect macro-F1",
        "",
        "| Aspect | Baseline | Fine-tuned | Delta |",
        "|---|---|---|---|",
    ]
    for row in rows[len(SCALAR_METRICS) : len(SCALAR_METRICS) + len(baseline.get("aspects") or [])]:
        lines.append(
            f"| {row['metric'].split('/', 1)[1]} | {fmt(row['baseline'])} | "
            f"{fmt(row['finetuned'])} | {fmt(row['delta'])} |"
        )
    lines += [
        "",
        "## Resources",
        "",
        "| Metric | Baseline | Fine-tuned |",
        "|---|---|---|",
    ]
    for row in rows[len(SCALAR_METRICS) + len(baseline.get("aspects") or []) :]:
        lines.append(
            f"| {row['label']} | {fmt(row['baseline'])} | {fmt(row['finetuned'])} |"
        )
    lines += [
        "",
        "> Scope: this report compares one base model with its own LoRA adapter on the",
        "> frozen HoASA test split with a shared prompt, parser and metric implementation.",
        "> Metrics are defined in `README.md`; INVALID predictions stay in the denominator.",
        "> No official ABSA leaderboard equivalence is claimed, and there is no",
        "> automatic multi-model runner.",
        "",
    ]
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {report_path.name}, {csv_path.name}")
    return {"report": str(report_path), "csv": str(csv_path), "rows": len(rows)}
