"""Strict JSON output parser and aspect-level metric computation.

Parsing rules (single implementation shared by every phase):

- `json.loads` with a duplicate-key-detecting `object_pairs_hook`.
- Surrounding whitespace and any key order are accepted; code fences or extra
  prose make the raw output unparseable.
- Malformed JSON, a non-object top level, or duplicate keys anywhere make the
  whole output invalid: every aspect scores `INVALID`.
- A parseable object (no duplicate keys) is scored per aspect: a missing key or
  a value outside the fixed label set makes only that aspect `INVALID`.
- Extra keys make `schema_valid` false while the canonical valid values still
  score.

Metrics: four fixed classes per aspect; `INVALID` is never a fifth macro class
but stays in the denominator and counts as a false negative for the gold class.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .spec import ASPECTS, INVALID, LABELS


@dataclass
class ParsedOutput:
    raw: str
    syntax_valid: bool
    output_valid: bool
    schema_valid: bool
    duplicate_keys: bool
    values: dict = field(default_factory=dict)
    missing_keys: list = field(default_factory=list)
    invalid_values: list = field(default_factory=list)
    extra_keys: list = field(default_factory=list)
    reason: str | None = None

    def validity(self) -> dict:
        return {
            "syntax_valid": self.syntax_valid,
            "output_valid": self.output_valid,
            "schema_valid": self.schema_valid,
            "duplicate_keys": self.duplicate_keys,
            "missing_keys": list(self.missing_keys),
            "invalid_values": list(self.invalid_values),
            "extra_keys": list(self.extra_keys),
            "reason": self.reason,
        }

    def parsed_payload(self):
        """Values dict when the output is a usable object, else None."""
        if not self.output_valid:
            return None
        return {aspect: self.values[aspect] for aspect in ASPECTS}


def _blank(raw: str, syntax_valid: bool, reason: str, **fields) -> ParsedOutput:
    return ParsedOutput(
        raw=raw,
        syntax_valid=syntax_valid,
        output_valid=False,
        schema_valid=False,
        duplicate_keys=bool(fields.get("duplicate_keys", False)),
        values={aspect: None for aspect in ASPECTS},
        missing_keys=list(fields.get("missing_keys", [])),
        invalid_values=list(fields.get("invalid_values", [])),
        extra_keys=list(fields.get("extra_keys", [])),
        reason=reason,
    )


def parse_output(raw) -> ParsedOutput:
    text = raw.strip() if isinstance(raw, str) else ""
    duplicates: list[str] = []

    def hook(pairs):
        seen = {}
        for key, value in pairs:
            if key in seen:
                duplicates.append(str(key))
            seen[key] = value
        return seen

    try:
        obj = json.loads(text, object_pairs_hook=hook)
    except (json.JSONDecodeError, ValueError, TypeError):
        return _blank(text, False, "malformed_json")

    if not isinstance(obj, dict):
        return _blank(text, True, "non_object")
    if duplicates:
        return _blank(text, True, "duplicate_keys", duplicate_keys=True)

    values: dict[str, str | None] = {}
    missing_keys = []
    invalid_values = []
    for aspect in ASPECTS:
        if aspect not in obj:
            missing_keys.append(aspect)
            values[aspect] = None
            continue
        value = obj[aspect]
        if isinstance(value, str) and value.strip() in LABELS:
            values[aspect] = value.strip()
        else:
            invalid_values.append(aspect)
            values[aspect] = None
    extra_keys = sorted(str(key) for key in obj if key not in ASPECTS)

    return ParsedOutput(
        raw=text,
        syntax_valid=True,
        output_valid=True,
        schema_valid=not missing_keys and not invalid_values and not extra_keys,
        duplicate_keys=False,
        values=values,
        missing_keys=missing_keys,
        invalid_values=invalid_values,
        extra_keys=extra_keys,
        reason=None,
    )


def compute_metrics(records: list[dict], parsed: list[ParsedOutput]) -> dict:
    """Per-aspect metrics for records (`id`/`target`) and their parses."""
    if len(records) != len(parsed):
        raise ValueError(
            f"records/parses length mismatch: {len(records)} != {len(parsed)}"
        )
    total = len(records)
    syntax_ok = sum(1 for item in parsed if item.syntax_valid)
    output_ok = sum(1 for item in parsed if item.output_valid)
    schema_ok = sum(1 for item in parsed if item.schema_valid)

    per_aspect = {}
    correct_total = 0
    for aspect in ASPECTS:
        confusion = {gold: {pred: 0 for pred in (*LABELS, INVALID)} for gold in LABELS}
        class_stats = {
            label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS
        }
        for record, item in zip(records, parsed):
            gold = record["target"][aspect]
            if gold not in LABELS:
                raise ValueError(f"record {record.get('id')!r} has invalid gold {gold!r}")
            value = item.values.get(aspect)
            prediction = value if value is not None else INVALID
            confusion[gold][prediction] += 1
            if prediction == gold:
                correct_total += 1
                class_stats[gold]["tp"] += 1
            else:
                class_stats[gold]["fn"] += 1
                if prediction in class_stats:
                    class_stats[prediction]["fp"] += 1

        per_class = {}
        for label in LABELS:
            tp = class_stats[label]["tp"]
            fp = class_stats[label]["fp"]
            fn = class_stats[label]["fn"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            per_class[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
            }
        correct = sum(confusion[label][label] for label in LABELS)
        per_aspect[aspect] = {
            "support": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "macro_f1": sum(item["f1"] for item in per_class.values()) / len(LABELS),
            "per_class": per_class,
            "confusion": confusion,
            "predicted_counts": {
                pred: sum(confusion[gold][pred] for gold in LABELS)
                for pred in (*LABELS, INVALID)
            },
        }

    exact = sum(
        1
        for record, item in zip(records, parsed)
        if item.schema_valid
        and all(item.values.get(aspect) == record["target"][aspect] for aspect in ASPECTS)
    )

    return {
        "examples": total,
        "aspects": list(ASPECTS),
        "labels": list(LABELS),
        "invalid_predictions": total - output_ok,
        "syntax_valid": syntax_ok,
        "schema_valid": schema_ok,
        "output_valid": output_ok,
        "syntax_valid_rate": syntax_ok / total if total else 0.0,
        "schema_valid_rate": schema_ok / total if total else 0.0,
        "output_valid_rate": output_ok / total if total else 0.0,
        "whole_review_exact": exact,
        "whole_review_exact_accuracy": exact / total if total else 0.0,
        "overall_aspect_correct": correct_total,
        "overall_aspect_total": total * len(ASPECTS),
        "overall_aspect_accuracy": (
            correct_total / (total * len(ASPECTS)) if total else 0.0
        ),
        "mean_aspect_macro_f1": (
            sum(per_aspect[aspect]["macro_f1"] for aspect in ASPECTS) / len(ASPECTS)
        ),
        "per_aspect": per_aspect,
    }
