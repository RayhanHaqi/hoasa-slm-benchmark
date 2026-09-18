"""Download, verify and prepare the pinned HoASA ABSA CSV files.

`hoasa prepare` downloads the exact bytes of one IndoNLU commit, verifies the
URL/commit, SHA-256, CSV header, row counts and label values, then writes
canonical JSONL records plus `dataset_stats.json`. Duplicate and empty findings
are reported only: rows are never deleted, reordered, deduplicated, re-split,
rebalanced or augmented.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .spec import ASPECTS, DATASET, LABELS, SPLITS, canonical_json, canonical_target, url_for

RAW_DIRNAME = "raw"
PREPARED_DIRNAME = "prepared"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def raw_filename(split: str) -> str:
    """Keep the upstream file name so raw/ maps 1:1 to the pinned commit."""
    return Path(DATASET["files"][split]).name


def raw_path(data_dir: str | Path, split: str) -> Path:
    return Path(data_dir) / RAW_DIRNAME / raw_filename(split)


def prepared_path(data_dir: str | Path, split: str) -> Path:
    return Path(data_dir) / PREPARED_DIRNAME / f"{split}.jsonl"


def stats_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / PREPARED_DIRNAME / "dataset_stats.json"


def fetch_bytes(url: str) -> bytes:
    """Default fetcher: plain HTTPS GET of the pinned raw URL."""
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def verify_raw(split: str, payload: bytes) -> dict:
    """Fail closed on hash/count/header/label problems. Returns {sha256, rows}."""
    expected_hash = DATASET["sha256"][split]
    actual_hash = sha256_bytes(payload)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{split}: SHA-256 mismatch for {url_for(split)}: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    header = tuple(reader.fieldnames or ())
    if header != DATASET["header"]:
        raise ValueError(
            f"{split}: CSV header mismatch: expected {list(DATASET['header'])}, "
            f"got {list(header)}"
        )

    rows = list(reader)
    expected_rows = DATASET["rows"][split]
    if len(rows) != expected_rows:
        raise ValueError(
            f"{split}: row count mismatch: expected {expected_rows}, got {len(rows)}"
        )

    for index, row in enumerate(rows):
        if None in row:
            raise ValueError(f"{split}: row {index} has more columns than the header")
        if not str(row.get("review") or "").strip():
            raise ValueError(f"{split}: row {index} has an empty review")
        for aspect in ASPECTS:
            value = row.get(aspect)
            if value is None or str(value) == "":
                raise ValueError(f"{split}: row {index} missing value for aspect {aspect!r}")
            if value not in LABELS:
                raise ValueError(
                    f"{split}: row {index} aspect {aspect!r} has disallowed label {value!r} "
                    f"(allowed: {list(LABELS)})"
                )
    return {"sha256": actual_hash, "rows": len(rows)}


def download_raw(data_dir: str | Path, *, fetcher=fetch_bytes, force: bool = False) -> dict:
    """Download (or reuse) and verify all raw CSVs; returns per-split verification."""
    results: dict[str, dict] = {}
    for split in SPLITS:
        destination = raw_path(data_dir, split)
        if destination.is_file() and not force:
            payload = destination.read_bytes()
        else:
            payload = fetcher(url_for(split))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        results[split] = verify_raw(split, payload)
        print(
            f"{split:5s}: {results[split]['rows']:5d} rows  "
            f"sha256={results[split]['sha256'][:12]}...  {destination}"
        )
    return results


def parse_rows(split: str, rows: list[dict]) -> list[dict]:
    """Prepared record list: id, input, label (canonical JSON), target (object)."""
    records = []
    for index, row in enumerate(rows):
        target = canonical_target(row)
        records.append(
            {
                "id": f"{split}-{index:06d}",
                "input": row["review"],
                "label": canonical_json(target),
                "target": target,
            }
        )
    return records


def load_csv(path: str | Path) -> list[dict]:
    reader = csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines())
    return list(reader)


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def per_aspect_distribution(records: list[dict]) -> dict:
    """Exact four-label counts per aspect (zero counts included)."""
    distribution: dict[str, dict[str, int]] = {}
    for aspect in ASPECTS:
        counts = Counter(record["target"][aspect] for record in records)
        distribution[aspect] = {label: counts.get(label, 0) for label in LABELS}
    return distribution


def duplicate_findings(splits: dict[str, list[dict]], examples: int = 5) -> dict:
    """Report exact duplicate reviews per split and across splits; never alter."""
    per_split: dict[str, dict] = {}
    counts_by_split = {
        split: Counter(record["input"] for record in records)
        for split, records in splits.items()
    }
    for split, counts in counts_by_split.items():
        duplicated = {text: count for text, count in counts.items() if count > 1}
        per_split[split] = {
            "duplicate_groups": len(duplicated),
            "extra_rows": sum(count - 1 for count in duplicated.values()),
        }
    membership: dict[str, set[str]] = defaultdict(set)
    for split, counts in counts_by_split.items():
        for text in counts:
            membership[text].add(split)
    cross_split = sum(1 for values in membership.values() if len(values) > 1)

    shown = []
    for split, records in splits.items():
        counts = counts_by_split[split]
        for record in records:
            if counts[record["input"]] > 1 and len(shown) < examples:
                if record["id"] not in {entry["id"] for entry in shown}:
                    shown.append(
                        {"id": record["id"], "split": split, "count": counts[record["input"]]}
                    )
    return {"per_split": per_split, "cross_split": cross_split, "examples": shown}


def empty_findings(splits: dict[str, list[dict]]) -> dict:
    empty_reviews = 0
    empty_cells = {aspect: 0 for aspect in ASPECTS}
    for records in splits.values():
        for record in records:
            if not str(record["input"]).strip():
                empty_reviews += 1
            for aspect in ASPECTS:
                if not str(record["target"].get(aspect, "")):
                    empty_cells[aspect] += 1
    return {"reviews": empty_reviews, "cells": empty_cells, "missing_values": 0}


def prepare(data_dir: str | Path, *, fetcher=fetch_bytes, force: bool = False) -> dict:
    """Full `hoasa prepare`: download/verify raw, write prepared JSONL + stats."""
    data_dir = Path(data_dir)
    raw_verification = download_raw(data_dir, fetcher=fetcher, force=force)
    splits = {split: parse_rows(split, load_csv(raw_path(data_dir, split))) for split in SPLITS}

    stats = {
        "source": {
            "name": DATASET["name"],
            "repo": DATASET["repo"],
            "commit": DATASET["commit"],
            "urls": {split: url_for(split) for split in SPLITS},
        },
        "raw": {
            split: {
                "path": str(raw_path(data_dir, split).relative_to(data_dir)),
                "sha256": raw_verification[split]["sha256"],
                "rows": raw_verification[split]["rows"],
            }
            for split in SPLITS
        },
        "prepared": {},
        "aspects": list(ASPECTS),
        "labels": list(LABELS),
        "per_aspect": {},
        "findings": {
            "duplicate_reviews": duplicate_findings(splits),
            "empty": empty_findings(splits),
        },
        "notes": (
            "report only: no cleaning, deduplication, re-splitting, balancing or "
            "augmentation; official splits and exact reviews/labels preserved"
        ),
    }

    for split in SPLITS:
        records = splits[split]
        destination = prepared_path(data_dir, split)
        write_jsonl(destination, records)
        stats["prepared"][split] = {
            "path": str(destination.relative_to(data_dir)),
            "sha256": sha256_file(destination),
            "rows": len(records),
        }
        stats["per_aspect"][split] = per_aspect_distribution(records)
        print(f"\nprepared/{split}.jsonl: {len(records)} records")

    destination = stats_path(data_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    duplicates = stats["findings"]["duplicate_reviews"]
    print(
        f"\ndataset_stats.json written to {destination}\n"
        f"duplicate extra rows: {duplicates['per_split']} "
        f"(cross-split {duplicates['cross_split']})"
    )
    return stats


def majority_labels(records: list[dict]) -> dict:
    """Per-aspect majority label from train targets; ties break on LABELS order."""
    majority = {}
    for aspect in ASPECTS:
        counts = Counter(record["target"][aspect] for record in records)
        majority[aspect] = min(
            LABELS, key=lambda label: (-counts.get(label, 0), LABELS.index(label))
        )
    return majority
