"""Synthetic dataset-prep tests: verification, canonical records, stats, no network."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hoasa_benchmark import data
from hoasa_benchmark.spec import ASPECTS, LABELS, SPLITS


def csv_payload(rows):
    header = ",".join(["review"] + list(ASPECTS))
    lines = [header]
    for review, values in rows:
        lines.append(",".join([review] + values))
    return ("\n".join(lines) + "\n").encode("utf-8")


def dict_rows(rows):
    return [
        {"review": review, **dict(zip(ASPECTS, values))} for review, values in rows
    ]


def synthetic_dataset(rows_per_split):
    """Patch-ready DATASET dict + payloads for all three splits."""
    dataset = {
        "name": "synthetic",
        "repo": "example/synthetic",
        "commit": "0" * 40,
        "files": {split: f"dataset/synthetic/{split}.csv" for split in SPLITS},
        "sha256": {},
        "rows": {},
        "url_template": "https://example.invalid/{commit}/{file}",
        "header": ("review",) + ASPECTS,
    }
    payloads = {}
    for split, rows in rows_per_split.items():
        payload = csv_payload(rows)
        dataset["sha256"][split] = hashlib.sha256(payload).hexdigest()
        dataset["rows"][split] = len(rows)
        payloads[split] = payload
    return dataset, payloads


def fetcher_for(payloads):
    return lambda url: payloads[url.rsplit("/", 1)[-1][: -len(".csv")]]


ROWS = {
    "train": [
        ("ulasan satu", ["neut"] * len(ASPECTS)),
        ("ulasan dua", ["neg"] * len(ASPECTS)),
        ("ulasan dua", ["neg"] * len(ASPECTS)),  # exact duplicate, must be preserved
    ],
    "val": [("ulasan validasi", ["pos"] * len(ASPECTS))],
    "test": [("ulasan uji", ["neg"] * len(ASPECTS))],
}


class VerifyRawTests(unittest.TestCase):
    def test_accepts_valid_payload(self):
        dataset, payloads = synthetic_dataset(ROWS)
        with mock.patch.dict(data.DATASET, dataset, clear=True):
            result = data.verify_raw("train", payloads["train"])
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["sha256"], dataset["sha256"]["train"])

    def test_rejects_wrong_hash(self):
        dataset, payloads = synthetic_dataset(ROWS)
        dataset["sha256"]["train"] = "f" * 64
        with mock.patch.dict(data.DATASET, dataset, clear=True):
            with self.assertRaises(ValueError):
                data.verify_raw("train", payloads["train"])

    def test_rejects_wrong_header(self):
        dataset, payloads = synthetic_dataset(ROWS)
        payload = payloads["train"].replace(b"review,ac", b"text,ac", 1)
        dataset["sha256"]["train"] = hashlib.sha256(payload).hexdigest()
        with mock.patch.dict(data.DATASET, dataset, clear=True):
            with self.assertRaises(ValueError):
                data.verify_raw("train", payload)

    def test_rejects_wrong_count(self):
        dataset, payloads = synthetic_dataset(ROWS)
        dataset["rows"]["train"] = 2
        with mock.patch.dict(data.DATASET, dataset, clear=True):
            with self.assertRaises(ValueError):
                data.verify_raw("train", payloads["train"])

    def test_rejects_disallowed_label(self):
        bad_rows = {"train": [("ulasan", ["positive"] + ["neut"] * (len(ASPECTS) - 1))]}
        dataset, payloads = synthetic_dataset(bad_rows)
        with mock.patch.dict(data.DATASET, dataset, clear=True):
            with self.assertRaises(ValueError):
                data.verify_raw("train", payloads["train"])

    def test_rejects_empty_review(self):
        bad_rows = {"train": [("", ["neut"] * len(ASPECTS))]}
        dataset, payloads = synthetic_dataset(bad_rows)
        with mock.patch.dict(data.DATASET, dataset, clear=True):
            with self.assertRaises(ValueError):
                data.verify_raw("train", payloads["train"])


class ParseRowsTests(unittest.TestCase):
    def test_ids_input_label_target_order(self):
        records = data.parse_rows("train", dict_rows(ROWS["train"]))
        first = records[0]
        self.assertEqual(list(first.keys()), ["id", "input", "label", "target"])
        self.assertEqual(first["id"], "train-000000")
        self.assertEqual(first["input"], "ulasan satu")
        target = json.loads(first["label"])
        self.assertEqual(target, {aspect: "neut" for aspect in ASPECTS})
        self.assertEqual(first["target"], target)
        self.assertEqual(
            first["label"], json.dumps(target, ensure_ascii=False, sort_keys=True)
        )
        self.assertEqual(records[2]["id"], "train-000002")


class MajorityTests(unittest.TestCase):
    def test_majority_and_tie_break(self):
        records = []
        for index, label in enumerate(["neg", "neg", "pos"]):
            target = {aspect: "neut" for aspect in ASPECTS}
            target["ac"] = label
            target["tv"] = "neg_pos" if index == 0 else "pos"
            records.append({"id": f"train-{index:06d}", "target": target})
        majority = data.majority_labels(records)
        self.assertEqual(majority["ac"], "neg")
        self.assertEqual(majority["tv"], "pos")
        self.assertEqual(majority["wifi"], "neut")


class PrepareTests(unittest.TestCase):
    def test_prepare_preserves_rows_and_reports_duplicates(self):
        dataset, payloads = synthetic_dataset(ROWS)
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            with mock.patch.dict(data.DATASET, dataset, clear=True):
                stats = data.prepare(data_dir, fetcher=fetcher_for(payloads))
            for split in SPLITS:
                records = data.load_jsonl(data.prepared_path(data_dir, split))
                self.assertEqual(len(records), len(ROWS[split]))
                self.assertEqual(records[0]["id"], f"{split}-000000")
            duplicates = stats["findings"]["duplicate_reviews"]
            self.assertEqual(duplicates["per_split"]["train"]["duplicate_groups"], 1)
            self.assertEqual(duplicates["per_split"]["train"]["extra_rows"], 1)
            self.assertEqual(stats["prepared"]["train"]["rows"], 3)
            self.assertTrue((data_dir / "prepared" / "dataset_stats.json").is_file())

    def test_stats_shape(self):
        dataset, payloads = synthetic_dataset(ROWS)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(data.DATASET, dataset, clear=True):
                stats = data.prepare(Path(tmp), fetcher=fetcher_for(payloads))
        for split in SPLITS:
            for aspect in ASPECTS:
                self.assertEqual(
                    sorted(stats["per_aspect"][split][aspect]), sorted(LABELS)
                )
        self.assertEqual(stats["source"]["commit"], "0" * 40)
        self.assertIn("report only", stats["notes"])


if __name__ == "__main__":
    unittest.main()
