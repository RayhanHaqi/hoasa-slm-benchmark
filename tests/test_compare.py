"""Comparison gate tests: mode/provenance/GPU/library parity, fail-closed."""

import json
import tempfile
import unittest
from pathlib import Path

from hoasa_benchmark.compare import compare
from hoasa_benchmark.parsing import compute_metrics, parse_output
from hoasa_benchmark.spec import ASPECTS, canonical_json

CUDA = {
    "probed": True,
    "available": True,
    "cuda_version": "12.8",
    "torch_version": "2.11.0",
    "device_count": 1,
    "devices": ["Fake GPU"],
}
LIBS = {
    "torch": "2.11.0",
    "transformers": "5.5.0",
    "peft": "0.20.0",
    "bitsandbytes": "0.50.2",
    "accelerate": "1.15.0",
}
DATASET = {"sha256": "abc", "rows": 1}  # test-split identity recorded in provenance
EVAL_DATASET = {**DATASET, "path": "/data/test.jsonl"}  # evaluation-local record


def metrics(mode, adapter=None, provenance=None, cuda=CUDA, libraries=LIBS, **overrides):
    records = [{"id": "test-000000", "target": {a: "neut" for a in ASPECTS}}]
    computed = compute_metrics(
        records, [parse_output(canonical_json(records[0]["target"]))]
    )
    payload = {
        "phase": "baseline" if mode == "base" else "finetuned",
        "mode": mode,
        "adapter": adapter,
        "provenance": provenance,
        "config_name": "unit",
        "base_model": "Org/Model",
        "kind": "language",
        "requested_revision": "rev1",
        "resolved_revision": "rev1",
        "system_prompt_sha256": "p1",
        "config_sha256": "c1",
        "dataset": dict(EVAL_DATASET),
        "decoding": {"max_new_tokens": 256},
        "precision": {"dtype": "bfloat16"},
        "cuda": cuda,
        "library_versions": libraries,
        "resource": {
            key: 1.0
            for key in (
                "peak_allocated_vram_gib",
                "peak_reserved_vram_gib",
                "mean_latency_ms",
                "median_latency_ms",
                "reviews_per_second",
                "output_tokens_per_second",
            )
        },
        **computed,
    }
    payload.update(overrides)
    return payload


def valid_pair():
    # train_metrics/manifest record the all-split dataset map; only the test
    # split is compared against the baseline's test-only record.
    provenance = {
        "train_metrics_sha256": "tm",
        "adapter": "/run/adapter",
        "config_sha256": "c1",
        "system_prompt_sha256": "p1",
        "dataset": {
            "train": {"sha256": "tr", "rows": 100},
            "val": {"sha256": "va", "rows": 10},
            "test": dict(DATASET),
        },
    }
    return metrics("base"), metrics("adapter", adapter="/run/adapter", provenance=provenance)


def write_pair(tmp, baseline, finetuned):
    run = Path(tmp) / "run"
    run.mkdir()
    (run / "baseline_metrics.json").write_text(json.dumps(baseline), encoding="utf-8")
    (run / "finetuned_metrics.json").write_text(json.dumps(finetuned), encoding="utf-8")
    return run


class CompareTests(unittest.TestCase):
    def test_valid_pair_writes_report_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_pair(tmp, *valid_pair())
            result = compare(run)
            self.assertTrue((run / "report.md").is_file())
            self.assertTrue((run / "comparison.csv").is_file())
            report = (run / "report.md").read_text(encoding="utf-8")
            self.assertIn("Fake GPU", report)
            self.assertIn("/run/adapter", report)
            self.assertGreater(result["rows"], 0)

    def test_baseline_must_be_base_mode(self):
        baseline, finetuned = valid_pair()
        baseline["mode"] = "adapter"
        with tempfile.TemporaryDirectory() as tmp:
            run = write_pair(tmp, baseline, finetuned)
            with self.assertRaises(RuntimeError):
                compare(run)

    def test_finetuned_must_be_adapter_mode(self):
        baseline, finetuned = valid_pair()
        finetuned["mode"] = "base"
        with tempfile.TemporaryDirectory() as tmp:
            run = write_pair(tmp, baseline, finetuned)
            with self.assertRaises(RuntimeError):
                compare(run)

    def test_finetuned_requires_adapter_path(self):
        baseline, finetuned = valid_pair()
        finetuned["adapter"] = None
        with tempfile.TemporaryDirectory() as tmp:
            run = write_pair(tmp, baseline, finetuned)
            with self.assertRaises(RuntimeError):
                compare(run)

    def test_provenance_mismatch_refused(self):
        for field, value in (
            ("config_sha256", "other"),
            ("system_prompt_sha256", "other"),
        ):
            with self.subTest(field=field):
                baseline, finetuned = valid_pair()
                finetuned["provenance"][field] = value
                with tempfile.TemporaryDirectory() as tmp:
                    run = write_pair(tmp, baseline, finetuned)
                    with self.assertRaises(RuntimeError):
                        compare(run)

    def test_provenance_test_split_mismatch_refused(self):
        for label, test_record in (
            ("sha256", {"sha256": "other", "rows": 1}),
            ("rows", {"sha256": "abc", "rows": 2}),
        ):
            with self.subTest(field=label):
                baseline, finetuned = valid_pair()
                finetuned["provenance"]["dataset"]["test"] = test_record
                with tempfile.TemporaryDirectory() as tmp:
                    run = write_pair(tmp, baseline, finetuned)
                    with self.assertRaises(RuntimeError):
                        compare(run)

    def test_provenance_missing_or_malformed_test_split_refused(self):
        for label, dataset in (
            ("missing test split", {"train": {"sha256": "tr", "rows": 100}}),
            ("test not a mapping", {"test": "abc"}),
            ("test missing sha256", {"test": {"rows": 1}}),
            ("flat dataset record", {"sha256": "abc", "rows": 1}),
            ("dataset not a mapping", None),
        ):
            with self.subTest(case=label):
                baseline, finetuned = valid_pair()
                finetuned["provenance"]["dataset"] = dataset
                with tempfile.TemporaryDirectory() as tmp:
                    run = write_pair(tmp, baseline, finetuned)
                    with self.assertRaises(RuntimeError):
                        compare(run)

    def test_provenance_test_path_difference_accepted(self):
        # Evaluation-local path differences are not part of the identity.
        baseline, finetuned = valid_pair()
        finetuned["provenance"]["dataset"]["test"]["path"] = "/other/test.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            run = write_pair(tmp, baseline, finetuned)
            self.assertGreater(compare(run)["rows"], 0)

    def test_missing_provenance_refused(self):
        baseline, finetuned = valid_pair()
        finetuned["provenance"] = None
        with tempfile.TemporaryDirectory() as tmp:
            run = write_pair(tmp, baseline, finetuned)
            with self.assertRaises(RuntimeError):
                compare(run)

    def test_cuda_mismatch_refused(self):
        baseline, finetuned = valid_pair()
        finetuned["cuda"] = dict(CUDA, devices=["Other GPU"])
        with tempfile.TemporaryDirectory() as tmp:
            run = write_pair(tmp, baseline, finetuned)
            with self.assertRaises(RuntimeError):
                compare(run)

    def test_missing_cuda_metadata_refused(self):
        baseline, finetuned = valid_pair()
        finetuned["cuda"] = None
        with tempfile.TemporaryDirectory() as tmp:
            run = write_pair(tmp, baseline, finetuned)
            with self.assertRaises(RuntimeError):
                compare(run)

    def test_library_version_mismatch_refused(self):
        baseline, finetuned = valid_pair()
        finetuned["library_versions"] = dict(LIBS, transformers="5.4.0")
        with tempfile.TemporaryDirectory() as tmp:
            run = write_pair(tmp, baseline, finetuned)
            with self.assertRaises(RuntimeError):
                compare(run)

    def test_dataset_and_revision_mismatch_refused(self):
        baseline, finetuned = valid_pair()
        finetuned["dataset"] = {"sha256": "other", "rows": 1}
        with tempfile.TemporaryDirectory() as tmp:
            run = write_pair(tmp, baseline, finetuned)
            with self.assertRaises(RuntimeError):
                compare(run)


if __name__ == "__main__":
    unittest.main()
