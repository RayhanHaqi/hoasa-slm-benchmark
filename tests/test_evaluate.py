"""Fine-tuned provenance: run-local adapter resolution and train_metrics checks."""

import json
import tempfile
import unittest
from pathlib import Path

from hoasa_benchmark import evaluate
from hoasa_benchmark.runtime import sha256_file

DATASET = {
    "train": {"sha256": "t", "rows": 1},
    "val": {"sha256": "v", "rows": 1},
    "test": {"sha256": "e", "rows": 1},
}


def config(adapter_subdir="adapter"):
    return {
        "name": "unit",
        "model": {"base_model": "Org/Model", "revision": "rev1", "kind": "language"},
        "training": {"adapter_subdir": adapter_subdir},
        "evaluation": {},
    }


def make_run(tmp, adapter_subdir="adapter", train_metrics_overrides=None):
    run = Path(tmp) / "run"
    adapter = run / adapter_subdir
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    manifest = {
        "config_sha256": "c1",
        "system_prompt_sha256": "p1",
        "model": {"revision": "rev1", "base_model": "Org/Model"},
        "dataset": DATASET,
    }
    train_metrics = {
        "config_sha256": "c1",
        "system_prompt_sha256": "p1",
        "dataset": DATASET,
        "base_model": "Org/Model",
        "requested_revision": "rev1",
        "resolved_revision": "rev1",
        "adapter_dir": str(adapter),
    }
    train_metrics.update(train_metrics_overrides or {})
    (run / "train_metrics.json").write_text(
        json.dumps(train_metrics), encoding="utf-8"
    )
    return run, adapter, manifest


class RunLocalAdapterTests(unittest.TestCase):
    def test_accepts_configured_run_local_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, adapter, _ = make_run(tmp)
            resolved = evaluate.resolve_run_local_adapter(run, config(), str(adapter))
            self.assertEqual(resolved, adapter.resolve())

    def test_refuses_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, _, _ = make_run(tmp)
            with self.assertRaises(RuntimeError):
                evaluate.resolve_run_local_adapter(run, config(), "base")

    def test_refuses_adapter_outside_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, _, _ = make_run(tmp)
            foreign = Path(tmp) / "foreign"
            foreign.mkdir()
            (foreign / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                evaluate.resolve_run_local_adapter(run, config(), str(foreign))

    def test_refuses_path_without_adapter_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, _, _ = make_run(tmp)
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(RuntimeError):
                evaluate.resolve_run_local_adapter(run, config(), str(empty))

    def test_refuses_escaping_adapter_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, _, _ = make_run(tmp)
            with self.assertRaises(RuntimeError):
                evaluate.resolve_run_local_adapter(
                    run, config(adapter_subdir="../../escape"), str(run / "adapter")
                )


class ProvenanceTests(unittest.TestCase):
    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, adapter, manifest = make_run(tmp)
            provenance = evaluate.verify_finetuned_provenance(
                run, config(), manifest, adapter.resolve()
            )
            self.assertEqual(provenance["adapter"], str(adapter.resolve()))
            self.assertEqual(provenance["dataset"], DATASET)
            self.assertEqual(
                provenance["train_metrics_sha256"],
                sha256_file(run / "train_metrics.json"),
            )

    def test_missing_train_metrics_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, adapter, manifest = make_run(tmp)
            (run / "train_metrics.json").unlink()
            with self.assertRaises(FileNotFoundError):
                evaluate.verify_finetuned_provenance(
                    run, config(), manifest, adapter.resolve()
                )

    def test_each_mismatch_refused(self):
        for field, value in (
            ("config_sha256", "other"),
            ("system_prompt_sha256", "other"),
            ("dataset", {"train": {"sha256": "x", "rows": 1}}),
            ("base_model", "Other/Model"),
            ("requested_revision", "rev2"),
            ("resolved_revision", "rev2"),
            ("adapter_dir", "/somewhere/else"),
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    run, adapter, manifest = make_run(
                        tmp, train_metrics_overrides={field: value}
                    )
                    with self.assertRaises(RuntimeError):
                        evaluate.verify_finetuned_provenance(
                            run, config(), manifest, adapter.resolve()
                        )

    def test_missing_resolved_revision_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, adapter, manifest = make_run(
                tmp, train_metrics_overrides={"resolved_revision": None}
            )
            provenance = evaluate.verify_finetuned_provenance(
                run, config(), manifest, adapter.resolve()
            )
            self.assertIsNone(provenance["resolved_revision"])


if __name__ == "__main__":
    unittest.main()
