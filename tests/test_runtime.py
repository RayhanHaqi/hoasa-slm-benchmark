"""Run-directory snapshot/verify tests, config invariants and prompt checks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from hoasa_benchmark import runtime
from hoasa_benchmark.spec import ASPECTS, LABELS, SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO_ROOT / "configs").glob("*.yaml"))


def synthetic_prepared_dir(root: Path) -> Path:
    prepared = root / "prepared"
    prepared.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (prepared / f"{split}.jsonl").write_text(
            json.dumps(
                {
                    "id": f"{split}-000000",
                    "input": "ulasan",
                    "label": "{}",
                    "target": {},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    (prepared / "dataset_stats.json").write_text('{"ok": true}\n', encoding="utf-8")
    return prepared


def minimal_config(revision="rev1", model="Org/Model", target_modules=None):
    return {
        "name": "unit-test",
        "model": {
            "base_model": model,
            "revision": revision,
            "kind": "language",
            "max_seq_length": 2048,
            "chat_template_kwargs": {},
        },
        "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "bias": "none", "target_modules": target_modules},
        "training": {"bf16": True, "fp16": False},
        "evaluation": {"max_user_tokens": 1536, "max_new_tokens": 256},
    }


class WorkspaceTests(unittest.TestCase):
    def test_snapshot_then_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = synthetic_prepared_dir(root)
            run_dir = root / "run"
            config = minimal_config()
            manifest = runtime.ensure_workspace(
                run_dir,
                config=config,
                config_path=None,
                prepared_dir=prepared,
                phase="baseline",
            )
            self.assertEqual(manifest["dataset"]["test"]["rows"], 1)
            self.assertTrue((run_dir / "run_manifest.json").is_file())
            self.assertTrue((run_dir / "environment.json").is_file())
            self.assertTrue((run_dir / "resolved_config.yaml").is_file())
            for split in ("train", "val", "test"):
                self.assertTrue((run_dir / f"{split}.jsonl").is_file())

            runtime.ensure_workspace(
                run_dir,
                config=config,
                config_path=None,
                prepared_dir=prepared,
                phase="train",
            )

    def test_revision_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = synthetic_prepared_dir(root)
            run_dir = root / "run"
            runtime.ensure_workspace(
                run_dir, config=minimal_config(), config_path=None,
                prepared_dir=prepared, phase="baseline",
            )
            with self.assertRaises(RuntimeError):
                runtime.ensure_workspace(
                    run_dir,
                    config=minimal_config(revision="rev2"),
                    config_path=None,
                    prepared_dir=prepared,
                    phase="train",
                )

    def test_model_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = synthetic_prepared_dir(root)
            run_dir = root / "run"
            runtime.ensure_workspace(
                run_dir, config=minimal_config(), config_path=None,
                prepared_dir=prepared, phase="baseline",
            )
            with self.assertRaises(RuntimeError):
                runtime.ensure_workspace(
                    run_dir,
                    config=minimal_config(model="Other/Model"),
                    config_path=None,
                    prepared_dir=prepared,
                    phase="train",
                )

    def test_dataset_snapshot_tamper_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = synthetic_prepared_dir(root)
            run_dir = root / "run"
            runtime.ensure_workspace(
                run_dir, config=minimal_config(), config_path=None,
                prepared_dir=prepared, phase="baseline",
            )
            (run_dir / "test.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                runtime.ensure_workspace(
                    run_dir, config=minimal_config(), config_path=None,
                    prepared_dir=prepared, phase="finetuned",
                )

    def test_stale_prepared_source_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = synthetic_prepared_dir(root)
            run_dir = root / "run"
            runtime.ensure_workspace(
                run_dir, config=minimal_config(), config_path=None,
                prepared_dir=prepared, phase="baseline",
            )
            (prepared / "test.jsonl").write_text("changed\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                runtime.ensure_workspace(
                    run_dir, config=minimal_config(), config_path=None,
                    prepared_dir=prepared, phase="finetuned",
                )

    def test_majority_run_manifest_has_no_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = synthetic_prepared_dir(root)
            run_dir = root / "majority"
            manifest = runtime.ensure_workspace(
                run_dir, config=None, config_path=None,
                prepared_dir=prepared, phase="majority-baseline",
            )
            self.assertIsNone(manifest["model"])
            self.assertFalse((run_dir / "resolved_config.yaml").exists())


class ConfigInvariantTests(unittest.TestCase):
    EXPECTED = {
        "01-qwen3-1.7b": (
            "Qwen/Qwen3-1.7B",
            "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "language",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        "02-qwen3.5-0.8b": (
            "Qwen/Qwen3.5-0.8B",
            "2fc06364715b967f1860aea9cf38778875588b17",
            "vision",
            "all-linear",
        ),
        "03-lfm2.5-1.2b-instruct": (
            "LiquidAI/LFM2.5-1.2B-Instruct",
            "0f604ada3f766f9f257460c4c9f0b5d6f69d431b",
            "language",
            ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
        ),
        "04-qwen3.5-2b": (
            "Qwen/Qwen3.5-2B",
            "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "vision",
            "all-linear",
        ),
    }

    def test_four_configs_present(self):
        self.assertEqual(len(CONFIGS), 4)

    def test_revisions_kinds_targets(self):
        for path in CONFIGS:
            config = runtime.load_config(path)
            base, revision, kind, targets = self.EXPECTED[config["name"]]
            self.assertEqual(config["model"]["base_model"], base)
            self.assertEqual(config["model"]["revision"], revision)
            self.assertEqual(config["model"]["kind"], kind)
            self.assertEqual(config["lora"]["target_modules"], targets)

    def test_shared_recipe(self):
        for path in CONFIGS:
            config = runtime.load_config(path)
            training = config["training"]
            self.assertEqual(training["per_device_train_batch_size"], 2)
            self.assertEqual(training["per_device_eval_batch_size"], 1)
            self.assertEqual(training["gradient_accumulation_steps"], 4)
            self.assertEqual(training["num_train_epochs"], 3)
            self.assertEqual(training["learning_rate"], 0.0002)
            self.assertEqual(training["lr_scheduler_type"], "cosine")
            self.assertEqual(training["warmup_ratio"], 0.05)
            self.assertEqual(training["weight_decay"], 0.01)
            self.assertEqual(training["seed"], 42)
            self.assertEqual(training["optim"], "adamw_8bit")
            self.assertTrue(training["bf16"])
            self.assertFalse(training["fp16"])
            self.assertFalse(config["model"]["load_in_4bit"])
            self.assertEqual(config["lora"]["r"], 16)
            self.assertEqual(config["lora"]["alpha"], 32)
            self.assertEqual(config["lora"]["dropout"], 0.0)
            self.assertNotIn("resume_from_checkpoint", training)
            self.assertNotIn("resume_from_checkpoint", config["model"])
            self.assertEqual(config["evaluation"]["max_user_tokens"], 1536)
            self.assertEqual(config["evaluation"]["max_new_tokens"], 256)
            self.assertFalse(config["evaluation"]["do_sample"])

    def test_vision_freeze_flags(self):
        for path in CONFIGS:
            config = runtime.load_config(path)
            if config["model"]["kind"] != "vision":
                continue
            lora = config["lora"]
            self.assertFalse(lora["finetune_vision_layers"])
            self.assertTrue(lora["finetune_language_layers"])
            self.assertTrue(lora["finetune_attention_modules"])
            self.assertTrue(lora["finetune_mlp_modules"])

    def test_chat_template_kwargs(self):
        expected = {
            "01-qwen3-1.7b": {"enable_thinking": False},
            "02-qwen3.5-0.8b": {"enable_thinking": False},
            "03-lfm2.5-1.2b-instruct": {},
            "04-qwen3.5-2b": {"enable_thinking": False},
        }
        for path in CONFIGS:
            config = runtime.load_config(path)
            self.assertEqual(
                config["model"]["chat_template_kwargs"], expected[config["name"]]
            )

    def test_config_hash_changes_with_revision(self):
        left = minimal_config(revision="a")
        right = minimal_config(revision="b")
        self.assertNotEqual(runtime.config_sha256(left), runtime.config_sha256(right))


class CudaMetadataTests(unittest.TestCase):
    @staticmethod
    def fake_torch(available=True, devices=("Fake GPU",), cuda_version="12.8", version="2.11.0"):
        return SimpleNamespace(
            __version__=version,
            version=SimpleNamespace(cuda=cuda_version),
            cuda=SimpleNamespace(
                is_available=lambda: available,
                device_count=lambda: len(devices),
                get_device_name=lambda index: devices[index],
            ),
        )

    def test_injected_available(self):
        metadata = runtime.cuda_metadata(self.fake_torch(devices=("GPU-A", "GPU-B")))
        self.assertEqual(
            metadata,
            {
                "probed": True,
                "available": True,
                "cuda_version": "12.8",
                "torch_version": "2.11.0",
                "device_count": 2,
                "devices": ["GPU-A", "GPU-B"],
            },
        )

    def test_injected_unavailable(self):
        metadata = runtime.cuda_metadata(self.fake_torch(available=False))
        self.assertTrue(metadata["probed"])
        self.assertFalse(metadata["available"])
        self.assertEqual(metadata["devices"], [])
        self.assertEqual(metadata["device_count"], 0)

    def test_environment_includes_cuda(self):
        environment = runtime.environment(self.fake_torch())
        self.assertEqual(environment["cuda"]["devices"], ["Fake GPU"])
        self.assertIn("packages", environment)


class PromptTests(unittest.TestCase):
    def test_prompt_mentions_every_aspect_and_label(self):
        for aspect in ASPECTS:
            self.assertIn(aspect, SYSTEM_PROMPT)
        for label in LABELS:
            self.assertIn(f"{label}:", SYSTEM_PROMPT)
        self.assertIn("ABSA", SYSTEM_PROMPT)
        self.assertIn("JSON", SYSTEM_PROMPT)
        self.assertIn("campuran", SYSTEM_PROMPT)

    def test_prompt_is_ascii_free_indonesian(self):
        self.assertIn("ulasan hotel", SYSTEM_PROMPT)
        self.assertIn("sentimen", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
