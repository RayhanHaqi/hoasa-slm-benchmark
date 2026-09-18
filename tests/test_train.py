"""Training-side invariants: vision PEFT fix, loss flags, BF16 gate, cache scan."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from hoasa_benchmark import models
from hoasa_benchmark.train import (
    _loss_policy,
    assert_no_vision_trainables,
    normalize_target_modules,
    peft_kwargs,
    trainable_parameter_summary,
)

VISION_LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.0,
    "bias": "none",
    "target_modules": "all-linear",
    "gradient_checkpointing": "unsloth",
    "random_state": 42,
    "finetune_vision_layers": False,
    "finetune_language_layers": True,
    "finetune_attention_modules": True,
    "finetune_mlp_modules": True,
}
LANGUAGE_LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj"],
    "gradient_checkpointing": "unsloth",
    "random_state": 42,
}


class FakeParam:
    def __init__(self, dtype, device_type="cuda", floating=True):
        self.dtype = dtype
        self.device = SimpleNamespace(type=device_type, index=0 if device_type == "cuda" else None)
        self._floating = floating

    def is_floating_point(self):
        return self._floating

    # Minimal in-place cast support for models.restore_marked_fp32_parameters.
    @property
    def data(self):
        return self

    @data.setter
    def data(self, value):
        self.dtype = value.dtype

    def to(self, dtype, *args, **kwargs):
        if dtype == self.dtype:
            return self
        return FakeParam(dtype, self.device.type, self._floating)


class FakeModule:
    def __init__(self, name, parameters=None, pre_set_compute_dtype=None):
        self.name = name
        self._parameters = dict(parameters or {})
        if pre_set_compute_dtype is not None:
            self._pre_set_compute_dtype = pre_set_compute_dtype

    def parameters(self, recurse=True):
        return list(self._parameters.values())


class FakeModel:
    def __init__(self, parameters, quantization_config=None, hf_device_map=None, modules=None):
        self._parameters = parameters
        self.config = SimpleNamespace(quantization_config=quantization_config)
        self.hf_device_map = hf_device_map
        self._modules = list(modules or [])

    def named_parameters(self):
        return list(self._parameters.items())

    def modules(self):
        return list(self._modules)


class PeftKwargsTests(unittest.TestCase):
    def test_vision_all_linear_passed_as_none_at_call_site(self):
        applied = peft_kwargs(VISION_LORA, "vision", for_peft_call=True)
        self.assertIsNone(applied["target_modules"])
        self.assertFalse(applied["finetune_vision_layers"])
        self.assertTrue(applied["finetune_language_layers"])

    def test_recorded_vision_kwargs_keep_requested_value(self):
        recorded = peft_kwargs(VISION_LORA, "vision")
        self.assertEqual(recorded["target_modules"], "all-linear")

    def test_language_targets_unchanged(self):
        applied = peft_kwargs(LANGUAGE_LORA, "language", for_peft_call=True)
        self.assertEqual(applied["target_modules"], ["q_proj", "k_proj"])
        self.assertNotIn("finetune_vision_layers", applied)

    def test_normalize_target_modules(self):
        self.assertEqual(normalize_target_modules("all-linear"), "all-linear")
        self.assertEqual(normalize_target_modules(("a", "b")), ["a", "b"])

    def test_loss_policy_is_full_sequence(self):
        policy = _loss_policy()
        self.assertFalse(policy["assistant_only_loss"])
        self.assertFalse(policy["completion_only_loss"])
        self.assertTrue(policy["full_sequence_loss"])
        self.assertTrue(policy["prediction_loss_only"])


class VisionTrainableTests(unittest.TestCase):
    def test_summary_flags_vision_like_names(self):
        class Model:
            def named_parameters(self):
                return [
                    ("base_model.model.language.layers.0.self_attn.q_proj.lora_A.weight",
                     torch.nn.Parameter(torch.zeros(2, 2))),
                    ("base_model.model.vision_tower.blocks.0.attn.qkv.lora_A.weight",
                     torch.nn.Parameter(torch.zeros(2, 2))),
                ]

        class Model2:
            def named_parameters(self):
                return [
                    ("base_model.model.language.layers.0.self_attn.q_proj.lora_A.weight",
                     torch.nn.Parameter(torch.zeros(2, 2))),
                ]

        summary = trainable_parameter_summary(Model())
        self.assertEqual(summary["vision_like_count"], 1)
        with self.assertRaises(RuntimeError):
            assert_no_vision_trainables(summary)

        clean = trainable_parameter_summary(Model2())
        self.assertEqual(clean["vision_like_count"], 0)
        assert_no_vision_trainables(clean)

    def test_unavailable_summary_fails_closed(self):
        with self.assertRaises(RuntimeError):
            assert_no_vision_trainables({"status": "unavailable"})


class Bf16GateTests(unittest.TestCase):
    def test_accepts_bf16_cuda(self):
        model = FakeModel({"layer.weight": FakeParam(torch.bfloat16)})
        stats = models.assert_bf16_placement(model, "unit")
        self.assertEqual(stats["base_parameter_dtypes"], {"torch.bfloat16": 1})
        self.assertEqual(stats["quantized_modules"], [])

    def test_rejects_non_bf16_base(self):
        model = FakeModel({"layer.weight": FakeParam(torch.float32)})
        with self.assertRaises(RuntimeError):
            models.assert_bf16_placement(model, "unit")

    def test_allows_adapter_trainables(self):
        model = FakeModel(
            {
                "base_model.model.layers.0.q_proj.base_layer.weight": FakeParam(torch.bfloat16),
                "base_model.model.layers.0.q_proj.lora_A.default.weight": FakeParam(torch.float32),
            }
        )
        stats = models.assert_bf16_placement(model, "unit", allow_adapter=True)
        self.assertEqual(stats["adapter_parameter_count"], 1)

    def test_adapter_trainable_on_cpu_still_fails(self):
        model = FakeModel(
            {
                "base_model.model.layers.0.q_proj.base_layer.weight": FakeParam(torch.bfloat16),
                "base_model.model.layers.0.q_proj.lora_A.default.weight": FakeParam(
                    torch.float32, device_type="cpu"
                ),
            }
        )
        with self.assertRaises(RuntimeError):
            models.assert_bf16_placement(model, "unit", allow_adapter=True)

    def test_adapter_trainable_on_meta_still_fails(self):
        model = FakeModel(
            {
                "base_model.model.layers.0.q_proj.base_layer.weight": FakeParam(torch.bfloat16),
                "base_model.model.layers.0.q_proj.lora_A.default.weight": FakeParam(
                    torch.bfloat16, device_type="meta"
                ),
            }
        )
        with self.assertRaises(RuntimeError):
            models.assert_bf16_placement(model, "unit", allow_adapter=True)

    def test_rejects_cpu_placement(self):
        model = FakeModel({"layer.weight": FakeParam(torch.bfloat16, device_type="cpu")})
        with self.assertRaises(RuntimeError):
            models.assert_bf16_placement(model, "unit")

    def test_rejects_quantization_config(self):
        model = FakeModel(
            {"layer.weight": FakeParam(torch.bfloat16)},
            quantization_config=SimpleNamespace(load_in_4bit=True),
        )
        with self.assertRaises(RuntimeError):
            models.assert_bf16_placement(model, "unit")

    def test_rejects_offload_device_map(self):
        model = FakeModel(
            {"layer.weight": FakeParam(torch.bfloat16)},
            hf_device_map={"layer": "cpu"},
        )
        with self.assertRaises(RuntimeError):
            models.assert_bf16_placement(model, "unit")


class UnslothFp32RestoreTests(unittest.TestCase):
    def test_restores_marked_fp32_direct_parameters(self):
        weight = FakeParam(torch.float32)
        bias = FakeParam(torch.float32)
        integer = FakeParam(torch.int64, floating=False)
        marked = FakeModule(
            "norm",
            {"weight": weight, "bias": bias, "count": integer},
            pre_set_compute_dtype=torch.float32,
        )
        model = FakeModel(
            {"norm.weight": weight, "norm.bias": bias, "norm.count": integer},
            modules=[marked],
        )
        restored = models.restore_marked_fp32_parameters(model)
        self.assertEqual(restored, 2)
        self.assertEqual(weight.dtype, torch.bfloat16)
        self.assertEqual(bias.dtype, torch.bfloat16)
        self.assertEqual(integer.dtype, torch.int64)
        stats = models.assert_bf16_placement(model, "unit")
        self.assertEqual(stats["base_parameter_dtypes"], {"torch.bfloat16": 2})

    def test_unmarked_fp32_still_fails_gate(self):
        weight = FakeParam(torch.float32)
        unmarked = FakeModule("norm", {"weight": weight})
        model = FakeModel({"norm.weight": weight}, modules=[unmarked])
        self.assertEqual(models.restore_marked_fp32_parameters(model), 0)
        self.assertEqual(weight.dtype, torch.float32)
        with self.assertRaises(RuntimeError):
            models.assert_bf16_placement(model, "unit")

    def test_bf16_marked_is_noop(self):
        weight = FakeParam(torch.bfloat16)
        marked = FakeModule(
            "norm", {"weight": weight}, pre_set_compute_dtype=torch.float32
        )
        model = FakeModel({"norm.weight": weight}, modules=[marked])
        self.assertEqual(models.restore_marked_fp32_parameters(model), 0)
        self.assertEqual(weight.dtype, torch.bfloat16)

    def test_no_mark_is_noop(self):
        weight = FakeParam(torch.bfloat16)
        plain = FakeModule("norm", {"weight": weight})
        model = FakeModel({"norm.weight": weight}, modules=[plain])
        self.assertEqual(models.restore_marked_fp32_parameters(model), 0)
        self.assertEqual(weight.dtype, torch.bfloat16)


class WeightFileScanTests(unittest.TestCase):
    def test_detects_and_rejects_weight_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "hub" / "models--org--model").mkdir(parents=True)
            self.assertEqual(models.weight_files(cache), [])
            models.assert_no_weight_files(cache)

            weight = cache / "hub" / "models--org--model" / "model.safetensors"
            weight.write_bytes(b"not a real weight")
            found = models.weight_files(cache)
            self.assertEqual(len(found), 1)
            self.assertTrue(found[0].endswith("model.safetensors"))
            with self.assertRaises(RuntimeError):
                models.assert_no_weight_files(cache)

    def test_tokenizer_like_files_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "tokenizer.json").write_text("{}", encoding="utf-8")
            (cache / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            models.assert_no_weight_files(cache)

    def test_configure_cache_forces_project_cache(self):
        import os
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HF_HOME": "/tmp/elsewhere"}, clear=False):
                models.configure_cache(Path(tmp))
                self.assertEqual(os.environ["HF_HOME"], str(Path(tmp)))
                self.assertEqual(
                    os.environ["HF_HUB_CACHE"], str(Path(tmp) / "hub")
                )


if __name__ == "__main__":
    unittest.main()
