"""Model metadata preflight and the fail-closed BF16 inference loader.

Preflight is metadata-only: it loads `AutoConfig` plus `AutoTokenizer` (language)
or `AutoProcessor` (vision) at the pinned revision inside the project-local
`.cache/huggingface` and never touches `AutoModel*`, so no weights are
downloaded. Weight files in that cache are treated as a hard error.

The loader used by baseline/evaluate is native transformers, BF16 only:
quantized modules/config, non-BF16 base weights or CPU/disk/meta placement make
it fail closed instead of silently falling back.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .prompting import build_generation_input, build_training_text, tokenizer_of
from .runtime import sha256_text
from .spec import SPLITS, SYSTEM_PROMPT

# Any file with one of these suffixes in the project model cache means something
# tried to download weights into a metadata preflight.
WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".gguf",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".h5",
    ".msgpack",
)


def configure_cache(cache_dir: str | Path) -> Path:
    """Point Hugging Face at the isolated, git-ignored project cache.

    Both `HF_HOME` and `HF_HUB_CACHE` are forced (not setdefault) so a metadata
    preflight can never read or write the user-level cache.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return cache_dir


def weight_files(cache_dir: str | Path) -> list[str]:
    """Every weight-like file currently inside the project cache (sorted)."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return []
    found = [
        str(path.relative_to(cache_dir))
        for path in cache_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    ]
    return sorted(found)


def assert_no_weight_files(cache_dir: str | Path) -> None:
    found = weight_files(cache_dir)
    if found:
        raise RuntimeError(
            "refusing metadata preflight: weight-like files exist in the project "
            f"cache {cache_dir}: {found}"
        )


def _percentiles(values: list[int]) -> dict:
    """Nearest-rank min/p50/p95/max over a list of token counts."""
    if not values:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)

    def rank(q: float) -> int:
        index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
        return int(ordered[index])

    return {"min": int(ordered[0]), "p50": rank(0.50), "p95": rank(0.95), "max": int(ordered[-1])}


def _resolved_revision(*objects) -> str | None:
    for obj in objects:
        if obj is None:
            continue
        for holder in (getattr(obj, "config", None), getattr(obj, "init_kwargs", None)):
            if holder is None:
                continue
            value = (
                holder.get("_commit_hash")
                if isinstance(holder, dict)
                else getattr(holder, "_commit_hash", None)
            )
            if value:
                return str(value)
    return None


def preflight(config: dict, prepared_dir: str | Path, cache_dir: str | Path, out_path: str | Path) -> dict:
    """Metadata-only preflight of one config; writes and returns the report."""
    configure_cache(cache_dir)
    assert_no_weight_files(cache_dir)

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    model_cfg = config["model"]
    evaluation_cfg = config.get("evaluation") or {}
    revision = model_cfg.get("revision")
    kind = model_cfg.get("kind", "language")
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    repo_kwargs = {"revision": revision, "trust_remote_code": trust_remote_code}

    auto_config = AutoConfig.from_pretrained(model_cfg["base_model"], **repo_kwargs)
    if kind == "vision":
        processor = AutoProcessor.from_pretrained(model_cfg["base_model"], **repo_kwargs)
        tokenizer = tokenizer_of(processor)
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], **repo_kwargs)

    import json

    from .data import load_jsonl

    max_user_tokens = int(evaluation_cfg.get("max_user_tokens", 1536))
    max_new_tokens = int(evaluation_cfg.get("max_new_tokens", 256))
    max_seq_length = int(model_cfg.get("max_seq_length", 2048))
    chat_template_kwargs = model_cfg.get("chat_template_kwargs") or {}

    splits_report = {}
    for split in SPLITS:
        path = Path(prepared_dir) / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing prepared split {path}; run `hoasa prepare` first"
            )
        records = load_jsonl(path)
        user_lengths, prompt_lengths, target_lengths, full_lengths = [], [], [], []
        truncated = shrunk = violations = target_missing = eos_present = eos_appended = 0
        for record in records:
            generation = build_generation_input(
                processor if processor is not None else tokenizer,
                record["input"],
                SYSTEM_PROMPT,
                max_user_tokens=max_user_tokens,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                chat_template_kwargs=chat_template_kwargs,
            )
            training = build_training_text(
                processor if processor is not None else tokenizer,
                record["input"],
                record["label"],
                SYSTEM_PROMPT,
                max_user_tokens=max_user_tokens,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                chat_template_kwargs=chat_template_kwargs,
            )
            user_lengths.append(generation["user_tokens"])
            prompt_lengths.append(generation["prompt_tokens"])
            target_lengths.append(training["target_tokens"])
            full_lengths.append(training["full_tokens"])
            truncated += int(generation["truncated"])
            shrunk += int(generation["reduced_below_limit"])
            violations += int(training["full_tokens"] > max_seq_length)
            target_missing += int(record["label"] not in training["text"])
            eos_present += int(training["eos_present"])
            eos_appended += int(training["eos_appended"])
        splits_report[split] = {
            "path": str(path),
            "rows": len(records),
            "user_tokens": _percentiles(user_lengths),
            "prompt_tokens": _percentiles(prompt_lengths),
            "target_tokens": _percentiles(target_lengths),
            "full_tokens": _percentiles(full_lengths),
            "truncated_reviews": truncated,
            "further_shrunk_reviews": shrunk,
            "full_length_violations": violations,
            "target_missing": target_missing,
            "eos_present": eos_present,
            "eos_appended": eos_appended,
        }

    assert_no_weight_files(cache_dir)
    report = {
        "config_name": config.get("name"),
        "base_model": model_cfg["base_model"],
        "kind": kind,
        "requested_revision": revision,
        "resolved_revision": _resolved_revision(auto_config, tokenizer) or revision,
        "config_type": getattr(auto_config, "model_type", None),
        "architectures": list(getattr(auto_config, "architectures", None) or []),
        "tokenizer_class": type(tokenizer).__name__,
        "processor_class": type(processor).__name__ if processor is not None else None,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "max_user_tokens": max_user_tokens,
        "max_new_tokens": max_new_tokens,
        "max_seq_length": max_seq_length,
        "chat_template_kwargs": chat_template_kwargs,
        "model_weights_loaded": False,
        "cache_dir": str(cache_dir),
        "cache_weight_files": weight_files(cache_dir),
        "splits": splits_report,
        "notes": (
            "metadata preflight only (AutoConfig + tokenizer/processor); no AutoModel*, "
            "no weight download; lengths are tokenizer measurements of the prepared splits"
        ),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    processor: object | None
    kind: str
    revision_info: dict
    precision: dict

    @property
    def prompt_source(self):
        return self.processor if self.processor is not None else self.tokenizer


def _is_adapter_parameter(name: str) -> bool:
    lowered = name.lower()
    return "lora_" in lowered or "modules_to_save" in lowered or "original_module" in lowered


def assert_bf16_placement(model, context: str, allow_adapter: bool = False) -> dict:
    """Fail closed unless the base is BF16 on CUDA with no quantization/offload.

    Placement (CUDA, no CPU/disk/meta) is required for every floating
    parameter; `allow_adapter` only exempts adapter trainables from the BF16
    dtype check (LoRA weights may be fp32).
    """
    import torch

    problems: list[str] = []
    config = getattr(model, "config", None)
    if getattr(config, "quantization_config", None) is not None:
        problems.append("model config carries a quantization_config")
    for flag in ("is_loaded_in_4bit", "is_loaded_in_8bit"):
        if getattr(model, flag, False):
            problems.append(f"{flag} is set")
    quantized_modules = sorted(
        {
            type(module).__name__
            for module in model.modules()
            if type(module).__module__.startswith("bitsandbytes")
        }
    )
    if quantized_modules:
        problems.append(f"bitsandbytes modules present: {quantized_modules}")

    base_dtypes: Counter = Counter()
    adapter_parameters = 0
    bad_dtype: list[str] = []
    non_cuda: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        # Placement is mandatory for every floating parameter; `allow_adapter`
        # only exempts adapter trainables from the BF16 dtype check, never from
        # CUDA/meta/CPU placement.
        device = parameter.device
        if getattr(device, "type", str(device)) != "cuda":
            non_cuda.append(f"{name}@{device}")
        if allow_adapter and _is_adapter_parameter(name):
            adapter_parameters += 1
            continue
        base_dtypes[str(parameter.dtype)] += 1
        if parameter.dtype != torch.bfloat16:
            bad_dtype.append(name)
    if bad_dtype:
        problems.append(
            f"{len(bad_dtype)} base parameters are not bfloat16 (e.g. {bad_dtype[:3]})"
        )
    if non_cuda:
        problems.append(
            f"{len(non_cuda)} parameters are not on CUDA (e.g. {non_cuda[:3]})"
        )

    device_map = getattr(model, "hf_device_map", None)
    offload: list[str] = []
    if isinstance(device_map, dict):
        for name, device in device_map.items():
            if str(device).strip().lower().split(":", 1)[0] in ("cpu", "disk", "meta"):
                offload.append(f"{name}@{device}")
    if offload:
        problems.append(f"device map offloads to CPU/disk/meta (e.g. {offload[:3]})")

    if problems:
        raise RuntimeError(
            f"BF16 fail-closed check failed for {context}: " + "; ".join(problems)
        )
    return {
        "base_parameter_dtypes": dict(base_dtypes),
        "adapter_parameter_count": adapter_parameters,
        "quantized_modules": [],
        "offload": [],
        "checked": "bf16+cuda, no quantization/offload",
    }


def restore_marked_fp32_parameters(model) -> int:
    """Undo Unsloth's deliberate fp32 pre-cast, only where Unsloth marks it.

    `Fast*Model.from_pretrained` sets `_pre_set_compute_dtype = torch.float32`
    on some modules (e.g. Qwen3.5 norms) and casts their parameters to fp32,
    which the strict full-BF16 gate rejects. Cast back to bfloat16 only the
    direct floating parameters of modules carrying that exact mark; unmarked
    non-BF16 parameters are left untouched so the gate still fails closed.
    Returns the number of parameters cast back to bfloat16.
    """
    import torch

    restored = 0
    for module in model.modules():
        if getattr(module, "_pre_set_compute_dtype", None) != torch.float32:
            continue
        for parameter in module.parameters(recurse=False):
            if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
                parameter.data = parameter.data.to(torch.bfloat16)
                restored += 1
    return restored


def load_for_inference(config: dict, adapter: str | None = None) -> LoadedModel:
    """Pinned native transformers load, BF16 base only, for baseline/evaluate."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    model_cfg = config["model"]
    kind = model_cfg.get("kind", "language")
    revision = model_cfg.get("revision")
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    repo_kwargs = {"revision": revision, "trust_remote_code": trust_remote_code}
    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
        **repo_kwargs,
    }

    if kind == "vision":
        processor = AutoProcessor.from_pretrained(model_cfg["base_model"], **repo_kwargs)
        tokenizer = tokenizer_of(processor)
        model = AutoModelForImageTextToText.from_pretrained(
            model_cfg["base_model"], **model_kwargs
        )
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], **repo_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model"], **model_kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    precision = assert_bf16_placement(
        model,
        context=f"{'adapter' if adapter else 'base'} {model_cfg['base_model']}",
        allow_adapter=bool(adapter),
    )
    model.eval()
    resolved = _resolved_revision(model, tokenizer, processor)
    revision_info = {
        "requested_revision": revision,
        "resolved_revision": resolved or revision,
        "resolved_revision_source": "model_config" if resolved else "requested",
    }
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        kind=kind,
        revision_info=revision_info,
        precision={"dtype": "bfloat16", "quantization": None, **precision},
    )
