"""Unsloth LoRA SFT: BF16 base, r16/alpha32, full-sequence loss, final epoch.

Import order (datasets, unsloth, trl) matches the reference implementation:
Unsloth patches TRL so the bound `SFTConfig`/`SFTTrainer` are the classes TRL's
own checks expect. Evaluation is loss-only (the local trainer subclass bypasses
Unsloth's logits-forcing `prediction_step`).

Policy: three epochs at lr 2e-4, cosine schedule, warmup 0.05, weight decay
0.01, seed 42, micro-batch 2 with gradient accumulation 4 (effective 8), eval
batch 1, `adamw_8bit`, validation loss and checkpoints each epoch, and the
adapter saved from the final epoch (epoch 3). No best-checkpoint selection and
no resume path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .prompting import build_training_text
from .runtime import ensure_workspace, prepared_dir_for
from .spec import SYSTEM_PROMPT

VISION_LIKE_PARAMETER_TOKENS = (
    "vision",
    "visual",
    "image",
    "pixel",
    "patch",
    "multi_modal",
    "multimodal",
    "mm_projector",
    "projector",
)


def normalize_target_modules(value):
    """Keep YAML strings (e.g. `all-linear`) as strings; copy lists/tuples."""
    if isinstance(value, str):
        return value
    return list(value)


def peft_kwargs(lora_cfg: dict, kind: str, *, for_peft_call: bool = False) -> dict:
    """Fast*Model.get_peft_model kwargs; vision adds the four freeze flags.

    With `for_peft_call=True` (the training call site), a vision config whose
    resolved target_modules is the literal `"all-linear"` is passed as `None`:
    Unsloth 2026.9.4 `FastVisionModel.get_peft_model` treats that literal as
    "force every finetune_* flag True" and silently overrides
    `finetune_vision_layers=False`; `None` routes module selection through
    `get_peft_regex` with the explicit flags instead. Recorded metrics keep the
    requested value.
    """
    target_modules = normalize_target_modules(lora_cfg["target_modules"])
    if for_peft_call and kind == "vision" and target_modules == "all-linear":
        target_modules = None
    kwargs = {
        "r": int(lora_cfg["r"]),
        "lora_alpha": int(lora_cfg["alpha"]),
        "lora_dropout": float(lora_cfg["dropout"]),
        "target_modules": target_modules,
        "bias": lora_cfg["bias"],
        "use_gradient_checkpointing": lora_cfg.get("gradient_checkpointing", "unsloth"),
        "random_state": int(lora_cfg.get("random_state", 42)),
    }
    if kind == "vision":
        kwargs.update(
            finetune_vision_layers=bool(lora_cfg.get("finetune_vision_layers", False)),
            finetune_language_layers=bool(
                lora_cfg.get("finetune_language_layers", True)
            ),
            finetune_attention_modules=bool(
                lora_cfg.get("finetune_attention_modules", True)
            ),
            finetune_mlp_modules=bool(lora_cfg.get("finetune_mlp_modules", True)),
        )
    return kwargs


def resolve_eos_token(processor, model=None) -> str | None:
    """Valid EOS token string for `SFTConfig.eos_token`, else None."""
    tokenizer = getattr(processor, "tokenizer", processor)

    def _valid(token):
        if not token:
            return None
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            return None
        if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
            return None
        return token

    token = _valid(getattr(tokenizer, "eos_token", None))
    if token is not None:
        return token

    eos_ids = getattr(getattr(model, "config", None), "eos_token_id", None)
    if eos_ids is None:
        return None
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]
    for eos_id in eos_ids:
        try:
            token = tokenizer.convert_ids_to_tokens(eos_id)
        except Exception:
            continue
        token = _valid(token)
        if token is not None:
            return token
    return None


def loss_only_sft_trainer_class(base):
    """SFTTrainer subclass with a logits-free `prediction_loss_only` eval step."""

    class LossOnlySFTTrainer(base):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if not prediction_loss_only:
                return super().prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys
                )

            import torch

            inputs = self._prepare_inputs(inputs)
            previous_return_logits = os.environ.get("UNSLOTH_RETURN_LOGITS")
            os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
            try:
                with torch.no_grad(), self.compute_loss_context_manager():
                    try:
                        num_items_in_batch = self._get_num_items_in_batch(
                            [inputs], self.args.device
                        )
                    except (AttributeError, TypeError):
                        num_items_in_batch = None
                    loss = self.compute_loss(
                        model,
                        inputs,
                        return_outputs=False,
                        num_items_in_batch=num_items_in_batch,
                    )
            finally:
                if previous_return_logits is None:
                    os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
                else:
                    os.environ["UNSLOTH_RETURN_LOGITS"] = previous_return_logits
            return loss.mean().detach(), None, None

    return LossOnlySFTTrainer


def _reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover
        pass


def _cuda_peak_gib() -> tuple[float, float]:
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated() / 1024**3,
        torch.cuda.max_memory_reserved() / 1024**3,
    )


def trainable_parameter_summary(model, name_limit: int = 200, vision_name_limit: int = 50) -> dict:
    """Post-PEFT trainable parameter counts/names, including vision-like names."""
    if not hasattr(model, "named_parameters"):
        return {
            "status": "unavailable",
            "reason": f"{type(model).__name__} exposes no named_parameters()",
        }
    names: list[str] = []
    sizes: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if not getattr(parameter, "requires_grad", False):
            continue
        names.append(name)
        sizes[name] = int(parameter.numel())
    names.sort()
    vision_like = [
        name
        for name in names
        if any(token in name.lower() for token in VISION_LIKE_PARAMETER_TOKENS)
    ]
    return {
        "status": "available",
        "count": len(names),
        "numel": sum(sizes.values()),
        "names": names[:name_limit],
        "names_truncated": len(names) > name_limit,
        "vision_like_count": len(vision_like),
        "vision_like_numel": sum(sizes[name] for name in vision_like),
        "vision_like_names": vision_like[:vision_name_limit],
    }


def assert_no_vision_trainables(summary: dict) -> None:
    if summary.get("status") != "available":
        raise RuntimeError(
            "cannot verify the vision freeze: trainable parameter summary is unavailable"
        )
    if summary["vision_like_count"]:
        raise RuntimeError(
            "vision-like parameters are trainable despite finetune_vision_layers=False: "
            f"{summary['vision_like_names']}"
        )


def _percentiles(values: list[int]) -> dict:
    import math

    if not values:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": int(ordered[0]),
        "p50": int(ordered[math.ceil(0.50 * len(ordered)) - 1]),
        "p95": int(ordered[math.ceil(0.95 * len(ordered)) - 1]),
        "max": int(ordered[-1]),
    }


def _loss_policy() -> dict:
    return {
        "assistant_only_loss": False,
        "completion_only_loss": False,
        "full_sequence_loss": True,
        "prediction_loss_only": True,
        "dataset_text_field": "text",
    }


def train(config: dict, run_dir: str | Path) -> dict:
    from datasets import Dataset
    import torch

    # Unsloth first: it patches trl so the bound SFTConfig/SFTTrainer are the
    # classes TRL's own isinstance checks expect (see module docstring).
    from unsloth import FastLanguageModel, FastVisionModel
    from trl import SFTConfig, SFTTrainer

    run_dir = Path(run_dir)
    manifest = ensure_workspace(
        run_dir,
        config=config,
        config_path=config.get("_config_path"),
        prepared_dir=prepared_dir_for(config),
        phase="train",
    )
    from .data import load_jsonl

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    evaluation_cfg = config.get("evaluation") or {}

    if bool(model_cfg.get("load_in_4bit", False)):
        raise RuntimeError("this benchmark is BF16-only: load_in_4bit must be false")
    if not bool(train_cfg.get("bf16", False)) or bool(train_cfg.get("fp16", False)):
        raise RuntimeError("this benchmark requires bf16=true and fp16=false")

    kind = model_cfg.get("kind", "language")
    if kind not in ("language", "vision"):
        raise ValueError(f"unknown model kind: {kind!r}")

    train_rows = load_jsonl(run_dir / "train.jsonl")
    val_rows = load_jsonl(run_dir / "val.jsonl")

    max_seq_length = int(model_cfg.get("max_seq_length", 2048))
    max_user_tokens = int(evaluation_cfg.get("max_user_tokens", 1536))
    max_new_tokens = int(evaluation_cfg.get("max_new_tokens", 256))
    chat_template_kwargs = model_cfg.get("chat_template_kwargs") or {}
    adapter_dir = run_dir / train_cfg.get("adapter_subdir", "adapter")
    trainer_dir = run_dir / train_cfg.get("trainer_subdir", "trainer")

    FastModel = FastVisionModel if kind == "vision" else FastLanguageModel
    print(f"Loading {model_cfg['base_model']} (kind={kind})...")
    model, processor = FastModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        revision=model_cfg.get("revision"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        use_exact_model_name=bool(model_cfg.get("use_exact_model_name", False)),
    )

    from .models import assert_bf16_placement, restore_marked_fp32_parameters

    # Unsloth deliberately pre-casts some modules (e.g. Qwen3.5 norms) to fp32;
    # undo exactly those, then let the strict gate judge everything else.
    restored_fp32_parameters = restore_marked_fp32_parameters(model)
    if restored_fp32_parameters:
        print(
            f"Restored {restored_fp32_parameters} Unsloth pre-cast fp32 parameter(s) "
            "to bfloat16 before the BF16 gate."
        )

    base_precision = assert_bf16_placement(
        model, context=f"Unsloth load of {model_cfg['base_model']}", allow_adapter=False
    )
    applied_lora = peft_kwargs(lora_cfg, kind, for_peft_call=True)
    model = FastModel.get_peft_model(model, **applied_lora)
    trainable_parameters = trainable_parameter_summary(model)
    if kind == "vision":
        assert_no_vision_trainables(trainable_parameters)
    precision = assert_bf16_placement(
        model, context=f"PEFT-wrapped {model_cfg['base_model']}", allow_adapter=True
    )

    tokenizer = getattr(processor, "tokenizer", processor)
    sequence_lengths = {}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        user_tokens, target_tokens, full_tokens = [], [], []
        truncated = violations = 0
        for row in rows:
            built = build_training_text(
                processor if kind == "vision" else tokenizer,
                row["input"],
                row["label"],
                SYSTEM_PROMPT,
                max_user_tokens=max_user_tokens,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                chat_template_kwargs=chat_template_kwargs,
            )
            user_tokens.append(built["user_tokens"])
            target_tokens.append(built["target_tokens"])
            full_tokens.append(built["full_tokens"])
            truncated += int(built["truncated"])
            violations += int(built["full_tokens"] > max_seq_length)
        if violations:
            raise RuntimeError(f"{split}: {violations} sequences exceed max_seq_length")
        sequence_lengths[split] = {
            "rows": len(rows),
            "user_tokens": _percentiles(user_tokens),
            "target_tokens": _percentiles(target_tokens),
            "full_tokens": _percentiles(full_tokens),
            "truncated_reviews": truncated,
            "full_length_violations": violations,
        }

    render_source = processor if kind == "vision" else tokenizer

    def format_example(row: dict) -> str:
        return build_training_text(
            render_source,
            row["input"],
            row["label"],
            SYSTEM_PROMPT,
            max_user_tokens=max_user_tokens,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
        )["text"]

    train_dataset = Dataset.from_list(
        [{"text": format_example(row)} for row in train_rows]
    )
    val_dataset = Dataset.from_list([{"text": format_example(row)} for row in val_rows])
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    sft_kwargs = dict(
        output_dir=str(trainer_dir),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        optim=train_cfg["optim"],
        seed=int(train_cfg["seed"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=int(train_cfg["save_total_limit"]),
        bf16=True,
        fp16=False,
        report_to=train_cfg.get("report_to", "none"),
        max_length=max_seq_length,
        dataset_text_field="text",
        prediction_loss_only=True,
        assistant_only_loss=False,
        completion_only_loss=False,
    )
    if kind == "vision":
        eos_token = resolve_eos_token(processor, model)
        if eos_token is not None:
            sft_kwargs["eos_token"] = eos_token
            print(f"EOS token: {eos_token}")

    trainer = loss_only_sft_trainer_class(SFTTrainer)(
        model=model,
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(**sft_kwargs),
    )

    setup_peak_allocated, setup_peak_reserved = _cuda_peak_gib()
    _reset_cuda_peak()

    import time

    started = time.perf_counter()
    trainer_stats = trainer.train()
    wall_train_seconds = time.perf_counter() - started
    print("\nTraining complete.")
    print(trainer_stats)

    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter: {adapter_dir}")

    log_history = trainer.state.log_history
    final_epoch = getattr(trainer.state, "epoch", None)
    if final_epoch is None:
        final_epoch = trainer_stats.metrics.get("epoch")
    final_eval = next(
        (entry for entry in reversed(log_history) if "eval_loss" in entry), {}
    )
    train_peak_allocated, train_peak_reserved = _cuda_peak_gib()

    from .runtime import cuda_metadata, library_versions

    metrics = {
        "phase": "train",
        "base_model": model_cfg["base_model"],
        "kind": kind,
        "config_name": config.get("name"),
        "config_sha256": manifest.get("config_sha256"),
        "system_prompt_sha256": manifest["system_prompt_sha256"],
        "dataset": manifest["dataset"],
        "dataset_stats_sha256": manifest.get("dataset_stats_sha256"),
        "requested_revision": model_cfg.get("revision"),
        "resolved_revision": None,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "adapter_dir": str(adapter_dir),
        "trainer_dir": str(trainer_dir),
        "train_metrics": trainer_stats.metrics,
        "log_history": log_history,
        "seed": int(train_cfg["seed"]),
        "num_train_epochs": float(train_cfg["num_train_epochs"]),
        "checkpoint_policy": "final_epoch",
        "best_checkpoint_selection": False,
        "final_epoch": final_epoch,
        "final_epoch_matches_policy": (
            final_epoch is not None
            and abs(float(final_epoch) - float(train_cfg["num_train_epochs"])) < 0.5
        ),
        "final_eval_epoch": final_eval.get("epoch"),
        "final_eval_loss": final_eval.get("eval_loss"),
        "loss": _loss_policy(),
        "precision": {
            "dtype": "bfloat16",
            "bf16": True,
            "fp16": False,
            "load_in_4bit": False,
            "unsloth_pre_cast_fp32_params_restored_to_bf16": restored_fp32_parameters,
            "base_check": base_precision,
            "peft_check": precision,
        },
        "lora": peft_kwargs(lora_cfg, kind),
        "lora_applied": applied_lora,
        "trainable_parameters": trainable_parameters,
        "vision_freeze": {
            "applicable": kind == "vision",
            "asserted_no_vision_trainables": (
                kind == "vision" and trainable_parameters.get("vision_like_count") == 0
            ),
        },
        "sequence_lengths": sequence_lengths,
        "train_peak_allocated_gib": train_peak_allocated,
        "train_peak_reserved_gib": train_peak_reserved,
        "setup_peak_allocated_gib": setup_peak_allocated,
        "setup_peak_reserved_gib": setup_peak_reserved,
        "wall_train_seconds": wall_train_seconds,
        "cuda": cuda_metadata(torch),
        "library_versions": library_versions(),
    }
    from .models import _resolved_revision

    resolved = _resolved_revision(model, tokenizer)
    metrics["resolved_revision"] = resolved or model_cfg.get("revision")
    metrics["resolved_revision_source"] = "model_config" if resolved else "requested"

    (run_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("Saved: train_metrics.json")
    return metrics
