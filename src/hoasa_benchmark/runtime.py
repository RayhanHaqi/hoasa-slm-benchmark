"""Config loading, run directories, environment and the run manifest.

One run directory per configuration: the first model phase copies the prepared
splits into it and writes `resolved_config.yaml`, `environment.json` and
`run_manifest.json`; later phases verify config projection, system prompt, model
identity and dataset hashes against that manifest and refuse to continue on any
mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import PACKAGE_NAME, __version__
from .spec import SPLITS, SYSTEM_PROMPT, canonical_json

DEFAULT_DATA_DIRNAME = "data"
DEFAULT_RUNS_DIRNAME = "runs"
MANIFEST_NAME = "run_manifest.json"
SCHEMA_VERSION = 1

# Packages recorded in environment.json / metrics. Read through importlib.metadata
# only: no heavy import happens here.
VERSION_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "datasets",
    "trl",
    "unsloth",
    "unsloth_zoo",
    "accelerate",
    "bitsandbytes",
    "PyYAML",
    "tqdm",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def project_root_for(config_path: str | Path) -> Path:
    """Project root: parent of a `configs/` directory, else the config's directory."""
    config_dir = Path(config_path).resolve().parent
    return config_dir.parent if config_dir.name == "configs" else config_dir


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config {path} must be a YAML mapping")

    root = project_root_for(path)
    paths = config.setdefault("paths", {})
    paths["data_dir"] = str(
        resolve_path(paths.get("data_dir", DEFAULT_DATA_DIRNAME), root)
    )
    paths["runs_dir"] = str(
        resolve_path(paths.get("runs_dir", DEFAULT_RUNS_DIRNAME), root)
    )
    config["_config_path"] = str(path.resolve())
    config["_project_root"] = str(root)
    return config


def data_dir_for(config: dict | None) -> Path:
    if config:
        return Path(config["paths"]["data_dir"])
    return Path(DEFAULT_DATA_DIRNAME).resolve()


def prepared_dir_for(config: dict | None) -> Path:
    return data_dir_for(config) / "prepared"


def cache_dir_for(config: dict | None) -> Path:
    root = Path(config.get("_project_root", Path.cwd())) if config else Path.cwd()
    return Path(root) / ".cache" / "huggingface"


def library_versions() -> dict:
    from importlib import metadata

    versions: dict = {"python": sys.version.split()[0]}
    for name in VERSION_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def cuda_metadata(torch_module=None) -> dict:
    """CUDA/torch identity for phase metadata and cross-phase parity checks.

    `torch_module` can be injected (tests, callers that already imported torch);
    otherwise torch is imported lazily and a missing installation yields a
    deterministic unprobed record instead of an error.
    """
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:  # pragma: no cover - torch-free environments
            torch_module = None
    if torch_module is None:
        return {
            "probed": False,
            "available": False,
            "cuda_version": None,
            "torch_version": None,
            "device_count": 0,
            "devices": [],
        }
    try:
        available = bool(torch_module.cuda.is_available())
    except Exception:  # pragma: no cover - driver/CUDA probing can fail
        available = False
    devices: list[str] = []
    if available:
        try:
            devices = [
                str(torch_module.cuda.get_device_name(index))
                for index in range(int(torch_module.cuda.device_count()))
            ]
        except Exception:  # pragma: no cover - driver/CUDA probing can fail
            devices = []
    return {
        "probed": True,
        "available": available,
        "cuda_version": getattr(torch_module.version, "cuda", None),
        "torch_version": getattr(torch_module, "__version__", None),
        "device_count": len(devices),
        "devices": devices,
    }


def environment(torch_module=None) -> dict:
    return {
        "created_at": utc_now_iso(),
        "package": {"name": PACKAGE_NAME, "version": __version__},
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cwd": os.getcwd(),
        "packages": library_versions(),
        "cuda": cuda_metadata(torch_module),
    }


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def model_identity(config: dict) -> dict:
    """Model/prompt-affecting configuration recorded in the manifest."""
    model_cfg = config["model"]
    lora_cfg = config.get("lora") or {}
    training_cfg = config.get("training") or {}
    evaluation_cfg = config.get("evaluation") or {}
    return {
        "base_model": model_cfg["base_model"],
        "revision": model_cfg.get("revision"),
        "kind": model_cfg.get("kind", "language"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "load_in_4bit": bool(model_cfg.get("load_in_4bit", False)),
        "bf16": bool(training_cfg.get("bf16", True)),
        "fp16": bool(training_cfg.get("fp16", False)),
        "max_seq_length": int(model_cfg.get("max_seq_length", 2048)),
        "max_user_tokens": int(evaluation_cfg.get("max_user_tokens", 1536)),
        "max_new_tokens": int(evaluation_cfg.get("max_new_tokens", 256)),
        "chat_template_kwargs": model_cfg.get("chat_template_kwargs") or {},
        "target_modules": lora_cfg.get("target_modules"),
        "lora_r": lora_cfg.get("r"),
        "lora_alpha": lora_cfg.get("alpha"),
        "lora_dropout": lora_cfg.get("dropout"),
        "lora_bias": lora_cfg.get("bias"),
        "finetune_vision_layers": lora_cfg.get("finetune_vision_layers"),
        "finetune_language_layers": lora_cfg.get("finetune_language_layers"),
        "finetune_attention_modules": lora_cfg.get("finetune_attention_modules"),
        "finetune_mlp_modules": lora_cfg.get("finetune_mlp_modules"),
    }


def config_projection(config: dict) -> dict:
    """Deterministic, machine-independent part of the config used for hashing."""
    return {
        key: config.get(key)
        for key in ("name", "model", "lora", "training", "evaluation")
    }


def config_sha256(config: dict) -> str:
    return sha256_text(canonical_json(config_projection(config)))


def dataset_hashes(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    hashes = {}
    for split in SPLITS:
        path = run_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset snapshot: {path}")
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for _ in handle)
        hashes[split] = {"sha256": sha256_file(path), "rows": rows}
    return hashes


def build_manifest(
    run_dir: Path, config: dict | None, config_path, phase: str
) -> dict:
    stats_file = run_dir / "dataset_stats.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "created_by": phase,
        "package": {"name": PACKAGE_NAME, "version": __version__},
        "config_path": str(config_path) if config_path else None,
        "config_name": config.get("name") if config else None,
        "config_sha256": config_sha256(config) if config else None,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "model": model_identity(config) if config else None,
        "dataset": dataset_hashes(run_dir),
        "dataset_stats_sha256": sha256_file(stats_file) if stats_file.is_file() else None,
        "environment_sha256": (
            sha256_file(run_dir / "environment.json")
            if (run_dir / "environment.json").is_file()
            else None
        ),
    }


def prepare_workspace(
    run_dir: str | Path,
    *,
    config: dict | None,
    config_path=None,
    prepared_dir: str | Path,
    phase: str,
) -> dict:
    """First phase in a run dir: snapshot prepared data and write run metadata."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = Path(prepared_dir)

    for split in SPLITS:
        source = prepared_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(
                f"missing prepared split {source}; run `hoasa prepare` first"
            )
        destination = run_dir / f"{split}.jsonl"
        if destination.is_file():
            if sha256_file(destination) != sha256_file(source):
                raise RuntimeError(
                    f"run snapshot {destination} differs from prepared source {source}; "
                    "delete the run directory or re-run `hoasa prepare`"
                )
        else:
            shutil.copyfile(source, destination)

    stats_source = prepared_dir / "dataset_stats.json"
    if stats_source.is_file():
        stats_destination = run_dir / "dataset_stats.json"
        if stats_destination.is_file():
            if sha256_file(stats_destination) != sha256_file(stats_source):
                raise RuntimeError(
                    f"run snapshot {stats_destination} differs from {stats_source}"
                )
        else:
            shutil.copyfile(stats_source, stats_destination)

    environment_file = run_dir / "environment.json"
    if not environment_file.is_file():
        write_json(environment_file, environment())

    if config is not None:
        snapshot = run_dir / "resolved_config.yaml"
        snapshot.write_text(
            f"# Snapshot of {config_path or 'in-memory config'} taken {utc_now_iso()}\n"
            + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    manifest = build_manifest(run_dir, config, config_path, phase)
    write_json(run_dir / MANIFEST_NAME, manifest)
    return manifest


def _differences(expected: dict, actual: dict) -> list[str]:
    differences = []
    for key, value in expected.items():
        if actual.get(key) != value:
            differences.append(f"{key}: manifest={value!r} current={actual.get(key)!r}")
    return differences


def verify_workspace(
    run_dir: str | Path,
    *,
    config: dict | None,
    prepared_dir: str | Path,
    phase: str,
) -> dict:
    """Later phase: refuse to continue unless config/prompt/model/data match."""
    run_dir = Path(run_dir)
    manifest_file = run_dir / MANIFEST_NAME
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"missing {manifest_file}; the first model phase must snapshot the run"
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported run manifest schema {manifest.get('schema_version')!r}"
        )

    if manifest.get("system_prompt_sha256") != sha256_text(SYSTEM_PROMPT):
        raise RuntimeError(
            "system prompt changed since this run directory was created; refusing to continue"
        )

    if config is None:
        if manifest.get("model") is not None:
            raise RuntimeError(
                "this run directory belongs to a model phase; run it with the matching CONFIG"
            )
    else:
        if config_sha256(config) != manifest.get("config_sha256"):
            raise RuntimeError(
                "resolved config changed since this run directory was created "
                f"(phase {phase!r}); refusing to continue"
            )
        differences = _differences(manifest.get("model") or {}, model_identity(config))
        if differences:
            raise RuntimeError(
                "model identity changed since this run directory was created: "
                + "; ".join(differences)
            )

    for split in SPLITS:
        path = run_dir / f"{split}.jsonl"
        recorded = (manifest.get("dataset") or {}).get(split)
        if recorded is None:
            raise RuntimeError(f"manifest records no dataset entry for split {split!r}")
        if not path.is_file():
            raise FileNotFoundError(f"run dataset snapshot missing: {path}")
        actual = {"sha256": sha256_file(path), "rows": _count_rows(path)}
        if actual != recorded:
            raise RuntimeError(
                f"run dataset snapshot {path} does not match the manifest: "
                f"{recorded} != {actual}"
            )
        source = Path(prepared_dir) / f"{split}.jsonl"
        if source.is_file():
            source_hash = sha256_file(source)
            if source_hash != recorded["sha256"]:
                raise RuntimeError(
                    f"prepared source {source} no longer matches the run snapshot "
                    f"({recorded['sha256']} != {source_hash}); re-run `hoasa prepare` "
                    "or start a fresh run directory"
                )
    return manifest


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def ensure_workspace(
    run_dir: str | Path,
    *,
    config: dict | None,
    config_path=None,
    prepared_dir: str | Path,
    phase: str,
) -> dict:
    """Snapshot on first use, verify on every later phase."""
    run_dir = Path(run_dir)
    if (run_dir / MANIFEST_NAME).is_file():
        return verify_workspace(
            run_dir, config=config, prepared_dir=prepared_dir, phase=phase
        )
    return prepare_workspace(
        run_dir,
        config=config,
        config_path=config_path,
        prepared_dir=prepared_dir,
        phase=phase,
    )
