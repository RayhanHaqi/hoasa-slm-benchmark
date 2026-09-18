# HOASA ABSA SLM Benchmark

Small-language-model benchmark for Indonesian hotel-review aspect-based
sentiment analysis (ABSA) on the IndoNLU **HoASA** dataset
(`hoasa_absa-airy`, Airy hotel reviews). Ten aspects, four labels
(`neg`, `neut`, `pos`, `neg_pos`), one shared Indonesian prompt, one shared
parser and one shared metric implementation for the base-model baseline, the
LoRA fine-tune and the majority baseline.

This repository is a self-contained local project: the commands below produce
artifacts locally, and local `runs/`, adapters, checkpoints, and weights remain
git-ignored. One benchmark is published here under
[`benchmarks/01-qwen3-1.7b/`](benchmarks/01-qwen3-1.7b/) — a real GPU run of
the Qwen3-1.7B base model and its LoRA fine-tune, plus the majority floor.

## Published benchmark

`benchmarks/01-qwen3-1.7b/` holds the complete lightweight artifact set of one
completed run: `Qwen/Qwen3-1.7B` @
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, full BF16 with no quantization,
LoRA `r=16`/`alpha=32`, 3 epochs, final-epoch-3 adapter, evaluated on the 286
labeled HoASA test rows on an `NVIDIA GeForce RTX 5060 Ti` (CUDA 12.8). Mean
latency is `model.generate` time per review and includes prefill. The labeled
test split is a property of the pinned commit; the
[Dataset](#dataset-pinned-unmodified) note applies, and no leaderboard
equivalence is claimed.

| Model | Mean aspect macro-F1 | Overall aspect accuracy | Whole-review exact accuracy | Schema-valid rate | Peak allocated VRAM (GiB) | Mean latency (ms/review) | Δ macro-F1 vs base |
|---|---|---|---|---|---|---|---|
| [Majority baseline](benchmarks/01-qwen3-1.7b/majority_metrics.json) | 0.221669 | 0.803497 | 0.000000 | 1.000000 | — | — | — |
| [Qwen3-1.7B base](benchmarks/01-qwen3-1.7b/baseline_metrics.json) | 0.342083 | 0.578671 | 0.017483 | 1.000000 | 3.307 | 972.273 | 0.000000 |
| [Qwen3-1.7B + LoRA](benchmarks/01-qwen3-1.7b/finetuned_metrics.json) | 0.689441 | 0.973427 | 0.779720 | 1.000000 | 3.397 | 1739.857 | +0.347358 |

Artifacts: [report.md](benchmarks/01-qwen3-1.7b/report.md),
[comparison.csv](benchmarks/01-qwen3-1.7b/comparison.csv),
[predictions (base)](benchmarks/01-qwen3-1.7b/baseline_predictions.jsonl),
[predictions (fine-tuned)](benchmarks/01-qwen3-1.7b/finetuned_predictions.jsonl),
[train_metrics.json](benchmarks/01-qwen3-1.7b/train_metrics.json),
[resolved_config.yaml](benchmarks/01-qwen3-1.7b/resolved_config.yaml),
[run_manifest.json](benchmarks/01-qwen3-1.7b/run_manifest.json),
[environment.json](benchmarks/01-qwen3-1.7b/environment.json),
[dataset_stats.json](benchmarks/01-qwen3-1.7b/dataset_stats.json),
[majority_metrics.json](benchmarks/01-qwen3-1.7b/majority_metrics.json).

## Reference

- Reference implementation (read-only): `/home/tilakoid/slm-specialist` at
  commit `d4f1f6bfcf39495af602649f0d532edd61a8b2ba`. Model-loading order,
  Unsloth peft kwargs, the loss-only SFT trainer subclass, EOS resolution and
  the vision `all-linear` PEFT fix are adapted from its
  `src/specialist/{model,train,evaluate}.py`.
- This project replaces the reference's GitHub issue-triage task with HoASA
  ABSA; no code or data is shared with the reference at runtime.

## Dataset (pinned, unmodified)

- Repository: `IndoNLP/indonlu`, commit
  `ce728f6926a36174b9923dfe49d6a6839b6e9bb7`.
- Direct raw CSVs (one per official split):

  | Split | File | Rows | SHA-256 |
  |---|---|---|---|
  | train | `dataset/hoasa_absa-airy/train_preprocess.csv` | 2283 | `752935b62235f1a719c5e526e4ac68b3ba452f84a2a6f911ef20cb855b23546d` |
  | val | `dataset/hoasa_absa-airy/valid_preprocess.csv` | 285 | `7109001762f0bd83526d3de224c0ba5302bfb781eee6c1334aac8039a188f4fa` |
  | test | `dataset/hoasa_absa-airy/test_preprocess.csv` | 286 | `ce2c7c3f30359e08d4637eb65a05b5a685b888165d40b3133c48c99e1102b094` |

- CSV header: `review,ac,air_panas,bau,general,kebersihan,linen,service,sunrise_meal,tv,wifi`.
- `hoasa prepare` verifies the pinned commit URL, SHA-256, header, row counts,
  every label value and missing/empty values before accepting bytes. Official
  splits are preserved exactly: no cleaning, deduplication, re-splitting,
  balancing or augmentation. Exact duplicate reviews and empty cells are
  reported in `dataset_stats.json` (report only).
- **Caveat:** the root IndoNLU README states the HoASA test split is usually
  distributed masked. In this pinned commit an unmasked, labeled
  `test_preprocess.csv` exists and is used for evaluation; this is a property of
  this specific commit and is recorded in `dataset_stats.json`.
- The pinned test CSV is labeled. Do not use any masked file if one is present.

## Task, prompt, parsing and metrics

- One Indonesian system prompt (`hoasa_benchmark.spec.SYSTEM_PROMPT`) defines
  ABSA, all ten aspects and the four labels (`neg_pos` = mixed positive and
  negative sentiment on the same aspect). It asks for exactly one plain JSON
  object with all ten keys, no code fences and no explanation. The user message
  contains only the review text — never gold labels.
- Parser: full `json.loads` with duplicate-key detection. Whitespace and key
  order are accepted; fences/prose are rejected. Malformed JSON, a non-object,
  or duplicate keys anywhere make all aspects `INVALID`. For a parseable object,
  a missing key or an out-of-set value makes only that aspect `INVALID`. Extra
  keys make `schema_valid` false while canonical valid values still score.
  A duplicate-key output can be `syntax_valid` (valid JSON syntax) but is not an
  `output_valid`/`schema_valid` result.
- Fixed classes per aspect: `[neg, neut, pos, neg_pos]`. `INVALID` is never a
  fifth macro class but always remains in the denominator and counts as a false
  negative for the gold class.
- Formulas:
  - per class: `F1 = 2*TP / (2*TP + FP + FN)`, `precision = TP/(TP+FP)`,
    `recall = TP/(TP+FN)`, all with `zero_division=0`;
  - `macro_f1(aspect) = mean(F1(neg), F1(neut), F1(pos), F1(neg_pos))`;
  - `mean_aspect_macro_f1 = mean(macro_f1(aspect) for the 10 aspects)`;
  - `overall_aspect_accuracy = correct aspect predictions / (rows * 10)`
    (INVALID counts as wrong);
  - `whole_review_exact_accuracy` requires `schema_valid` **and** all ten
    aspects correct;
  - `syntax_valid_rate`, `schema_valid_rate`, `output_valid_rate` are fractions
    of rows over the test set.
- There is no claim of equivalence with any official leaderboard: this project
  defines its own parser and aggregation.
- Majority baseline: per-aspect train majority, serialized as canonical JSON and
  then parsed/evaluated through the same code path.
- Predictions JSONL rows: `id`, `gold`, `parsed`, `raw_output`, `validity`,
  `input_tokens`, `output_tokens`, `latency_ms`.
- Model metrics additionally include peak allocated/reserved VRAM (GiB, counters
  reset after model load), mean/median `model.generate` latency, reviews/second
  and output tokens/second — the latter divides generated tokens by total
  `model.generate` time, so it **includes prefill** and is not pure decode
  throughput.

## Precision, truncation, training policy

- Precision: **full BF16 base weights, no quantization** at any phase
  (`load_in_4bit=false`, `fp16=false`). Runtime checks fail closed on a
  quantization config, bitsandbytes modules, non-BF16 base parameters, or
  CPU/disk/meta placement; there is no fallback model or quantization mode.
- One shared training/inference helper truncates the review alone to
  `max_user_tokens=1536`, renders the model's native chat template and reserves
  `max_new_tokens=256` inside `max_seq_length=2048`, shrinking the user text
  further when needed. Fine-tuning uses the identical truncated user, appends
  the assistant target plus EOS and asserts the full system/user/assistant/EOS
  sequence fits; the target is never right-truncated.
- Training recipe: LoRA `r=16`, `alpha=32`, `dropout=0`, `bias=none`, gradient
  checkpointing (`unsloth`), 3 epochs, `lr=2e-4`, cosine schedule,
  `warmup_ratio=0.05`, `weight_decay=0.01`, `seed=42`, micro-batch 2 with
  gradient accumulation 4 (effective 8), eval batch 1, `adamw_8bit`,
  validation loss and checkpoints every epoch.
- Loss: explicit full-sequence language-model loss
  (`assistant_only_loss=false`, `completion_only_loss=false`,
  `prediction_loss_only=true`); both flags are stored in `train_metrics.json`.
- Checkpoint policy: **final epoch (epoch 3) adapter**, no best-checkpoint
  selection, no resume path.
- Vision models (`Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`) are loaded with
  `FastVisionModel` but only language/attention/MLP LoRA is trained; the
  literal `all-linear` target is passed as `None` to `get_peft_model` so Unsloth
  honors `finetune_vision_layers=false`, and training fails closed if any
  vision-like parameter is trainable.
- Metadata preflight loads only `AutoConfig` + `AutoTokenizer`/`AutoProcessor`
  at the pinned revision inside the project-local `.cache/huggingface` cache and
  rejects weight files (`*.safetensors`, `*.bin`, `*.gguf`, `*.pt`, …). It never
  imports `AutoModel*` and never downloads weights.

## Environment

Validated with conda env `github-triage` (Python 3.11.16):
`torch 2.11.0+cu128`, `transformers 5.5.0`, `peft 0.20.0`, `datasets 4.3.0`,
`trl 0.24.0`, `unsloth 2026.9.4`, `unsloth_zoo 2026.9.3`,
`accelerate 1.15.0`, `bitsandbytes 0.50.2`, `PyYAML 6.0.3`, `tqdm 4.70.1`.
`scikit-learn` is intentionally not used; all metrics are implemented here.

```bash
pip install -e . --no-deps          # editable install, only PyYAML is required
```

Heavy imports (torch/transformers/peft/unsloth/trl/datasets/tqdm) are lazy:
`prepare`, `majority-baseline` and every `--help` work without them.

## Commands

```bash
hoasa prepare                                   # download/verify + write data/prepared
hoasa preflight CONFIG                          # metadata-only tokenizer/length report
hoasa majority-baseline --run-dir runs/majority # train-majority floor
hoasa baseline CONFIG --run-dir DIR             # base model, shared evaluator
hoasa train CONFIG --run-dir DIR                # Unsloth BF16 LoRA SFT
hoasa evaluate CONFIG --checkpoint DIR/adapter --run-dir DIR
hoasa compare --run-dir DIR                     # report.md + comparison.csv
```

`hoasa baseline` evaluates the pinned base model only. `hoasa evaluate` requires
the run-local adapter produced by `hoasa train` (`<run-dir>/<training.adapter_subdir>`,
default `DIR/adapter`); `base`, model ids and adapter directories outside the run
directory are refused, so `finetuned_metrics.json` can only describe this run's
adapter. Before loading the adapter the evaluator cross-checks run-local
`train_metrics.json` and `run_manifest.json` on config hash, system prompt hash,
dataset hashes, base model and requested/pinned revision, and records a
provenance block in the metrics.

Four configs in `configs/` (exact revisions):

1. `01-qwen3-1.7b.yaml` — `Qwen/Qwen3-1.7B` @
   `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, language, targets
   `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`,
   `enable_thinking=false`.
2. `02-qwen3.5-0.8b.yaml` — `Qwen/Qwen3.5-0.8B` @
   `2fc06364715b967f1860aea9cf38778875588b17`, vision (frozen tower),
   `all-linear`, `enable_thinking=false`.
3. `03-lfm2.5-1.2b-instruct.yaml` — `LiquidAI/LFM2.5-1.2B-Instruct` @
   `0f604ada3f766f9f257460c4c9f0b5d6f69d431b`, language, targets
   `q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3`, no chat-template kwargs.
4. `04-qwen3.5-2b.yaml` — `Qwen/Qwen3.5-2B` @
   `15852e8c16360a2fea060d615a32b45270f8a8fc`, vision (frozen tower),
   `all-linear`, `enable_thinking=false`. No resume field.

## Run layout and artifacts

```
data/raw/                 pinned CSV bytes
data/prepared/            train.jsonl, val.jsonl, test.jsonl, dataset_stats.json
data/preflight/<name>.json
runs/<run>/               explicit --run-dir
  train.jsonl val.jsonl test.jsonl dataset_stats.json   # snapshot (first phase)
  resolved_config.yaml environment.json run_manifest.json
  baseline_metrics.json baseline_predictions.jsonl
  adapter/ trainer/ train_metrics.json
  finetuned_metrics.json finetuned_predictions.jsonl
  report.md comparison.csv
```

The first model phase snapshots the prepared splits and writes the run
metadata; every later phase re-verifies config projection, system prompt, model
identity/revision and dataset hashes and refuses to continue on any mismatch.
Every phase records `environment.json` (package versions plus
`torch.version.cuda`, CUDA availability and device names) and the evaluation
metrics carry the same CUDA/library identity. `hoasa compare` requires baseline
`mode=base` and fine-tuned `mode=adapter` with a matching provenance block, and
fails closed if the two runs differ in CUDA/GPU identity or the parity ML
library versions (torch, transformers, peft, bitsandbytes, accelerate) — a
comparison across machines is refused rather than reported.
`data/`, `runs/`, `.cache/`, adapters, checkpoints and weights are git-ignored.
There is no automatic four-model runner — run each configuration explicitly.

## Validation status

- `python -m unittest discover -s tests -v` (94 tests, all passing) covers the
  parser (perfect, key order, wrong/missing/invalid values, extra keys,
  malformed, non-object, duplicates, all-invalid), metrics (zero support,
  INVALID denominator, empty input), synthetic prepare helpers, truncation with
  a fake tokenizer, the PEFT vision fix/config invariants/loss flags, run
  snapshot verification, run-local adapter resolution and train_metrics
  provenance, the BF16/CUDA placement gate (adapter trainables exempt from the
  dtype check only, never from placement), and the comparison gates
  (mode/provenance/CUDA/library parity, with injected deterministic metadata).
- The GPU path (`baseline`, `train`, `evaluate`, `compare`) completed for the
  `01-qwen3-1.7b.yaml` configuration on the RTX 5060 Ti host; its artifacts are
  the [published benchmark](#published-benchmark) linked above. The other three
  configs are not published and are outside this reported benchmark — no zero
  rows, placeholder scores, or run-status claims are made for them here.
