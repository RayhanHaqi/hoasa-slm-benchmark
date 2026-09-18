# Baseline vs fine-tuned comparison

- Run directory: `runs/04-qwen3.5-2b`
- Config: `04-qwen3.5-2b`
- Base model: `Qwen/Qwen3.5-2B`
- Requested revision: `15852e8c16360a2fea060d615a32b45270f8a8fc` (resolved `15852e8c16360a2fea060d615a32b45270f8a8fc`)
- Test dataset sha256: `10ea1e8c87ca2b3d7c436874a7b6a5c9fc28e01113d60d47bda00bfe510609a9` (286 rows)
- Adapter: `/home/tilakoid/hoasa-slm-benchmark/runs/04-qwen3.5-2b/adapter` (train_metrics sha256 `e83959b9f5f6c6e2a63dcd910719d1a1e3eddb6dd6bf85ab5d8b83ba9d4cd96d`)
- GPUs: `['NVIDIA GeForce RTX 5060 Ti']` (CUDA `12.8`, torch `2.11.0+cu128`) — identical for both runs
- Decoding: `{'split': 'test', 'max_user_tokens': 1536, 'max_new_tokens': 256, 'do_sample': False, 'max_seq_length': 2048, 'chat_template_kwargs': {'enable_thinking': False}}`
- Policy: final-epoch adapter (epoch 3), BF16 base, no quantization; no best-checkpoint selection.

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| Mean aspect macro-F1 | 0.475039 | 0.676261 | 0.201222 |
| Overall aspect accuracy | 0.789860 | 0.937413 | 0.147552 |
| Whole-review exact accuracy (schema-valid) | 0.125874 | 0.734266 | 0.608392 |
| Syntax-valid rate | 1.000000 | 0.965035 | -0.034965 |
| Schema-valid rate | 0.986014 | 0.965035 | -0.020979 |
| Output-valid rate | 1.000000 | 0.965035 | -0.034965 |

## Per-aspect macro-F1

| Aspect | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| ac | 0.524722 | 0.715767 | 0.191045 |
| air_panas | 0.464669 | 0.639635 | 0.174966 |
| bau | 0.303548 | 0.698754 | 0.395205 |
| general | 0.246380 | 0.592441 | 0.346061 |
| kebersihan | 0.555147 | 0.691675 | 0.136528 |
| linen | 0.390399 | 0.610573 | 0.220174 |
| service | 0.631584 | 0.709118 | 0.077534 |
| sunrise_meal | 0.494177 | 0.674463 | 0.180286 |
| tv | 0.607684 | 0.689331 | 0.081647 |
| wifi | 0.532077 | 0.740855 | 0.208778 |

## Resources

| Metric | Baseline | Fine-tuned |
|---|---|---|
| Peak allocated VRAM (GiB) | 4.196743 | 4.259400 |
| Peak reserved VRAM (GiB) | 4.228516 | 4.292969 |
| Mean generate latency (ms/review) | 1435.629193 | 6408.723340 |
| Median generate latency (ms/review) | 1441.168618 | 6366.074434 |
| Reviews/second | 0.696559 | 0.156037 |
| Output tokens/second (includes prefill) | 54.032009 | 39.945553 |

> Scope: this report compares one base model with its own LoRA adapter on the
> frozen HoASA test split with a shared prompt, parser and metric implementation.
> Metrics are defined in `README.md`; INVALID predictions stay in the denominator.
> No official ABSA leaderboard equivalence is claimed, and there is no
> automatic multi-model runner.
