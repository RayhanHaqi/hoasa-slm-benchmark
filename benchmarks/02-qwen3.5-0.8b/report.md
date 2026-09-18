# Baseline vs fine-tuned comparison

- Run directory: `runs/02-qwen3.5-0.8b`
- Config: `02-qwen3.5-0.8b`
- Base model: `Qwen/Qwen3.5-0.8B`
- Requested revision: `2fc06364715b967f1860aea9cf38778875588b17` (resolved `2fc06364715b967f1860aea9cf38778875588b17`)
- Test dataset sha256: `10ea1e8c87ca2b3d7c436874a7b6a5c9fc28e01113d60d47bda00bfe510609a9` (286 rows)
- Adapter: `/home/tilakoid/hoasa-slm-benchmark/runs/02-qwen3.5-0.8b/adapter` (train_metrics sha256 `f0d37619354396e76484e2f34c65ea3ada7f61c7e21a13c8855001ebea50e89a`)
- GPUs: `['NVIDIA GeForce RTX 5060 Ti']` (CUDA `12.8`, torch `2.11.0+cu128`) — identical for both runs
- Decoding: `{'split': 'test', 'max_user_tokens': 1536, 'max_new_tokens': 256, 'do_sample': False, 'max_seq_length': 2048, 'chat_template_kwargs': {'enable_thinking': False}}`
- Policy: final-epoch adapter (epoch 3), BF16 base, no quantization; no best-checkpoint selection.

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| Mean aspect macro-F1 | 0.252686 | 0.692106 | 0.439421 |
| Overall aspect accuracy | 0.315385 | 0.975524 | 0.660140 |
| Whole-review exact accuracy (schema-valid) | 0.000000 | 0.793706 | 0.793706 |
| Syntax-valid rate | 1.000000 | 1.000000 | 0.000000 |
| Schema-valid rate | 0.986014 | 1.000000 | 0.013986 |
| Output-valid rate | 1.000000 | 1.000000 | 0.000000 |

## Per-aspect macro-F1

| Aspect | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| ac | 0.111579 | 0.690685 | 0.579107 |
| air_panas | 0.391300 | 0.656158 | 0.264857 |
| bau | 0.064346 | 0.715765 | 0.651419 |
| general | 0.129492 | 0.615007 | 0.485515 |
| kebersihan | 0.405648 | 0.715559 | 0.309911 |
| linen | 0.276981 | 0.663277 | 0.386296 |
| service | 0.341690 | 0.715201 | 0.373510 |
| sunrise_meal | 0.286738 | 0.658639 | 0.371901 |
| tv | 0.274742 | 0.744796 | 0.470054 |
| wifi | 0.244340 | 0.745978 | 0.501638 |

## Resources

| Metric | Baseline | Fine-tuned |
|---|---|---|
| Peak allocated VRAM (GiB) | 1.660311 | 1.700630 |
| Peak reserved VRAM (GiB) | 1.681641 | 1.722656 |
| Mean generate latency (ms/review) | 1158.471390 | 6036.982935 |
| Median generate latency (ms/review) | 1147.552485 | 6032.734095 |
| Reviews/second | 0.863206 | 0.165646 |
| Output tokens/second (includes prefill) | 62.072393 | 42.405288 |

> Scope: this report compares one base model with its own LoRA adapter on the
> frozen HoASA test split with a shared prompt, parser and metric implementation.
> Metrics are defined in `README.md`; INVALID predictions stay in the denominator.
> No official ABSA leaderboard equivalence is claimed, and there is no
> automatic multi-model runner.
