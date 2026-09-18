# Baseline vs fine-tuned comparison

- Run directory: `runs/01-qwen3-1.7b`
- Config: `01-qwen3-1.7b`
- Base model: `Qwen/Qwen3-1.7B`
- Requested revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (resolved `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`)
- Test dataset sha256: `10ea1e8c87ca2b3d7c436874a7b6a5c9fc28e01113d60d47bda00bfe510609a9` (286 rows)
- Adapter: `/home/tilakoid/hoasa-slm-benchmark/runs/01-qwen3-1.7b/adapter` (train_metrics sha256 `547853b95989b25a170060bfdd38264797d79b78aa2119a2460d4c8b430a8f43`)
- GPUs: `['NVIDIA GeForce RTX 5060 Ti']` (CUDA `12.8`, torch `2.11.0+cu128`) — identical for both runs
- Decoding: `{'split': 'test', 'max_user_tokens': 1536, 'max_new_tokens': 256, 'do_sample': False, 'max_seq_length': 2048, 'chat_template_kwargs': {'enable_thinking': False}}`
- Policy: final-epoch adapter (epoch 3), BF16 base, no quantization; no best-checkpoint selection.

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| Mean aspect macro-F1 | 0.342083 | 0.689441 | 0.347358 |
| Overall aspect accuracy | 0.578671 | 0.973427 | 0.394755 |
| Whole-review exact accuracy (schema-valid) | 0.017483 | 0.779720 | 0.762238 |
| Syntax-valid rate | 1.000000 | 1.000000 | 0.000000 |
| Schema-valid rate | 1.000000 | 1.000000 | 0.000000 |
| Output-valid rate | 1.000000 | 1.000000 | 0.000000 |

## Per-aspect macro-F1

| Aspect | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| ac | 0.294372 | 0.706803 | 0.412431 |
| air_panas | 0.297902 | 0.652137 | 0.354235 |
| bau | 0.314550 | 0.720041 | 0.405492 |
| general | 0.146203 | 0.550895 | 0.404692 |
| kebersihan | 0.476002 | 0.704953 | 0.228952 |
| linen | 0.341730 | 0.680829 | 0.339099 |
| service | 0.466102 | 0.721125 | 0.255023 |
| sunrise_meal | 0.381679 | 0.666849 | 0.285170 |
| tv | 0.373525 | 0.744796 | 0.371271 |
| wifi | 0.328766 | 0.745978 | 0.417212 |

## Resources

| Metric | Baseline | Fine-tuned |
|---|---|---|
| Peak allocated VRAM (GiB) | 3.306608 | 3.396643 |
| Peak reserved VRAM (GiB) | 3.394531 | 3.492188 |
| Mean generate latency (ms/review) | 972.273065 | 1739.856811 |
| Median generate latency (ms/review) | 973.465835 | 1733.500266 |
| Reviews/second | 1.028518 | 0.574760 |
| Output tokens/second (includes prefill) | 75.351504 | 44.276612 |

> Scope: this report compares one base model with its own LoRA adapter on the
> frozen HoASA test split with a shared prompt, parser and metric implementation.
> Metrics are defined in `README.md`; INVALID predictions stay in the denominator.
> No official ABSA leaderboard equivalence is claimed, and there is no
> automatic multi-model runner.
