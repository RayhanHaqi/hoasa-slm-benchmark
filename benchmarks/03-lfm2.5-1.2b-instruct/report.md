# Baseline vs fine-tuned comparison

- Run directory: `runs/03-lfm2.5-1.2b-instruct`
- Config: `03-lfm2.5-1.2b-instruct`
- Base model: `LiquidAI/LFM2.5-1.2B-Instruct`
- Requested revision: `0f604ada3f766f9f257460c4c9f0b5d6f69d431b` (resolved `0f604ada3f766f9f257460c4c9f0b5d6f69d431b`)
- Test dataset sha256: `10ea1e8c87ca2b3d7c436874a7b6a5c9fc28e01113d60d47bda00bfe510609a9` (286 rows)
- Adapter: `/home/tilakoid/hoasa-slm-benchmark/runs/03-lfm2.5-1.2b-instruct/adapter` (train_metrics sha256 `6c9a5d47cb53d9a735f6320e57511985f3ac5e656f3cd484ac4ce2382665b1ad`)
- GPUs: `['NVIDIA GeForce RTX 5060 Ti']` (CUDA `12.8`, torch `2.11.0+cu128`) — identical for both runs
- Decoding: `{'split': 'test', 'max_user_tokens': 1536, 'max_new_tokens': 256, 'do_sample': False, 'max_seq_length': 2048, 'chat_template_kwargs': {}}`
- Policy: final-epoch adapter (epoch 3), BF16 base, no quantization; no best-checkpoint selection.

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| Mean aspect macro-F1 | 0.056927 | 0.664728 | 0.607801 |
| Overall aspect accuracy | 0.031469 | 0.964336 | 0.932867 |
| Whole-review exact accuracy (schema-valid) | 0.000000 | 0.730769 | 0.730769 |
| Syntax-valid rate | 0.643357 | 1.000000 | 0.356643 |
| Schema-valid rate | 0.052448 | 1.000000 | 0.947552 |
| Output-valid rate | 0.643357 | 1.000000 | 0.356643 |

## Per-aspect macro-F1

| Aspect | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| ac | 0.149238 | 0.693158 | 0.543920 |
| air_panas | 0.028846 | 0.571138 | 0.542292 |
| bau | 0.038462 | 0.688840 | 0.650378 |
| general | 0.076437 | 0.545557 | 0.469121 |
| kebersihan | 0.081142 | 0.685523 | 0.604381 |
| linen | 0.040846 | 0.638652 | 0.597806 |
| service | 0.062362 | 0.710231 | 0.647869 |
| sunrise_meal | 0.037037 | 0.682814 | 0.645777 |
| tv | 0.020000 | 0.685387 | 0.665387 |
| wifi | 0.034897 | 0.745978 | 0.711081 |

## Resources

| Metric | Baseline | Fine-tuned |
|---|---|---|
| Peak allocated VRAM (GiB) | 2.239425 | 2.311240 |
| Peak reserved VRAM (GiB) | 2.263672 | 2.332031 |
| Mean generate latency (ms/review) | 1003.184299 | 692.037389 |
| Median generate latency (ms/review) | 1112.576663 | 689.989299 |
| Reviews/second | 0.996826 | 1.445009 |
| Output tokens/second (includes prefill) | 131.992285 | 106.930639 |

> Scope: this report compares one base model with its own LoRA adapter on the
> frozen HoASA test split with a shared prompt, parser and metric implementation.
> Metrics are defined in `README.md`; INVALID predictions stay in the denominator.
> No official ABSA leaderboard equivalence is claimed, and there is no
> automatic multi-model runner.
