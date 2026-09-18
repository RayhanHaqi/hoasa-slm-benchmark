"""Canonical task specification: aspects, labels, prompt and the dataset pin.

One source of truth for prepare, preflight, majority baseline, training and
evaluation so the label space, serialization and system prompt are byte-identical
between the base-model baseline and every fine-tuned evaluation.
"""

from __future__ import annotations

import json

# Canonical aspect order (ascending by name; integer JSON keys are never used so
# `sort_keys=True` and this order agree).
ASPECTS = (
    "ac",
    "air_panas",
    "bau",
    "general",
    "kebersihan",
    "linen",
    "service",
    "sunrise_meal",
    "tv",
    "wifi",
)

# Fixed four-class label set. `neg_pos` means mixed sentiment on the *same*
# aspect. `INVALID` is an evaluation-only marker, never a model class.
LABELS = ("neg", "neut", "pos", "neg_pos")
INVALID = "INVALID"

SPLITS = ("train", "val", "test")

# Single Indonesian ABSA system prompt, shared by baseline, fine-tuning and the
# majority baseline. It asks for exactly one plain JSON object with all keys and
# forbids fences, prose and explanation. Gold labels never appear in the user
# message (the user turn is only the review text).
SYSTEM_PROMPT = (
    "Anda melakukan analisis sentimen berbasis aspek (ABSA) pada ulasan hotel "
    "berbahasa Indonesia.\n"
    "\n"
    "Untuk setiap ulasan, tentukan sentimen untuk SEMUA 10 aspek berikut:\n"
    "- ac: pendingin ruangan (air conditioner)\n"
    "- air_panas: ketersediaan dan kualitas air panas\n"
    "- bau: bau tidak sedap di kamar atau area hotel\n"
    "- general: kesan umum terhadap hotel secara keseluruhan\n"
    "- kebersihan: kebersihan kamar dan area hotel\n"
    "- linen: seprai, handuk, dan linen lainnya\n"
    "- service: pelayanan staf dan resepsionis\n"
    "- sunrise_meal: makanan sarapan (sunrise meal)\n"
    "- tv: televisi di kamar\n"
    "- wifi: koneksi internet nirkabel\n"
    "\n"
    "Setiap aspek diberi tepat satu label dari:\n"
    "- neg: sentimen negatif\n"
    "- neut: sentimen netral, tidak disebutkan, atau tidak dapat disimpulkan\n"
    "- pos: sentimen positif\n"
    "- neg_pos: sentimen campuran, yaitu positif dan negatif pada aspek yang sama\n"
    "\n"
    "Balas HANYA dengan satu objek JSON biasa yang memuat semua 10 kunci aspek "
    "di atas sebagai nilai salah satu label. Urutan kunci bebas. Jangan menulis "
    "penjelasan, kalimat pembuka, markdown, atau blok kode. Contoh bentuk "
    'balasan: {"ac": "<label>", "air_panas": "<label>", "bau": "<label>", '
    '"general": "<label>", "kebersihan": "<label>", "linen": "<label>", '
    '"service": "<label>", "sunrise_meal": "<label>", "tv": "<label>", '
    '"wifi": "<label>"}'
)

# Pinned IndoNLU HoASA source. Direct raw CSVs from one commit; the commit is
# part of every URL and every hash is verified before the bytes are accepted.
DATASET = {
    "name": "hoasa_absa-airy",
    "repo": "IndoNLP/indonlu",
    "commit": "ce728f6926a36174b9923dfe49d6a6839b6e9bb7",
    "files": {
        "train": "dataset/hoasa_absa-airy/train_preprocess.csv",
        "val": "dataset/hoasa_absa-airy/valid_preprocess.csv",
        "test": "dataset/hoasa_absa-airy/test_preprocess.csv",
    },
    "sha256": {
        "train": "752935b62235f1a719c5e526e4ac68b3ba452f84a2a6f911ef20cb855b23546d",
        "val": "7109001762f0bd83526d3de224c0ba5302bfb781eee6c1334aac8039a188f4fa",
        "test": "ce2c7c3f30359e08d4637eb65a05b5a685b888165d40b3133c48c99e1102b094",
    },
    "rows": {"train": 2283, "val": 285, "test": 286},
    "url_template": "https://raw.githubusercontent.com/IndoNLP/indonlu/{commit}/{file}",
    "header": ("review",) + ASPECTS,
}


def url_for(split: str) -> str:
    """Pinned raw URL for one split (commit is interpolated into the URL)."""
    return DATASET["url_template"].format(
        commit=DATASET["commit"], file=DATASET["files"][split]
    )


def canonical_json(obj) -> str:
    """Canonical JSON string: UTF-8, sorted keys (== canonical aspect order)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def canonical_target(row) -> dict:
    """Aspect -> label mapping in canonical aspect order."""
    return {aspect: row[aspect] for aspect in ASPECTS}
