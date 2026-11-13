"""Capability check: did the LISA fine-tune hurt general ability in these languages?

Multiple-choice reading (Belebele) and knowledge (AfriMMLU) questions in English,
Hausa, Yoruba and Igbo, scored by log-likelihood through AuditKIT
(MCQAdapter, method "mcq_loglikelihood", metric "acc"): each answer option is
scored by the model's own probability, so no judge and no API cost.

The same question sample is used for both models, so the comparison is paired:
the change is measured question by question with a paired bootstrap interval.
"""

from __future__ import annotations

import ast
import csv
import json
import random

import auditkit as ak

from .generate import _fresh_auditkit_cache

BELEBELE = {"eng": "eng_Latn", "hau": "hau_Latn", "yor": "yor_Latn", "ibo": "ibo_Latn"}
LETTERS = "ABCD"


def belebele_samples(rows: list[dict], lang: str) -> list[ak.Sample]:
    return [ak.Sample(
        input=f"Passage: {r['flores_passage']}\nQuestion: {r['question']}\nAnswer:",
        choices=[r[f"mc_answer{i}"] for i in range(1, 5)],
        target=int(r["correct_answer_num"]) - 1,
        id=f"belebele/{lang}/{r['link']}#{r['question_number']}",
        task=f"belebele:{lang}", metadata={"bench": "belebele", "lang": lang},
    ) for r in rows]


def afrimmlu_samples(rows: list[dict], lang: str) -> list[ak.Sample]:
    out = []
    for i, r in enumerate(rows):
        choices = ast.literal_eval(r["choices"])
        out.append(ak.Sample(
            input=f"Question: {r['question']}\nAnswer:", choices=list(choices),
            target=LETTERS.index(r["answer"].strip()),
            id=f"afrimmlu/{lang}/{i}", task=f"afrimmlu:{lang}",
            metadata={"bench": "afrimmlu", "lang": lang, "subject": r["subject"]},
        ))
    return out


def load_samples(langs=("eng", "hau", "yor", "ibo"), n: int = 200, seed: int = 0) -> list[ak.Sample]:
    """``n`` questions per benchmark and language, the same draw every time."""
    from huggingface_hub import hf_hub_download

    samples = []
    for lang in langs:
        path = hf_hub_download("facebook/belebele", f"data/{BELEBELE[lang]}.jsonl", repo_type="dataset")
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        samples += random.Random(seed).sample(belebele_samples(rows, lang), n)
        path = hf_hub_download("masakhane/afrimmlu", f"data/{lang}/test.tsv", repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        samples += random.Random(seed).sample(afrimmlu_samples(rows, lang), n)
    return samples


def score(samples: list[ak.Sample], model) -> dict[str, bool]:
    """Correct-or-not per sample id. ``model`` is an AuditKIT model or spec."""
    with _fresh_auditkit_cache():
        result = ak.evaluate(samples, model=model, scorers=["acc"],
                             adapter=ak.MCQAdapter(method="mcq_loglikelihood"))
    return {p.sample_id: bool(p.correct) for p in result.predictions}


def paired_table(samples: list[ak.Sample], before: dict[str, bool], after: dict[str, bool],
                 n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """Accuracy before and after per (benchmark, lang), with a paired
    bootstrap 95% interval on the change."""
    groups: dict[tuple[str, str], list[str]] = {}
    for s in samples:
        groups.setdefault((s.metadata["bench"], s.metadata["lang"]), []).append(s.id)
    rng = random.Random(seed)
    table = []
    for (bench, lang), ids in sorted(groups.items()):
        diffs = [int(after[i]) - int(before[i]) for i in ids]
        boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n_boot))
        table.append({"bench": bench, "lang": lang, "n": len(ids),
                      "before": sum(before[i] for i in ids) / len(ids),
                      "after": sum(after[i] for i in ids) / len(ids),
                      "change": sum(diffs) / len(diffs),
                      "change_ci": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1])})
    return table


def to_markdown(table: list[dict]) -> str:
    lines = ["| Benchmark | Lang | n | Base | LISA | Change [95% CI] |", "|---|---|---|---|---|---|"]
    for t in table:
        lo, hi = t["change_ci"]
        lines.append(f"| {t['bench']} | {t['lang']} | {t['n']} | {t['before']:.1%} | {t['after']:.1%} | "
                     f"{t['change']:+.1%} [{lo:+.1%}, {hi:+.1%}] |")
    return "\n".join(lines) + "\n"
