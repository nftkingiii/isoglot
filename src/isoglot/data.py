"""Load the public TukaBench prompt sets as AuditKIT samples.

Only the three JailbreakBench-derived sets are public on the Hub
(McGill-NLP/tukabench, cc-by-nc-4.0). They are evaluation data: nothing here
may be used for training.
"""

from __future__ import annotations

import csv
from pathlib import Path

import auditkit as ak

from . import HARMFUL_SETS, LANGS, SETS

REPO_ID = "McGill-NLP/tukabench"
REVISION = "main"


def tukabench_path(set_name: str, lang: str, revision: str = REVISION) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, f"data/{set_name}/{lang}.csv",
                                repo_type="dataset", revision=revision))


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_samples(rows: list[dict], set_name: str, lang: str) -> list[ak.Sample]:
    """One Sample per prompt. The prompt sent to the model is the translation;
    the English goal travels in metadata for the judge."""
    kind = "harmful" if set_name in HARMFUL_SETS else "benign"
    samples = []
    for row in rows:
        prompt = (row.get("Goal_Translation") or "").strip()
        if not prompt:
            continue
        idx = row["Index"].strip()
        samples.append(ak.Sample(
            input=prompt,
            id=f"{set_name}/{lang}/{idx}",
            task=f"{set_name}:{lang}",
            metadata={"set": set_name, "lang": lang, "index": idx, "kind": kind,
                      "goal_en": row["Goal"].strip()},
        ))
    return samples


def load_tukabench(sets=SETS, langs=LANGS, limit: int | None = None,
                   revision: str = REVISION) -> list[ak.Sample]:
    samples = []
    for set_name in sets:
        for lang in langs:
            rows = read_rows(tukabench_path(set_name, lang, revision))
            samples.extend(to_samples(rows[:limit] if limit else rows, set_name, lang))
    return samples
