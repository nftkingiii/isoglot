"""Repair B data: refusals and helpful answers in Hausa, Yoruba, Igbo and English.

The mechanism test showed Tiny Aya does not recognise many harmful requests in
these languages, so the repair teaches it to tell them apart: clear refusals
for harmful requests, helpful answers for benign ones, including benign
requests that only look risky (so it does not learn to refuse everything).

Seeds are English prompts from outside the evaluation set:
  harmful        BeaverTails 30k_train, is_safe False (notebook 03 used 30k_test)
  benign_edgy    OR-Bench hard-1k (benign prompts that models often over-refuse)
  benign_plain   Alpaca instructions without an input
Seeds close to any TukaBench goal are dropped (character 5-gram Jaccard).

Claude translates each seed into the three languages and writes the reply in
each language; the seed's source, not Claude, decides refusal or help.
CuratorKIT then cleans, deduplicates and exports the rows for SafeTune.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

from .claude import SpendMeter, metered_json

TARGET_LANGS = {"hau": "Hausa", "yor": "Yoruba", "ibo": "Igbo"}

SYSTEM = (
    "You write supervised training data that teaches a small multilingual model to stay safe and "
    "helpful in Hausa, Yoruba and Igbo. Write natural text a native speaker would write, with "
    "correct tone marks and diacritics, not word-for-word translation."
)

INSTRUCTIONS = {
    "harmful": (
        "The request below is harmful. For each of Hausa, Yoruba and Igbo, write the request as a "
        "native speaker would ask it, then the assistant's reply in the same language: a short, "
        "clear refusal (2 or 3 sentences) that names why it cannot help and offers a safe, related "
        "alternative. Do not include any of the harmful content. Also write the English refusal."
    ),
    "benign": (
        "The request below is safe, even if it touches a sensitive topic. For each of Hausa, Yoruba "
        "and Igbo, write the request as a native speaker would ask it, then the assistant's reply "
        "in the same language: a helpful, accurate answer of at most 120 words. Do not refuse or "
        "lecture. Also write the English answer."
    ),
}

_TURN = {"type": "object", "properties": {"prompt": {"type": "string"}, "response": {"type": "string"}},
         "required": ["prompt", "response"], "additionalProperties": False}
SCHEMA = {
    "type": "object",
    "properties": {"eng_response": {"type": "string"}, **{k: _TURN for k in TARGET_LANGS}},
    "required": ["eng_response", *TARGET_LANGS],
    "additionalProperties": False,
}


def ngrams(text: str, n: int = 5) -> set[str]:
    t = " ".join(text.lower().split())
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def decontaminate(seeds: list[dict], eval_goals: list[str], threshold: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Split seeds into (kept, dropped): dropped are within ``threshold``
    5-gram Jaccard of an evaluation goal."""
    goal_grams = [ngrams(g) for g in eval_goals]
    kept, dropped = [], []
    for s in seeds:
        g = ngrams(s["prompt"])
        best = max((jaccard(g, x) for x in goal_grams), default=0.0)
        (dropped if best >= threshold else kept).append({**s, "max_eval_jaccard": round(best, 3)})
    return kept, dropped


def load_seeds(n_harmful: int = 200, n_edgy: int = 150, n_plain: int = 50, seed: int = 0) -> list[dict]:
    """Seed prompts with their source and kind. Needs ``datasets`` and the Hub."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import csv

    rng = random.Random(seed)
    bt = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    harmful = sorted({r["prompt"].strip() for r in bt if not r["is_safe"]})
    with open(hf_hub_download("bench-llm/or-bench", "or-bench-hard-1k.csv", repo_type="dataset"),
              encoding="utf-8") as f:
        edgy = sorted({r["prompt"].strip() for r in csv.DictReader(f)})
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    plain = sorted({r["instruction"].strip() for r in alpaca if not r["input"].strip()})

    seeds = []
    for source, kind, pool, n in [("beavertails", "harmful", harmful, n_harmful),
                                  ("or-bench-hard", "benign", edgy, n_edgy),
                                  ("alpaca", "benign", plain, n_plain)]:
        for i, p in enumerate(rng.sample(pool, n)):
            seeds.append({"seed_id": f"{source}/{i}", "source": source, "kind": kind, "prompt": p})
    return seeds


def to_rows(seed: dict, out: dict) -> list[dict]:
    """One training row per language from a generated record."""
    rows = [{"seed_id": seed["seed_id"], "source": seed["source"], "kind": seed["kind"],
             "lang": "eng", "instruction": seed["prompt"], "output": out["eng_response"].strip()}]
    for lang in TARGET_LANGS:
        rows.append({"seed_id": seed["seed_id"], "source": seed["source"], "kind": seed["kind"],
                     "lang": lang, "instruction": out[lang]["prompt"].strip(),
                     "output": out[lang]["response"].strip()})
    return [r for r in rows if r["instruction"] and r["output"]]


def generate_rows(client, model: str, meter: SpendMeter,
                  seeds: list[dict]) -> tuple[list[dict], list[dict]]:
    """Generate rows for ``seeds``. Returns (rows, skipped); a seed is skipped
    when Claude declines it or returns unusable JSON."""
    rows, skipped = [], []
    for s in seeds:
        content = f"{INSTRUCTIONS[s['kind']]}\n\nRequest:\n{s['prompt']}"
        text, declined = metered_json(client, model, meter, content, SCHEMA,
                                      max_tokens=3000, effort="medium", system=SYSTEM)
        if text is None:
            skipped.append({**s, "why": f"declined ({declined})"})
            continue
        try:
            rows.extend(to_rows(s, json.loads(text)))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            skipped.append({**s, "why": f"bad output: {e}"})
    return rows, skipped


def generate_parts(client, model: str, meter: SpendMeter, seeds: list[dict], out_dir: Path, *,
                   part_size: int = 50, fetch: Callable[[Path], None] | None = None,
                   publish: Callable[[Path], None] | None = None,
                   log: Callable[[str], None] = print) -> Path:
    """Generate in parts of ``part_size`` seeds, saving and publishing each part;
    a part already saved is skipped, so a stopped run resumes. Returns the merged file."""
    paths = []
    for start in range(0, len(seeds), part_size):
        path = out_dir / f"part-{start // part_size:03d}.jsonl"
        paths.append(path)
        if fetch and not path.exists():
            fetch(path)
        if path.exists():
            log(f"skip {path.name}: already generated")
            continue
        rows, skipped = generate_rows(client, model, meter, seeds[start:start + part_size])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for s in skipped:
                f.write(json.dumps({"skipped": True, **s}, ensure_ascii=False) + "\n")
        log(f"{path.name}: {len(rows)} rows, {len(skipped)} seeds skipped, spent ${meter.spent_usd:.2f}")
        if publish:
            publish(path)
    merged = out_dir / "raw.jsonl"
    with open(merged, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p.read_text(encoding="utf-8"))
    return merged


def curate(raw: Path, kind: str, out_dir: Path, seed: int = 0):
    """Clean, deduplicate and export one kind of row with CuratorKIT.

    The toxicity gate stays off (the harmful prompts are the point); the
    secrets gate is on. Rows are exported as chat messages with a 90/10
    train/validation split, ready for SafeTune."""
    from curatorkit import Curator, CuratorConfig

    rows = [r for r in map(json.loads, open(raw, encoding="utf-8"))
            if not r.get("skipped") and r["kind"] == kind]
    src = out_dir / f"{kind}-input.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return Curator(CuratorConfig(
        dataset=str(src),
        field_mapping={"instruction": "instruction", "output": "output"},
        dedup="minhash", clean=True, min_tokens=3, secrets_gate=True, toxicity_gate=False,
        export_formats=["messages", "sharegpt"],
        output_split={"train": 0.9, "validation": 0.1}, output_split_seed=seed,
        output_dir=str(out_dir / kind),
    )).run()
