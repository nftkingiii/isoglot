"""Stage 1: generate answers with AuditKIT and save them as JSONL.

Generation and judging are separate so a run can be re-judged (or judged by a
second judge) without spending GPU time again.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import auditkit as ak

from . import __version__


def generate(samples: list[ak.Sample], model_id: str, *, max_tokens: int = 512,
             seed: int = 0, batch_size: int = 1, hf_token: str | None = None,
             **model_opts) -> list[ak.Sample]:
    config = ak.RunConfig(temperature=0.0, max_tokens=max_tokens, seed=seed)
    model = hf_model(model_id, batch_size=batch_size,
                     hf_token=hf_token or os.environ.get("HF_TOKEN"), **model_opts)
    return ak.generate(samples, model=model, config=config)


def hf_model(model_id: str, *, batch_size: int = 1, **opts):
    """AuditKIT's hf: model, made safe to batch.

    AuditKIT's hf: backend ignores RunConfig.batch_size; a batch_size model
    option is forwarded to the transformers pipeline call instead. Batched
    decoding then needs a pad token and left padding, which AuditKIT does not
    set, so the pipeline is loaded here and its tokenizer fixed before use.
    """
    from auditkit.model import AutoModel

    if batch_size > 1:
        opts["batch_size"] = batch_size
    model = AutoModel.resolve(f"hf:{model_id}", **opts)
    if batch_size > 1:
        model._ensure_pipeline()  # no public hook for the tokenizer
        tok = model._pipeline.tokenizer
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    return model


def save_answers(answers: list[ak.Sample], path: Path, *, model_id: str,
                 variant: str, max_tokens: int, seed: int) -> Path:
    """One JSON row per answer, with enough run context to reproduce it."""
    run = {"model_id": model_id, "variant": variant, "max_tokens": max_tokens,
           "seed": seed, "temperature": 0.0, "isoglot": __version__,
           "auditkit": getattr(ak, "__version__", "unknown"),
           "python": platform.python_version(),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in answers:
            f.write(json.dumps({"id": s.id, "prompt": s.input,
                                "response": s.actual_output, **s.metadata,
                                "run": run}, ensure_ascii=False) + "\n")
    return path


def load_answers(path: Path) -> list[ak.Sample]:
    """Read saved answers back as precomputed AuditKIT samples."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta = {k: r[k] for k in ("set", "lang", "index", "kind", "goal_en")}
            meta["variant"] = r["run"]["variant"]
            samples.append(ak.Sample(input=r["prompt"], id=r["id"],
                                     task=f"{r['set']}:{r['lang']}",
                                     actual_output=r["response"] or "", metadata=meta))
    return samples
