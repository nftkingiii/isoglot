"""Stage 1: generate answers with AuditKIT and save them as JSONL.

Generation and judging are separate so a run can be re-judged (or judged by a
second judge) without spending GPU time again.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import auditkit as ak

from . import __version__


def generate(samples: list[ak.Sample], model_id: str, *, max_tokens: int = 512,
             seed: int = 0, batch_size: int = 1, hf_token: str | None = None,
             model=None, **model_opts) -> list[ak.Sample]:
    """Answer samples. Pass an already-loaded ``model`` to avoid reloading it."""
    config = ak.RunConfig(temperature=0.0, max_tokens=max_tokens, seed=seed)
    if model is None:
        model = hf_model(model_id, batch_size=batch_size,
                         hf_token=hf_token or os.environ.get("HF_TOKEN"), **model_opts)
    with _fresh_auditkit_cache():
        return ak.generate(samples, model=model, config=config)


@contextmanager
def _fresh_auditkit_cache():
    """Run with an empty, throwaway AuditKIT result cache.

    AuditKIT caches runs by a fingerprint of the model name and run settings.
    Generation also depends on state that fingerprint cannot see (steering
    hooks, a fine-tuned adapter loaded under the same name, the pipeline batch
    size), so a cached run could replay answers from a different model state.
    """
    with tempfile.TemporaryDirectory() as tmp:
        old = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = tmp
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = old


def run_chunks(model, model_id: str, out_dir: Path, *, sets, langs, variant: str,
               max_tokens: int = 512, seed: int = 0,
               fetch: Callable[[Path], None] | None = None,
               publish: Callable[[Path], None] | None = None,
               log: Callable[[str], None] = print) -> Path:
    """Generate one (set, lang) chunk at a time and save each as soon as it is done.

    A chunk whose file already holds every answer is skipped, so a run that
    dies part-way can be restarted. ``fetch`` may restore a chunk from remote
    storage before the check; ``publish`` uploads each finished chunk.
    Returns the merged answers file.
    """
    from .data import load_tukabench

    chunk_paths = []
    for set_name in sets:
        for lang in langs:
            path = out_dir / variant / set_name / f"{lang}.jsonl"
            chunk_paths.append(path)
            samples = load_tukabench([set_name], [lang])
            if fetch and not path.exists():
                fetch(path)
            if _count_lines(path) == len(samples):
                log(f"skip {set_name}/{lang}: already done")
                continue
            t0 = time.time()
            answers = generate(samples, model_id, max_tokens=max_tokens, seed=seed, model=model)
            save_answers(answers, path, model_id=model_id, variant=variant,
                         max_tokens=max_tokens, seed=seed)
            empty = sum(not (s.actual_output or "").strip() for s in answers)
            log(f"done {set_name}/{lang}: {len(answers)} answers, {empty} empty, "
                f"{(time.time() - t0) / 60:.1f} min")
            if publish:
                publish(path)

    merged = out_dir / variant / "answers.jsonl"
    with open(merged, "w", encoding="utf-8") as out:
        for path in chunk_paths:
            out.write(path.read_text(encoding="utf-8"))
    return merged


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


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
