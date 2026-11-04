"""isoglot generate | judge | report"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DEFAULT_MODEL, LANGS, SETS


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="isoglot")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="answer TukaBench prompts (GPU)")
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--variant", default="base")
    g.add_argument("--sets", nargs="+", default=list(SETS))
    g.add_argument("--langs", nargs="+", default=list(LANGS))
    g.add_argument("--limit", type=int, help="prompts per (set, lang); for smoke tests")
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="auto")
    g.add_argument("--batch-size", type=int, default=1)
    g.add_argument("--out", type=Path, required=True)

    j = sub.add_parser("judge", help="label saved answers (calls the judge endpoint)")
    j.add_argument("answers", type=Path)
    j.add_argument("--judge-model", required=True)
    j.add_argument("--base-url")
    j.add_argument("--out", type=Path, required=True)

    r = sub.add_parser("report", help="per-language table from judged rows")
    r.add_argument("judged", type=Path)

    a = p.parse_args(argv)
    if a.cmd == "generate":
        from .data import load_tukabench
        from .generate import generate, save_answers

        samples = load_tukabench(a.sets, a.langs, limit=a.limit)
        answers = generate(samples, a.model, max_tokens=a.max_tokens, seed=a.seed,
                           batch_size=a.batch_size, device=a.device)
        print(save_answers(answers, a.out, model_id=a.model, variant=a.variant,
                           max_tokens=a.max_tokens, seed=a.seed))
    elif a.cmd == "judge":
        from .generate import load_answers
        from .judge import TukaJudge, judge_answers, openai_client

        judge = TukaJudge(openai_client(a.judge_model, a.base_url), a.judge_model)
        rows = judge_answers(load_answers(a.answers), judge)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(a.out)
    else:
        from .metrics import summarize, to_markdown

        with open(a.judged, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        print(to_markdown(summarize(rows)))


if __name__ == "__main__":
    main()
