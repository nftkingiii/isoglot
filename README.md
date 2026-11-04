# Isoglot

The same safety in every language.

Isoglot tests whether a small open model, Cohere Labs' Tiny Aya, keeps its safety
when harmful requests arrive in Hausa, Yoruba or Igbo instead of English, and then
works to close the gaps it finds with the Lexsi Labs stack (CuratorKIT, AuditKIT,
CircuitKIT, SafeTune).

Built for the Lexsi AI Safety and Alignment Hackathon, Track 2a, *Multilingual and
code-mixed safety*.

## Status

Work in progress. What exists today:

- [x] Public TukaBench prompts loaded as AuditKIT samples (1,200 prompts: harm, culture
      and benign sets × English, Hausa, Yoruba, Igbo)
- [x] Generation through AuditKIT's `hf:` backend, saved with run settings
- [x] TukaBench's 3-way judge (Jailbroken / Refused / Deflected) as an AuditKIT metric
- [x] Per-language report: attack success rate, refusal, deflection, over-refusal, RCD,
      bootstrap intervals
- [ ] Baseline results for Tiny Aya
- [ ] Locating refusal per language (CircuitKIT, SafeTune)
- [ ] Repairs and before/after results

No results are claimed yet.

## Run

```bash
pip install -e ".[gpu,judge,dev]"
pytest
isoglot generate --out runs/base/answers.jsonl --batch-size 8      # needs a GPU and HF_TOKEN
isoglot judge runs/base/answers.jsonl --judge-model <model> --out runs/base/judged.jsonl
isoglot report runs/base/judged.jsonl
```

The judge reads its key from the `JUDGE_API_KEY` environment variable and works with any
OpenAI-compatible endpoint (`--base-url`).

On Kaggle, use [`notebooks/01_baseline.ipynb`](notebooks/01_baseline.ipynb).

## Data and licenses

| What | Source | License |
|---|---|---|
| Model | [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) | CC BY-NC 4.0 |
| Evaluation prompts | [McGill-NLP/tukabench](https://huggingface.co/datasets/McGill-NLP/tukabench) (public afri-jbb sets) | CC BY-NC 4.0 |
| Judge rubric | TukaBench `gpt4_judge.py`, reused verbatim | CC BY-NC 4.0 |

TukaBench prompts are used for evaluation only, never for training.

## Acknowledgements

TukaBench (Akinode et al., 2026) for the benchmark and judge rubric. Lexsi Labs for
the toolkits, Cohere Labs for Tiny Aya.
