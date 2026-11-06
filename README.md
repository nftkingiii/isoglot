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
- [x] TukaBench's 3-way judge prompt (Jailbroken / Refused / Deflected) as an AuditKIT metric,
      judged by Claude Sonnet 5.5 with a hard spend cap
- [x] Baseline answers generated: 1,200 Tiny Aya answers, 0 empty (Kaggle T4, 82 min)
- [x] Per-language report: attack success rate, refusal, deflection, over-refusal, RCD,
      bootstrap intervals
- [x] Baseline judged (Claude Sonnet 5.5, $4.01): refusals lost in Hausa, Yoruba and Igbo mostly
      become confused answers, not harmful ones (results table in the HF article, coming)
- [ ] Locating refusal per language: [`notebooks/03_mechanism.ipynb`](notebooks/03_mechanism.ipynb)
      (SafeTune activation extraction; English refusal direction from BeaverTails vs Alpaca)
- [ ] Repairs and before/after results

Numbers are reported only once their run and judge are recorded.

## Run

```bash
pip install -e ".[gpu,judge,dev]"
pytest
isoglot generate --out runs/base/answers.jsonl --batch-size 8      # needs a GPU and HF_TOKEN
isoglot judge runs/base/answers.jsonl --cap 5 --out runs/base/judged.jsonl  # needs ANTHROPIC_API_KEY
isoglot report runs/base/judged.jsonl
```

The judge reads its key from the `ANTHROPIC_API_KEY` environment variable and stops once it
has spent `--cap` dollars. TukaBench judged with GPT-4.1; Isoglot uses Claude Sonnet 5.5 with
the same prompt, so rates are not one-to-one comparable with the paper's.

On Kaggle, use [`notebooks/01_baseline.ipynb`](notebooks/01_baseline.ipynb) (GPU) to generate
and [`notebooks/02_judge.ipynb`](notebooks/02_judge.ipynb) (CPU) to judge. Both upload
results chunk by chunk to a private Hugging Face dataset and resume if stopped.

## Data and licenses

| What | Source | License |
|---|---|---|
| Model | [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) | CC BY-NC 4.0 |
| Evaluation prompts | [McGill-NLP/tukabench](https://huggingface.co/datasets/McGill-NLP/tukabench) (public afri-jbb sets) | CC BY-NC 4.0 |
| Judge rubric | TukaBench `gpt4_judge.py`, reused verbatim | CC BY-NC 4.0 |
| Judge model | Claude Sonnet 5.5 (`claude-sonnet-5-5`), Anthropic API | Anthropic terms |

TukaBench prompts are used for evaluation only, never for training.

## Acknowledgements

TukaBench (Akinode et al., 2026) for the benchmark and judge rubric. Lexsi Labs for
the toolkits, Cohere Labs for Tiny Aya.
