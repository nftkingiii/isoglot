---
license: cc-by-nc-4.0
language: [en, ha, yo, ig]
tags: [safety, red-teaming, multilingual, african-languages, tukabench, lexsi-hackathon]
pretty_name: Isoglot runs
---

# Isoglot runs

> **Content warning.** This dataset contains harmful requests and some harmful model answers in
> English, Hausa, Yoruba and Igbo. It exists to measure and repair model safety. Do not use it to
> train models to comply with harmful requests.

Every artifact behind Isoglot, a study of Tiny Aya's safety gap in African languages for the Lexsi
AI Safety and Alignment Hackathon. Code: [github.com/nftkingiii/isoglot](https://github.com/nftkingiii/isoglot).

| Folder | Contents |
|---|---|
| `base/`, `steer/`, `harden/` | One run each of Tiny Aya Global (base), with refusal steering (steer) and fine-tuned with LISA (harden): 1,200 answers to the public TukaBench prompts per run (`answers.jsonl` plus 12 chunks), every judge label with its reason (`judged.jsonl`), and `report.md`. `steer/steer_config.json` records the steering calibration. |
| `mechanism/` | Per-layer separation tables (`tables.json`), the plot (`separation_by_layer.png`) and the English refusal directions for all 36 layers (`english_refusal_directions.npz`). |
| `traindata/` | Repair B data: `raw/` (944 generated rows and the 4 skipped seeds) and `curated/` (CuratorKIT exports in chat format with train/validation splits, manifest, dataset card, rejected rows, checksums and provenance). |

**Judging:** TukaBench's 3-way prompt (Jailbroken / Refused / Deflected), run by Claude Sonnet 5.5
as an AuditKIT metric. Each judged row records the judge model, prompt hash and AuditKIT
fingerprint. About 7% of calls were served by a fallback model after a safety classifier declined;
20-22 answers per run stayed unjudged (`label: JudgeError`).

**Sources and licenses:** TukaBench public sets (CC BY-NC 4.0, evaluation only), BeaverTails
(CC BY-NC 4.0), Alpaca (CC BY-NC 4.0), OR-Bench (CC BY 4.0), Tiny Aya Global outputs (CC BY-NC 4.0).
This dataset is released under CC BY-NC 4.0.
