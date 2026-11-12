---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
language: [en, ha, yo, ig]
tags: [safety, alignment, multilingual, african-languages, lisa, safetune, lexsi-hackathon]
library_name: transformers
pipeline_tag: text-generation
---

# isoglot-tiny-aya-lisa

[Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) fine-tuned to refuse harmful
requests as reliably in Hausa, Yoruba and Igbo as in English. Built for the Lexsi AI Safety and
Alignment Hackathon (Track 2a, multilingual and code-mixed safety).

- **Method:** SafeTune `LisaTrainer` (LISA, rho 0.1, 20 alignment / 20 fine-tune steps, 2 epochs,
  LoRA, lr 1e-4), merged, fp16.
- **Data:** 944 rows (refusals to harmful requests and helpful answers to benign ones, 118 per kind
  and language in English, Hausa, Yoruba and Igbo), written by Claude Sonnet 5.5 from BeaverTails,
  OR-Bench hard-1k and Alpaca seeds, cleaned and exported by CuratorKIT. In
  [NFTKING/isoglot-runs](https://huggingface.co/datasets/NFTKING/isoglot-runs) under `traindata/`.
- **Code:** [github.com/nftkingiii/isoglot](https://github.com/nftkingiii/isoglot), notebooks 05 and 06.

## Results on TukaBench (public sets, judged with TukaBench's prompt by Claude Sonnet 5.5)

| Harmful prompts, refused | English | Hausa | Igbo | Yoruba |
|---|---|---|---|---|
| Tiny Aya Global | 79.5% | 65.6% | 54.6% | 39.7% |
| **This model** | **88.2%** | **87.7%** | **84.6%** | **84.0%** |

| Benign prompts, over-refused | English | Hausa | Igbo | Yoruba |
|---|---|---|---|---|
| Tiny Aya Global | 10% | 22% | 14% | 11% |
| **This model** | **12%** | **37%** | **39%** | **35%** |

The refusal gap to English closes (Yoruba 40% to 84%) and harmful compliance stays at 2-4%, but
benign requests in Hausa, Yoruba and Igbo are over-refused more often than before. Use it as a
research artifact, not as a deployed assistant.

## License

CC BY-NC 4.0, inherited from Tiny Aya Global. Non-commercial use only.
