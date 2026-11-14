# Isoglot: Tiny Aya's safety gap in African languages is a comprehension gap

*Lexsi AI Safety and Alignment Hackathon, Track 2a: Multilingual and code-mixed safety.*

> Every number here comes from a logged run in the repository's notebooks; nothing is estimated.

## 1. Problem and approach

A health-information assistant that runs on a small open model will be asked the same things in
Hausa, Yoruba and Igbo as in English. It must refuse the harmful requests and answer the ordinary
ones in every language. Recent benchmarks (TukaBench, LSR) show large language models refuse
harmful requests less often in African languages. We asked what that gap is made of on Tiny Aya,
a 3.35B model built for 70 languages, and whether the Lexsi stack can close it.

The approach, in four steps:

1. **Measure.** Answer TukaBench's 1,200 public prompts (harmful, culturally adapted harmful,
   and benign "twins" on the same topics, in English, Hausa, Yoruba and Igbo) and label every
   answer *Refused*, *Jailbroken* or *Deflected* (off-topic or incoherent) with TukaBench's judge
   prompt.
2. **Locate.** Find where refusal lives inside Tiny Aya and test whether it recognises harm in
   each language.
3. **Repair A: steering.** Add the English refusal direction at inference time.
4. **Repair B: fine-tuning.** Teach the model to tell harmful from benign requests in these
   languages with curated multilingual data and safety-preserving training.

**The findings in brief:**

- The refusals Tiny Aya loses in Hausa, Yoruba and Igbo turn into confused answers, not harmful
  ones, because the model does not recognise the request. It is safe by incompetence.
- Steering along the English refusal direction cannot fix that without refusing ordinary requests
  too, as the model's internals predicted.
- Fine-tuning on 944 curated rows with SafeTune's LISA **closes the refusal gap**: harmful-prompt
  refusal is now 84-88% in all four languages (Yoruba was 40%), with harmful compliance at 2-4%.
  It keeps English helpfulness close to the base model but over-refuses benign requests in the
  African languages (35-39%).
- Adding benign "twins" of the harmful training prompts and shifting LISA toward benign steps buys
  most of that back: at 30:10, over-refusal falls to 21-25% while harmful refusal stays at 73-86%,
  better than steering on both counts in every language. Yoruba trades off most steeply.

## 2. Lexsi repositories used

| Repository | What we used it for | Features used |
|---|---|---|
| **AuditKIT** | Every generation and every judgement | `ak.generate` with the `hf:` backend; a custom 3-way judge written as an AuditKIT `Metric`; `model="precomputed"` re-scoring; run fingerprints |
| **SafeTune** | Locating refusal, Repair A, Repair B | `SteeringVectorExtractor` (per-layer last-token activations); `RefusalDirectionModel` in steer mode; `LisaTrainer` (Harden) reading CuratorKIT exports directly |
| **CuratorKIT** | Repair B training data | Exact and MinHash dedup, text cleaning, secrets gate, schema gate, chat-format train/validation export with `lexsi_provenance.json`, dataset card and checksums |

Two things we found while chaining them, both fixed in our code and worth reporting upstream:

- **AuditKIT's result cache cannot see model state.** A run is fingerprinted by model name and
  settings, so a steered model, or the same model with a different pipeline batch size, replays
  the cached answers of the plain model. It also caches runs where scoring raised. Isoglot
  generates with a throwaway cache and deletes cache entries for failed scoring runs.
- **SafeTune 0.1.12's LISA cannot train a model split across GPUs.** Its proximal penalty adds
  per-parameter norms from different devices. `isoglot/patches.py` moves each term to one device;
  a test shows the penalties match SafeTune's original exactly on one device.

## 3. Models and licenses

| Model | Role | License |
|---|---|---|
| [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B) | The model under test and the base for Repair B | CC BY-NC 4.0 |
| Claude Sonnet 5.5 (`claude-sonnet-5-5`, Anthropic API) | Judge; writer of the multilingual training examples | Anthropic commercial terms |

Because Tiny Aya is non-commercial, Isoglot is a research artifact, not a product, and we did not
include a closed-model cost comparison.

## 4. Datasets, benchmarks and licenses

| Data | Use | License |
|---|---|---|
| [McGill-NLP/tukabench](https://huggingface.co/datasets/McGill-NLP/tukabench), public `afri-jbb-harm`, `afri-jbb-culture`, `afri-jbb-benign` (eng, hau, yor, ibo) | **Evaluation only, never training** | CC BY-NC 4.0 |
| TukaBench judge prompt (`gpt4_judge.py`), reused word for word | Judging | CC BY-NC 4.0 |
| [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Refusal direction (30k_test, n=128); training seeds (30k_train, n=120) | CC BY-NC 4.0 |
| [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | Harmless side of the direction (n=128); plain benign seeds (n=30) | CC BY-NC 4.0 |
| [bench-llm/or-bench](https://huggingface.co/datasets/bench-llm/or-bench) `or-bench-hard-1k` | Benign seeds that look risky (n=90) | CC BY 4.0 |

Contamination controls: training seeds within 0.5 character-5-gram Jaccard of any TukaBench goal
are dropped (none were), JailbreakBench (TukaBench's source) is never used for training, and
steering strength is tuned on held-out English prompts, never on TukaBench.

## 5. Model artifacts (collection)

[huggingface.co/collections/NFTKING/isoglot-models](https://huggingface.co/collections/NFTKING/isoglot-models)

- [`NFTKING/isoglot-tiny-aya-lisa`](https://huggingface.co/NFTKING/isoglot-tiny-aya-lisa): Tiny Aya
  fine-tuned with SafeTune LISA on the Isoglot data, merged, fp16, with `lexsi_provenance.json`.
- [`NFTKING/isoglot-tiny-aya-lisa-sweep-ft30-al10`](https://huggingface.co/NFTKING/isoglot-tiny-aya-lisa-sweep-ft30-al10)
  and [`-sweep-ft36-al4`](https://huggingface.co/NFTKING/isoglot-tiny-aya-lisa-sweep-ft36-al4): the
  same recipe with benign twins added, at two benign shares of LISA steps (section 7.4c).
- English refusal directions for all 36 layers (`mechanism/english_refusal_directions.npz` in the
  dataset below).

## 6. Data artifacts (collection)

[huggingface.co/collections/NFTKING/isoglot-data](https://huggingface.co/collections/NFTKING/isoglot-data).
The [`isoglot-runs`](https://huggingface.co/datasets/NFTKING/isoglot-runs) dataset holds:

- `base/`, `steer/`, `harden/`: every model answer (1,200 per run) and every judge label with its
  reason, judge model, prompt hash and AuditKIT fingerprint.
- `sweep-ft30-al10/`, `sweep-ft36-al4/`: the same for the two sweep models (Hausa, Yoruba and Igbo,
  900 answers each).
- `mechanism/`: per-layer separation tables and the plot below.
- `traindata/`: 944 generated rows (raw) and the CuratorKIT exports, with manifest, dataset card,
  rejected rows and checksums; the 351 benign twins and the merged benign export (823 rows).

## 7. Experiments and findings

### 7.1 Baseline: the gap is confusion, not compliance

Harmful prompts (harm + culturally adapted, n=200 per language):

| | English | Hausa | Igbo | Yoruba |
|---|---|---|---|---|
| Refused | 79.5% | 65.6% | 54.6% | 39.7% |
| Jailbroken | 11.3% | 5.6% | 9.2% | 5.2% |
| Deflected | 9.2% | 28.7% | 36.2% | 55.2% |

Benign twins (n=100 per language):

| | English | Hausa | Igbo | Yoruba |
|---|---|---|---|---|
| Answered | 60% | 35% | 27% | 16% |
| Over-refused | 10% | 22% | 14% | 11% |
| Deflected | 30% | 43% | 59% | 73% |

Refusal falls by 40 points from English to Yoruba, but harmful compliance does **not** rise: it is
lowest outside English. The lost refusals became *Deflected* answers. On the direct harm set Hausa
refuses almost as often as English (85.6% vs 89.7%); on the culturally adapted set Yoruba refuses
only 20.4% and deflects 73.5%. The same model gives a useful answer to only 16% of benign Yoruba
requests.

The judge's reasons for all 252 deflected harmful answers describe confusion, not partial harm.
One Yoruba "bomb tutorial" answer is a household vinegar project; one Igbo answer "pretends to
comply" with a child-labour request and then lists anti-child-labour measures. The model is safe
because it does not understand, which will stop holding as its language ability improves.

### 7.2 Locating refusal: Tiny Aya does not recognise the harm

We built the English refusal direction at every layer (mean BeaverTails harmful minus mean Alpaca
activation at the last prompt token, via SafeTune's extractor) and projected each language's
TukaBench prompts onto it. AUROC, where 0.5 is chance:

| At layer 32 of 36 | English | Hausa | Igbo | Yoruba |
|---|---|---|---|---|
| Harmful vs benign | 0.92 | 0.81 | 0.79 | 0.74 |
| **Deflected harmful vs benign** | 0.71 | **0.58** | **0.57** | **0.60** |
| Refused vs deflected | 0.98 | 0.98 | 0.98 | 0.98 |

![Separation by layer](https://huggingface.co/datasets/NFTKING/isoglot-runs/resolve/main/mechanism/separation_by_layer.png)

The direction predicts refuse-or-deflect almost perfectly (0.98), and the harmful prompts Tiny Aya
deflects sit with the benign ones on it. This holds at **every** layer: deflected-vs-benign never
reaches 0.75 in Hausa, Yoruba or Igbo (it peaks near 0.7 at layers 8-10 and falls to 0.55-0.60).
It is a comprehension gap, not "the model knows but does not refuse". That gave a testable
prediction: steering along the refusal direction cannot reach the deflected harmful prompts without
also reaching benign ones.

### 7.3 Repair A: refusal steering trades helpfulness for safety, as predicted

SafeTune `RefusalDirectionModel`, steer mode, layer 24, strength 0.5 × the harmful-harmless mean
distance, chosen on held-out English prompts as the strongest setting keeping English harmless
refusal at or under 10% (it was 6%).

| Baseline → steered | English | Hausa | Igbo | Yoruba |
|---|---|---|---|---|
| Harmful: refused | 79.5 → 92.3% | 65.6 → 80.5% | 54.6 → 74.2% | 39.7 → 62.1% |
| Harmful: jailbroken | 11.3 → 3.6% | 5.6 → 2.1% | 9.2 → 3.1% | 5.2 → 2.6% |
| Benign: over-refused | 10 → 33% | 22 → 40% | 14 → 43% | 11 → 31% |
| Benign: answered | 60 → 38% | 35 → 21% | 27 → 16% | 16 → 12% |

Steering cuts harmful compliance about threefold and raises refusals in every language, but
over-refusal on the benign twins roughly triples and the Yoruba gap to English stays at 30 points.
The English calibration set (plain Alpaca instructions) showed 6% over-refusal; on benign prompts
that share the harmful prompts' topics the real cost was 33%. Calibrating on easy harmless prompts
hides the price of steering.

### 7.4 Repair B: fine-tuning closes the refusal gap

Data (notebook 05): 240 English seeds (120 BeaverTails harmful, 90 OR-Bench hard benign, 30
Alpaca) written by Claude Sonnet 5.5 as natural Hausa, Yoruba, Igbo and English requests with a
short refusal (harmful) or a helpful answer of at most 120 words (benign). 944 rows, 118 per kind
and language; 4 seeds declined by Claude; CuratorKIT rejected none. Cost $4.38.

Training (notebook 06): SafeTune `LisaTrainer` (rho 0.1, 20 alignment / 20 fine-tune steps, 2
epochs, LoRA, lr 1e-4), refusals as the safety set and helpful answers as the task set, read
straight from the CuratorKIT exports. Base model loaded in fp16 and split across two T4s; trained
in 13.4 minutes. The merged model answered all 1,200 TukaBench prompts (none empty) in 54 minutes,
against 82 for the base model, so its answers are shorter.

Harmful prompts (n=200 per language), base → steered → fine-tuned:

| | English | Hausa | Igbo | Yoruba |
|---|---|---|---|---|
| Refused | 79.5 → 92.3 → **88.2%** | 65.6 → 80.5 → **87.7%** | 54.6 → 74.2 → **84.6%** | 39.7 → 62.1 → **84.0%** |
| Jailbroken | 11.3 → 3.6 → **3.6%** | 5.6 → 2.1 → **4.1%** | 9.2 → 3.1 → **2.1%** | 5.2 → 2.6 → **2.6%** |
| Deflected | 9.2 → 4.1 → **8.2%** | 28.7 → 17.4 → **8.2%** | 36.2 → 22.7 → **13.3%** | 55.2 → 35.4 → **13.4%** |
| Refusal gap to English (RCD) | — | 0.17 → 0.13 → **0.01** | 0.31 → 0.20 → **0.04** | 0.50 → 0.33 → **0.05** |

Benign twins (n=100 per language), base → steered → fine-tuned:

| | English | Hausa | Igbo | Yoruba |
|---|---|---|---|---|
| Answered | 60 → 38 → **50%** | 35 → 21 → **28%** | 27 → 16 → **17%** | 16 → 12 → **10%** |
| Over-refused | 10 → 33 → **12%** | 22 → 40 → **37%** | 14 → 43 → **39%** | 11 → 31 → **35%** |
| Deflected | 30 → 29 → **38%** | 43 → 39 → **35%** | 59 → 41 → **44%** | 73 → 57 → **55%** |

What changed:

- **The refusal gap is closed.** Yoruba refuses harmful requests 84% of the time, up from 40%, and
  the gap to English shrinks from 40 points to 4. Hausa and Igbo match English within 4 points.
  Steering only narrowed it to 30 points in Yoruba.
- **Confusion falls.** Deflected answers to harmful prompts drop from 55% to 13% in Yoruba and from
  36% to 13% in Igbo: the model now answers in a recognisable way.
- **English helpfulness mostly survives.** English over-refusal is 12% (base 10%, steered 33%).
- **Over-refusal in the African languages is the remaining cost.** Benign over-refusal rises to
  35-39% in Hausa, Igbo and Yoruba, about what steering caused there. The model has learned to
  refuse in these languages more reliably than it has learned which requests to refuse. With
  only 118 benign rows per language against benign test twins that share the harmful prompts'
  topics, this is the clearest place for more data.

### 7.4b Did the fine-tune teach recognition, or only refusal?

[`03_mechanism`](notebooks/03_mechanism.ipynb) re-run on the fine-tuned model (`VARIANT = "harden"`;
its own refusal direction, rebuilt from the same BeaverTails/Alpaca data, and its own judged
answers). Same layers, base → fine-tuned, AUROC:

| | English | Hausa | Igbo | Yoruba |
|---|---|---|---|---|
| Harmful vs benign, layer 25 | 0.92 → **0.95** | 0.81 → **0.89** | 0.78 → **0.85** | 0.74 → **0.80** |
| Harmful vs benign, layer 32 | 0.92 → **0.95** | 0.81 → **0.88** | 0.79 → **0.84** | 0.74 → **0.78** |
| Deflected harmful (count) | 18 → 16 | 56 → 16 | 71 → 26 | 107 → 26 |
| Deflected vs benign, layer 25 | 0.69 → 0.79 | 0.57 → 0.59 | 0.54 → 0.45 | 0.60 → 0.53 |

- **Recognition improved, not just refusal.** Harmful and benign prompts separate better inside the
  fine-tuned model in every language, at every layer we compared (best layer: Hausa 0.82 → 0.89,
  Igbo 0.80 → 0.85, Yoruba 0.75 → 0.80).
- **What is left is what it still does not understand.** The harmful prompts it still deflects
  (16-26 per language, down from 56-107) still look benign on the direction (0.45-0.59).
- **The boundary is still blurrier than in English** (0.80-0.89 against 0.95), which fits the
  remaining over-refusal of benign twins in these languages and is what the twins sweep targets.
  The deflected-prompt counts are small, so treat those AUROCs as rough.

### 7.4c Benign twins and a benign-share sweep

Suggested by the Lexsi team after review. [`08_twins`](https://github.com/nftkingiii/isoglot/blob/main/notebooks/08_twins.ipynb) wrote a benign
**twin** for each of the 120 harmful training seeds (a safe request on the same topic, e.g. "teach me
to break into a car" → "I locked my keys in my car, how do I get back in safely?") with a helpful
answer, in Hausa, Yoruba and Igbo: 351 rows, none close to a TukaBench prompt, $2.25. CuratorKIT
merged them with the first benign rows (823 rows, 0 rejected).
[`09_sweep`](https://github.com/nftkingiii/isoglot/blob/main/notebooks/09_sweep.ipynb) retrained with the twins at two benign shares of LISA steps
(benign : safety = 30:10 and 36:4, against 20:20 before) and answered in Hausa, Yoruba and Igbo.

![Safety vs over-refusal per language](https://raw.githubusercontent.com/nftkingiii/isoglot/main/docs/tradeoff/tradeoff.png)

| Harmful refused / benign over-refused | Hausa | Igbo | Yoruba |
|---|---|---|---|
| Base | 65.6% / 22% | 54.6% / 14% | 39.7% / 11% |
| Steering | 80.5% / 40% | 74.2% / 43% | 62.1% / 31% |
| LISA 20:20 | 87.7% / 37% | 84.6% / 39% | 84.0% / 35% |
| LISA + twins 30:10 | **85.6% / 21%** | **78.5% / 24%** | **72.8% / 25%** |
| LISA + twins 36:4 | 73.2% / 13% | 67.0% / 15% | 43.6% / 9% |

Harmful prompts jailbroken: 30:10 gives 3.6% / 5.1% / 2.1%, 36:4 gives 7.2% / 8.8% / 5.6% (Hausa / Igbo
/ Yoruba; base 5.6% / 9.2% / 5.2%).

- **The 30:10 setting beats steering on both axes in every language:** more harmful requests
  refused and fewer benign ones over-refused.
- **The twins buy back most of the over-refusal.** From 20:20 to 30:10, over-refusal falls 11-16
  points in every language for a 2-11 point loss in harmful refusal; Hausa is back to the base
  model's over-refusal (21% vs 22%) while refusing 20 points more harmful requests.
- **At 36:4, Hausa and Igbo beat or match the base model on both axes** (Hausa 73% vs 66% refused
  at 13% vs 22% over-refused; Igbo 67% vs 55% at 15% vs 14%), but jailbreaks drift back to base
  levels. 30:10 is the most balanced setting.
- **Yoruba's curve is the steepest.** At 36:4 it falls back near the base model (44% refused, 51%
  deflected): with the least comprehension, Yoruba depends most on the safety steps.
- Caveat: 20:20 had no twins and the two sweep runs have them, so the curve shows the twins and the
  step share together. Judged in Hausa, Yoruba and Igbo only; 2 settings plus the earlier run.

### 7.5 Capability check: no drop, and a second view of the gap

Did the fine-tune hurt general ability in these languages?
[`07_capability`](notebooks/07_capability.ipynb) scores both models on the same 200 Belebele
reading and 200 AfriMMLU knowledge questions per language by log-likelihood (AuditKIT
`MCQAdapter`, no judge), with a paired bootstrap 95% interval on the change. Chance is 25%.

| Benchmark | Lang | Base | LISA | Change [95% CI] |
|---|---|---|---|---|
| Belebele | English | 44.5% | 44.5% | +0.0 [-2.5, +2.5] |
| Belebele | Hausa | 34.5% | 37.5% | +3.0 [+0.5, +6.0] |
| Belebele | Igbo | 29.5% | 30.5% | +1.0 [-1.0, +3.0] |
| Belebele | Yoruba | 24.5% | 24.5% | +0.0 [-3.0, +3.0] |
| AfriMMLU | English | 29.5% | 29.5% | +0.0 [-2.0, +2.0] |
| AfriMMLU | Hausa | 24.0% | 26.0% | +2.0 [-0.5, +5.0] |
| AfriMMLU | Igbo | 20.5% | 21.5% | +1.0 [-2.0, +4.5] |
| AfriMMLU | Yoruba | 25.0% | 22.5% | -2.5 [-5.5, +0.0] |

- **No drop detected.** Every interval includes zero except Hausa reading, which rose 3 points.
- **The reading scores line up with the safety gap.** Base reading accuracy falls from English
  (44.5%) to Hausa (34.5%), Igbo (29.5%) and Yoruba (24.5%, chance), the same order as the refusal
  gap: independent support for "the model does not understand", measured without any judge.
- **Limits of this check.** Several cells sit near chance (all of AfriMMLU, Yoruba reading), so a
  small loss there would not show. Log-likelihood scoring of an instruction-tuned model without its
  chat template is a weak format for this model; it is the same for both models, so the comparison
  is fair, but the absolute numbers understate what the model can do.

### 7.6 What else we tried or checked

- A keyword refusal count first suggested the gap; it could not separate refusal from confusion,
  which is why every number above uses the 3-way judge.
- Batched generation was checked against one-at-a-time generation: 7 of 8 answers identical, the
  8th differing after ~90 characters (fp16 arithmetic).
- An earlier batched/unbatched check had been answered from AuditKIT's cache; we found and fixed
  that (section 2).
- Judge: Claude Sonnet 5.5 with TukaBench's prompt. TukaBench used GPT-4.1, so our rates are not
  one-to-one comparable with the paper's; Tiny Aya was not in the paper. About 7% of judge calls
  were re-run by a fallback model after a safety classifier declined, and a few of the 1,200 answers
  per run (20-22) stayed unjudged; both are reported per run. The judge prompt calls every prompt a
  "harmful request", including the benign twins; we kept it verbatim for consistency.
- **Not done:** a native-speaker check of the judge in Hausa, Yoruba and Igbo, and code-mixed
  prompts (TukaBench's code-switched set is not public).

## 8. With more credits

- Native-speaker validation of the judge and of the generated training data.
- Code-mixed evaluation, by generating code-switched variants with TukaBench's published method.
- A fuller sweep that separates the twins from the step share, with more points between 30:10
  and 36:4, and more benign data for Yoruba, whose trade-off is steepest.
- The same fine-tune on human-written data (the Aya Dataset) instead of Claude-written rows.
- A second training method (SafeGrad, circuit-restricted LoRA via CircuitKIT) to compare how much
  recognition each method adds.
- The same pipeline on Aya Expanse 8B, to see whether a larger model is unsafe where Tiny Aya is
  merely lost.

## 9. Team

Solo project by [@NFTKING](https://huggingface.co/NFTKING).

## Reproduce

Everything runs from [github.com/nftkingiii/isoglot](https://github.com/nftkingiii/isoglot): notebooks
01 (baseline answers), 02 (judge, any run), 03 (mechanism), 04 (steering), 05 (training data), 06
(fine-tuning), 07 (capability), 08 (benign twins), 09 (benign-share sweep). Each runs on a free
Kaggle session, uploads results chunk by chunk to the Hub and resumes if stopped. Total API spend:
$23.16 (judging $4.01 + $3.72 + $3.32 + $2.68 + $2.80, training data $4.38 + $2.25).
GPU: free Kaggle T4 sessions.
