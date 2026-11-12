Isoglot asks what Tiny Aya's safety gap in African languages is made of, and whether the Lexsi stack can close it. A small open model serving people in Hausa, Yoruba and Igbo must refuse harmful requests and answer ordinary ones in every language, yet recent benchmarks show models refuse far less in these languages.

We answered TukaBench's 1,200 public prompts (harmful, culturally adapted harmful, and benign "twins" on the same topics, in English, Hausa, Yoruba and Igbo) with Tiny Aya Global through AuditKIT, and labelled every answer Refused, Jailbroken or Deflected with TukaBench's judge prompt (Claude Sonnet 5.5 as judge, run as an AuditKIT metric).

Finding 1: the gap is confusion, not compliance. Refusal of harmful prompts falls from 79.5% in English to 39.7% in Yoruba, but harmful compliance does not rise (11.3% English, 5.2% Yoruba). The lost refusals become off-topic or incoherent answers: a Yoruba "bomb tutorial" becomes a vinegar project. The model is safe by incompetence.

Finding 2: Tiny Aya does not recognise the harm. Using SafeTune's activation extractor, we built the English refusal direction at all 36 layers from BeaverTails and Alpaca, outside the test set. It predicts refuse-or-deflect almost perfectly (AUROC 0.98), and the harmful prompts the model deflects sit with benign ones on it (AUROC 0.57-0.60) at every layer. This predicted that steering could not reach them without hitting benign prompts.

Repair A, SafeTune refusal-direction steering, tuned on held-out English prompts only, confirmed it: harmful compliance fell about threefold and refusals rose, but over-refusal of benign twins roughly tripled (English 10% to 33%) and the Yoruba gap to English stayed at 30 points.

Repair B teaches the distinction. Claude wrote 944 rows of refusals and helpful answers in four languages from BeaverTails, OR-Bench and Alpaca seeds (none near a TukaBench prompt); CuratorKIT cleaned, deduplicated and exported them; SafeTune's LISA trainer read the export directly and fine-tuned Tiny Aya with LoRA. The refusal gap closed: harmful-prompt refusal is now 84-88% in all four languages (Yoruba 40% to 84%, gap to English 4 points), harmful compliance 2-4%, and English over-refusal stays near the base model (12% vs 10%). The remaining cost is over-refusal of benign requests in the African languages (35-39%), the clearest target for more data.

Along the way we found and fixed two issues in the stack: AuditKIT's result cache cannot see model state, so a steered model replayed unsteered answers; and SafeTune 0.1.12's LISA cannot train a model split across GPUs. Both fixes are tested and documented.

Everything is reproducible from six resumable Kaggle notebooks on free T4s, with models, answers, judge labels, training data and per-layer tables on Hugging Face. Total API spend: $15.43.
