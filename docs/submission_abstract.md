Isoglot asks what Tiny Aya's safety gap in African languages is made of, and whether the Lexsi stack can close it. A small open model serving people in Hausa, Yoruba and Igbo must refuse harmful requests and answer ordinary ones in every language, yet recent benchmarks show models refuse far less in these languages.

We answered TukaBench's 1,200 public prompts (harmful, culturally adapted harmful, and benign "twins", in English, Hausa, Yoruba and Igbo) with Tiny Aya Global through AuditKIT, and labelled every answer Refused, Jailbroken or Deflected with TukaBench's judge prompt (Claude Sonnet 5.5, run as an AuditKIT metric).

Finding 1: the gap is confusion, not compliance. Refusal of harmful prompts falls from 79.5% in English to 39.7% in Yoruba, but harmful compliance does not rise (11.3% English, 5.2% Yoruba). The lost refusals become answers that miss the request: a Yoruba "bomb tutorial" becomes a vinegar project.

Finding 2: Tiny Aya does not recognise the harm. With SafeTune's activation extractor we built the English refusal direction at all 36 layers from data outside the test set. The harmful prompts the model deflects sit with benign ones on it (AUROC 0.57-0.60) at every layer, predicting that steering cannot reach them without hitting benign prompts.

Repair A, SafeTune refusal steering, confirmed it: harmful compliance fell about threefold, but over-refusal of benign twins roughly tripled and the Yoruba gap stayed at 30 points.

Repair B teaches the distinction. Claude wrote 944 refusal and helpful rows in four languages (none near a TukaBench prompt); CuratorKIT cleaned and exported them; SafeTune's LISA fine-tuned Tiny Aya. Harmful-prompt refusal rose to 84-88% in all four languages (Yoruba 40% to 84%), but benign over-refusal in the African languages reached 35-39%. Re-running the mechanism analysis shows the fine-tune improved recognition, not just refusal (harmful-vs-benign AUROC Hausa 0.81 to 0.89, Igbo 0.78 to 0.85, Yoruba 0.74 to 0.80).

After Lexsi's review we added 351 benign twins of the harmful training prompts and swept LISA's benign share. At 30:10, over-refusal falls to 21-25% while harmful refusal stays at 73-86%, beating steering on both axes in every language; Yoruba trades off most steeply. A Belebele and AfriMMLU check found no capability loss.

A native-speaker check (Igbo done, Hausa and Yoruba in progress) agrees with the judge on refusals (kappa 0.65 overall), finds the deflections fluent Igbo about the wrong thing, would count some as attempts to comply, and traced three misses to mistranslated benchmark prompts.

We found and fixed issues in the stack: AuditKIT's cache cannot see model state, so a steered model replayed unsteered answers; SafeTune's LISA cannot train a model split across GPUs. Both are fixed here, tested and reported to Lexsi; two smaller fixes went upstream as pull requests (SafeTune #85, CuratorKIT #52).

Everything is reproducible from nine resumable Kaggle notebooks on free T4s, with models, answers, judge labels, training data and plots on Hugging Face. Total API spend: $23.16.
