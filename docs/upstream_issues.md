# Issues found in the Lexsi stack while building Isoglot

Versions: AuditKIT 1.1.3, SafeTune 0.1.12, CuratorKIT 0.1.24, transformers 5.22, peft (Kaggle, Nov 2026).
Project: https://github.com/nftkingiii/isoglot (Track 2a).

## 1. AuditKIT: the result cache replays runs from a different model state

**What happens.** `Runner.run()` (`runner.py`, `DiskCache().get(fingerprint)` before running,
`cache.set(...)` after) returns a cached `RunResult` whenever the fingerprint matches.
`RunSpec.fingerprint()` hashes the model's `identity()` (its name / `_model_name`), the scenario,
adapter, metrics and `RunConfig`, but not anything else about the model. So:

- **Steering hooks are invisible.** Install a forward hook (for example SafeTune's
  `RefusalDirectionModel`) on the model inside an `hf:` backend and call `ak.generate` again: you
  get the unsteered answers back from cache, silently.
- **The pipeline batch size is invisible.** A `batch_size` passed to the `hf:` model (the only way
  to batch, see 1b) is not in the fingerprint, so a batched-vs-unbatched comparison returns the
  first run's answers twice. Our own consistency check passed this way (8 s vs 107 s gave it away).
- **Failed scoring is cached.** A run where a scorer raised (spend cap, network error) is cached
  with those failures, so re-running the same evaluation replays the errors instead of retrying.

**Repro (CPU, seconds; verified with AuditKIT 1.1.3, SafeTune 0.1.12).**
```python
import os, tempfile
os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp()          # start from an empty cache
import auditkit as ak, numpy as np, safetune, torch
from auditkit.model import AutoModel

m = AutoModel.resolve("hf:sshleifer/tiny-gpt2", device="cpu")
s = [ak.Sample(input="Hello there", id="a"), ak.Sample(input="Tell me a story", id="b")]
cfg = ak.RunConfig(max_tokens=8)
plain = ak.generate(s, model=m, config=cfg)

m._ensure_pipeline()
d = np.random.default_rng(0).normal(size=m._pipeline.model.config.n_embd)
safetune.steer.RefusalDirectionModel(m._pipeline.model, torch.tensor(d, dtype=torch.float32),
                                     mode="steer", strength=1000.0, layers=[0, 1]).install()
steered = ak.generate(s, model=m, config=cfg)
out = lambda r: [x.actual_output for x in r]
print("replayed from cache:", out(plain) == out(steered))            # True

os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp()          # same steered call, empty cache
print("steering really changes output:", out(plain) != out(ak.generate(s, model=m, config=cfg)))  # True
```

**Our workaround.** Generation runs inside a throwaway cache (`XDG_CACHE_HOME` pointed at a
temporary directory) - `src/isoglot/generate.py`, `_fresh_auditkit_cache`. After judging, cache
entries for runs with any unscored sample are deleted - `src/isoglot/judge.py`, `_forget`.
Test: `tests/test_generate_smoke.py::test_generation_is_never_replayed_from_cache`.

**Suggested fix.** Let callers opt out (`RunConfig(cache=False)` or `ak.generate(..., cache=False)`),
include extra model kwargs such as `batch_size` in `HFGenModel.identity()`, and do not `cache.set`
a result whose `failed_count > 0`.

### 1b. AuditKIT `hf:` backend: `RunConfig.batch_size` is ignored, and batching lacks left padding

`HFGenModel.generate` passes all prompts to the transformers pipeline without a `batch_size`, so it
generates one prompt at a time whatever `RunConfig.batch_size` says. Passing `batch_size` as a model
kwarg does reach the pipeline, but then tokenizers without a pad token fail, and decoder-only models
need `padding_side="left"`, which is not set. Workaround: `src/isoglot/generate.py`, `hf_model`
(sets pad token and left padding on the pipeline tokenizer). Suggested fix: forward
`RunConfig.batch_size` to the pipeline and set left padding for causal LMs.

## 2. SafeTune: LISA cannot train a model split across GPUs

**What happens.** `LisaOptimizerWrapper.apply_proximal_penalty` (`core/optim/lisa.py`) starts
`penalty = 0.0` and adds `torch.norm(param - anchor[name]) ** 2` for every trainable parameter.
With the model on two GPUs (`device_map="auto"`, needed to train Tiny Aya on two 15 GB T4s), the
second term is on another device and the addition raises:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!
  safetune/core/optim/lisa.py, in apply_proximal_penalty
    penalty += (self.rho / 2) * torch.norm(param - self.finetune_weights[name]) ** 2
```

`switch_state` has the same pattern when it sums `sum_drift`.

**Our fix (same math).** Move each term to the loss device before summing; make the drift a float.
`src/isoglot/patches.py`, `patch_lisa_multi_device()`:

```python
penalty = torch.zeros((), device=loss.device, dtype=torch.float32)
for name, param in self.model.named_parameters():
    if param.requires_grad:
        term = (self.rho / 2) * torch.norm(param - anchor[name]) ** 2
        penalty = penalty + term.to(loss.device, torch.float32)
modified_loss = modified_loss + penalty * scale
```

Test: `tests/test_patches.py` checks the patched penalties equal SafeTune's original on one device.
With the patch, LISA trained Tiny Aya across two T4s in 13.4 minutes.

## Smaller notes

- **Kaggle + peft:** Kaggle ships torchao 0.10; peft refuses torchao < 0.16 when SafeTune wraps the
  model in LoRA. `pip uninstall -y torchao` fixes it (SafeTune's README mentions this for Colab; it
  applies to Kaggle too).
- **CuratorKIT:** `secrets_gate=True` needs `detect-secrets`, which the base install does not
  include; `curatorkit[hygiene]` also pulls detoxify. Installing `detect-secrets` alone was enough.
- **SafeTune LISA with an fp32 base on T4:** `device_map="auto"` fills GPU 0 first and leaves no room
  for activations; an fp16 base with `max_memory` per GPU works, with LoRA weights kept in fp32 by peft.
- **CuratorKIT tests:** `tests/unit/test_hygiene.py` is skipped entirely when detect-secrets is not
  installed. `TestSecretsGateRealDetection` calls `pytest.importorskip("detect_secrets")` in its class
  body, which runs at import time and skips the whole module, so its 58 tests (mocked so they need no
  hygiene packages) report as 1 skipped. Moving the call into a fixture or a `skipif` marker on that
  class would let the rest run. Our test for PR #52 lives in its own file for this reason.
