"""CPU smoke test of the real hf: generation path with a tiny public model.
Skipped when torch/transformers are not installed."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import auditkit as ak  # noqa: E402

from isoglot.generate import generate  # noqa: E402


def test_tiny_model_generates():
    samples = [ak.Sample(input="Hello", id="afri-jbb-harm/eng/0", task="afri-jbb-harm:eng",
                         metadata={"set": "afri-jbb-harm", "lang": "eng", "index": "0",
                                   "kind": "harmful", "goal_en": "Hello"})]
    out = generate(samples, "sshleifer/tiny-gpt2", max_tokens=8, device="cpu")
    assert out[0].id == samples[0].id and isinstance(out[0].actual_output, str)
    assert out[0].metadata["goal_en"] == "Hello"


def test_batched_generation_runs():
    samples = [ak.Sample(input=t, id=f"s{i}", task="t", metadata={"goal_en": t})
               for i, t in enumerate(["Hello", "A much longer prompt than the first one"])]
    out = generate(samples, "sshleifer/tiny-gpt2", max_tokens=8, batch_size=2, device="cpu")
    assert [s.id for s in out] == ["s0", "s1"] and all(isinstance(s.actual_output, str) for s in out)


def test_generation_is_never_replayed_from_cache():
    """A steering hook changes the model without changing AuditKIT's
    fingerprint; the second run must still generate, not replay the first."""
    pytest.importorskip("safetune")
    import numpy as np

    from isoglot.generate import hf_model
    from isoglot.steer import install_steering

    samples = [ak.Sample(input=t, id=f"s{i}", task="t", metadata={"goal_en": t})
               for i, t in enumerate(["Hello there", "Tell me a story"])]
    model = hf_model("sshleifer/tiny-gpt2", device="cpu")
    plain = [s.actual_output for s in generate(samples, "sshleifer/tiny-gpt2", max_tokens=8, model=model)]
    model._ensure_pipeline()
    hidden = model._pipeline.model.config.n_embd
    # not np.ones: LayerNorm subtracts the mean over dimensions, which removes a
    # uniform vector entirely
    direction = np.random.default_rng(0).normal(size=hidden)
    steer = install_steering(model, direction, strength=1000.0, layers=[0, 1])
    try:
        steered = [s.actual_output for s in generate(samples, "sshleifer/tiny-gpt2", max_tokens=8, model=model)]
    finally:
        steer.remove()
    assert steered != plain
    again = [s.actual_output for s in generate(samples, "sshleifer/tiny-gpt2", max_tokens=8, model=model)]
    assert again == plain  # hooks removed: back to the plain model


def test_pick_strength():
    from isoglot.steer import pick_strength

    results = [{"strength": 0, "harmful_refusal": 0.5, "harmless_refusal": 0.0},
               {"strength": 5, "harmful_refusal": 0.8, "harmless_refusal": 0.05},
               {"strength": 10, "harmful_refusal": 0.95, "harmless_refusal": 0.30}]
    assert pick_strength(results)["strength"] == 5
    assert pick_strength(results, max_harmless_refusal=0.0)["strength"] == 0
