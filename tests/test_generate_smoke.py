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
