import numpy as np
import pytest

from isoglot.mechanism import (auroc, language_table, mean_diff_directions, pick_layer,
                               project, verdict)


def test_auroc_basics():
    assert auroc([3, 4], [1, 2]) == 1.0
    assert auroc([1, 2], [3, 4]) == 0.0
    assert auroc([1, 2], [1, 2]) == 0.5
    assert np.isnan(auroc([], [1]))


def test_direction_points_to_refusal_and_projects():
    rng = np.random.default_rng(0)
    d = 16
    axis = np.zeros(d); axis[3] = 1.0
    pos = {0: rng.normal(size=(50, d)) + 3 * axis}
    neg = {0: rng.normal(size=(50, d))}
    direction = mean_diff_directions(pos, neg)
    assert np.isclose(np.linalg.norm(direction[0]), 1.0) and direction[0][3] > 0.9
    p_pos, p_neg = project(pos, direction)[0], project(neg, direction)[0]
    assert auroc(p_pos, p_neg) > 0.95


def _rows_and_proj(deflected_shift):
    """Benign at 0, refused harmful at 3, deflected harmful at ``deflected_shift``."""
    rng = np.random.default_rng(1)
    rows, scores = [], []
    for kind, label, mu, n in [("benign", "Jailbroken", 0, 40), ("harmful", "Refused", 3, 40),
                               ("harmful", "Deflected", deflected_shift, 40)]:
        for _ in range(n):
            rows.append({"kind": kind, "label": label})
            scores.append(rng.normal(mu, 0.5))
    return rows, {5: np.array(scores)}


def test_recognised_harm_reads_as_steering_candidate():
    rows, proj = _rows_and_proj(deflected_shift=3)
    t = language_table(proj, rows)[0]
    assert t["harm_vs_benign"] > 0.95 and t["deflected_vs_benign"] > 0.95
    assert 0.3 < t["refused_vs_deflected"] < 0.7
    assert verdict(t).startswith("harm recognised")


def test_unrecognised_harm_reads_as_comprehension_gap():
    rows, proj = _rows_and_proj(deflected_shift=0)
    t = language_table(proj, rows)[0]
    assert t["deflected_vs_benign"] < 0.7 and t["refused_vs_deflected"] > 0.95
    assert verdict(t).startswith("harm not recognised")


def test_too_few_deflections():
    assert verdict({"n_deflected": 3, "deflected_vs_benign": 0.9}).startswith("too few")


def test_pick_layer_skips_early_layers():
    table = [{"layer": 0, "harm_vs_benign": 0.99}, {"layer": 5, "harm_vs_benign": 0.9},
             {"layer": 8, "harm_vs_benign": 0.95}]
    assert pick_layer(table, n_layers=10) == 8


def test_last_token_acts_with_safetune():
    pytest.importorskip("safetune")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from isoglot.mechanism import last_token_acts

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    acts = last_token_acts(model, tok, ["a", "a longer prompt here", "b"], chunk=2, batch_size=2)
    assert len(acts) == model.config.n_layer
    assert all(a.shape == (3, model.config.n_embd) for a in acts.values())
