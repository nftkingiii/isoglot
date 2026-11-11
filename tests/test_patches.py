"""The patched LISA methods must give the same numbers as SafeTune's on one device."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetune")


def _wrapper_pair():
    from safetune.core.optim.lisa import LisaOptimizerWrapper

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    return model, LisaOptimizerWrapper(model, rho=0.5, warmup_steps=0)


def _nudge(model, k):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * k)


def _penalties(w, model):
    loss = torch.tensor(1.0)
    w.current_step = 1
    out = []
    for state, k in (("alignment", 1), ("finetune", 2), ("alignment", 3)):
        w.switch_state(state)
        _nudge(model, k)
        with w.apply_proximal_penalty(loss, scale=0.5) as modified:
            out.append(float(modified))
    return out


def test_patch_matches_original_math():
    from safetune.core.optim.lisa import LisaOptimizerWrapper

    model_a, wa = _wrapper_pair()
    original = _penalties(wa, model_a)

    saved = (LisaOptimizerWrapper.switch_state, LisaOptimizerWrapper.apply_proximal_penalty)
    from isoglot.patches import patch_lisa_multi_device
    try:
        patch_lisa_multi_device()
        model_b, wb = _wrapper_pair()
        patched = _penalties(wb, model_b)
    finally:
        LisaOptimizerWrapper.switch_state, LisaOptimizerWrapper.apply_proximal_penalty = saved

    assert patched == pytest.approx(original, rel=1e-6)
    assert all(p > 1.0 for p in patched)  # the penalty actually applied
