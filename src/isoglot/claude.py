"""Metered Claude calls, shared by the judge and the training-data generator.

Every call is charged to a SpendMeter and refused once its cap is reached.
Keys are read from the environment only.
"""

from __future__ import annotations

import os

# USD per 1M tokens (input, output), Anthropic first-party list prices, cached
# 2026-10-06. A fallback can be served by an Opus model; any model not listed
# here is charged at the highest listed rate.
CLAUDE_PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-5": (5.00, 25.00),
}


class BudgetExceeded(RuntimeError):
    """Raised before a call once the spend cap has been reached."""


class SpendMeter:
    """Running cost of calls, from the token usage each response reports.

    Prices are USD per million tokens. The cap is checked before every call,
    so a run stops at most one call past it.
    """

    def __init__(self, cap_usd: float, input_per_m: float, output_per_m: float):
        self.cap_usd = cap_usd
        self.input_per_m = input_per_m
        self.output_per_m = output_per_m
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.fallbacks = 0
        self.refusals = 0
        self.spent_usd = 0.0

    def check(self) -> None:
        if self.spent_usd >= self.cap_usd:
            raise BudgetExceeded(f"spend cap ${self.cap_usd:.2f} reached "
                                 f"(${self.spent_usd:.2f} after {self.calls} calls)")

    def add(self, input_tokens: int, output_tokens: int,
            rates: tuple[float, float] | None = None) -> None:
        """Record one call. ``rates`` overrides the default prices, e.g. when a
        fallback model with different pricing served the request."""
        inp, out = rates or (self.input_per_m, self.output_per_m)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.spent_usd += (input_tokens * inp + output_tokens * out) / 1e6
        self.calls += 1


def client_from_env(api_key_env: str = "ANTHROPIC_API_KEY"):
    import anthropic

    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"set {api_key_env} in the environment (never in code or notebooks)")
    return anthropic.Anthropic(api_key=key)


def metered_json(client, model: str, meter: SpendMeter, content: str, schema: dict, *,
                 max_tokens: int, effort: str, system: str | None = None) -> tuple[str | None, str | None]:
    """One Claude call with structured JSON output, charged to ``meter``.

    Thinking is off (``between_tools``, valid at effort high or below).
    ``fallbacks: "default"`` lets the API re-run a call a safety classifier
    declined on another model. Returns ``(text, None)``, or
    ``(None, category)`` when the request was declined even after fallback.

    Uses the Messages API rather than the Batches API (half price) because
    server-side fallbacks are rejected on Batches.
    """
    meter.check()
    kwargs = {"system": system} if system else {}
    resp = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "between_tools"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": content}],
        **kwargs,
    )
    fell_back = resp.model != model
    meter.fallbacks += fell_back
    top = max(CLAUDE_PRICES.values())
    meter.add(resp.usage.input_tokens, resp.usage.output_tokens,
              rates=CLAUDE_PRICES.get(resp.model, top) if fell_back else None)
    if resp.stop_reason == "refusal":
        meter.refusals += 1
        return None, (resp.stop_details.category if resp.stop_details else None)
    return next((b.text for b in resp.content if b.type == "text"), ""), None
