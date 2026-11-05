import json

import auditkit as ak
import pytest

from isoglot.data import read_rows, to_samples
from isoglot.generate import load_answers, save_answers
from isoglot.judge import LABELS, TukaJudge, build_message, judge_answers, parse_label
from isoglot.metrics import rate_ci, summarize, to_markdown

CSV = "Index,Goal,Goal_Translation\n0,Do a bad thing,Yi wani mummunan abu\n1,Another,  \n2,Third,Na uku\n"


def test_rows_to_samples(tmp_path):
    path = tmp_path / "hau.csv"
    path.write_text(CSV, encoding="utf-8")
    samples = to_samples(read_rows(path), "afri-jbb-harm", "hau")
    assert [s.id for s in samples] == ["afri-jbb-harm/hau/0", "afri-jbb-harm/hau/2"]  # blank skipped
    assert samples[0].input == "Yi wani mummunan abu"
    assert samples[0].metadata == {"set": "afri-jbb-harm", "lang": "hau", "index": "0",
                                   "kind": "harmful", "goal_en": "Do a bad thing"}
    benign = to_samples(read_rows(path), "afri-jbb-benign", "hau")
    assert benign[0].metadata["kind"] == "benign"


@pytest.mark.parametrize("raw,label", [
    ('{"label": "Refused", "reason": "said no"}', "Refused"),
    ('Sure: {"label": "Jailbroken", "reason": "x"}', "Jailbroken"),
    ("The answer is Deflected.", "Deflected"),
    ("Refused or Jailbroken, unclear", None),   # ambiguous text is a judge error
    ("", None),
])
def test_parse_label(raw, label):
    assert parse_label(raw)[0] == label


def test_message_uses_english_goal():
    msg = build_message("Do a bad thing", "Ba zan iya ba")
    assert "USER PROMPT:\nDo a bad thing" in msg and "MODEL RESPONSE:\nBa zan iya ba" in msg


def _answers():
    rows = []
    for lang, outs in {"eng": ["no", "no", "yes"], "hau": ["yes", "yes", "no"]}.items():
        for i, out in enumerate(outs):
            rows.append(ak.Sample(input=f"p{i}", id=f"afri-jbb-harm/{lang}/{i}",
                                  task=f"afri-jbb-harm:{lang}", actual_output=out,
                                  metadata={"set": "afri-jbb-harm", "lang": lang, "index": str(i),
                                            "kind": "harmful", "goal_en": f"g{i}"}))
    rows.append(ak.Sample(input="b", id="afri-jbb-benign/hau/0", task="afri-jbb-benign:hau",
                          actual_output="", metadata={"set": "afri-jbb-benign", "lang": "hau",
                                                      "index": "0", "kind": "benign", "goal_en": "b"}))
    return rows


def fake_call(message):
    response = message.split("MODEL RESPONSE:\n", 1)[1]
    label = {"yes": "Jailbroken", "no": "Refused"}.get(response, "garbage")
    return json.dumps({"label": label, "reason": "fake"}) if label != "garbage" else "???"


def test_judge_through_auditkit():
    rows = judge_answers(_answers(), TukaJudge(fake_call, "fake-judge"))
    labels = {r["id"]: r["label"] for r in rows}
    assert labels["afri-jbb-harm/eng/2"] == "Jailbroken"
    assert labels["afri-jbb-harm/hau/2"] == "Refused"
    assert labels["afri-jbb-benign/hau/0"] == "Deflected"  # empty answer, judge not called
    assert all(r["judge_model"] == "fake-judge" and r["fingerprint"] for r in rows)


def test_judge_identity_changes_with_model():
    a = TukaJudge(fake_call, "judge-a").identity()
    b = TukaJudge(fake_call, "judge-b").identity()
    assert a != b and a["rubric"] == "tukabench-3way"


def test_summary_and_rcd():
    rows = judge_answers(_answers(), TukaJudge(fake_call, "fake-judge"))
    table = {(t["kind"], t["lang"]): t for t in summarize(rows)}
    eng, hau = table[("harmful", "eng")], table[("harmful", "hau")]
    assert eng["refused"] == pytest.approx(2 / 3) and hau["refused"] == pytest.approx(1 / 3)
    assert hau["jailbroken"] == pytest.approx(2 / 3)
    assert hau["rcd"] == pytest.approx(0.5) and eng["rcd"] == pytest.approx(0.0)
    assert "| harmful | hau |" in to_markdown(summarize(rows))


def test_judge_errors_are_not_deflections():
    rows = [{"kind": "harmful", "lang": "eng", "label": "Refused"},
            {"kind": "harmful", "lang": "eng", "label": "JudgeError"}]
    t = summarize(rows)[0]
    assert t["judge_errors"] == 1 and t["refused"] == 1.0 and t["deflected"] == 0.0


def test_rate_ci_bounds():
    mean, lo, hi = rate_ci([1, 0, 1, 1, 0, 0, 1, 0])
    assert lo <= mean <= hi and mean == 0.5


def test_save_load_roundtrip(tmp_path):
    path = save_answers(_answers(), tmp_path / "a.jsonl", model_id="m", variant="base",
                        max_tokens=8, seed=0)
    back = load_answers(path)
    assert [s.id for s in back] == [s.id for s in _answers()]
    assert back[0].metadata["variant"] == "base" and back[0].actual_output == "no"


def test_run_chunks_resumes(tmp_path, monkeypatch):
    import isoglot.data as data
    import isoglot.generate as gen

    def fake_load(sets, langs, **kw):
        return to_samples([{"Index": str(i), "Goal": f"g{i}", "Goal_Translation": f"p{i}"}
                           for i in range(3)], sets[0], langs[0])

    calls = []

    def fake_generate(samples, model_id, **kw):
        calls.append(samples[0].task)
        return [s.__class__(**{**s.__dict__, "actual_output": "ok"}) for s in samples]

    monkeypatch.setattr(data, "load_tukabench", fake_load)
    monkeypatch.setattr(gen, "generate", fake_generate)
    published = []
    kw = dict(sets=["afri-jbb-harm"], langs=["eng", "hau"], variant="base",
              publish=published.append, log=lambda m: None)

    merged = gen.run_chunks(None, "m", tmp_path, **kw)
    assert calls == ["afri-jbb-harm:eng", "afri-jbb-harm:hau"] and len(published) == 2
    assert len(merged.read_text(encoding="utf-8").splitlines()) == 6

    # a half-written chunk is redone; a complete one is skipped
    (tmp_path / "base/afri-jbb-harm/hau.jsonl").write_text("{}\n", encoding="utf-8")
    calls.clear()
    gen.run_chunks(None, "m", tmp_path, **kw)
    assert calls == ["afri-jbb-harm:hau"]


def test_labels_constant():
    assert LABELS == ("Jailbroken", "Refused", "Deflected")
