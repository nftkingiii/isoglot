import pytest

from isoglot.capability import afrimmlu_samples, belebele_samples, paired_table, to_markdown


def test_belebele_rows_to_samples():
    row = {"link": "u", "question_number": "2", "flores_passage": "P", "question": "Q",
           "mc_answer1": "a", "mc_answer2": "b", "mc_answer3": "c", "mc_answer4": "d",
           "correct_answer_num": "3"}
    s = belebele_samples([row], "hau")[0]
    assert s.choices == ["a", "b", "c", "d"] and s.target == 2
    assert s.input == "Passage: P\nQuestion: Q\nAnswer:" and s.task == "belebele:hau"


def test_afrimmlu_rows_to_samples():
    row = {"question": "2p = 24?", "choices": "['p = 4', 'p = 8', 'p = 12', 'p = 24']",
           "answer": "C", "subject": "math"}
    s = afrimmlu_samples([row], "yor")[0]
    assert s.choices[2] == "p = 12" and s.target == 2 and s.metadata["subject"] == "math"


def test_paired_table():
    samples = belebele_samples([{"link": "u", "question_number": str(i), "flores_passage": "P",
                                 "question": "Q", **{f"mc_answer{k}": str(k) for k in range(1, 5)},
                                 "correct_answer_num": "1"} for i in range(4)], "eng")
    ids = [s.id for s in samples]
    before = dict(zip(ids, [True, True, False, False]))
    after = dict(zip(ids, [True, False, False, True]))
    t = paired_table(samples, before, after)[0]
    assert t["before"] == 0.5 and t["after"] == 0.5 and t["change"] == 0.0
    lo, hi = t["change_ci"]
    assert lo <= 0.0 <= hi
    assert "| belebele | eng | 4 | 50.0% | 50.0% |" in to_markdown([t])


def test_mcq_scoring_through_auditkit():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from isoglot.capability import score

    samples = afrimmlu_samples([{"question": "One plus one is", "choices": "['two', 'blue', 'cat', 'run']",
                                 "answer": "A", "subject": "x"}], "eng")
    out = score(samples, "hf:sshleifer/tiny-gpt2")
    assert list(out) == [samples[0].id] and isinstance(out[samples[0].id], bool)
