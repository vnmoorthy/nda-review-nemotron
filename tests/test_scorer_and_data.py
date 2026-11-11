"""Fast checks that need no AWS and no dataset download.

    pytest -q
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab"))

from contractnli_scorer import parse_answer, score_record  # noqa: E402

GOLD = json.dumps({
    "nda-1": {"label": "Entailment", "evidence": [3, 4]},
    "nda-2": {"label": "Contradiction", "evidence": [7]},
    "nda-3": {"label": "NotMentioned", "evidence": []},
})


def scores(pred):
    out = score_record({"model_response": pred, "reference_answer": GOLD})
    return {m["name"]: m["value"] for m in out["metrics_list"]}


def test_gold_scores_perfectly():
    s = scores(GOLD)
    assert s == {"label_correct": 1.0, "evidence_f1": 1.0, "contradiction_correct": 1.0, "json_valid": 1.0}


def test_right_verdicts_wrong_clauses():
    pred = json.dumps({
        "nda-1": {"label": "Entailment", "evidence": [9]},
        "nda-2": {"label": "Contradiction", "evidence": [7]},
        "nda-3": {"label": "NotMentioned", "evidence": []},
    })
    s = scores(pred)
    assert s["label_correct"] == 1.0
    assert 0 < s["evidence_f1"] < 1.0  # verdicts alone are not enough


def test_majority_guess_scores_zero_evidence():
    pred = json.dumps({k: {"label": "NotMentioned", "evidence": []} for k in ("nda-1", "nda-2", "nda-3")})
    s = scores(pred)
    assert s["evidence_f1"] == 0.0
    assert s["contradiction_correct"] == 0.0


def test_unparseable_reply_scores_zero():
    s = scores("I think the contract is fine.")
    assert s["json_valid"] == 0.0 and s["label_correct"] == 0.0


def test_parser_handles_fences_and_think_blocks():
    obj, ok = parse_answer('<think>hmm {"x":1}</think>```json\n{"nda-1": {"label": "Entailment", "evidence": [1]}}\n```')
    assert ok and obj["nda-1"]["evidence"] == [1]


def test_demo_data_matches_published_baselines():
    meta = json.loads((ROOT / "demo" / "data" / "index.json").read_text())
    published = {
        "tuned": "tuned_reference_metrics.json",
        "frontier": "frontier_metrics.json",
        "base": "base_metrics.json",
    }
    for key, fname in published.items():
        ref = json.loads((ROOT / "lab" / "baselines" / fname).read_text())
        for metric in ("label_correct", "evidence_f1", "contradiction_correct", "json_valid"):
            assert abs(meta["summary"][key][metric] - ref[metric]) < 0.02, (key, metric)
    assert len(meta["contracts"]) == 123
    assert len(list((ROOT / "demo" / "data" / "c").glob("*.json"))) == 123
    assert len(meta["checklist"]) == 17
