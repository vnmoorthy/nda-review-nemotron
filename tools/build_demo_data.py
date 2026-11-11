"""Build the static data behind the web demo.

Joins the ContractNLI test split (123 real NDAs) with the per-contract model outputs
saved in lab/baselines/, re-scores every output with the lab's own scorer, and writes:

    demo/data/index.json        one row per contract: title, size, per-model scores
    demo/data/c/<i>.json        one file per contract: numbered spans, the expert
                                answer, and each model's parsed verdicts + citations

Re-scoring is a check, not decoration: if a recomputed score drifts from the score
stored next to the generation, the build fails. So every number the demo shows is
reproducible from the raw model output.

    python tools/build_demo_data.py            # needs data/contract-nli (see README)
"""

from __future__ import annotations

import json
import os
import pathlib
import statistics
import sys
from urllib.parse import unquote

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAB = ROOT / "lab"
os.environ.setdefault("CONTRACTNLI_DIR", str(ROOT / "data" / "contract-nli"))
sys.path.insert(0, str(LAB))

import contractnli as C  # noqa: E402
from contractnli_scorer import parse_answer, score_record  # noqa: E402

MODELS = {
    # key in the demo -> (rows file, display name)
    "tuned": ("tuned_reference_rows.json", "Nemotron 3.5 Bolt E4B + SFT (this repo)"),
    "frontier": ("frontier_rows.json", "Claude Sonnet 5 (zero-shot)"),
    "base": ("base_rows.json", "Nemotron 3.5 Bolt E4B (untuned)"),
}
METRICS = ("label_correct", "evidence_f1", "contradiction_correct", "json_valid")


def gold_json(doc, keys):
    g = C.gold_for(doc)
    return {k: {"label": g[k]["choice"], "evidence": list(g[k]["spans"])} for k in keys if k in g}


def flat(scored):
    return {m["name"]: m["value"] for m in scored["metrics_list"]}


def build(out_dir: pathlib.Path = ROOT / "demo" / "data") -> dict:
    C.ensure_dataset(str(ROOT / "data"))
    docs, labels = C.load("test")
    keys = list(labels)

    rows = {}
    for key, (fname, _) in MODELS.items():
        rows[key] = {int(r["index"]): r for r in json.loads((LAB / "baselines" / fname).read_text())}
        assert len(rows[key]) == len(docs), f"{fname}: {len(rows[key])} rows for {len(docs)} docs"

    (out_dir / "c").mkdir(parents=True, exist_ok=True)
    index, per_model = [], {k: [] for k in MODELS}

    for i, doc in enumerate(docs):
        gold = gold_json(doc, keys)
        title = unquote(doc["file_name"].rsplit(".", 1)[0]).replace("_", " ").strip()
        entry = {"i": i, "title": title, "spans": len(doc["spans"]),
                 "chars": len(doc["text"]), "scores": {}}
        detail = {
            "i": i, "title": entry["title"], "source": doc.get("url", ""),
            "spans": [[n, t] for n, t in C.doc_spans(doc)],
            "gold": gold, "models": {},
        }
        for key in MODELS:
            row = rows[key][i]
            scored = flat(score_record({"model_response": row["generated"],
                                        "reference_answer": json.dumps(gold)}, i))
            for m in METRICS:  # the check: recomputed == stored
                stored = float(row[m])
                assert abs(scored[m] - stored) < 1e-3, (key, i, m, scored[m], stored)
            parsed, ok = parse_answer(row["generated"])
            detail["models"][key] = {"ok": ok, "answer": parsed or {}, "scores": scored,
                                     "raw": None if ok else row["generated"][:2000]}
            entry["scores"][key] = scored
            per_model[key].append(scored)
        (out_dir / "c" / f"{i}.json").write_text(json.dumps(detail, separators=(",", ":")))
        index.append(entry)

    summary = {k: {m: round(100 * statistics.mean(s[m] for s in per_model[k]), 2) for m in METRICS}
               for k in MODELS}
    meta = {
        "models": {k: name for k, (_, name) in MODELS.items()},
        "summary": summary,
        "checklist": {k: {"short": v["short_description"], "hypothesis": v["hypothesis"]}
                      for k, v in labels.items()},
        "contracts": index,
    }
    (out_dir / "index.json").write_text(json.dumps(meta, separators=(",", ":")))
    return summary


if __name__ == "__main__":
    s = build()
    for k, v in s.items():
        print(f"{k:9s} " + "  ".join(f"{m}={v[m]:.2f}" for m in METRICS))
