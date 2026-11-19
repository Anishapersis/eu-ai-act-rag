"""Evaluate the v2 assistant: retrieval hit rate, answer keywords and refusals.

Run locally (downloads the embedding model and TinyLlama on first use):
    python eval/evaluate.py            # full run, writes eval/results.json
    python eval/evaluate.py --retrieval-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rag  # noqa: E402

HERE = Path(__file__).resolve().parent


def evaluate(assistant: rag.AIActAssistant, questions: dict, retrieval_only: bool = False) -> dict:
    rows = []
    for item in questions["in_scope"]:
        hits = assistant.retrieve(item["q"])
        pages = [d.metadata["page"] for d, _ in hits]
        row = {"q": item["q"], "retrieved_pages": pages,
               "hit": any(p in item["pages"] for p in pages),
               "best_score": round(max((s for _, s in hits), default=0.0), 3)}
        if not retrieval_only:
            t0 = time.time()
            ans = assistant.ask(item["q"])
            row |= {"answer": ans.text, "refused": ans.refused, "seconds": round(time.time() - t0, 1),
                    "keywords_ok": all(k.lower() in ans.text.lower() for k in item["keywords"])}
        rows.append(row)

    oos = []
    for q in questions["out_of_scope"]:
        if retrieval_only:
            best = max((s for _, s in assistant.retrieve(q)), default=0.0)
            oos.append({"q": q, "best_score": round(best, 3), "refused": best < assistant.min_score})
        else:
            ans = assistant.ask(q)
            oos.append({"q": q, "best_score": round(ans.best_score, 3), "refused": ans.refused, "answer": ans.text})

    n = len(rows)
    summary = {"retrieval_hit_at_k": round(sum(r["hit"] for r in rows) / n, 2),
               "k": assistant.k, "min_score": assistant.min_score,
               "out_of_scope_refusal_rate": round(sum(o["refused"] for o in oos) / len(oos), 2)}
    if not retrieval_only:
        summary["answer_keyword_accuracy"] = round(sum(r["keywords_ok"] for r in rows) / n, 2)
        summary["in_scope_wrongly_refused"] = sum(r["refused"] for r in rows)
    return {"summary": summary, "in_scope": rows, "out_of_scope": oos}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()
    from langchain_huggingface import HuggingFaceEmbeddings

    store = rag.get_vectorstore(HuggingFaceEmbeddings(model_name=rag.EMBED_MODEL))
    gen = (lambda s, u: "") if args.retrieval_only else rag.tinyllama_generator()
    report = evaluate(rag.AIActAssistant(store, gen), json.loads((HERE / "questions.json").read_text()),
                      args.retrieval_only)
    (HERE / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
