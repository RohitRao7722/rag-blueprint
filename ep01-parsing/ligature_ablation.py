"""Episode 1: does the ligature alone change retrieval?

Takes pypdf's text and replaces only the ligature characters with their letters
("ﬁ" -> "fi"). No whitespace changes, so every chunk keeps the same boundaries;
the ligature is the only variable. Run compare_parsers.py first (it writes the texts).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE.parent / "ep00-rag-map"))
import run as ep0  # noqa: E402

from compare_parsers import QUESTIONS, norm  # noqa: E402

EXPAND = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}


def ranks(model, pages):
    chunks = ep0.chunk([{"page": i + 1, "text": t} for i, t in enumerate(pages)])
    vectors = ep0.embed_and_store(model, chunks)
    out = {}
    for question, phrase in QUESTIONS.items():
        q = np.array(list(model.query_embed([question])))[0]
        q /= np.linalg.norm(q)
        scores = vectors @ q
        order = list(np.argsort(-scores))
        out[question] = [(order.index(c["id"]) + 1, round(float(scores[c["id"]]), 4))
                         for c in chunks if phrase in norm(c["text"])]
    return chunks, out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    raw = (OUT / "lewis__pypdf.txt").read_text(encoding="utf-8").split("\n\n=== PAGE BREAK ===\n\n")
    fixed = [re.sub("[" + "".join(EXPAND) + "]", lambda m: EXPAND[m.group(0)], p) for p in raw]

    from fastembed import TextEmbedding
    model = TextEmbedding(ep0.MODEL)
    chunks_a, a = ranks(model, raw)
    chunks_b, b = ranks(model, fixed)
    assert [len(c["text"].split()) for c in chunks_a] == [len(c["text"].split()) for c in chunks_b], "boundaries moved"

    changed = sum(ca["text"] != cb["text"] for ca, cb in zip(chunks_a, chunks_b))
    print(f"{len(chunks_a)} chunks, identical boundaries; {changed} chunks contain at least one ligature")
    for question in QUESTIONS:
        print(f"\nQ: {question}\n  with ligatures   (rank, score): {a[question]}\n  ligatures fixed  (rank, score): {b[question]}")
    OUT.mkdir(exist_ok=True)
    (OUT / "ligature_ablation.json").write_text(json.dumps(
        {"chunks": len(chunks_a), "chunks_with_ligatures": changed, "with_ligatures": a, "ligatures_fixed": b},
        indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
