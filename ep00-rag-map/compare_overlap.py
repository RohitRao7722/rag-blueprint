"""Episode 0, slide 5: the same question with and without chunk overlap.

Reuses the parse/chunk/embed/retrieve steps from run.py and reports, for each
setting, where the chunks that contain the answer phrase end up in the ranking.
Run run.py first so the paper is already downloaded.
"""
import json
import sys

import numpy as np
from fastembed import TextEmbedding

import run

ANSWER_PHRASE = "dense vector index of Wikipedia"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    question = sys.argv[1] if len(sys.argv) > 1 else "What does RAG use as its non-parametric memory?"
    pages = run.parse(run.load_pdf())
    model = TextEmbedding(run.MODEL)
    results = {}
    for overlap in (0, run.OVERLAP_WORDS):
        run.OVERLAP_WORDS = overlap          # chunk() reads the module constant
        chunks = run.chunk(pages)
        vectors = run.embed_and_store(model, chunks)
        q = np.array(list(model.query_embed([question])))[0]
        q /= np.linalg.norm(q)
        order = list(np.argsort(-(vectors @ q)))
        answer_ranks = sorted(order.index(c["id"]) + 1 for c in chunks if ANSWER_PHRASE in c["text"])
        results[f"overlap_{overlap}"] = {"chunks": len(chunks), "answer_chunk_ranks": answer_ranks}
        print(f"overlap {overlap:>2}: {len(chunks)} chunks; chunks containing the answer rank {answer_ranks}")
    run.OUT.mkdir(exist_ok=True)
    run.save("8_overlap_comparison.json", {"question": question, "answer_phrase": ANSWER_PHRASE, **results})


if __name__ == "__main__":
    main()
