"""RAG map, Episode 0: one question traced through a minimal RAG pipeline.

No framework. Each step writes its output to out/ so you can inspect it.
Corpus: Lewis et al. 2020, "Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks" (arXiv 2005.11401). The PDF is downloaded at run time and not committed.

Usage:  python run.py "your question"
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader

PAPER_URL = "https://arxiv.org/pdf/2005.11401"
MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_WORDS = 200      # words per chunk
OVERLAP_WORDS = 40     # words shared between neighbouring chunks
TOP_K = 5
LLM_MODEL = "gpt-6-luna"   # override with the OPENAI_MODEL env var

# Optional providers (set LLM_PROVIDER=groq or google). Both accept the OpenAI client
# with a different base URL. The slides' numbers were produced with the default: OpenAI.
PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key": "GROQ_API_KEY",
             "model": "openai/gpt-oss-20b"},
    "google": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key": "GEMINI_API_KEY",
               "model": "gemini-3.8-flash"},
}

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"


def load_env():
    """Read KEY=value lines from .env. Values here override the shell environment."""
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#") and value.strip():
                os.environ[key.strip()] = value.strip().strip('"')


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# 1. Load ---------------------------------------------------------------------
def load_pdf():
    DATA.mkdir(exist_ok=True)
    pdf = DATA / "lewis2020.pdf"
    if not pdf.exists():
        req = urllib.request.Request(PAPER_URL, headers={"User-Agent": "rag-blueprint-demo"})
        try:
            pdf.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        except OSError as e:
            sys.exit(f"Could not download the paper from {PAPER_URL}: {e}")
    return pdf


# 2. Parse: one text string per page ----------------------------------------
def parse(pdf):
    pages = [p.extract_text() or "" for p in PdfReader(pdf).pages]
    return [{"page": i + 1, "text": t} for i, t in enumerate(pages)]


# 3. Chunk: fixed-size word windows with overlap, within each page ----------
def chunk(pages):
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for p in pages:
        words = p["text"].split()
        for start in range(0, len(words), step):
            piece = words[start:start + CHUNK_WORDS]
            if len(piece) < 20:   # skip tiny tails
                continue
            chunks.append({"id": len(chunks), "page": p["page"], "text": " ".join(piece)})
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks


# 4. Embed + 5. Store: a NumPy matrix, one normalised row per chunk ---------
def embed_and_store(model, chunks):
    vectors = np.array(list(model.embed([c["text"] for c in chunks])))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


# 6. Retrieve: cosine similarity, top-k -------------------------------------
def retrieve(model, vectors, chunks, question):
    q = np.array(list(model.query_embed([question])))[0]
    q /= np.linalg.norm(q)
    scores = vectors @ q
    top = np.argsort(-scores)[:TOP_K]
    return q, [{"rank": r + 1, "score": round(float(scores[i]), 4), **chunks[i]}
               for r, i in enumerate(top)]


# 7. Build prompt ------------------------------------------------------------
def build_prompt(question, hits):
    context = "\n\n".join(f"[{h['rank']}] (page {h['page']}) {h['text']}" for h in hits)
    return (
        "Answer the question using only the context below. "
        "Cite the context numbers you used. If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


# 8. Generate: the LLM answers from the prompt -------------------------------
def generate(prompt):
    from openai import OpenAI
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nOPENAI_API_KEY not set: skipping generation (prompt saved in out/5_prompt.txt).")
            return None
        model = os.environ.get("OPENAI_MODEL", LLM_MODEL)
        resp = OpenAI().responses.create(model=model, input=prompt)
        return {"provider": provider, "model": model, "answer": resp.output_text}
    if provider not in PROVIDERS:
        sys.exit(f"LLM_PROVIDER must be openai, groq or google (got {provider!r}).")
    p = PROVIDERS[provider]
    if not os.environ.get(p["key"]):
        print(f"\n{p['key']} not set: skipping generation (prompt saved in out/5_prompt.txt).")
        return None
    model = os.environ.get("LLM_MODEL", p["model"])
    client = OpenAI(api_key=os.environ[p["key"]], base_url=p["base_url"])
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return {"provider": provider, "model": model, "answer": resp.choices[0].message.content}


def count_tokens(text):
    """Token count with OpenAI's o200k_base tokenizer (other models tokenize differently)."""
    import tiktoken
    return len(tiktoken.get_encoding("o200k_base").encode(text))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_env()
    question = sys.argv[1] if len(sys.argv) > 1 else "What does RAG use as its non-parametric memory?"
    OUT.mkdir(exist_ok=True)

    pages = parse(load_pdf())
    save("1_pages.json", pages)

    chunks = chunk(pages)
    save("2_chunks.json", chunks)

    model = TextEmbedding(MODEL)
    vectors = embed_and_store(model, chunks)
    save("3_vectors_sample.json", {
        "model": MODEL, "dimensions": int(vectors.shape[1]), "num_chunks": int(vectors.shape[0]),
        "chunk_0_first_8_values": [round(float(x), 4) for x in vectors[0][:8]],
    })

    q_vec, hits = retrieve(model, vectors, chunks, question)
    save("4_retrieval.json", {"question": question,
                              "query_first_8_values": [round(float(x), 4) for x in q_vec[:8]],
                              "hits": hits})

    prompt = build_prompt(question, hits)
    (OUT / "5_prompt.txt").write_text(prompt, encoding="utf-8")

    answer = generate(prompt)
    if answer:
        save("6_answer.json", {"question": question, **answer})

    full_text = "\n".join(p["text"] for p in pages)
    stats = {"pages": len(pages), "words": len(full_text.split()),
             "tokens_o200k": count_tokens(full_text), "chunks": len(chunks),
             "chunk_words": CHUNK_WORDS, "overlap_words": OVERLAP_WORDS, "top_k": TOP_K}
    save("0_stats.json", stats)

    print(json.dumps(stats, indent=2))
    print(f"\nQ: {question}")
    for h in hits:
        print(f"#{h['rank']}  {h['score']:.3f}  p{h['page']}  {h['text'][:110]}...")
    if answer:
        print(f"\nA ({answer['model']}): {answer['answer']}")


if __name__ == "__main__":
    main()
