"""Episode 1: the same PDFs through several parsers, measured.

Corpora (downloaded at run time, not committed):
  lewis   - Lewis et al. 2020, arXiv 2005.11401 (single column; ligatures, hyphens, Table 1)
  barnett - Barnett et al. 2024, arXiv 2401.05856 (two columns; reading order)

Then Episode 0's pipeline (chunk/embed/retrieve, unchanged) runs on each parser's
Lewis text, so the only thing that varies is the parser.

Usage:  python compare_parsers.py            (all parsers)
        python compare_parsers.py --no-docling
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
sys.path.insert(0, str(HERE.parent / "ep00-rag-map"))
import run as ep0  # noqa: E402  (Episode 0's chunk/embed/retrieve)

PAPERS = {
    "lewis": "https://arxiv.org/pdf/2005.11401",
    "barnett": "https://arxiv.org/pdf/2401.05856",
}
LIGATURES = "ﬀﬁﬂﬃﬄﬅﬆ"   # ﬀ ﬁ ﬂ ﬃ ﬄ ﬅ ﬆ
QUESTIONS = {   # question -> phrase that marks a chunk containing the answer
    "What does RAG use as its non-parametric memory?": "dense vector index of Wikipedia",
    "Which retriever does RAG use?": "Dense Passage Retriever",
    "How many documents are in the Wikipedia index?": "21M documents",
}


def download(name):
    DATA.mkdir(exist_ok=True)
    pdf = DATA / f"{name}.pdf"
    if not pdf.exists():
        req = urllib.request.Request(PAPERS[name], headers={"User-Agent": "rag-blueprint-demo"})
        try:
            pdf.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        except OSError as e:
            sys.exit(f"Could not download {PAPERS[name]}: {e}")
    return pdf


# Parsers: each returns one string per page --------------------------------------
def parse_pypdf(pdf):
    from pypdf import PdfReader
    return [p.extract_text() or "" for p in PdfReader(pdf).pages]


def parse_pymupdf_default(pdf):
    import pymupdf
    with pymupdf.open(pdf) as doc:
        return [page.get_text() for page in doc]


def parse_pymupdf_ligatures_expanded(pdf):
    """Same library, one flag changed: ligatures expanded into their letters."""
    import pymupdf
    flags = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES
    with pymupdf.open(pdf) as doc:
        return [page.get_text(flags=flags) for page in doc]


def parse_pdfplumber(pdf, **kwargs):
    import pdfplumber
    with pdfplumber.open(pdf) as doc:
        return [page.extract_text(**kwargs) or "" for page in doc.pages]


_converter = None


def parse_docling(pdf):
    global _converter
    from docling.document_converter import DocumentConverter
    if _converter is None:            # model loading happens once, outside the timed runs
        _converter = DocumentConverter()
    doc = _converter.convert(str(pdf)).document
    return [doc.export_to_markdown(page_no=n) for n in range(1, doc.num_pages() + 1)]


PARSERS = {
    "pypdf": parse_pypdf,
    "pymupdf (default)": parse_pymupdf_default,
    "pymupdf (ligatures expanded)": parse_pymupdf_ligatures_expanded,
    "pdfplumber (default)": parse_pdfplumber,
    "pdfplumber (x_tolerance=1.5)": lambda pdf: parse_pdfplumber(pdf, x_tolerance=1.5),
    "docling": parse_docling,
}


# Measurements --------------------------------------------------------------------
def norm(text):
    return re.sub(r"\s+", " ", text)


def measure(name, pages):
    text = "\n".join(pages)
    flat = norm(text)
    m = {
        "words": len(text.split()),
        "ligature_chars": sum(text.count(c) for c in LIGATURES),
        # a lowercase letter, a hyphen, whitespace, a lowercase letter: a word split at a line end
        "line_break_hyphens": len(re.findall(r"[a-z]-\s+[a-z]", text)),
        # glyphs the parser could not map to a character, written as "(cid:NN)"
        "unmapped_glyphs": len(re.findall(r"\(cid:\d+\)", text)),
    }
    if name == "lewis":
        m["arxiv_stamp_in_text"] = "arXiv:2005.11401" in flat
        m["table1_rag_seq_row_intact"] = bool(
            re.search(r"RAG-Seq\.?\s*\|?\s*44\.5\s*\|?\s*56\.8\s*/\s*68\.0\s*\|?\s*45\.2\s*\|?\s*52\.2", flat))
    else:
        m["arxiv_stamp_in_text"] = "arXiv:2401.05856" in flat
        # each sentence spans a line break inside one column; interleaved columns would break it
        m["left_column_sentence_intact"] = (
            "case studies involving the implementation of a RAG system. This presents the challenges" in flat)
        m["right_column_sentence_intact"] = (
            "the retrieval system works using embeddings that provide a compressed semantic" in flat)
    return m


def retrieval(model, pages):
    """Episode 0's pipeline on this parser's pages; ranks of chunks containing each answer."""
    chunks = ep0.chunk([{"page": i + 1, "text": t} for i, t in enumerate(pages)])
    vectors = ep0.embed_and_store(model, chunks)
    result = {"chunks": len(chunks)}
    for question, phrase in QUESTIONS.items():
        q = np.array(list(model.query_embed([question])))[0]
        q /= np.linalg.norm(q)
        order = list(np.argsort(-(vectors @ q)))
        ranks = sorted(order.index(c["id"]) + 1 for c in chunks if phrase in norm(c["text"]))
        result[question] = ranks
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parsers = dict(PARSERS)
    if "--no-docling" in sys.argv:
        parsers.pop("docling")
    OUT.mkdir(exist_ok=True)
    from fastembed import TextEmbedding
    model = TextEmbedding(ep0.MODEL)

    pdfs = {paper: download(paper) for paper in PAPERS}
    for parse in parsers.values():    # warm-up: imports and model loading are not parsing time
        parse(pdfs["barnett"])

    report = {}
    for paper, pdf in pdfs.items():
        for name, parse in parsers.items():
            start = time.perf_counter()
            pages = parse(pdf)
            seconds = round(time.perf_counter() - start, 2)
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            (OUT / f"{paper}__{slug}.txt").write_text("\n\n=== PAGE BREAK ===\n\n".join(pages), encoding="utf-8")
            entry = {"seconds": seconds, "pages": len(pages), **measure(paper, pages)}
            if paper == "lewis":
                entry["retrieval"] = retrieval(model, pages)
            report.setdefault(paper, {})[name] = entry
            print(f"{paper:8} {name:30} {seconds:>7}s  " + json.dumps({k: v for k, v in entry.items() if k != 'retrieval'}))

    (OUT / "parsers_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nRetrieval on each parser's Lewis text (ranks of chunks containing the answer):")
    for name, entry in report["lewis"].items():
        r = entry["retrieval"]
        print(f"  {name:30} chunks={r['chunks']:<3} " + "  ".join(f"Q{i+1}={r[q]}" for i, q in enumerate(QUESTIONS)))


if __name__ == "__main__":
    main()
