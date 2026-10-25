"""RAG Blueprint: interactive companion to Episodes 0 and 1 of the LinkedIn series.

Tab 1: see what six parser setups extract from the same PDF.
Tab 2: ask a document a question and watch retrieval rank the chunks.
"""
import os

# Before NumPy loads anywhere: one maths thread per process. By default OpenBLAS starts a thread (and a memory
# buffer) per CPU core in every process; with parallel parser workers that ran a 24-core test machine out of memory.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import html
import json
import re
from pathlib import Path

import gradio as gr
import pandas as pd

import rag_core as core

# On Hugging Face (ZeroGPU), embeddings run on a GPU: ZeroGPU only starts apps that use one. Same model
# (bge-small-en-v1.5) through sentence-transformers; elsewhere the app uses fastembed on CPU.
ON_SPACES = bool(os.environ.get("SPACE_ID"))   # set by Hugging Face in every Space
if ON_SPACES:
    import spaces
    from sentence_transformers import SentenceTransformer

    _st_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")   # ZeroGPU: place on cuda at startup

    def _gpu_seconds(texts):
        # The first call after the Space wakes up also starts the GPU; a 10 s budget was too short for that.
        # Quota is charged for the time actually used, not this budget.
        return min(90, 30 + len(texts) // 100)

    @spaces.GPU(duration=_gpu_seconds)
    def _embed_on_gpu(texts):
        return _st_model.encode(texts, normalize_embeddings=True, batch_size=64)

    core.set_embedder(_embed_on_gpu)

core.warm_up()   # parser libraries load once here, not in every worker on every upload

MAX_MB, MAX_PAGES, MAX_QUESTION = 10, 30, 500
UPLOAD = "Upload your own PDF"
UPLOADED_DOC = "The PDF you uploaded in tab 1"
SESSION_TTL = 3600   # seconds; Gradio also deletes session state when the tab is closed or refreshed

# Local testing only: set RAG_TEST_LOG=1 to record what the app shows, so results can be cross-checked.
# Off by default and never enabled on the hosted app. API keys are never recorded.
TEST_LOG = Path(__file__).parent / "test_log.jsonl" if os.environ.get("RAG_TEST_LOG") == "1" else None


def log_event(kind, **data):
    if TEST_LOG:
        from datetime import datetime
        with TEST_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "event": kind, **data},
                               ensure_ascii=False, default=str) + "\n")
DOCLING = json.loads((Path(__file__).parent / "docling_results.json").read_text(encoding="utf-8"))
RAG_PAPER = next(k for k in core.PAPERS if "RAG paper" in k)
PARSER_NAMES = list(core.PARSERS)
EXAMPLES = ["What does RAG use as its non-parametric memory?",
            "Which retriever does RAG use?",
            "How many documents are in the Wikipedia index?"]

BLUE, ORANGE = "#0B3A75", "#FF9F43"
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Space Grotesk"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(button_primary_background_fill=ORANGE, button_primary_text_color="#0B2A55")
CSS = f"""
#header {{ background: {BLUE}; color: #EAF4FF; padding: 22px 26px; border-radius: 10px;
  background-image: linear-gradient(rgba(255,255,255,.08) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.08) 1px, transparent 1px);
  background-size: 24px 24px; }}
#header h1 {{ color: #EAF4FF; margin: 0 0 6px; }}
#header p, #header a {{ color: #BFDBFF; margin: 0; }}
#header .accent {{ color: {ORANGE}; }}
.step {{ font-family: 'IBM Plex Mono', monospace; color: {ORANGE}; font-weight: 600; letter-spacing: 1px; }}
.bar {{ height: 10px; background: #E2E8F0; border-radius: 5px; overflow: hidden; margin: 6px 0 10px; }}
.bar > div {{ height: 100%; background: {ORANGE}; }}
.hit {{ border: 1px solid #CBD5E1; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; }}
.hit .meta {{ font-family: 'IBM Plex Mono', monospace; font-size: 14px; }}
"""


# Tab 1 --------------------------------------------------------------------------
def pdf_for(source, upload):
    if source != UPLOAD:
        return core.download(core.PAPERS[source]), False
    if upload is None:
        raise gr.Error("Choose a PDF to upload first.")
    if os.path.getsize(upload) > MAX_MB * 1024 * 1024:
        os.remove(upload)
        raise gr.Error(f"Please use a PDF under {MAX_MB} MB.")
    return upload, True


def parse(source, upload, prev_upload=None, progress=gr.Progress()):
    pdf, uploaded = pdf_for(source, upload)
    file_name = Path(pdf).name
    try:
        data = Path(pdf).read_bytes()
    finally:
        if uploaded and os.path.exists(pdf):   # the upload is deleted before any parser runs; parsers use memory
            os.remove(pdf)
    try:
        n_pages, no_text, image_only = core.inspect_pdf(data, MAX_PAGES)
    except core.BadPDF as e:
        log_event("tab1_rejected", document=file_name if uploaded else source, reason=str(e))
        raise gr.Error(str(e))
    results, rows, ok_rows, errors = {}, [], [], {}
    progress((0, len(PARSER_NAMES)), desc=f"Running {len(PARSER_NAMES)} parser setups in parallel")
    finished = lambda names, n, total: progress((n, total), desc=f"Finished {', '.join(names)} ({n} of {total})")
    for name, (pages, m, error) in core.run_all_safe(data, finished).items():
        if error:
            errors[name] = error
            rows.append({"parser": name, "status": "failed"})
        else:
            results[name] = pages
            ok_rows.append({"parser": name, **m})
            rows.append({"parser": name, "status": "ok", **m})
    if not results:
        raise gr.Error("None of the parsers could read this PDF. " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
    summary, pick = core.findings(ok_rows, results, errors, n_pages, no_text, image_only)
    first = "pypdf" if "pypdf" in results else next(iter(results))
    other = pick if pick in results and pick != first else next((n for n in results if n != first), first)
    page = gr.update(maximum=max(2, n_pages), value=1)   # a slider needs maximum > minimum
    # keep the uploaded PDF's extracted text (not the file) for tab 2, in this session only
    upload_text = {"name": file_name, "pages": results} if uploaded else prev_upload
    log_event("tab1_compare", document=file_name if uploaded else source, n_pages=n_pages, no_text_pages=no_text,
              image_only_pages=image_only,
              table=rows, errors=errors, findings=summary, default_pair=[first, other])
    choices = list(results)
    return (summary, pd.DataFrame(rows), results, gr.update(choices=choices, value=first),
            gr.update(choices=choices, value=other), page, show(results, first, 1), show(results, other, 1),
            upload_text)


def show(results, parser, page):
    if not results or parser not in results:
        return []
    pages = results[parser]
    if not pages:
        return [("(no text)", None)]
    text = pages[min(max(int(page), 1), len(pages)) - 1] or "(no text on this page)"
    log_event("tab1_view_page", parser=parser, page=int(page), text=text)
    return core.highlight(text)


def toggle_upload(source):
    return gr.update(visible=source == UPLOAD)


def clear_tab1():
    return "", None, {}, [], []


# Tab 2 --------------------------------------------------------------------------
_cache = {}


def index(doc, parser, chunk_words, overlap, upload_text):
    if doc == UPLOADED_DOC:                  # visitor's PDF: built per request, never cached across sessions
        if not upload_text:
            raise gr.Error("Upload a PDF in tab 1 and click Compare parsers first.")
        if parser not in upload_text["pages"]:
            raise gr.Error(f"{parser} failed on your PDF in tab 1. Choose another parser in Settings.")
        chunks = core.chunk(upload_text["pages"][parser], chunk_words, overlap)
        if not chunks:
            raise gr.Error("This parser extracted no usable text from your PDF. Try another parser in Settings.")
        return chunks, core.embed([c["text"] for c in chunks])
    key = (doc, parser, chunk_words, overlap)
    if key not in _cache:
        pages, _ = core.run_parser(parser, core.download(core.PAPERS[doc]))
        chunks = core.chunk(pages, chunk_words, overlap)
        _cache[key] = (chunks, core.embed([c["text"] for c in chunks]))
    return _cache[key]


def visible(text):
    """Show broken characters as � instead of passing invisible control codes to the page."""
    return "".join("�" if core.is_broken_char(ch) else ch for ch in text)


def hits_html(hits, n_chunks):
    hosted = (" Here the model runs on a GPU in full precision, so scores can differ from the slides by about 0.001; "
              "the rankings are the same (checked)." if ON_SPACES else "")
    cards = [f"<p class='meta'>Searched {n_chunks} chunks. Score = cosine similarity between the question and the chunk "
             "(1.0 would be identical meaning). Retrieval always returns the closest chunks, even when none of "
             f"them answers the question.{hosted}</p>"]
    for h in hits:
        cards.append(
            f"<div class='hit'><div class='meta'><b>#{h['rank']}</b> · score {h['score']:.3f} · page {h['page']}</div>"
            f"<div class='bar'><div style='width:{max(0.0, h['score']) * 100:.1f}%'></div></div>"
            f"<div>{html.escape(visible(h['text'][:300]))}…</div></div>")
    return "".join(cards)


def trace(question, doc, upload_text, parser, chunk_words, overlap, top_k):
    if not question.strip():
        raise gr.Error("Type a question first, or click one of the examples.")
    if len(question) > MAX_QUESTION:
        raise gr.Error(f"Please keep the question under {MAX_QUESTION} characters.")
    if overlap >= chunk_words:
        raise gr.Error("Overlap must be smaller than the chunk size.")
    try:
        chunks, vectors = index(doc, parser, int(chunk_words), int(overlap), upload_text)
        hits = core.retrieve(vectors, chunks, question, int(top_k))
    except gr.Error:
        raise
    except Exception as e:
        if not ON_SPACES:
            raise
        log_event("tab2_gpu_error", error=f"{type(e).__name__}: {str(e)[:200]}")
        print(f"tab 2 embedding step failed: {type(e).__name__}: {str(e)[:200]}", flush=True)   # no user content
        raise gr.Error("The free GPU that computes the embeddings didn't respond this time: it may be starting up, "
                       "busy, or your daily free GPU time on Hugging Face may be used up. Please try again in a "
                       "minute; logging in to Hugging Face gives you more GPU time.")
    prompt = core.build_prompt(question, hits)
    log_event("tab2_retrieve", document=upload_text["name"] if doc == UPLOADED_DOC and upload_text else doc,
              parser=parser, chunk_words=int(chunk_words), overlap=int(overlap), top_k=int(top_k),
              question=question, n_chunks=len(chunks), hits=hits, prompt=prompt)
    return hits_html(hits, len(chunks)), visible(prompt)


def answer(prompt, provider, llm_model, api_key):
    if not prompt:
        raise gr.Error("Find the best chunks first.")
    if not api_key.strip():
        raise gr.Error(f"Paste your own {provider} key to generate an answer (used for this request only, never stored).")
    try:
        result = core.generate(prompt, provider, llm_model, api_key.strip())
        log_event("tab2_answer", provider=provider, model=llm_model, prompt=prompt, answer=result)   # no key
        return result
    except Exception as e:          # wrong key, rate limit, model unavailable: show the provider's message
        msg = str(e).replace(api_key.strip(), "[key]")
        msg = re.sub(r"\b(sk|gsk|AIza)[A-Za-z0-9_\-*.]{6,}", "[key]", msg)   # some errors echo part of the key
        log_event("tab2_answer_error", provider=provider, model=llm_model, error=f"{type(e).__name__}: {msg[:300]}")
        raise gr.Error(f"{provider} returned an error: {type(e).__name__}: {msg[:300]}")


def provider_changed(provider):
    p = core.PROVIDERS[provider]
    return gr.update(choices=p["models"], value=p["models"][0]), p["note"]


# Layout -------------------------------------------------------------------------
with gr.Blocks(title="RAG Blueprint") as demo:
    gr.HTML("<div id='header'><h1>RAG <span class='accent'>Blueprint</span></h1>"
            "<p>What happens to a document before an LLM can answer from it? Try each step yourself. "
            "Companion to my LinkedIn series · code on "
            "<a href='https://github.com/RohitRao7722/rag-blueprint'>GitHub</a></p></div>")

    with gr.Tab("1 · See what parsers do to a PDF"):
        gr.Markdown("A PDF stores glyphs at positions on a page, not words and paragraphs. A **parser** rebuilds the "
                    "text, and each one guesses differently. Here six parser setups read the same PDF.")
        gr.HTML("<span class='step'>STEP 1 · PICK A PDF</span>")
        source = gr.Radio([*core.PAPERS, UPLOAD], value=RAG_PAPER, show_label=False)
        upload = gr.File(file_types=[".pdf"], type="filepath", visible=False,
                         label=f"Your PDF (up to {MAX_MB} MB / {MAX_PAGES} pages; text PDFs only, scans need OCR). "
                               "The file is deleted right after parsing. Its extracted text stays in memory only "
                               "while this page is open (at most an hour), so you can ask it questions in tab 2.")
        compare_btn = gr.Button("Compare parsers", variant="primary")
        gr.HTML("<span class='step'>WHAT THE PARSERS DID</span>")
        summary = gr.Markdown()
        gr.HTML("<span class='step'>STEP 2 · COMPARE TWO PARSERS ON ONE PAGE</span>")
        with gr.Row():
            parser_a = gr.Dropdown(PARSER_NAMES, value="pypdf", label="Parser A")
            parser_b = gr.Dropdown(PARSER_NAMES, value="pdfplumber (default)", label="Parser B")
            page = gr.Slider(1, 2, value=1, step=1, label="Page")
        with gr.Row():
            text_a = gr.HighlightedText(label="Parser A", combine_adjacent=True, show_legend=True)
            text_b = gr.HighlightedText(label="Parser B", combine_adjacent=True, show_legend=True)
        with gr.Accordion("All the numbers", open=False):
            table = gr.Dataframe(interactive=False)
            gr.Markdown("**Line-break hyphens** counts a hyphen followed by a line break, as in “knowl- edge”; "
                        "a few may be real hyphenated words that happen to wrap. **Spaces before hyphens** counts "
                        "cases like “well -defined”. **Broken characters** are invisible control codes "
                        "where a letter should be. **Look-alike characters** are ligatures (“ﬁ”), Arabic "
                        "presentation forms and Chinese radicals that look like normal letters but aren't. "
                        "**pymupdf4llm** outputs Markdown, so its word count includes formatting symbols such as # "
                        "and **.")
        with gr.Accordion("What about a layout-aware parser (Docling)?", open=False):
            gr.Markdown("Docling uses vision models to find paragraphs, tables and reading order before extracting "
                        "text. It's too slow to run live here (about 2 s per page on my laptop CPU), so these are "
                        "its measured results on the two papers:")
            gr.Dataframe(pd.DataFrame([{"paper": p, **r["metrics"]} for p, r in DOCLING.items()]), interactive=False)
        state = gr.State({}, time_to_live=SESSION_TTL)
        upload_text = gr.State(None, time_to_live=SESSION_TTL)   # uploaded PDF's text, for tab 2

        source.change(toggle_upload, source, upload)
        outputs = [summary, table, state, parser_a, parser_b, page, text_a, text_b, upload_text]
        # clear first, so a failed parse never leaves the previous PDF's results on screen
        # progress is shown only in the findings box, so the controls and text panels stay readable
        compare_btn.click(clear_tab1, None, [summary, table, state, text_a, text_b], show_progress="hidden").then(
            parse, [source, upload, upload_text], outputs, show_progress_on=[summary])
        parser_a.change(show, [state, parser_a, page], text_a, show_progress="hidden")
        parser_b.change(show, [state, parser_b, page], text_b, show_progress="hidden")
        page.change(show, [state, parser_a, page], text_a, show_progress="hidden").then(
            show, [state, parser_b, page], text_b, show_progress="hidden")

    with gr.Tab("2 · Ask a document a question"):
        gr.Markdown("Retrieval finds the chunks of a document closest in meaning to your question. The document is "
                    "split into chunks and embedded with bge-small-en-v1.5. Watch which chunks come first, and "
                    "whether they actually answer the question.\n\n"
                    "<small>**English only:** bge-small-en-v1.5 is an English embedding model. Questions about "
                    "documents in other languages (Hindi, Arabic, Chinese…) won't retrieve well, even when the "
                    "answer is in the document. Tab 1 still works for any language.</small>")
        gr.HTML("<span class='step'>STEP 1 · PICK A DOCUMENT AND ASK</span>")
        doc = gr.Radio([*core.PAPERS, UPLOADED_DOC], value=RAG_PAPER, label="Document")
        question = gr.Textbox(EXAMPLES[0], label="Your question")
        gr.Examples([[q] for q in EXAMPLES], inputs=question, label="Example questions for the RAG paper")
        with gr.Accordion("Settings (optional)", open=False):
            with gr.Row():
                parser = gr.Dropdown(PARSER_NAMES, value="pypdf", label="Parser")
                chunk_words = gr.Slider(80, 400, value=200, step=20, label="Chunk size (words)")
                overlap = gr.Slider(0, 100, value=40, step=10, label="Overlap (words)")
                top_k = gr.Slider(1, 10, value=5, step=1, label="Chunks to show")
        find_btn = gr.Button("Find the best chunks", variant="primary")
        hits = gr.HTML()
        with gr.Accordion("See the exact prompt an LLM would get", open=False):
            prompt = gr.Textbox(show_label=False, lines=10)

        gr.HTML("<span class='step'>STEP 2 (OPTIONAL) · GENERATE AN ANSWER WITH YOUR OWN KEY</span>")
        first = next(iter(core.PROVIDERS))
        with gr.Row():
            provider = gr.Radio(list(core.PROVIDERS), value=first, label="Provider")
            llm_model = gr.Dropdown(core.PROVIDERS[first]["models"], value=core.PROVIDERS[first]["models"][0],
                                    label="Model")
        provider_note = gr.Markdown(core.PROVIDERS[first]["note"])
        key = gr.Textbox(type="password", label="API key: used for this one request, never stored or logged")
        gen_btn = gr.Button("Generate answer")
        out = gr.Textbox(label="Answer", lines=4)

        find_btn.click(trace, [question, doc, upload_text, parser, chunk_words, overlap, top_k], [hits, prompt],
                       show_progress_on=[hits])
        provider.change(provider_changed, provider, [llm_model, provider_note])
        gen_btn.click(answer, [prompt, provider, llm_model, key], out)

    gr.Markdown("<small>One question, one embedding model: these demos show how each step behaves. "
                "They are not benchmarks.</small>")

    # show results straight away, so the first screen already teaches something
    demo.load(parse, [source, upload, upload_text], outputs, show_progress_on=[summary])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)                  # one heavy PDF can't stall everyone else
    demo.launch(theme=THEME, css=CSS, max_file_size=f"{MAX_MB}mb")
