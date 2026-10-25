# Episode 1: Parsing

A PDF stores glyphs at positions on a page, not words and paragraphs. A parser rebuilds the text, and each one guesses differently. This episode measures what different parsers do to the same documents, and whether it changes retrieval.

**Try it yourself:** [huggingface.co/spaces/Rohitrao7722/rag-blueprint](https://huggingface.co/spaces/Rohitrao7722/rag-blueprint). Upload a PDF and compare six parser setups side by side, then ask it a question. The code is in [`app/`](app/).

## Run the experiment

```bash
pip install -r requirements.txt
python compare_parsers.py              # all parsers, including Docling
python compare_parsers.py --no-docling # skip Docling's large install
python ligature_ablation.py            # the controlled ligature test (run compare_parsers.py first)
```

It downloads two papers from arXiv at run time and reuses Episode 0's chunk/embed/retrieve code unchanged (`../ep00-rag-map/run.py`), so the only thing that varies is the parser. Outputs go to `out/`.

## What it measured

Two papers: Lewis et al. 2020 ([arXiv 2005.11401](https://arxiv.org/abs/2005.11401), one column) and Barnett et al. 2024 ([arXiv 2401.05856](https://arxiv.org/abs/2401.05856), two columns, CC BY 4.0).

| Parser setup | Lewis: words | Ligatures | Line-break hyphens | Barnett: column order |
|---|---|---|---|---|
| pypdf | 9,871 | 114 | 47 | correct |
| PyMuPDF (default) | 9,886 | 114 | 47 | correct |
| PyMuPDF (ligatures expanded) | 9,886 | 0 | 47 | correct |
| pdfplumber (default) | 3,211 | 0 | 46 | interleaved |
| pdfplumber (`x_tolerance=1.5`) | 10,046 | 0 | 46 | interleaved |
| Docling | 10,061 | 0 | 0 | correct |

- **One setting can change the result as much as switching tools.** pdfplumber's default spacing lost the spaces on page 1 of Lewis (81 words); `x_tolerance=1.5` gave 401. In the stress test below, the default also lost words on 6 other PDFs. PyMuPDF keeps ligatures by default (its documented behaviour); clearing one flag removes them.
- **Reading order is a guess.** On Barnett page 2, pdfplumber reads straight across both columns: "case studies involving the implementation of a RAG system. shown in Figure 1 as two separate processes…". pypdf and PyMuPDF read the left column first.
- **The ligature-only test** (`ligature_ablation.py`): pypdf's text with only "ﬁ"→"fi" and similar replaced, so every chunk keeps the same boundaries. For three questions, one answer chunk moved up (rank 2 → 1) and one moved down (3 → 5; that question's best chunk stayed at rank 2). Clean text is still right (keyword search and readers need it), but it doesn't automatically improve dense retrieval. Measure it.

## Stress test of the app (31 files)

Run before publishing, on Windows and in a Linux container: real documents (an IRS form and publication, Wikipedia in Hindi, Arabic, Chinese and Japanese, arXiv papers, lecture slides, a Gazette of India notice) and generated hostile files (scans, password-protected, truncated, empty, renamed images and HTML, over-size, prompt injection, embedded JavaScript, rotated pages). Result: 0 crashes, 12 files rejected with a clear message, 19 parsed with findings checked against the actual text.

Found along the way:

| Document | What happened |
|---|---|
| Hindi Wikipedia | pypdf and pdfplumber turned the vowel sign ि into an invisible control character: "आधिकारिक" became "आ�धका�रक" (3,840 times). PyMuPDF kept the letters but wrote vowel signs twice ("बड़ाा", 156 times); the app now flags that as "repeated marks" |
| Arabic Wikipedia | pdfplumber wrote words backwards; pypdf and pdfplumber kept 26,959 Arabic presentation-form characters, which look right but don't match a search for the normal letters |
| Chinese Wikipedia | pypdf and pdfplumber returned Kangxi radicals such as ⽣ instead of 生 (about 320) |
| Gazette of India notice | every parser read अधिसूचना as अजधसूचना: the PDF's own font data is wrong, so comparing parsers can't reveal it |
| Rotated pages | pdfplumber and PyMuPDF4LLM lost text |

## Ground-truth test: 17 languages, 4 PDF makers, 85 PDFs (`gt/`)

The opening paragraph of Wikipedia's "India" article in 17 languages, turned into PDFs by Chrome, Typst, PyMuPDF and Microsoft Word (85 files), then read back by five parser setups and compared with the original text (share of 20 random 12-character snippets still findable). Headline: for Latin, Cyrillic and CJK scripts every parser is fine; for Thai, Arabic and Indic scripts pypdf and pdfplumber lose most of the text while PyMuPDF keeps 0.75–1.00; and on Word- and MuPDF-made files in the nine Indian languages every parser returns nearly the same wrong text (letter agreement between parsers 0.87–1.00), because the file's own character map is wrong. Details, numbers and caveats in [`gt/README.md`](gt/README.md).

## Caveats

- Two papers and three questions for the retrieval tests; one embedding model (bge-small-en-v1.5, English only). It shows how the steps behave; it is not a benchmark.
- Timings depend on the machine. On a laptop CPU, Docling took about 2 s per page.
- pdfplumber's table extraction (`extract_tables`) was not tested; only its text extraction.
- PyMuPDF's `TEXT_DEHYPHENATE` flag had no visible effect in my runs, so I make no claim about it.

## The app

`app/` is the Gradio app deployed on Hugging Face. To run it locally: `pip install -r app/requirements-local.txt` then `python app/app.py` (`app/requirements.txt` is the hosted version's, with PyTorch for the GPU). On the hosted version, embeddings run on ZeroGPU through sentence-transformers; locally they run on CPU through fastembed. Same model either way: identical rankings and scores within 0.0005 in my check. Uploaded files are deleted before parsing; their extracted text stays in memory only while the page is open.
