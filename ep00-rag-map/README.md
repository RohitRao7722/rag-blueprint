# Episode 0: The RAG map

One question traced through a basic RAG pipeline, with the output of every step saved to disk. The corpus is the paper that introduced the name "retrieval-augmented generation": Lewis et al., 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* ([arXiv 2005.11401](https://arxiv.org/abs/2005.11401)).

About 150 lines of plain Python. No framework, so each step is visible:

| Step | How | Where |
|---|---|---|
| Parse | `pypdf` text extraction, one string per page | `run.py: parse()` |
| Chunk | 200-word windows, 40-word overlap, within each page | `run.py: chunk()` |
| Embed | `BAAI/bge-small-en-v1.5` via `fastembed`, runs locally, 384 dimensions | `run.py: embed_and_store()` |
| Store | A normalised NumPy matrix in memory (not a real vector database; fine for one paper) | same |
| Retrieve | Cosine similarity, top 5 | `run.py: retrieve()` |
| Prompt | Question + the 5 chunks, with an instruction to cite them | `run.py: build_prompt()` |
| Generate | OpenAI Responses API, `gpt-6-luna` by default | `run.py: generate()` |

## Run it

With pip (Python 3.12):

```bash
pip install -r requirements.txt
python run.py                      # default question
python run.py "your own question"
```

Or with Docker, nothing else to install (image ~670 MB; the embedding model and tokenizer are baked in, the paper downloads at run time):

```bash
docker build -t rag-blueprint-ep00 .
docker run --rm -v "$PWD/out:/app/out" rag-blueprint-ep00                        # run.py
docker run --rm -v "$PWD/out:/app/out" rag-blueprint-ep00 compare_overlap.py
docker run --rm -v "$PWD/out:/app/out" --env-file .env rag-blueprint-ep00        # with the LLM step
```

`-v` puts the outputs in your own `out/` folder. The key is passed at run time with `--env-file` and is never part of the image. On Windows PowerShell use `"${PWD}/out:/app/out"`.

The paper is downloaded from arXiv on the first run into `data/` (not committed: arXiv papers keep their own licence). Each step writes to `out/`:

```
0_stats.json           pages, words, tokens, chunk count, settings
1_pages.json           parsed text per page
2_chunks.json          every chunk with its page number
3_vectors_sample.json  dimensions and the first values of chunk 0's vector
4_retrieval.json       top-5 chunks with scores
5_prompt.txt           the exact prompt sent to the LLM
6_answer.json          the LLM's answer (only if a key is set)
```

The generation step needs an API key. Copy `.env.example` to `.env` and paste your key; without a key the script stops after writing the prompt. Note: values in `.env` **override** the shell environment, so the key in the file is always the one used.

Three providers are supported, chosen with `LLM_PROVIDER`:

| `LLM_PROVIDER` | Key variable | Default model | Notes |
|---|---|---|---|
| `openai` (default) | `OPENAI_API_KEY` | `gpt-6-luna` | What the slides used (Responses API) |
| `groq` | `GROQ_API_KEY` | `openai/gpt-oss-20b` | Has a free plan with rate limits |
| `google` | `GEMINI_API_KEY` | `gemini-3.8-flash` | Google AI Studio; has a free tier. Google states free-tier content is used to improve its products |

Groq and Google use the OpenAI client through their OpenAI-compatible endpoints (Chat Completions). Override the model with `LLM_MODEL`. Answers from other models will differ from the slides; retrieval doesn't depend on the provider.

Two more scripts reproduce the comparisons on the slides:

```bash
python compare_overlap.py   # slide 5: same question, overlap 0 vs 40 words  -> out/8_overlap_comparison.json
python no_context.py        # slide 8: same question, no retrieved chunks     -> out/7_no_context.json
```

## What the run produced (the numbers on the slides)

Question: *"What does RAG use as its non-parametric memory?"* The answer in the paper: "a dense vector index of Wikipedia".

- Corpus: 19 pages, 9,871 words, 18,213 tokens (OpenAI `o200k_base`), 66 chunks. Chunks don't cross page breaks, so 19 of them are shorter page-end pieces.
- Parsing artifacts: 109 "ﬁ" and 5 "ﬂ" ligature characters, hyphenated line breaks ("knowl- edge"), page numbers and the arXiv stamp inside the text.

Top 5 by cosine similarity:

| Rank | Score | Page | Section | Contains the answer? |
|---|---|---|---|---|
| 1 | 0.778 | 9 | Discussion | No |
| 2 | 0.768 | 2 | Introduction | Yes |
| 3 | 0.754 | 1 | Abstract | Yes |
| 4 | 0.751 | 17 | Code-version note | No |
| 5 | 0.735 | 7 | Results | Partial (names the Wikipedia index, not as "memory") |

All five scores sit within 0.043; #1 and #2 are 0.010 apart. A third chunk that contains the answer sentence (chunk 0, page 1) ranked 17th. We tested whether the author list in that chunk was the cause: removing it moved the score from 0.699 to 0.701 and the rank stayed 17th, so we don't claim a cause.

Overlap comparison (`compare_overlap.py`): with no overlap, 59 chunks and the answer chunks ranked 4 and 14; with 40-word overlap, 66 chunks and ranks 2, 3 and 17. Changing the overlap moves every chunk boundary, so this is one question, not a rule.

Generation (`gpt-6-luna`, with the 5 chunks): *"RAG uses a dense vector index of Wikipedia as its non-parametric memory, accessed by a pre-trained neural retriever (DPR). [2][3]"* — the run shown on the slides. Later runs of the same script gave *"…accessed through a pre-trained neural retriever. [2][3]"* and *"…accessed with a pre-trained neural retriever. [2]"*: the same claim each time, but the wording and the cited chunks vary. LLM output varies between runs; the retrieval does not.

Without context (`no_context.py`): *"RAG uses an external knowledge corpus—typically documents or passages stored in a searchable index—as its non-parametric memory."* A re-run gave *"RAG uses an external collection of documents, typically stored in a vector database or index, as its non-parametric memory."* Both generic; neither names Wikipedia or DPR. The specifics came from the retrieved chunks. That holds for these `gpt-6-luna` runs only: with no context, Google's `gemini-3.8-flash` answered *"a dense vector index of an external knowledge corpus (such as Wikipedia passages)"*, so that model already knows this paper from training. This is exactly the caveat on slide 8; the no-context test only means something for documents the model hasn't seen. This paper is famous and likely in the model's training data, so grounding matters most for documents the model has never seen.

## Caveats

- One question, one embedding model, one chunking setting. It shows how the steps behave. It is not a benchmark.
- A NumPy array is not a vector database. Episode 4 covers real indexes.
- The paper's 18K tokens fit in one prompt; Anthropic's guidance for Claude is that under ~200K tokens you can skip RAG. We used RAG here to see the mechanics.

## Sources quoted on the slides

- Lewis et al., 2020, arXiv 2005.11401
- Barnett et al., 2024, *Seven Failure Points When Engineering a RAG System*, arXiv 2401.05856
- Anthropic, Sep 2024, *Introducing Contextual Retrieval*
- Liu et al., *Lost in the Middle*, TACL 2024
- LangChain docs, *Build a RAG agent*
