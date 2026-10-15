"""Episode 0, slide 8: ask the LLM the same question with no retrieved context.

Compares against out/6_answer.json from run.py. Needs OPENAI_API_KEY (see README).
"""
import os
import sys

import run


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    run.load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set.")
    from openai import OpenAI
    question = sys.argv[1] if len(sys.argv) > 1 else "What does RAG use as its non-parametric memory?"
    prompt = f"Answer the question in one or two sentences.\n\nQuestion: {question}\nAnswer:"
    model = os.environ.get("OPENAI_MODEL", run.LLM_MODEL)
    answer = OpenAI().responses.create(model=model, input=prompt).output_text
    run.OUT.mkdir(exist_ok=True)
    run.save("7_no_context.json", {"question": question, "model": model, "prompt": prompt, "answer": answer})
    print(f"A ({model}, no context): {answer}")


if __name__ == "__main__":
    main()
