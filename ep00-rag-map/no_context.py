"""Episode 0, slide 8: ask the LLM the same question with no retrieved context.

Compares against out/6_answer.json from run.py. Uses the same provider settings
as run.py (OpenAI by default; see README).
"""
import sys

import run


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    run.load_env()
    question = sys.argv[1] if len(sys.argv) > 1 else "What does RAG use as its non-parametric memory?"
    prompt = f"Answer the question in one or two sentences.\n\nQuestion: {question}\nAnswer:"
    result = run.generate(prompt)
    if result is None:
        sys.exit("No API key set for the chosen provider.")
    run.OUT.mkdir(exist_ok=True)
    run.save("7_no_context.json", {"question": question, "prompt": prompt, **result})
    print(f"A ({result['provider']}, {result['model']}, no context): {result['answer']}")


if __name__ == "__main__":
    main()
