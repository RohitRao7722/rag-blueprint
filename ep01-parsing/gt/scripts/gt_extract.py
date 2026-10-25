"""Step 1: run the app's six parser setups on every generated PDF; save raw texts."""
import sys, json, time, traceback
sys.path.insert(0, r"d:/MY/linkedin-posts/experiments/ep01-parsing/app")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from gt_common import *
import rag_core as core
out = {}
for pdf in sorted(PDF.glob("*.pdf")):
    data = pdf.read_bytes(); row = {}
    for name, fn in core.PARSERS.items():
        t0 = time.time()
        try:
            t = fn(data)
            if isinstance(t, (list, tuple)): t = "\n".join(t)
            row[name] = {"text": t, "s": round(time.time() - t0, 2)}
        except Exception as e:
            row[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
    out[pdf.stem] = row
    print(pdf.stem, {k: (len(v.get("text", "")) if "text" in v else "ERR") for k, v in row.items()}, flush=True)
(ROOT / "extracted.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
