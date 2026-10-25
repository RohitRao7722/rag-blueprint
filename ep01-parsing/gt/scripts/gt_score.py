"""Step 2: score each parser output against the ground-truth text."""
import sys, json, re, random, unicodedata
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from gt_common import *
ex = json.loads((ROOT / "extracted.json").read_text(encoding="utf-8"))
WS = re.compile(r"\s+")
DOUBLE = re.compile(r"([\u0900-\u0903\u093a-\u094f\u0981-\u0983\u09bc-\u09cd\u0a01-\u0a03\u0a3c-\u0a4d\u0a81-\u0a83\u0abc-\u0acd\u0b82\u0bbe-\u0bcd\u0c00-\u0c04\u0c3e-\u0c4d\u0c81-\u0c83\u0cbc-\u0ccd\u0d00-\u0d03\u0d3b-\u0d4d\u0e31\u0e34-\u0e3a\u0e47-\u0e4e])\1")
def norm(s): return WS.sub("", unicodedata.normalize("NFC", s))
def broken(s): return sum(1 for c in s if unicodedata.category(c) in ("Cc", "Co", "Cs", "Cn") and c not in "\n\t\r")
def phrases(gt, n=20, k=12):
    rnd = random.Random(0); flat = norm(gt)
    return [flat[i:i + k] for i in (rnd.randrange(0, len(flat) - k) for _ in range(n))]
rows = []
for stem, parsers in sorted(ex.items()):
    code, gen = stem.split("__", 1)
    gt = text(code); gtn = norm(gt); ph = phrases(gt); gt_double = len(DOUBLE.findall(gt))
    for name, r in parsers.items():
        if "text" not in r:
            rows.append((code, gen, name, None, None, None, None, r["error"])); continue
        t = r["text"]; tn = norm(t)
        sim = Levenshtein.normalized_similarity(gtn, tn)
        hit = sum(p in tn for p in ph) / len(ph)
        rows.append((code, gen, name, round(sim, 3), round(hit, 2), len(DOUBLE.findall(t)) - gt_double, broken(t), ""))
(ROOT / "scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
# summary: per language × generator, best/worst parser
print(f"{'lang':4} {'gen':13} {'parser':30} {'sim':>5} {'hit':>5} {'dbl':>4} {'brk':>5}")
for r in rows:
    print(f"{r[0]:4} {r[1]:13} {r[2]:30} {str(r[3]):>5} {str(r[4]):>5} {str(r[5]):>4} {str(r[6]):>5} {r[7]}")
