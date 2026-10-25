"""Fetch plain-text Wikipedia extracts as ground truth for generated test PDFs."""
import json, sys, re, unicodedata, requests
from pathlib import Path
OUT = Path(sys.argv[1])
# (code, language, article title)
ARTICLES = [
    ("en", "English", "India"), ("hi", "Hindi", "भारत"), ("mr", "Marathi", "भारत"),
    ("bn", "Bengali", "ভারত"), ("ta", "Tamil", "இந்தியா"), ("te", "Telugu", "భారతదేశం"),
    ("gu", "Gujarati", "ભારત"), ("pa", "Punjabi", "ਭਾਰਤ"), ("kn", "Kannada", "ಭಾರತ"),
    ("ml", "Malayalam", "ഇന്ത്യ"), ("ur", "Urdu", "بھارت"), ("ar", "Arabic", "الهند"),
    ("th", "Thai", "ประเทศอินเดีย"), ("zh", "Chinese", "印度"), ("ja", "Japanese", "インド"),
    ("ko", "Korean", "인도"), ("ru", "Russian", "Индия"),
]
s = requests.Session(); s.headers["User-Agent"] = "rag-blueprint-parser-test/0.1 (research)"
import time
for code, lang, title in ARTICLES:
    if (OUT / f"{code}.txt").exists(): continue
    for attempt in range(5):
      time.sleep(2 + attempt * 3)
      r = s.get(f"https://{code}.wikipedia.org/w/api.php", params=dict(action="query", prop="extracts",
        explaintext=1, exsectionformat="plain", redirects=1, titles=title, format="json"), timeout=30)
      if r.status_code == 200 and r.text.startswith("{"): break
      print(code, 'retry', r.status_code)
    pages = r.json()["query"]["pages"]
    text = next(iter(pages.values())).get("extract", "")
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 80]
    out, n = [], 0
    for p in paras:
        out.append(p); n += len(p)
        if n > 2200: break
    body = unicodedata.normalize("NFC", "\n\n".join(out))
    (OUT / f"{code}.txt").write_text(body, encoding="utf-8")
    print(code, lang, len(body), "chars", len(out), "paras")
