"""Generator 2: Typst (its own PDF writer and HarfBuzz-based shaping)."""
import sys, typst
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from gt_common import *
PDF.mkdir(parents=True, exist_ok=True)
def esc(s):  # escape Typst markup characters
    for c in r"\#*_`$<>@[]~/=-+":
        s = s.replace(c, "\\" + c)
    return s
for code in langs():
    d = "rtl" if code in RTL else "ltr"
    body = "\n\n".join(esc(p) for p in text(code).split("\n\n"))
    src = (f'#set page(paper: "a4", margin: 2cm)\n#set text(font: "{font(code)}", size: 12pt, lang: "{code}", dir: {d})\n'
           f'#set par(justify: false)\n{body}\n')
    f = PDF / f"{code}__typst.typ"; f.write_text(src, encoding="utf-8")
    out = PDF / f"{code}__typst.pdf"
    try:
        typst.compile(str(f), output=str(out), font_paths=[r"C:\Windows\Fonts"])
        print(code, out.stat().st_size)
    except Exception as e:
        print(code, "FAILED", str(e)[:200])
    f.unlink()
