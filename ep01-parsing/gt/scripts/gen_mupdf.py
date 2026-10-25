"""Generator 3: PyMuPDF Story (MuPDF's HTML layout engine and PDF writer)."""
import html, sys, pymupdf
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from gt_common import *
PDF.mkdir(parents=True, exist_ok=True)
for code in langs():
    d = ' dir="rtl"' if code in RTL else ""
    body = "".join(f"<p>{html.escape(p)}</p>" for p in text(code).split("\n\n"))
    story = pymupdf.Story(f'<body{d} lang="{code}">{body}</body>', user_css="body{font-size:12pt}")
    out = PDF / f"{code}__mupdf.pdf"
    w = pymupdf.DocumentWriter(str(out))
    more = True
    while more:
        dev = w.begin_page(pymupdf.paper_rect("a4"))
        more, _ = story.place(pymupdf.paper_rect("a4") + (57, 57, -57, -57))
        story.draw(dev); w.end_page()
    w.close()
    print(code, out.stat().st_size)
