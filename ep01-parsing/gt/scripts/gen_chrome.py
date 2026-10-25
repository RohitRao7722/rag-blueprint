"""Generator 1: Chrome print-to-PDF, plain and with underlined links (tests the link hypothesis)."""
import html, subprocess, sys, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from gt_common import *
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PDF.mkdir(parents=True, exist_ok=True)
def page(code, links):
    paras = []
    for p in text(code).split("\n\n"):
        words = p.split(" ")
        if links:  # every 4th word becomes an underlined link, like Wikipedia
            words = [f'<a href="https://example.org/{i}">{html.escape(w)}</a>' if i % 4 == 1 else html.escape(w)
                     for i, w in enumerate(words)]
        else:
            words = [html.escape(w) for w in words]
        paras.append("<p>" + " ".join(words) + "</p>")
    d = ' dir="rtl"' if code in RTL else ""
    return (f'<!doctype html><html lang="{code}"{d}><meta charset="utf-8"><style>'
            f'body{{font-family:"{font(code)}";font-size:12pt;margin:2cm}} a{{color:#0645ad}}</style>'
            + "".join(paras) + "</html>")
for code in langs():
    for links in (False, True):
        name = f"{code}__chrome{'_links' if links else ''}"
        h = PDF / f"{name}.html"; h.write_text(page(code, links), encoding="utf-8")
        out = PDF / f"{name}.pdf"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out}", h.as_uri()], capture_output=True, timeout=120)
        h.unlink()
        print(name, out.exists() and out.stat().st_size)
