"""Shared settings for the ground-truth parser test."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent   # the gt/ folder
TEXT, PDF = ROOT / "text", ROOT / "pdf"
FONTS = {  # one Windows system font per script
    "en": "Calibri", "ru": "Arial", "ar": "Arial", "ur": "Arial", "th": "Leelawadee UI",
    "zh": "Microsoft YaHei", "ja": "Yu Gothic", "ko": "Malgun Gothic",
}
RTL = {"ar", "ur"}
def font(code): return FONTS.get(code, "Nirmala UI")
def langs(): return sorted(p.stem for p in TEXT.glob("*.txt"))
def text(code): return (TEXT / f"{code}.txt").read_text(encoding="utf-8")
