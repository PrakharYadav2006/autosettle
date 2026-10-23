"""
Reading a garage estimate / invoice  ->  UNTRUSTED candidate data
==================================================================
The already-loaded vision model (Qwen2.5-VL) only TRANSCRIBES the printed rows into JSON.
It decides nothing: whether a line is legitimate, which repair method applies, what is paid,
and whether an offer changes are all decided in Python by
pricing_rules.check_garage_estimate() + price_claim().

Formats: photo of the estimate (jpg/png/webp/heic...) or a PDF (first 2 pages, rendered with
pypdfium2:  pip install pypdfium2).

What this cannot do: prove the document is genuine or that the garage issued it.  The
result is labelled "contents checked against claim evidence and pricing rules", never
"verified".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageOps

MAX_PAGES = 2
PROMPT = (
    "This image should be a vehicle repair estimate or invoice from a garage. Copy what is printed; "
    "do not calculate, correct or guess anything.\n"
    "Return ONLY this JSON:\n"
    '{"readable": true, "lines": [{"description": "row text", "part": "panel name as printed, or null", '
    '"type": "part|labour|paint|other", "qty": number or null, "rate": number or null, "amount": number}], '
    '"subtotal": number or null, "tax_total": number or null, "grand_total": number or null}\n'
    "Rules: one entry for every printed row that has an amount. type: \"part\" = a new spare part, "
    "\"labour\" = fitting / denting / repair labour, \"paint\" = painting, \"other\" = tax, consumables, "
    "discounts, totals or anything else. Write numbers without currency symbols or commas. "
    "If this is not a repair estimate or you cannot read it, return {\"readable\": false, \"lines\": []}."
)


class DocumentUnsupported(Exception):
    """The server cannot open this document type (e.g. PDF without pypdfium2)."""


def document_pages(path: Path, max_pages: int = MAX_PAGES) -> List[Image.Image]:
    if path.suffix.lower() == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise DocumentUnsupported("PDF reading needs pypdfium2 on the server (pip install pypdfium2)") from exc
        pdf = pdfium.PdfDocument(str(path))
        try:
            return [pdf[i].render(scale=2).to_pil().convert("RGB") for i in range(min(len(pdf), max_pages))]
        finally:
            pdf.close()
    with Image.open(path) as im:
        return [ImageOps.exif_transpose(im).convert("RGB")]


def parse_reply(text: str) -> Optional[Dict]:
    """The first {...} object in the model's reply, or None.  Structure is validated later."""
    if not isinstance(text, str):
        return None
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        v = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, dict) else None


def extract(path: Path, vlm) -> Dict:
    """-> {"status": "ok" | "unreadable" | "no_reader", "doc": {...} | None, "pages": n, "raw": [...]}"""
    if vlm is None:
        return {"status": "no_reader", "doc": None, "pages": 0, "raw": []}
    pages = document_pages(path)
    read = getattr(vlm, "read_document", None) or vlm.ask
    raws, docs = [], []
    for img in pages:
        reply = read(img, PROMPT)
        raws.append(str(reply)[:4000])
        docs.append(parse_reply(reply))
    if not docs or any(d is None for d in docs):
        return {"status": "unreadable", "doc": None, "pages": len(pages), "raw": raws}
    merged: Dict = {"readable": all(d.get("readable") is not False for d in docs), "lines": []}
    for d in docs:
        if isinstance(d.get("lines"), list):
            merged["lines"].extend(d["lines"])
        else:
            merged["readable"] = False
    for k in ("subtotal", "tax_total", "grand_total"):   # totals: from the last page that prints them
        vals = [d.get(k) for d in docs if d.get(k) not in (None, "")]
        merged[k] = vals[-1] if vals else None
    return {"status": "ok" if merged["readable"] and merged["lines"] else "unreadable",
            "doc": merged, "pages": len(pages), "raw": raws}
