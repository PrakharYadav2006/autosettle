"""
Licence plate: locate with the ALREADY-LOADED Grounding DINO, read with EasyOCR on the CPU
=========================================================================================
Best-effort and never load-bearing:
  * the plate box comes from the same DINO model used for parts (one extra prompt, no new model);
  * OCR is EasyOCR with gpu=False, created ONCE per process (no extra GPU memory);
  * any failure (no plate, low confidence, OCR missing, an exception) returns
    status "not_identified" - the claim continues exactly as before;
  * the result is shown to the customer and staff as evidence only.  It never approves a claim,
    sets a price, or authenticates anyone; comparing it with the policy registration is reported
    as a vehicle-registration match / mismatch, nothing more.

Install:  pip install easyocr      (first use downloads its small text models once)
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

MIN_OCR_CONF = 0.40          # mean EasyOCR confidence needed to call a plate "identified"
MIN_LEN, MAX_LEN = 6, 11     # Indian registrations: DL3CAB1234 (10), 22BH1234AA (10), HR26BK9999 ...
CROP_PAD = 0.10
OCR_MIN_HEIGHT = 120         # px: small plate crops are enlarged before OCR (and for display)
MAX_CANDIDATES = 3           # plate boxes tried (best detection confidence first)
NOT_IDENTIFIED = "License plate not identified"
INDIAN = re.compile(r"^([A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4}|\d{2}BH\d{4}[A-Z]{1,2})$")

_READER = None
_READER_FAILED = False
_LOCK = threading.Lock()


def get_reader():
    """The process-wide EasyOCR reader on the CPU, or None if EasyOCR is unavailable."""
    global _READER, _READER_FAILED
    with _LOCK:
        if _READER is None and not _READER_FAILED:
            try:
                import easyocr
                _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception as exc:  # noqa: BLE001 - not installed / model download failed
                print(f"[plate] EasyOCR unavailable ({type(exc).__name__}: {exc}) - plates will show "
                      f"'{NOT_IDENTIFIED}'.  pip install easyocr")
                _READER_FAILED = True
        return _READER


def normalize(text: str) -> str:
    """Upper-case letters and digits only ('dl 3c ab-1234' -> 'DL3CAB1234')."""
    return "".join(ch for ch in str(text or "").upper() if ch.isalnum())


def _fix_common_confusions(t: str) -> str:
    """Only for strings that then match the Indian pattern: O/0, I/1, S/5, B/8, Z/2 in the
    positions where the format says letters vs digits.  Otherwise the raw read is kept."""
    if INDIAN.match(t) or len(t) < 8:
        return t
    to_digit = {"O": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
    to_alpha = {v: k for k, v in to_digit.items() if k not in ("D", "L", "G")}
    c = list(t)
    for i in (0, 1):                               # state code: letters
        c[i] = to_alpha.get(c[i], c[i])
    for i in range(len(c) - 4, len(c)):            # last four: digits
        c[i] = to_digit.get(c[i], c[i])
    fixed = "".join(c)
    return fixed if INDIAN.match(fixed) else t


def _crop(img: Image.Image, bbox, pad: float = CROP_PAD) -> Image.Image:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    box = (max(0, int(x1 - pad * w)), max(0, int(y1 - pad * h)),
           min(img.width, int(x2 + pad * w)), min(img.height, int(y2 + pad * h)))
    c = img.crop(box)
    if c.height and c.height < OCR_MIN_HEIGHT:
        f = OCR_MIN_HEIGHT / c.height
        c = c.resize((max(1, int(c.width * f)), OCR_MIN_HEIGHT), Image.LANCZOS)
    return c


def ocr(crop: Image.Image) -> Tuple[Optional[str], Optional[float]]:
    """(normalised text, mean confidence) or (None, None)."""
    reader = get_reader()
    if reader is None:
        return None, None
    import numpy as np
    hits = reader.readtext(np.array(crop.convert("RGB")))
    if not hits:
        return None, None
    hits = sorted(hits, key=lambda h: (h[0][0][1] // 25, h[0][0][0]))   # rows top->bottom, then left->right
    text = _fix_common_confusions(normalize("".join(h[1] for h in hits)))
    conf = float(sum(h[2] for h in hits) / len(hits))
    return (text or None), round(conf, 3)


def not_identified(reason: str, crop_path: Optional[str] = None, **extra) -> Dict:
    return {"status": "not_identified", "text": None, "confidence": None, "message": NOT_IDENTIFIED,
            "reason": reason, "crop_path": crop_path, **extra}


def read_from_images(parts_det, images: Iterable[Tuple[Image.Image, Dict]], out_dir: Path) -> Dict:
    """images: (PIL image, info) pairs - one for a photo, keyframes for a video.
    Locate plates with DINO, OCR the best few crops on the CPU, keep the best readable one."""
    try:
        if parts_det is None or not hasattr(parts_det, "find"):
            return not_identified("plate detector not available")
        cands: List[Tuple[float, Image.Image, Dict, Dict]] = []
        for img, info in images:
            for d in parts_det.find(img, ["license plate"]) or []:
                cands.append((float(d.get("confidence") or 0), img, d, info))
        if not cands:
            return not_identified("no licence plate visible")
        cands.sort(key=lambda c: -c[0])
        out_dir.mkdir(parents=True, exist_ok=True)
        best_unread = None
        for k, (det_conf, img, d, info) in enumerate(cands[:MAX_CANDIDATES]):
            crop = _crop(img, d["bbox"])
            path = out_dir / ("plate.jpg" if k == 0 else f"plate_{k}.jpg")
            crop.save(path, quality=95)
            text, conf = ocr(crop)
            if text and conf is not None and conf >= MIN_OCR_CONF and MIN_LEN <= len(text) <= MAX_LEN:
                if k:
                    path.replace(out_dir / "plate.jpg")
                    path = out_dir / "plate.jpg"
                return {"status": "identified", "text": text, "confidence": conf, "message": None,
                        "reason": None if INDIAN.match(text) else "read, but not in the usual Indian format",
                        "detection_confidence": round(det_conf, 3), "crop_path": str(path), **info}
            if best_unread is None:
                why = ("text recognition is not installed" if get_reader() is None else
                       "plate found but the text could not be read" if not text else
                       f"plate text unclear (confidence {conf:.2f})")
                best_unread = not_identified(why, str(path), detection_confidence=round(det_conf, 3),
                                             unread_text=text, unread_confidence=conf, **info)
        return best_unread
    except Exception as exc:  # noqa: BLE001 - never break an assessment over a plate
        return not_identified(f"plate reading failed ({type(exc).__name__})")


def registration_check(read: Dict, policy_plate: Optional[str]) -> Optional[str]:
    """'match' | 'mismatch' | None.  A vehicle-registration comparison only - not identity."""
    if not policy_plate or read.get("status") != "identified":
        return None
    return "match" if normalize(read["text"]) == normalize(policy_plate) else "mismatch"
