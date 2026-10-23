"""
Licence plate: locate with the ALREADY-LOADED Grounding DINO, read with OCR on the CPU
======================================================================================
OCR engines (AUTOSETTLE_OCR=auto|rapid|easy, default auto):
  * RapidOCR (rapidocr_onnxruntime) - PaddleOCR's detection + recognition models run through
    ONNX Runtime: no PaddlePaddle install, CPU-only, ~0.3 s per plate crop.  Tried first.
  * EasyOCR - fallback when RapidOCR is missing or its read is not a valid plate.
A box is only treated as a plate if its SHAPE is plate-like (wide, small) - DINO sometimes
boxes a lamp or a whole corner when asked for "license plate".  A read counts as identified
only if it is a complete Indian registration (e.g. DL3CAB1234, 22BH1234AA); partial reads
(grass, mud, angle) are shown as "not identified".
Best-effort and never load-bearing:
  * the plate box comes from the same DINO model used for parts (one extra prompt, no new model);
  * OCR is EasyOCR with gpu=False, created ONCE per process (no extra GPU memory);
  * any failure (no plate, low confidence, OCR missing, an exception) returns
    status "not_identified" - the claim continues exactly as before;
  * the result is shown to the customer and staff as evidence only.  It never approves a claim,
    sets a price, or authenticates anyone; comparing it with the policy registration is reported
    as a vehicle-registration match / mismatch, nothing more.

Install:  pip install rapidocr_onnxruntime easyocr   (EasyOCR downloads its models on first use)
"""
from __future__ import annotations

import os
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
PLATE_ASPECT = (1.2, 7.0)    # width / height: two-line (bikes, square plates) up to long single-line
PLATE_MAX_AREA = 0.08        # a "plate" covering more than 8 % of a claim photo is a lamp or a corner
PLATE_MIN_PX = 12
OCR_MODE = os.environ.get("AUTOSETTLE_OCR", "auto")
NOT_IDENTIFIED = "License plate not identified"
INDIAN = re.compile(r"^([A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4}|\d{2}BH\d{4}[A-Z]{1,2})$")

_ENGINES: Optional[List[Tuple[str, object]]] = None    # [(name, read(crop) -> [(box, text, conf)])]
_LOCK = threading.Lock()


def _rapid_engine():
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR()

    def read(img):
        import numpy as np
        res, _ = eng(np.array(img.convert("RGB")))
        return [(r[0], str(r[1]), float(r[2])) for r in (res or [])]
    return read


def _easy_engine():
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(img):
        import numpy as np
        return [(r[0], str(r[1]), float(r[2])) for r in reader.readtext(np.array(img.convert("RGB")))]
    return read


def get_engines() -> List[Tuple[str, object]]:
    """OCR engines available on this machine, in the order they are tried (created once)."""
    global _ENGINES
    with _LOCK:
        if _ENGINES is None:
            _ENGINES = []
            order = {"rapid": ["rapidocr"], "easy": ["easyocr"]}.get(OCR_MODE, ["rapidocr", "easyocr"])
            for name in order:
                try:
                    _ENGINES.append((name, _rapid_engine() if name == "rapidocr" else _easy_engine()))
                except Exception as exc:  # noqa: BLE001 - not installed / model download failed
                    print(f"[plate] {name} unavailable ({type(exc).__name__}: {str(exc)[:120]})")
            if not _ENGINES:
                print(f"[plate] no OCR engine - plates will show '{NOT_IDENTIFIED}'. "
                      "pip install rapidocr_onnxruntime easyocr")
        return _ENGINES


def get_reader():
    """Kept for callers that warm the OCR up at start-up."""
    return get_engines() or None


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


def _join(hits: List[Tuple]) -> Tuple[str, float]:
    """Text boxes -> one string: rows top-to-bottom, left-to-right.  Overlapping boxes that repeat
    characters ('DL3' + '3CAB1234') are joined without the repeat."""
    if not hits:
        return "", 0.0
    def geom(h):
        xs = [p[0] for p in h[0]]
        ys = [p[1] for p in h[0]]
        return min(xs), max(xs), (min(ys) + max(ys)) / 2, max(ys) - min(ys)
    hs = sorted(((geom(h), normalize(h[1]), h[2]) for h in hits if normalize(h[1])), key=lambda g: g[0][2])
    if not hs:
        return "", 0.0
    med_h = sorted(g[0][3] for g in hs)[len(hs) // 2] or 1
    rows: List[List] = []
    for g in hs:
        if rows and abs(g[0][2] - rows[-1][0][0][2]) <= 0.5 * med_h:
            rows[-1].append(g)
        else:
            rows.append([g])
    text, prev = "", None
    for row in rows:
        for g in sorted(row, key=lambda g: g[0][0]):
            t = g[1]
            if prev is not None and g[0][0] < prev[0][1]:          # boxes overlap horizontally
                k = next((k for k in range(min(len(text), len(t)), 0, -1) if text.endswith(t[:k])), 0)
                t = t[k:]
            text += t
            prev = g
        prev = None
    conf = sum(g[2] for g in hs) / len(hs)
    return text, round(conf, 3)


def ocr(crop: Image.Image) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """(normalised text, confidence, engine).  The first engine that returns a VALID Indian
    registration wins; otherwise the most confident partial read is returned (for display)."""
    best: Tuple[Optional[str], Optional[float], Optional[str]] = (None, None, None)
    for name, read in get_engines():
        try:
            raw, conf = _join(read(crop))
        except Exception as exc:  # noqa: BLE001 - one engine failing never stops the other
            print(f"[plate] {name} failed: {type(exc).__name__}")
            continue
        text = _fix_common_confusions(normalize(raw))
        if not text:
            continue
        if INDIAN.match(text) and conf >= MIN_OCR_CONF:
            return text, conf, name
        if best[1] is None or conf > best[1]:
            best = (text, conf, name)
    return best


def plate_shaped(bbox, img_w: int, img_h: int) -> bool:
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return (w >= PLATE_MIN_PX and h >= PLATE_MIN_PX and PLATE_ASPECT[0] <= w / h <= PLATE_ASPECT[1]
            and (w * h) / max(img_w * img_h, 1) <= PLATE_MAX_AREA)


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
        odd = 0
        for img, info in images:
            for d in parts_det.find(img, ["license plate"]) or []:
                if plate_shaped(d["bbox"], img.width, img.height):
                    cands.append((float(d.get("confidence") or 0), img, d, info))
                else:
                    odd += 1          # a lamp / corner / whole bumper boxed as "plate"
        if not cands:
            return not_identified("no licence plate visible" + (f" ({odd} non-plate-shaped box(es) ignored)"
                                                               if odd else ""))
        cands.sort(key=lambda c: -c[0])
        out_dir.mkdir(parents=True, exist_ok=True)
        best_unread = None
        for k, (det_conf, img, d, info) in enumerate(cands[:MAX_CANDIDATES]):
            crop = _crop(img, d["bbox"])
            path = out_dir / ("plate.jpg" if k == 0 else f"plate_{k}.jpg")
            crop.save(path, quality=95)
            text, conf, engine = ocr(crop)
            if (text and conf is not None and conf >= MIN_OCR_CONF and MIN_LEN <= len(text) <= MAX_LEN
                    and INDIAN.match(text)):
                if k:
                    path.replace(out_dir / "plate.jpg")
                    path = out_dir / "plate.jpg"
                return {"status": "identified", "text": text, "confidence": conf, "message": None,
                        "reason": None, "ocr_engine": engine,
                        "detection_confidence": round(det_conf, 3), "crop_path": str(path), **info}
            if best_unread is None:
                why = ("text recognition is not installed" if not get_engines() else
                       "plate found but the text could not be read" if not text else
                       f"only part of the plate could be read ('{text}')" if not INDIAN.match(text) else
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
