"""
Auto-Settle AI — geometric fixes for Grounding DINO part labels
==============================================================

Zero-shot DINO makes two predictable mistakes on side-view close-ups:

1. The long strip under the doors gets labelled "front/rear bumper".
   A bumper never looks like a thin strip running under the doors, so a
   "bumper" box that is >= 6x wider than tall is relabelled "side skirt".

2. Two adjacent doors come back as ONE "car door" box (often the whole photo).
   If that box contains >= 2 door handles, there are >= 2 doors. We find the
   vertical door seam between each pair of handles (a long continuous vertical
   edge, found with OpenCV) and split the box there. The door bottom is also
   clipped at the top of the side skirt.

Only needs numpy + opencv, so it can be tested without a GPU.
Used by grounding_dino_parts.py; run directly to test on a JSON + image:
    python refine_parts.py car.jpg outputs/parts.json
"""

from __future__ import annotations

import copy
import json
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BUMPERS = {"front bumper", "rear bumper"}
STRIP_ASPECT = 6.0       # w/h above this -> a strip, not a bumper
SEAM_MIN_CONTINUITY = 0.5  # seam must be a strong edge in >= 50% of rows
WHOLE_FRAME = 0.85


def _w(b) -> float:
    return b[2] - b[0]


def _h(b) -> float:
    return b[3] - b[1]


def _area(b) -> float:
    return max(0.0, _w(b)) * max(0.0, _h(b))


def containment(inner, outer) -> float:
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(_area(inner), 1.0)


def _finalize(d: Dict, img_w: int, img_h: int) -> None:
    b = d["bbox"] = [round(float(v), 1) for v in d["bbox"]]
    d["bbox_norm"] = [round(b[0] / img_w, 4), round(b[1] / img_h, 4),
                      round(b[2] / img_w, 4), round(b[3] / img_h, 4)]
    d["whole_frame"] = _area(b) / (img_w * img_h) > WHOLE_FRAME


def find_seam(gray: np.ndarray, x_lo: int, x_hi: int, y_lo: int, y_hi: int) -> Optional[Tuple[int, float]]:
    """Column of the longest continuous vertical edge inside the box, or None."""
    if x_hi - x_lo < 12 or y_hi - y_lo < 20:
        return None
    roi = gray[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
    sob = np.abs(cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3))
    strong = (sob > np.percentile(sob, 90)).astype(np.uint8)
    strong = cv2.dilate(strong, np.ones((1, 3), np.uint8))  # tolerate a slight tilt
    cont = strong.mean(axis=0).astype(np.float32)           # share of rows with an edge
    cont[:3] = 0
    cont[-3:] = 0                                            # ignore border artefacts
    x = int(np.argmax(cont))
    if cont[x] < SEAM_MIN_CONTINUITY:
        return None
    return x_lo + x, round(float(cont[x]), 3)


def refine_parts(image_rgb: np.ndarray, parts: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """Returns (new part list, human-readable notes). Input list is not modified."""
    img_h, img_w = image_rgb.shape[:2]
    dets = copy.deepcopy(parts)
    notes: List[str] = []

    # 1) bumper-labelled strips -> side skirt
    for d in dets:
        b = d["bbox"]
        if d["label"] in BUMPERS and _h(b) > 0 and _w(b) / _h(b) >= STRIP_ASPECT:
            notes.append(f"relabelled '{d['label']}' {[int(v) for v in b]} -> 'side skirt' "
                         f"(strip, {(_w(b) / _h(b)):.0f}:1)")
            d["refined_from"] = d["label"]
            d["label"] = "side skirt"

    # 2) clip door bottoms at the top of the side skirt
    skirts = [d for d in dets if d["label"] == "side skirt"]
    for door in [d for d in dets if d["label"] == "car door"]:
        db = door["bbox"]
        for s in skirts:
            sb = s["bbox"]
            if containment(sb, db) >= 0.8 and sb[1] > db[1] + 0.5 * _h(db) and sb[1] < db[3]:
                notes.append(f"clipped door bottom {int(db[3])} -> {int(sb[1])} (top of side skirt)")
                db[3] = sb[1]
                door.setdefault("refined_from", "car door")

    # 3) split a door box that holds >= 2 handles at the seam(s) between them
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    handles = [d for d in dets if d["label"] == "door handle"]
    for door in [d for d in dets if d["label"] == "car door"]:
        db = door["bbox"]
        hs = sorted((h for h in handles if containment(h["bbox"], db) >= 0.8),
                    key=lambda h: h["bbox"][0])
        if len(hs) < 2:
            continue
        cuts = []
        for a, b in zip(hs, hs[1:]):
            x_lo, x_hi = int(a["bbox"][2]), int(b["bbox"][0])
            y_lo, y_hi = int(max(a["bbox"][3], b["bbox"][3])), int(db[3])
            seam = find_seam(gray, x_lo, x_hi, y_lo, y_hi)
            if seam:
                cuts.append(seam[0])
                notes.append(f"door seam found at x={seam[0]} (continuity {seam[1]})")
            else:
                notes.append(f"no clear door seam between x={x_lo} and x={x_hi} - door not split")
        if not cuts:
            continue
        dets.remove(door)
        xs = [db[0]] + cuts + [db[2]]
        for i, (x1, x2) in enumerate(zip(xs, xs[1:]), 1):
            nd = copy.deepcopy(door)
            nd["bbox"] = [x1, db[1], x2, db[3]]
            nd["refined_from"] = f"car door (split at seam, piece {i} of {len(xs) - 1})"
            dets.append(nd)
        notes.append(f"split one 'car door' box into {len(xs) - 1} doors")

    for d in dets:
        _finalize(d, img_w, img_h)
    dets.sort(key=lambda d: -d["confidence"])
    return dets, notes


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python refine_parts.py car.jpg parts.json")
        sys.exit(1)
    img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    data = json.load(open(sys.argv[2]))
    parts, notes = refine_parts(img, data["parts"] if isinstance(data, dict) else data)
    print("\n".join(notes) or "no changes")
    for p in parts:
        print(f"  {p['label']:<14} conf={p['confidence']:.3f} bbox={[int(v) for v in p['bbox']]}")
