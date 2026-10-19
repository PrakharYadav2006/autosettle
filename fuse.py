"""
Auto-Settle AI — Stage 3: Fuse YOLO damage boxes with Grounding DINO part boxes
==============================================================================

Uses CONTAINMENT (share of the damage box that lies inside the part box), not
IoU. A small dent inside a big door has IoU ~0.02 but containment ~1.0.

What this filter DOES catch: damage boxes off the car (road, background,
glass if no glass part is listed).
What it does NOT catch: a reflection on a clean door flagged as a dent - that
box is inside the door. Stage 4 (VLM) must confirm/reject those.

Inputs
------
parts.json  : output of grounding_dino_parts.py
yolo.json   : list of {"label", "confidence", "bbox": [x1, y1, x2, y2]} (pixels)

    # teammate: dump ultralytics results like this
    r = model(image_path)[0]
    dets = [{"label": r.names[int(c)], "confidence": float(s), "bbox": b.tolist()}
            for b, s, c in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)]
    json.dump(dets, open("yolo.json", "w"))

Usage
-----
    python fuse.py car.jpg parts.json yolo.json --out-dir outputs
Outputs: fused.json, fused_annotated.jpg, crop_<n>_<key>.jpg (one per match)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageOps

GREEN, RED, GREY, WHITE = (0, 200, 0), (0, 0, 255), (160, 160, 160), (255, 255, 255)
ORANGE = (0, 140, 255)

# Damage types that belong to a specific part: a broken lamp IS the headlight,
# even if its box sticks out of the headlight box or also overlaps the bumper.
PREFERRED_PARTS = {
    "lamp broken": {"headlight", "taillight", "fog light"},
    "glass shatter": {"windshield", "side mirror", "headlight", "taillight"},
    "tire flat": {"wheel"},
}
PREFERRED_MIN_CONTAINMENT = 0.3
# Lamps / glass are checked by the VLM whenever the photo shows damage,
# even if YOLO found nothing on them (they are expensive and easy to miss).
INSPECT_PARTS = {"headlight", "taillight", "fog light", "side mirror", "windshield"}
# For the per-part claim: the worst damage on a part decides repair vs replace.
SEVERITY_RANK = {"lamp broken": 5, "glass shatter": 5, "tire flat": 4, "crack": 3, "dent": 2, "scratch": 1}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def containment(inner, outer) -> float:
    """Fraction of `inner` box area that lies inside `outer`."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(box_area(inner), 1.0)


def severity_hint(part_coverage: float) -> str:
    """Rough first guess; Stage 4 VLM gives the real severity."""
    return "minor" if part_coverage < 0.15 else "moderate" if part_coverage < 0.40 else "severe"


def part_key(label: str) -> str:
    return label.replace("car ", "").replace(" ", "_")


WHOLE_FRAME = 0.85  # a "part" box covering >85% of the photo tells us nothing
# Body panels: a real door box never contains a bumper / side skirt / fender box.
PANELS = {"car door", "front bumper", "rear bumper", "fender", "quarter panel",
          "side skirt", "car hood", "trunk"}


def too_broad(p: Dict, parts: List[Dict], img_area: float) -> bool:
    """Part box is useless as a location: whole photo, or it swallows another panel."""
    if box_area(p["bbox"]) / img_area >= WHOLE_FRAME:
        return True
    return any(q is not p and q["label"] in PANELS and q["label"] != p["label"]
               and containment(q["bbox"], p["bbox"]) >= 0.85 for q in parts)


def fuse(damages: List[Dict], parts: List[Dict], image_wh, min_containment: float = 0.6,
         min_damage_conf: float = 0.25) -> Dict:
    img_area = float(image_wh[0] * image_wh[1])
    matched, rejected = [], []
    for d in damages:
        if d["confidence"] < min_damage_conf:
            rejected.append({**d, "reason": f"damage confidence < {min_damage_conf}"})
            continue
        scored = [(containment(d["bbox"], p["bbox"]), p) for p in parts]
        inside = [(c, p) for c, p in scored if c >= min_containment]
        has_preferred = any(p["label"] in PREFERRED_PARTS.get(d["label"], ())
                            and c >= PREFERRED_MIN_CONTAINMENT for c, p in scored)
        if not inside and not has_preferred:
            best = round(max((c for c, _ in scored), default=0.0), 3)
            rejected.append({**d, "reason": f"not inside any part (best containment {best})"})
            continue
        preferred = [(c, p) for c, p in scored
                     if p["label"] in PREFERRED_PARTS.get(d["label"], ())
                     and c >= PREFERRED_MIN_CONTAINMENT]
        specific = [(c, p) for c, p in inside if not too_broad(p, parts, img_area)]
        if preferred:  # type-aware: lamp damage -> lamp part
            c, p = max(preferred, key=lambda cp: cp[0])
            uncertain = False
        else:
            pool = specific or inside
            # most specific part = smallest box that still contains the damage
            c, p = min(pool, key=lambda cp: box_area(cp[1]["bbox"]))
            uncertain = not specific
        coverage = min(1.0, box_area(d["bbox"]) / max(box_area(p["bbox"]), 1.0))
        spans = [q["label"] for cq, q in scored
                 if q is not p and cq >= 0.3 and not too_broad(q, parts, img_area)]
        matched.append({
            "key": f"{part_key(p['label'])}-{d['label'].lower().replace(' ', '_')}",
            "part": p["label"],
            "damage": d["label"],
            "damage_confidence": round(d["confidence"], 4),
            "part_confidence": p["confidence"],
            "containment": round(c, 3),
            "part_coverage": round(coverage, 4),
            # coverage of a whole-frame box is meaningless -> no severity guess
            "severity_hint": None if uncertain else severity_hint(coverage),
            "part_uncertain": uncertain,
            "part_uncertain_reason": ("only an over-broad part box (whole photo / swallows other panels) contains this damage - "
                                      "let the VLM name the panel") if uncertain else None,
            "spans_other_parts": spans,
            "damage_bbox": d["bbox"],
            "part_bbox": p["bbox"],
            "source": d.get("source", "full"),
            "needs_vlm_verification": True,  # always - reflections pass this filter
        })
    matched.sort(key=lambda m: -m["damage_confidence"])

    # One claim line per part: all damages on it, the worst one decides repair vs replace
    claims: Dict[str, Dict] = {}
    for m in matched:
        c = claims.setdefault(m["part"], {"part": m["part"], "damages": [], "part_uncertain": False})
        c["damages"].append(m["damage"])
        c["part_uncertain"] |= m["part_uncertain"]
    for c in claims.values():
        c["primary_damage"] = max(c["damages"], key=lambda x: SEVERITY_RANK.get(x, 0))
        c["damages"] = sorted(set(c["damages"]), key=lambda x: -SEVERITY_RANK.get(x, 0))

    # Lamps / glass with no detected damage, in a photo that DOES show damage -> VLM check
    inspect = []
    if matched:
        damaged = {m["part"] for m in matched}
        seen = set()
        for p in parts:
            if (p["label"] in INSPECT_PARTS and p["label"] not in damaged
                    and not too_broad(p, parts, img_area)):
                k = (p["label"], tuple(int(v) for v in p["bbox"]))
                if k not in seen:
                    seen.add(k)
                    inspect.append({"part": p["label"], "bbox": p["bbox"],
                                    "reason": "no damage found by YOLO on this lamp/glass part - "
                                              "VLM to confirm it is intact"})
    return {"matched": matched, "claims": list(claims.values()), "inspect": inspect,
            "rejected": rejected,
            "escalate": len(matched) == 0,
            "escalate_reason": "no damage located on a vehicle part" if not matched else None}


def crop(image: Image.Image, bbox, pad: float = 0.30) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = bbox
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return image.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                       min(w, int(x2 + px)), min(h, int(y2 + py))))


def draw(image: Image.Image, parts: List[Dict], fused: Dict) -> Image.Image:
    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    H, W = canvas.shape[:2]
    t = max(2, round(min(H, W) / 400))
    s = max(0.45, min(H, W) / 1400)
    tt = max(1, round(s * 1.6))
    placed: List[tuple] = []  # label rectangles already drawn

    def free(r):
        return all(r[2] <= q[0] or r[0] >= q[2] or r[3] <= q[1] or r[1] >= q[3] for q in placed)

    def box(b, color, text, thick=t):
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
        (tw, th), base = cv2.getTextSize(text, FONT, s, tt)
        lh = th + base + 6
        lx = min(max(0, x1), max(0, W - tw - 6))
        # try above the box, then inside its top, then keep moving down until free
        candidates = [y1 - lh, y1] + [y1 + k * lh for k in range(1, 40)]
        ly = next((y for y in candidates if 0 <= y <= H - lh and free((lx, y, lx + tw + 6, y + lh))),
                  max(0, y1 - lh))
        placed.append((lx, ly, lx + tw + 6, ly + lh))
        cv2.rectangle(canvas, (lx, ly), (lx + tw + 6, ly + lh), color, -1)
        cv2.putText(canvas, text, (lx + 3, ly + th + 2), FONT, s, WHITE, tt, cv2.LINE_AA)

    img_area = float(W * H)
    inspected = {(i["part"], tuple(i["bbox"])) for i in fused.get("inspect", [])}
    for p in parts:
        # skip boxes that say nothing (whole photo / swallows other panels) and
        # parts that are already drawn in orange as "check ..."
        if too_broad(p, parts, img_area) or (p["label"], tuple(p["bbox"])) in inspected:
            continue
        box(p["bbox"], GREEN, p["label"], max(1, t - 1))
    for r in fused["rejected"]:
        box(r["bbox"], GREY, f"rejected {r['label']}", max(1, t - 1))
    for i in fused.get("inspect", []):
        box(i["bbox"], ORANGE, f"check {i['part']}")
    for m in sorted(fused["matched"], key=lambda m: m["damage_confidence"]):
        box(m["damage_bbox"], RED, f"{m['damage']} {m['damage_confidence']:.2f}")
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def main() -> int:
    ap = argparse.ArgumentParser(description="Fuse YOLO damage + DINO parts")
    ap.add_argument("image")
    ap.add_argument("parts_json")
    ap.add_argument("yolo_json")
    ap.add_argument("--min-containment", type=float, default=0.6)
    ap.add_argument("--min-damage-conf", type=float, default=0.25)
    ap.add_argument("--pad", type=float, default=0.30, help="crop padding around damage (0.30 = 30%%)")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    try:
        image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
        parts_data = json.loads(Path(args.parts_json).read_text())
        parts = parts_data["parts"] if isinstance(parts_data, dict) else parts_data
        damages = json.loads(Path(args.yolo_json).read_text())
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    fused = fuse(damages, parts, image.size, args.min_containment, args.min_damage_conf)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(fused["matched"], 1):
        p = out / f"crop_{i}_{m['key']}.jpg"
        crop(image, m["damage_bbox"], args.pad).save(p, quality=95)
        m["crop_path"] = str(p)
    draw(image, parts, fused).save(out / "fused_annotated.jpg", quality=95)
    (out / "fused.json").write_text(json.dumps(fused, indent=2))

    print(f"Matched  : {len(fused['matched'])}")
    for m in fused["matched"]:
        print(f"  {m['key']:<24} conf={m['damage_confidence']:.2f} containment={m['containment']} "
              f"coverage={m['part_coverage']} severity~{m['severity_hint']}"
              + ("  [PART UNCERTAIN - ask VLM]" if m["part_uncertain"] else ""))
    for c in fused["claims"]:
        print(f"  CLAIM  {c['part']:<14} {', '.join(c['damages'])}  (primary: {c['primary_damage']})")
    for i in fused["inspect"]:
        print(f"  CHECK  {i['part']:<14} {i['reason']}")
    print(f"Rejected : {len(fused['rejected'])}")
    for r in fused["rejected"]:
        print(f"  {r['label']:<12} {r['reason']}")
    if fused["escalate"]:
        print(f"ESCALATE : {fused['escalate_reason']}")
    print(f"Saved    : {out / 'fused.json'}, {out / 'fused_annotated.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
