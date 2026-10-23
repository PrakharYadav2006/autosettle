"""
Grounding DINO part labels - baseline vs new, scored by a HUMAN review (no invented labels)
=========================================================================================

Step 1  (GPU laptop)  run both label methods on the SAME photos with the SAME loaded model:
    python tools/eval_dino_parts.py run  test_photos/  eval_out/
  -> eval_out/phrase/<photo>.json|.jpg      original labels (text-threshold phrase)
     eval_out/tokens/<photo>.json|.jpg      new labels (token evidence + ambiguity + recheck)
     eval_out/review.csv                    ONE row per box (boxes from both runs matched by IoU),
                                            plus one "__parts_present__" row per photo

Step 2  (a person)  open review.csv next to the two annotated images and fill:
    true_label   the real part in that box (a name from the vocabulary), or "none" if it is not a
                 vehicle part / not on the claim vehicle
    box_ok       y = box fits the part, n = badly localised (too big / cut off / two parts)
    For each photo's "__parts_present__" row, list the parts that are visible (true_label column,
    separated by ";") - this is what "missed" is measured against.
  Rules: decide from the photo only, BEFORE looking at which method said what (columns
  label_phrase / label_tokens can be hidden while reviewing).

Step 3  score:
    python tools/eval_dino_parts.py score eval_out/review.csv
  -> per method: boxes, correct label, wrong label (with hood/door and headlight/taillight
     confusions), false positives, badly localised, missed parts, ambiguity flags (and how often
     the true label was among the candidates), plus every box where the two methods disagree.

Nothing here reports accuracy unless review.csv has been filled in by a person.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRS = [("car hood", "car door"), ("headlight", "taillight"), ("front bumper", "rear bumper")]
MATCH_IOU = 0.7
FIELDS = ["photo", "box_id", "bbox", "label_phrase", "conf_phrase", "label_tokens", "conf_tokens",
          "tokens_ambiguous", "tokens_candidates", "tokens_resolution", "true_label", "box_ok", "notes"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ar = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    return inter / max(ar(a) + ar(b) - inter, 1e-6)


def merge_boxes(phrase: List[Dict], tokens: List[Dict]) -> List[Dict]:
    """One review row per physical box: a phrase box and a tokens box with IoU >= MATCH_IOU are
    the same box (the box regressor is identical; only labels / NMS differ)."""
    rows, used = [], set()
    for p in phrase:
        j = max(range(len(tokens)), key=lambda k: iou(p["bbox"], tokens[k]["bbox"]), default=None)
        t = tokens[j] if j is not None and j not in used and iou(p["bbox"], tokens[j]["bbox"]) >= MATCH_IOU else None
        if t is not None:
            used.add(j)
        rows.append((p, t))
    rows += [(None, t) for k, t in enumerate(tokens) if k not in used]
    out = []
    for p, t in rows:
        b = (p or t)["bbox"]
        out.append({"bbox": json.dumps([int(v) for v in b]),
                    "label_phrase": p["label"] if p else "", "conf_phrase": p["confidence"] if p else "",
                    "label_tokens": t["label"] if t else "", "conf_tokens": t["confidence"] if t else "",
                    "tokens_ambiguous": "y" if t and t.get("label_ambiguous") else "",
                    "tokens_candidates": ";".join([t["label"]] + [a["label"] for a in t.get("label_alternatives", [])])
                    if t and t.get("label_ambiguous") else "",
                    "tokens_resolution": (t or {}).get("label_resolution") or ""})
    return out


def run(photos: Path, out: Path, model: str = "base") -> None:
    from grounding_dino_parts import PartDetector
    det = PartDetector(model_size=model)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    imgs = sorted(p for p in photos.iterdir() if p.suffix.lower() in IMG_EXTS)
    for img in imgs:
        res = {}
        for mode in ("phrase", "tokens"):
            det.label_mode = mode
            annotated, r = det.detect(str(img), vehicle_type="car")
            d = out / mode
            d.mkdir(exist_ok=True)
            (d / f"{img.stem}.json").write_text(json.dumps(r, indent=2))
            annotated.save(d / f"{img.stem}.jpg", quality=90)
            res[mode] = r["parts"]
            print(f"{img.name:<28} {mode:<6} {len(r['parts']):>3} boxes  ambiguous: "
                  f"{', '.join(r.get('ambiguous_parts') or []) or '-'}  "
                  f"rechecks {r['timing_s'].get('label_rechecks', 0):.2f}s")
        rows.append({"photo": img.name, "box_id": "__parts_present__", **{k: "" for k in FIELDS[2:]}})
        for i, r in enumerate(merge_boxes(res["phrase"], res["tokens"]), 1):
            rows.append({"photo": img.name, "box_id": i, **r, "true_label": "", "box_ok": "", "notes": ""})
    with open(out / "review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(imgs)} photos.  Fill {out / 'review.csv'} (true_label, box_ok, parts present), then:\n"
          f"  python tools/eval_dino_parts.py score {out / 'review.csv'}")


def _norm(x: Optional[str]) -> str:
    return " ".join(str(x or "").strip().lower().split())


def score(review: Path) -> Dict:
    rows = list(csv.DictReader(open(review, newline="", encoding="utf-8-sig")))   # Excel adds a BOM
    boxes = [r for r in rows if r["box_id"] != "__parts_present__"]
    present = {r["photo"]: {_norm(x) for x in (r["true_label"] or "").split(";") if _norm(x)}
               for r in rows if r["box_id"] == "__parts_present__"}
    unreviewed = [r for r in boxes if not _norm(r["true_label"])]
    if unreviewed or not present or any(not v for v in present.values()):
        raise SystemExit(f"review.csv is not complete: {len(unreviewed)} box(es) without true_label, "
                         f"{sum(1 for v in present.values() if not v)} photo(s) without parts present. "
                         "No accuracy is reported from an unfinished review.")
    report = {}
    for mode in ("phrase", "tokens"):
        m = Counter()
        conf = Counter()
        found = {}
        for r in boxes:
            lab, truth = _norm(r[f"label_{mode}"]), _norm(r["true_label"])
            if not lab:
                continue
            m["boxes"] += 1
            if truth == "none":
                m["false_positive"] += 1
                continue
            found.setdefault(r["photo"], set())
            if lab == truth:
                m["correct_label"] += 1
                found[r["photo"]].add(truth)
            else:
                m["wrong_label"] += 1
                conf[f"{truth} -> {lab}"] += 1
            if _norm(r["box_ok"]) == "n":
                m["badly_localised"] += 1
            if mode == "tokens" and r["tokens_ambiguous"] == "y":
                m["flagged_ambiguous"] += 1
                cands = {_norm(c) for c in r["tokens_candidates"].split(";")}
                m["ambiguous_truth_in_candidates" if truth in cands else "ambiguous_truth_not_in_candidates"] += 1
                m["ambiguous_label_was_wrong" if lab != truth else "ambiguous_label_was_right"] += 1
        m["missed_parts"] = sum(len(present[p] - found.get(p, set())) for p in present)
        for a, b in PAIRS:
            m[f"{a} <-> {b} confusions"] = conf[f"{a} -> {b}"] + conf[f"{b} -> {a}"]
        report[mode] = {"metrics": dict(m), "confusions": dict(conf)}
    diffs = []
    for r in boxes:
        lp, lt, truth = _norm(r["label_phrase"]), _norm(r["label_tokens"]), _norm(r["true_label"])
        if lp != lt:
            ok = lambda lab: (lab == truth) or (not lab and truth == "none")   # noqa: E731 - removing a false box is right
            verdict = ("improvement" if ok(lt) and not ok(lp) else
                       "regression" if ok(lp) and not ok(lt) else
                       "both wrong" if not ok(lp) and not ok(lt) else "other")
            diffs.append({"photo": r["photo"], "box": r["box_id"], "phrase": lp or "-", "tokens": lt or "-",
                          "truth": truth, "tokens_ambiguous": r["tokens_ambiguous"] == "y", "verdict": verdict})
    report["disagreements"] = diffs
    return report


def print_report(rep: Dict) -> None:
    keys = sorted(set(rep["phrase"]["metrics"]) | set(rep["tokens"]["metrics"]))
    print(f"\n{'metric':<44}{'phrase (old)':>14}{'tokens (new)':>14}")
    for k in keys:
        print(f"{k:<44}{rep['phrase']['metrics'].get(k, 0):>14}{rep['tokens']['metrics'].get(k, 0):>14}")
    print(f"\nBoxes where the methods disagree ({len(rep['disagreements'])}):")
    for d in rep["disagreements"]:
        print(f"  {d['photo']:<24} box {d['box']:<4} old={d['phrase']:<14} new={d['tokens']:<14} "
              f"truth={d['truth']:<14} {d['verdict']}{' (flagged ambiguous)' if d['tokens_ambiguous'] else ''}")


def draw(review: Path, photos: Path, out: Path) -> None:
    """One picture per photo with every review box drawn and NUMBERED like the box_id column,
    so a reviewer can fill review.csv without guessing which box is which."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    rows = list(csv.DictReader(open(review, newline="", encoding="utf-8-sig")))
    out.mkdir(parents=True, exist_ok=True)
    colours = [(255, 80, 80), (80, 200, 255), (255, 200, 0), (120, 255, 120), (255, 120, 255), (255, 255, 255)]
    for photo in dict.fromkeys(r["photo"] for r in rows):
        src = photos / photo
        if not src.is_file():
            print(f"missing photo {src}")
            continue
        img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        d = ImageDraw.Draw(img)
        w = max(2, img.width // 300)
        try:
            font = ImageFont.truetype("arial.ttf", max(14, img.width // 40))
        except OSError:
            font = ImageFont.load_default()
        for r in rows:
            if r["photo"] != photo or r["box_id"] == "__parts_present__":
                continue
            b = json.loads(r["bbox"])
            c = colours[int(r["box_id"]) % len(colours)]
            d.rectangle(b, outline=c, width=w)
            tag = str(r["box_id"])
            tx, ty = b[0] + 3, b[1] + 3
            tb = d.textbbox((tx, ty), tag, font=font)
            d.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill=(0, 0, 0))
            d.text((tx, ty), tag, fill=c, font=font)
        img.save(out / (Path(photo).stem + "_numbered.jpg"), quality=90)
        print(f"{photo} -> {out / (Path(photo).stem + '_numbered.jpg')}")


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) == 4 and a[0] == "draw":
        draw(Path(a[1]), Path(a[2]), Path(a[3]))
        return 0
    if len(a) >= 3 and a[0] == "run":
        run(Path(a[1]), Path(a[2]), *(a[3:4] or []))
        return 0
    if len(a) == 2 and a[0] == "score":
        rep = score(Path(a[1]))
        print_report(rep)
        out = Path(a[1]).with_name("report.json")
        out.write_text(json.dumps(rep, indent=2))
        print(f"\nSaved {out}")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
