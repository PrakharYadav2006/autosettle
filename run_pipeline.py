"""
Auto-Settle AI — run Stages 1-3 on ANY photo (or a whole folder) in one command
==============================================================================

    photo -> Grounding DINO (parts) + YOLO CarDD (damage) -> fuse -> crops

Both models are loaded once, so a folder of 15 test photos runs quickly.

Usage
-----
    python run_pipeline.py car.jpg                    # downloads CarDD v2.0 weights once
    python run_pipeline.py test_photos --out-dir results
    python run_pipeline.py car.jpg --weights cardd.pt # or use a local weights file
    python run_pipeline.py claim.mp4                  # video -> see video_pipeline.py
    python run_pipeline.py activa.jpg --plate HR26BK9999        # scooter (type from policy)
    python run_pipeline.py bike.jpg --vehicle-type motorcycle   # force a type
    (default --vehicle-type auto: DINO decides car / motorcycle / scooter per photo)

Outputs per photo (in <out-dir>\\<photo name>\\):
    parts.json, yolo.json, fused.json, fused_annotated.jpg, crop_<n>_<key>.jpg
Plus <out-dir>\\summary.csv - one row per photo, for your accuracy table.

Needs these files in the same folder:
    grounding_dino_parts.py, refine_parts.py, yolo_damage.py, fuse.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from fuse import crop, draw, fuse
from grounding_dino_parts import PartDetector
from yolo_damage import DamageDetector, list_images, load_image
import pricing_rules
from severity import SeverityAssessor, assess_fused, load_vlm
from vehicles import family, is_two_wheeler, norm_type

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp"}


MISMATCH_CONF = 0.50   # only trust the auto type check when DINO is fairly sure


def decide_vehicle_type(image, parts_det, requested: str = "auto", expected=None):
    """requested (CLI) > expected (policy) > auto-detected > car.
    Returns (type, detection info, mismatch message or None)."""
    req = norm_type(requested)
    info = None
    if req is None or expected:   # auto mode, or we have a policy to check against
        info = parts_det.detect_vehicle_type(image)
    detected = (info or {}).get("vehicle_type")
    vt = req or norm_type(expected) or detected or "car"
    mismatch = None
    if (expected and detected and (info["confidence"] or 0) >= MISMATCH_CONF
            and family(expected) != family(detected)):
        mismatch = f"photo shows a {detected} but the policy is for a {norm_type(expected)}"
    return vt, info, mismatch


def run_one(img_path: Path, parts_det, damage_det, out_root: Path, pad: float,
            details: bool = False, assessor: SeverityAssessor = None,
            vehicle_type: str = "auto", expected_type=None):
    out = out_root / img_path.stem
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    image = load_image(str(img_path))

    vt, type_info, mismatch = decide_vehicle_type(image, parts_det, vehicle_type, expected_type)
    _, parts_res = parts_det.detect(str(img_path), vehicle_type=vt)
    # two-wheelers: CarDD never saw bikes -> zoomed YOLO pass on every part
    dmg_res = damage_det.detect(str(img_path), zoom_parts=parts_res["parts"],
                                zoom_all=is_two_wheeler(vt))

    fused = fuse(dmg_res["damages"], parts_res["parts"], image.size, vehicle_type=vt)
    fused["vehicle_type_detection"] = type_info
    fused["vehicle_mismatch"] = mismatch
    for i, m in enumerate(fused["matched"], 1):
        p = out / f"crop_{i}_{m['key']}.jpg"
        crop(image, m["damage_bbox"], pad).save(p, quality=95)
        m["crop_path"] = str(p)
    for i, ins in enumerate(fused["inspect"], 1):
        p = out / f"inspect_{i}_{ins['part'].replace(' ', '_')}.jpg"
        crop(image, ins["bbox"], 0.15).save(p, quality=95)
        ins["crop_path"] = str(p)

    if assessor is not None:   # Stage 4: severity (VLM facts + rules, or area rules)
        assess_fused(fused, image, assessor)

    (out / "parts.json").write_text(json.dumps(parts_res, indent=2))
    (out / "yolo.json").write_text(json.dumps(dmg_res["damages"], indent=2))
    (out / "fused.json").write_text(json.dumps(fused, indent=2))
    draw(image, parts_res["parts"], fused).save(out / "fused_annotated.jpg", quality=95)

    total = time.perf_counter() - t0
    row = {
        "photo": img_path.name,
        "vehicle_type": vt + (f" (auto {type_info['confidence']})" if type_info and type_info.get("vehicle_type")
                              and not norm_type(vehicle_type) and not expected_type else ""),
        "parts": ", ".join(p["label"] for p in parts_res["parts"]) or "-",
        "damages_found": len(dmg_res["damages"]),
        "matched": " | ".join(m["key"] + (" (uncertain)" if m["part_uncertain"] else "")
                              for m in fused["matched"]) or "-",
        "claims": " | ".join(f"{c['part']}: {'+'.join(c['damages'])}" for c in fused["claims"]) or "-",
        "severity": " | ".join(f"{m['key']}={m.get('severity') or ('not damage' if m.get('vlm_confirmed') is False else '?')}"
                               + ("(review)" if m.get("severity_review") else "")
                               for m in fused["matched"]) or "-",
        "vlm_check": ", ".join(i["part"] for i in fused["inspect"]) or "-",
        "vlm_missed": ", ".join(fused.get("vlm_missed_damage", [])) or "-",
        "rejected": len(fused["rejected"]),
        "escalate": fused["escalate"],
        "dino_s": parts_res["timing_s"]["inference"],
        "yolo_s": dmg_res["inference_s"],
        "total_s": round(total, 3),
        "fixes": " ; ".join(parts_res.get("refine_notes", [])) or "-",
        # fill these two in by hand after looking at fused_annotated.jpg
        "parts_correct(y/n)": "",
        "damage_correct(y/n)": "",
    }
    if details:
        return row, fused, parts_res
    return row


def add_vehicle_args(ap) -> None:
    g = ap.add_argument_group("vehicle / policy (enables pricing)")
    g.add_argument("--plate", help="licence plate -> demo policy lookup")
    g.add_argument("--make"); g.add_argument("--model"); g.add_argument("--year", type=int)
    g.add_argument("--cc", type=int, help="engine cc (deductible slab)")
    g.add_argument("--zero-dep", action="store_true", help="policy has zero-depreciation add-on")
    g.add_argument("--vehicle-type", choices=["auto", "car", "motorcycle", "scooter"], default="auto",
                   help="auto = Grounding DINO decides per photo (default)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-Settle AI - full Stage 1-3 pipeline")
    ap.add_argument("input", help="a car photo, a folder of photos, or a video (.mp4/.mov/.avi)")
    ap.add_argument("--weights", help="CarDD .pt file (default: auto-download CarDD v2.0)")
    ap.add_argument("--dino", choices=["tiny", "base"], default="base")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--pad", type=float, default=0.30, help="crop padding (0.30 = 30%%)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--severity", choices=["rules", "vlm"], default="rules",
                    help="rules = measured damage area; vlm = Qwen2.5-VL facts + rules + 3-view vote")
    ap.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    add_vehicle_args(ap)
    args, extra = ap.parse_known_args()

    if Path(args.input).suffix.lower() in VIDEO_EXTS:
        import video_pipeline  # video mode: same models, multi-frame consensus
        return video_pipeline.main(sys.argv[1:])
    if extra:
        ap.error(f"unrecognized arguments: {' '.join(extra)}")

    images = list_images(args.input)
    if not images:
        print(f"[ERROR] no images found in {args.input}", file=sys.stderr)
        return 1

    try:
        print("Loading models (first time downloads DINO)...")
        parts_det = PartDetector(model_size=args.dino)
        damage_det = DamageDetector(args.weights, args.yolo_conf)
        if args.severity == "vlm":
            print(f"Loading VLM {args.vlm_model} (first time downloads ~7 GB)...")
        assessor = SeverityAssessor(load_vlm("qwen" if args.severity == "vlm" else "none",
                                             args.vlm_model))
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    vehicle, suspicious = pricing_rules.resolve_vehicle(
        args.make, args.model, args.year, args.cc, args.zero_dep, args.plate,
        vehicle_type=None if args.vehicle_type == "auto" else args.vehicle_type)
    expected = (vehicle or {}).get("vehicle_type") if args.vehicle_type == "auto" else None
    if vehicle is None and not suspicious:
        print("(No --make/--plate given: damage only, no pricing.)")
    rows = []
    for img in images:
        try:
            row, fused, _ = run_one(img, parts_det, damage_det, out_root, args.pad, details=True,
                                    assessor=assessor, vehicle_type=args.vehicle_type,
                                    expected_type=expected)
            if fused["vehicle_mismatch"]:
                print(f"[WARN] {img.name}: {fused['vehicle_mismatch']}")
            if vehicle is not None or suspicious:
                v = dict(vehicle or {})
                v.setdefault("vehicle_type", None)
                v["vehicle_type"] = v["vehicle_type"] or fused["vehicle_type"]
                sus = list(suspicious) + ([fused["vehicle_mismatch"]] if fused["vehicle_mismatch"] else [])
                q = pricing_rules.price_claim(pricing_rules.items_from_fused(fused), v, sus)
                (out_root / img.stem / "quote.json").write_text(json.dumps(q, indent=2))
                row["quote"] = q["status"]
                row["payable_inr"] = q["totals"].get("insurer_payable_inr", "-")
                rng = q["totals"].get("settlement_range")
                row["offer_range_inr"] = (f"{rng['opening_offer_inr']}-{rng['maximum_offer_inr']}"
                                          if rng else "-")
        except Exception as exc:  # noqa: BLE001 - one bad photo must not stop the batch
            print(f"[ERROR] {img.name}: {exc}", file=sys.stderr)
            continue
        rows.append(row)
        flag = "  ESCALATE" if row["escalate"] else ""
        print(f"{row['photo']:<28} {row['total_s']:>6.2f}s  claims: {row['claims']}{flag}")
        print(f"{'':<36}severity: {row['severity']}")
        if row["vlm_missed"] != "-":
            print(f"{'':<36}VLM thinks YOLO MISSED a break on: {row['vlm_missed']} (human check)")
        if row["vlm_check"] != "-":
            print(f"{'':<36}VLM check (no damage found, confirm intact): {row['vlm_check']}")
        if "quote" in row:
            print(f"{'':<36}PRICING: {row['quote']}  payable Rs {row['payable_inr']}  "
                  f"offer range Rs {row['offer_range_inr']}  (details: quote.json)")

    if rows:
        with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{len(rows)} photo(s) done. Open {out_root / 'summary.csv'} and each "
              f"folder's fused_annotated.jpg; mark the two 'correct' columns y/n.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
