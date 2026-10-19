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

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp"}


def run_one(img_path: Path, parts_det, damage_det, out_root: Path, pad: float,
            details: bool = False):
    out = out_root / img_path.stem
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    _, parts_res = parts_det.detect(str(img_path))
    dmg_res = damage_det.detect(str(img_path), zoom_parts=parts_res["parts"])
    image = load_image(str(img_path))

    fused = fuse(dmg_res["damages"], parts_res["parts"], image.size)
    for i, m in enumerate(fused["matched"], 1):
        p = out / f"crop_{i}_{m['key']}.jpg"
        crop(image, m["damage_bbox"], pad).save(p, quality=95)
        m["crop_path"] = str(p)
    for i, ins in enumerate(fused["inspect"], 1):
        p = out / f"inspect_{i}_{ins['part'].replace(' ', '_')}.jpg"
        crop(image, ins["bbox"], 0.15).save(p, quality=95)
        ins["crop_path"] = str(p)

    (out / "parts.json").write_text(json.dumps(parts_res, indent=2))
    (out / "yolo.json").write_text(json.dumps(dmg_res["damages"], indent=2))
    (out / "fused.json").write_text(json.dumps(fused, indent=2))
    draw(image, parts_res["parts"], fused).save(out / "fused_annotated.jpg", quality=95)

    total = time.perf_counter() - t0
    row = {
        "photo": img_path.name,
        "parts": ", ".join(p["label"] for p in parts_res["parts"]) or "-",
        "damages_found": len(dmg_res["damages"]),
        "matched": " | ".join(m["key"] + (" (uncertain)" if m["part_uncertain"] else "")
                              for m in fused["matched"]) or "-",
        "claims": " | ".join(f"{c['part']}: {'+'.join(c['damages'])}" for c in fused["claims"]) or "-",
        "vlm_check": ", ".join(i["part"] for i in fused["inspect"]) or "-",
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-Settle AI - full Stage 1-3 pipeline")
    ap.add_argument("input", help="a car photo, a folder of photos, or a video (.mp4/.mov/.avi)")
    ap.add_argument("--weights", help="CarDD .pt file (default: auto-download CarDD v2.0)")
    ap.add_argument("--dino", choices=["tiny", "base"], default="base")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--pad", type=float, default=0.30, help="crop padding (0.30 = 30%%)")
    ap.add_argument("--out-dir", default="results")
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
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    vehicle, suspicious = pricing_rules.resolve_vehicle(
        args.make, args.model, args.year, args.cc, args.zero_dep, args.plate)
    if vehicle is None and not suspicious:
        print("(No --make/--plate given: damage only, no pricing.)")
    rows = []
    for img in images:
        try:
            row, fused, _ = run_one(img, parts_det, damage_det, out_root, args.pad, details=True)
            if vehicle is not None or suspicious:
                q = pricing_rules.price_claim(pricing_rules.items_from_fused(fused), vehicle or {}, suspicious)
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
