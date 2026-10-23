#!/usr/bin/env python3
"""Run a small, controlled Grounding DINO comparison on a photo ZIP.

Put this file in the same directory as grounding_dino_parts.py, refine_parts.py,
and vehicles.py. It reuses one loaded model and processes configurations and
photos sequentially, which is friendlier to an 8 GB GPU than loading several
models at once.

This script does NOT decide which configuration is accurate. It creates
annotated outputs and reviewer sheets so the same human criteria can be applied
to every configuration. Mean confidence and number of boxes are not accuracy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw, ImageOps

from grounding_dino_parts import PartDetector, load_image
from vehicles import CAR_PARTS


@dataclass(frozen=True)
class Config:
    name: str
    description: str
    box: float
    text: float
    nms: float
    cross_label_nms: float
    prompt: Optional[str] = None


def default_configs() -> list[Config]:
    baseline_prompt = " . ".join(CAR_PARTS) + " ."
    descriptive_parts = {
        "car hood": "car hood (front bonnet panel)",
        "car door": "car door (side door panel)",
        "headlight": "headlight (front lamp assembly)",
        "taillight": "taillight (rear lamp assembly)",
    }
    descriptive_prompt = " . ".join(descriptive_parts.get(p, p) for p in CAR_PARTS) + " ."
    return [
        Config("01_baseline", "Current settings and current part prompt", .35, .25, .50, .80,
               baseline_prompt),
        Config("02_recall", "Lower detection thresholds; may add missed parts and false boxes",
               .30, .20, .50, .80, baseline_prompt),
        Config("03_precision", "Higher box threshold; may remove weak or small parts",
               .40, .25, .50, .80, baseline_prompt),
        Config("04_keep_overlaps", "Less cross-label suppression for adjacent/overlapping parts",
               .35, .25, .50, .95, baseline_prompt),
        Config("05_context_words", "Adds front/rear/bonnet/side-door wording to confusing labels",
               .35, .25, .50, .80, descriptive_prompt),
    ]


def list_photos(source: Path, extract_to: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() == ".zip":
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                # Reject absolute paths and parent traversal in archive names.
                target = (extract_to / member.filename).resolve()
                if not target.is_relative_to(extract_to.resolve()):
                    raise ValueError(f"Unsafe path in ZIP: {member.filename}")
            archive.extractall(extract_to)
        root = extract_to
    elif source.is_dir():
        root = source
    else:
        raise FileNotFoundError(f"Photo ZIP or folder not found: {source}")

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    photos = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    if not photos:
        raise ValueError(f"No supported photos found in {root}")
    return photos


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_stem(photo: Path) -> str:
    # Keep ordinary names readable, while avoiding path separators.
    return "_".join(photo.stem.replace("\\", "_").replace("/", "_").split())[:120]


def make_contact_sheet(paths: list[Path], output: Path, title: str) -> None:
    if not paths:
        return
    columns = 3
    cell_w, cell_h, label_h = 420, 300, 34
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * (cell_h + label_h)), "#172033")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail((cell_w - 12, cell_h - 12))
        x = (index % columns) * cell_w + (cell_w - img.width) // 2
        y = (index // columns) * (cell_h + label_h) + 6
        sheet.paste(img, (x, y))
        draw.text(((index % columns) * cell_w + 6,
                   (index // columns) * (cell_h + label_h) + cell_h),
                  path.stem[:52], fill="white")
    draw.text((8, max(2, rows * (cell_h + label_h) - 22)), title[:100], fill="#93e9df")
    sheet.save(output, quality=88)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare five DINO configurations on the same photos")
    parser.add_argument("photos", help="Path to test_photos.zip or an extracted photo folder")
    parser.add_argument("--out", default="dino_sweep_output", help="Output folder (default: ./dino_sweep_output)")
    parser.add_argument("--model", choices=("tiny", "base"), default="base",
                        help="Use the same model size as the existing audit unless you intentionally want to compare another model")
    parser.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    parser.add_argument("--limit", type=int, default=0, help="Optional photo limit for a quick smoke run; 0 means all photos")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Run this on the GPU laptop, or use --device cpu for a slow smoke run.")

    out = Path(args.out).resolve()
    package = out / "dino_sweep_results"
    if package.exists():
        raise FileExistsError(f"Output already exists; choose a new --out folder or remove it: {package}")
    package.mkdir(parents=True)
    source = Path(args.photos).resolve()
    photos = list_photos(source, out / "_extracted_photos")
    if args.limit > 0:
        photos = photos[:args.limit]
    configs = default_configs()

    print(f"Photos: {len(photos)} | configs: {len(configs)} | model: {args.model} | device: {device}")
    print("Loading one model; each config/photo is processed sequentially.")
    detector = PartDetector(model_size=args.model, parts=None, device=device, warmup=True,
                            vehicle_type="car")

    config_rows: list[dict] = []
    detection_rows: list[dict] = []
    image_review_rows: list[dict] = []
    image_paths_by_config: dict[str, list[Path]] = {}

    try:
        for cfg in configs:
            cfg_dir = package / cfg.name
            cfg_dir.mkdir(parents=True)
            detector.box_threshold = cfg.box
            detector.text_threshold = cfg.text
            detector.nms_iou = cfg.nms
            detector.cross_label_iou = cfg.cross_label_nms
            detector.prompt = cfg.prompt or detector.prompt
            infer_times: list[float] = []
            counts: list[int] = []
            annotated_paths: list[Path] = []

            print(f"\n[{cfg.name}] {cfg.description}")
            for index, photo in enumerate(photos, start=1):
                image = load_image(str(photo))
                try:
                    annotated, result = detector.detect(str(photo), vehicle_type="car")
                    # detect() keeps the configured prompt and settings in this output.
                    result["sweep_config"] = {
                        "name": cfg.name, "description": cfg.description,
                        "box_threshold": cfg.box, "text_threshold": cfg.text,
                        "nms_iou": cfg.nms, "cross_label_iou": cfg.cross_label_nms,
                    }
                    relative_name = safe_stem(photo)
                    image_path = cfg_dir / f"{relative_name}_annotated.jpg"
                    json_path = cfg_dir / f"{relative_name}_parts.json"
                    annotated.save(image_path, quality=94)
                    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    annotated_paths.append(image_path)
                    infer_times.append(float(result["timing_s"]["inference"]))
                    counts.append(len(result["parts"]))

                    labels = "; ".join(f"{d['label']} ({d['confidence']:.3f})" for d in result["parts"])
                    image_review_rows.append({
                        "config": cfg.name, "photo": photo.name,
                        "detected_parts": labels,
                        "missed_parts_found_by_reviewer": "",
                        "false_or_wrongly_labelled_parts": "",
                        "all_material_detections_correct_YN": "",
                        "review_notes": "",
                    })
                    for det_i, det in enumerate(result["parts"], start=1):
                        detection_rows.append({
                            "config": cfg.name, "photo": photo.name,
                            "detection_index": det_i, "predicted_part": det["label"],
                            "confidence": det["confidence"], "bbox_xyxy": json.dumps(det["bbox"]),
                            "label_correct_YN": "", "box_localization_correct_YN": "",
                            "correct_part_if_wrong": "", "reviewer_note": "",
                        })
                finally:
                    del image
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    gc.collect()
                print(f"  {index:02d}/{len(photos)} {photo.name}: {len(result['parts'])} detections")

            image_paths_by_config[cfg.name] = annotated_paths
            make_contact_sheet(annotated_paths, cfg_dir / "contact_sheet.jpg", cfg.description)
            config_rows.append({
                "config": cfg.name, "description": cfg.description,
                "box_threshold": cfg.box, "text_threshold": cfg.text,
                "nms_iou": cfg.nms, "cross_label_iou": cfg.cross_label_nms,
                "photos": len(photos), "detections_total": sum(counts),
                "detections_per_photo_mean": round(statistics.mean(counts), 3) if counts else 0,
                "inference_seconds_median": round(statistics.median(infer_times), 3) if infer_times else 0,
                "accuracy": "NOT COMPUTED - requires human labels",
            })

    finally:
        detector.close()

    review_dir = package / "review"
    review_dir.mkdir()
    write_csv(review_dir / "config_summary.csv", list(config_rows[0]), config_rows)
    write_csv(review_dir / "image_review.csv", [
        "config", "photo", "detected_parts", "missed_parts_found_by_reviewer",
        "false_or_wrongly_labelled_parts", "all_material_detections_correct_YN", "review_notes",
    ], image_review_rows)
    write_csv(review_dir / "detection_review.csv", [
        "config", "photo", "detection_index", "predicted_part", "confidence", "bbox_xyxy",
        "label_correct_YN", "box_localization_correct_YN", "correct_part_if_wrong", "reviewer_note",
    ], detection_rows)

    guide = """# DINO configuration sweep results

Five configurations were run on the same photos using one loaded Grounding DINO model. The model was not fine-tuned; this is an inference/prompt/settings comparison.

## Configurations

1. `01_baseline`: current thresholds and prompt (box .35, text .25, NMS .50, cross-label NMS .80).
2. `02_recall`: lower detection thresholds (.30/.20); can increase false detections.
3. `03_precision`: higher box threshold (.40/.25); can miss small or subtle parts.
4. `04_keep_overlaps`: cross-label NMS .95 to reduce suppression of overlapping labels; may retain duplicates.
5. `05_context_words`: adds front/rear/bonnet/side-door wording for commonly confused parts.

## How to audit strictly

- Review each original alongside that configuration's annotated image. The original photos are included under `_extracted_photos` in the output folder (the ZIP includes them for reproducibility).
- In `detection_review.csv`, mark label correctness and box localization separately. A box can be in the right place but have the wrong part label.
- In `image_review.csv`, record missed parts as well as false or wrongly labelled detections. Do not mark an image fully correct just because its visible boxes look plausible; inspect the whole photo for missed parts.
- For strict image-level correctness, mark `all_material_detections_correct_YN=Y` only when all material detections have the right part and location and no material false detection is present. Record missed real parts separately; any material miss should also make the image-level result N.
- A higher number of detections or higher average confidence does not mean better performance. Choose based on the same human labels across all configurations, then examine confusion pairs such as hood/door and headlight/taillight.

The CSV starts blank by design. This runner cannot infer ground truth from DINO's own output. Do not report the config summary's box counts or confidence as accuracy.
"""
    (review_dir / "README_audit.md").write_text(guide, encoding="utf-8")

    # Include the source photos in the package to keep each overlay auditable.
    originals_dir = package / "original_photos"
    originals_dir.mkdir()
    for photo in photos:
        dest = originals_dir / photo.name
        if dest.exists():
            dest = originals_dir / f"{photo.parent.name}_{photo.name}"
        dest.write_bytes(photo.read_bytes())

    zip_path = out / "dino_sweep_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in package.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(out))

    print(f"\nDone. Review and return this file: {zip_path}")
    print("The winner must be chosen from the completed human review sheets, not from confidence or box count.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)
