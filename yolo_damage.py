"""
Auto-Settle AI — Stage 2: damage detection with the CarDD-trained YOLOv8 model
=============================================================================

Takes ANY car photo (or a folder of photos) and writes yolo.json in the format
fuse.py expects:
    [{"label": "dent", "confidence": 0.95, "bbox": [x1, y1, x2, y2], "mask_area_px": 5120}, ...]

You need your teammate's trained weights file (e.g. cardd.pt). The generic
yolov8n.pt does NOT know what a dent is - it only knows cars, people, dogs...

Usage
-----
    python yolo_damage.py car.jpg                                     # auto-downloads CarDD v2.0
    python yolo_damage.py car.jpg --weights cardd.pt                 # -> yolo.json
    python yolo_damage.py car.jpg --weights cardd.pt --out my.json
    python yolo_damage.py photos\\ --weights cardd.pt --out-dir outputs  # -> outputs\\<name>_yolo.json

Install once:  pip install ultralytics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Per-class confidence cut-offs. Broken lamps / glass are rarer in CarDD and the
# model is less sure of them, but missing one is costly (lamps are expensive),
# so they get a lower bar. Every match still goes to the VLM to confirm.
CLASS_CONF = {"lamp broken": 0.10, "glass shatter": 0.10, "tire flat": 0.15}

# Small, high-value parts that get a second, zoomed-in YOLO pass. YOLO shrinks
# the whole photo to 640 px, so a headlight may be only ~100 px wide - too small
# to see a cracked lens. Cropping to the part first gives YOLO ~6x more pixels.
from vehicles import LAMP_GLASS_PARTS
ZOOM_PARTS = LAMP_GLASS_PARTS   # small, expensive parts get a zoomed second YOLO pass
ZOOM_PAD = 0.25
ZOOM_MIN_SIDE = 64   # px in the original photo


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_dets(dets: List[Dict], iou_thr: float = 0.5) -> List[Dict]:
    """Per-class NMS: drop a box if a higher-confidence box of the same class overlaps it."""
    keep: List[Dict] = []
    for d in sorted(dets, key=lambda d: -d["confidence"]):
        if all(k["label"] != d["label"] or iou(k["bbox"], d["bbox"]) < iou_thr for k in keep):
            keep.append(d)
    return keep


def load_image(image_path: str) -> Image.Image:
    """Same loading as the DINO script, so both models see identical pixels.
    (Phone photos carry an EXIF rotation; if one model applied it and the other
    didn't, the boxes would not line up.)"""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def results_to_dets(r) -> List[Dict]:
    """Convert one ultralytics Results object to fuse.py's format."""
    dets: List[Dict] = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    areas: Optional[list] = None
    if getattr(r, "masks", None) is not None and r.masks is not None:
        # segmentation model: real damage area (in pixels of the ORIGINAL image)
        areas = []
        for poly in r.masks.xy:
            if len(poly) < 3:
                areas.append(0)
                continue
            x, y = poly[:, 0], poly[:, 1]  # shoelace formula
            areas.append(int(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))))
    for i in range(len(xyxy)):
        d = {
            "label": str(r.names[int(cls[i])]).lower().replace("_", " "),
            "confidence": round(float(conf[i]), 4),
            "bbox": [round(float(v), 1) for v in xyxy[i]],
        }
        if areas is not None:
            d["mask_area_px"] = areas[i]
        dets.append(d)
    dets.sort(key=lambda d: -d["confidence"])
    return dets


# CarDD v2.0 checkpoint on Hugging Face (the one your teammate uses)
HF_REPO, HF_FILE, HF_SUBFOLDER = "abdullahg7/cardd-yolov8s", "best.pt", "v2.0"


def resolve_weights(weights: Optional[str]) -> str:
    """Local .pt file if given, otherwise download the CarDD checkpoint once (cached)."""
    if weights:
        return weights
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, subfolder=HF_SUBFOLDER)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not download CarDD weights from {HF_REPO}: {exc}. "
                           f"Pass a local file with --weights instead.") from exc


class DamageDetector:
    def __init__(self, weights: Optional[str] = None, conf: float = 0.25,
                 device: Optional[str] = None) -> None:
        weights = resolve_weights(weights)
        if not Path(weights).is_file():
            raise FileNotFoundError(
                f"Weights not found: {weights}  - get the CarDD .pt file from your teammate")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics not installed - run: pip install ultralytics") from exc
        self.model = YOLO(weights)
        self.conf = conf
        self.class_conf = {k: min(v, conf) for k, v in CLASS_CONF.items()}
        self.predict_conf = min([conf] + list(self.class_conf.values()))
        self.device = device  # None = ultralytics picks the GPU automatically
        self.classes = [str(n).lower() for n in self.model.names.values()]
        if not ({"dent", "scratch"} & set(self.classes)):
            print(f"[WARN] this model's classes are {self.classes} - it does not look like a "
                  f"car-damage model. Are you sure these are the CarDD weights?", file=sys.stderr)

    def _predict(self, image: Image.Image, offset=(0, 0), source: str = "full") -> List[Dict]:
        r = self.model.predict(image, conf=self.predict_conf, device=self.device, verbose=False)[0]
        out = []
        for d in results_to_dets(r):
            if d["confidence"] < self.class_conf.get(d["label"], self.conf):
                continue
            ox, oy = offset
            b = d["bbox"]
            d["bbox"] = [round(b[0] + ox, 1), round(b[1] + oy, 1), round(b[2] + ox, 1), round(b[3] + oy, 1)]
            d["source"] = source
            out.append(d)
        return out

    def detect(self, image_path: str, zoom_parts: Optional[List[Dict]] = None,
               zoom_all: bool = False) -> Dict:
        """zoom_parts: DINO part dicts; small high-value ones get a zoomed second pass.
        zoom_all=True (two-wheelers): zoom on EVERY part - bike parts are small in the
        frame and CarDD never saw them, so a closer look raises recall."""
        image = load_image(image_path)
        w, h = image.size
        t0 = time.perf_counter()
        dets = self._predict(image)
        zoomed = []
        for p in zoom_parts or []:
            if not zoom_all and p["label"] not in ZOOM_PARTS:
                continue
            x1, y1, x2, y2 = p["bbox"]
            px, py = (x2 - x1) * ZOOM_PAD, (y2 - y1) * ZOOM_PAD
            box = (max(0, int(x1 - px)), max(0, int(y1 - py)), min(w, int(x2 + px)), min(h, int(y2 + py)))
            # tiny crops get blown up into noise -> false "glass shatter" on reflections
            if box[2] - box[0] < ZOOM_MIN_SIDE or box[3] - box[1] < ZOOM_MIN_SIDE:
                continue
            found = self._predict(image.crop(box), offset=box[:2], source=f"zoom:{p['label']}")
            zoomed.append({"part": p["label"], "found": len(found)})
            dets += found
        infer_s = time.perf_counter() - t0
        return {"image": str(image_path), "image_size": [w, h], "inference_s": round(infer_s, 4),
                "zoom_passes": zoomed, "damages": merge_dets(dets)}


def list_images(path: str) -> List[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    return [p]


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-Settle AI - YOLO damage detection -> yolo.json")
    ap.add_argument("input", help="a car photo or a folder of photos")
    ap.add_argument("--weights", help="CarDD .pt file (default: auto-download CarDD v2.0)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="yolo.json", help="output file (single image)")
    ap.add_argument("--out-dir", default="outputs", help="output folder (folder input)")
    args = ap.parse_args()

    images = list_images(args.input)
    if not images:
        print(f"[ERROR] no images found in {args.input}", file=sys.stderr)
        return 1
    try:
        det = DamageDetector(args.weights, args.conf)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    for img in images:
        try:
            res = det.detect(str(img))
        except Exception as exc:  # noqa: BLE001 - keep going through a folder
            print(f"[ERROR] {img.name}: {exc}", file=sys.stderr)
            continue
        if len(images) == 1:
            out = Path(args.out)
        else:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            out = Path(args.out_dir) / f"{img.stem}_yolo.json"
        out.write_text(json.dumps(res["damages"], indent=2))
        print(f"{img.name}: {len(res['damages'])} damage(s) in {res['inference_s']:.3f}s -> {out}")
        for d in res["damages"]:
            extra = f"  mask={d['mask_area_px']}px" if "mask_area_px" in d else ""
            print(f"    {d['label']:<14} conf={d['confidence']:.2f}  bbox={[int(v) for v in d['bbox']]}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
