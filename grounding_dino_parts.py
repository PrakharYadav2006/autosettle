"""
Auto-Settle AI — Stage 1: Vehicle COMPONENT detection with Grounding DINO
=======================================================================

Parts only. Damage comes from the CarDD-trained YOLOv8 (Stage 2) and the two
are fused in fuse.py (Stage 3).

Outputs (in --out-dir):
    annotated_parts.jpg   green boxes + label + confidence
    parts.json            [{label, confidence, bbox, bbox_norm}, ...] + timing

Usage
-----
    python grounding_dino_parts.py car.jpg                    # tiny model
    python grounding_dino_parts.py car.jpg --model base --runs 20
    python grounding_dino_parts.py car.jpg --compare --runs 20  # tiny vs base table
    python grounding_dino_parts.py car.jpg --parts "car door,bumper,side mirror"
    python grounding_dino_parts.py bike.jpg --vehicle-type auto      # car / motorcycle / scooter
    python grounding_dino_parts.py activa.jpg --vehicle-type scooter

Two-wheelers: no retraining - DINO is zero-shot, we only switch the TEXT prompt
(see vehicles.py).  The car-only geometric fixes (refine_parts) are skipped.

From code:
    from grounding_dino_parts import detect_parts
    annotated_img, result = detect_parts("car.jpg")
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import part_labels                    # token-evidence labels + honest ambiguity (same folder)
from refine_parts import refine_parts  # geometric label fixes (same folder)
from vehicles import CAR_PARTS, VEHICLE_PARTS, norm_type

MODEL_IDS = {
    "tiny": "IDEA-Research/grounding-dino-tiny",
    "base": "IDEA-Research/grounding-dino-base",
}

# Team's agreed component list. Note: DINO is unreliable at telling FRONT from
# REAR bumper from a close-up photo - expect both labels on one box sometimes.
# Cross-label NMS below keeps only the higher-scoring one; let the VLM decide
# front/rear in Stage 4 if it matters for pricing.
DEFAULT_PARTS: List[str] = CAR_PARTS   # see vehicles.py for motorcycle / scooter lists
# Giving DINO the RIGHT name for nearby things (handle, side skirt, wheel) stops it
# forcing them into the closest wrong label. Close-up photos are still weak:
# for the demo, also take one wide shot of the car side.

WHOLE_FRAME = 0.85  # box covering >85% of the photo = DINO guessed "the whole car"
# "tokens" (default): each box's label from the probabilities of the tokens that distinguish
# the part names, with ambiguity kept (part_labels.py).  "phrase": the original behaviour
# (HF text_threshold phrase + substring match), kept for A/B evaluation and as a fallback.
LABEL_MODE = os.environ.get("AUTOSETTLE_DINO_LABELS", "tokens")

GREEN = (0, 200, 0)  # BGR
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def canonicalize(phrase: str, vocab: List[str]) -> Optional[str]:
    """Map DINO's raw phrase (may be partial or merged) to one prompt label."""
    p = phrase.strip().lower()
    if not p:
        return None
    if p in vocab:
        return p
    contained = [v for v in vocab if v in p]       # "car door fender" -> longest match
    if contained:
        return max(contained, key=len)
    containing = [v for v in vocab if p in v]      # "mirror" -> "side mirror"
    if containing:
        return min(containing, key=len)
    return None  # phrase not in our vocab -> drop it


def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def load_image(image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read image '{path}': {exc}") from exc


def build_prompt(labels: List[str]) -> str:
    return " . ".join(l.strip().lower() for l in labels) + " ."


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class PartDetector:
    def __init__(
        self,
        model_size: str = "tiny",
        parts: Optional[List[str]] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        nms_iou: float = 0.5,
        cross_label_iou: float = 0.8,
        device: Optional[str] = None,
        warmup: bool = True,
        refine: bool = True,
        vehicle_type: str = "car",
        label_mode: Optional[str] = None,
    ) -> None:
        if model_size not in MODEL_IDS:
            raise ValueError(f"model_size must be one of {list(MODEL_IDS)}")
        self.model_size = model_size
        self.model_id = MODEL_IDS[model_size]
        self.vehicle_type = norm_type(vehicle_type) or "car"
        self.custom_parts = bool(parts)
        self.parts = [p.strip().lower() for p in (parts or VEHICLE_PARTS[self.vehicle_type]) if p.strip()]
        self.prompt = build_prompt(self.parts)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou = nms_iou
        self.cross_label_iou = cross_label_iou
        self.refine = refine
        self.label_mode = label_mode or LABEL_MODE
        self._tok_cache: Dict[str, Tuple] = {}

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            print("[WARN] CUDA not available - running on CPU; latency is NOT comparable "
                  "to a GPU YOLO run.", file=sys.stderr)

        t0 = time.perf_counter()
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = (AutoModelForZeroShotObjectDetection
                          .from_pretrained(self.model_id).to(self.device).eval())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load {self.model_id}: {exc}") from exc
        self.load_time_s = time.perf_counter() - t0

        if warmup:  # first CUDA call is slow (kernel setup) - keep it out of metrics
            self._forward(Image.new("RGB", (640, 480), (128, 128, 128)))
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _forward(self, image: Image.Image):
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        self._sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        self._sync()
        return inputs, outputs, time.perf_counter() - t0

    def _postprocess(self, inputs, outputs, image: Image.Image) -> List[Dict]:
        if self.label_mode == "tokens":
            try:
                return self._postprocess_tokens(inputs, outputs, image)
            except Exception as exc:  # noqa: BLE001 - tokenizer/shape surprise: old path, logged
                print(f"[WARN] token label scoring failed ({type(exc).__name__}: {exc}) - phrase labels used",
                      file=sys.stderr)
        return self._postprocess_phrase(inputs, outputs, image)

    def _token_maps(self):
        key = self.prompt
        if key not in self._tok_cache:
            enc = self.processor.tokenizer(self.prompt, return_offsets_mapping=True)
            self._tok_cache[key] = part_labels.label_token_ids(enc["offset_mapping"], enc["input_ids"],
                                                               self.prompt, self.parts)
        return self._tok_cache[key]

    def _box_scores(self, inputs, outputs):
        """Per-query token probabilities (Q x T) and xyxy-normalised boxes, like HF post-processing."""
        n_tok = inputs.input_ids.shape[1]
        probs = outputs.logits.sigmoid()[0, :, :n_tok].float().cpu().numpy()
        cxcywh = outputs.pred_boxes[0].float().cpu().numpy()
        xyxy = np.stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                         cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2], axis=1)
        return probs, xyxy

    def _postprocess_tokens(self, inputs, outputs, image: Image.Image) -> List[Dict]:
        """Same box threshold and NMS as before; the LABEL comes from part_labels.score_labels()
        (distinctive tokens), and confusable near-ties are kept as ambiguity, not hidden."""
        probs, xyxy = self._box_scores(inputs, outputs)
        pos, distinct = self._token_maps()
        w, h = image.size
        qscore = probs.max(axis=1)
        cands = []
        for q in np.where(qscore > self.box_threshold)[0]:
            ranked = part_labels.score_labels(probs[q], pos, distinct)
            if not ranked or ranked[0][1] < self.text_threshold:
                continue                     # a box, but no part name grounded on it
            x1, y1, x2, y2 = (float(np.clip(xyxy[q][0] * w, 0, w)), float(np.clip(xyxy[q][1] * h, 0, h)),
                              float(np.clip(xyxy[q][2] * w, 0, w)), float(np.clip(xyxy[q][3] * h, 0, h)))
            if box_area([x1, y1, x2, y2]) <= 0:
                continue
            dec = part_labels.decide(ranked)
            cands.append({**dec, "confidence": float(qscore[q]), "bbox": [x1, y1, x2, y2],
                          "raw_phrase": dec["label"]})
        keep: List[Dict] = []
        for lbl in {c["label"] for c in cands}:           # 1) per-label NMS (unchanged)
            same = [c for c in cands if c["label"] == lbl]
            k = nms(torch.tensor([c["bbox"] for c in same]), torch.tensor([c["confidence"] for c in same]),
                    self.nms_iou).tolist()
            keep += [same[i] for i in k]
        keep = part_labels.cross_label_nms(keep, self.cross_label_iou)   # 2) cross-label, alternatives kept
        dets = []
        for d in sorted(keep, key=lambda d: -d["confidence"]):
            b = [round(v, 1) for v in d["bbox"]]
            d.update(bbox=b, confidence=round(d["confidence"], 4),
                     bbox_norm=[round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)],
                     whole_frame=box_area(b) / (w * h) > WHOLE_FRAME)
            dets.append(d)
        return dets

    def recheck(self, image: Image.Image, det: Dict, names: List[str], pad: float = 0.25) -> Optional[Dict[str, float]]:
        """One extra DINO pass on a crop around an ambiguous box, prompting ONLY the candidate
        names.  Returns {name: score} for the crop box that best matches the original region."""
        W, H = image.size
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
        cx2, cy2 = min(W, int(x2 + pad * bw)), min(H, int(y2 + pad * bh))
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return None
        crop = image.crop((cx1, cy1, cx2, cy2))
        target = [(x1 - cx1) / crop.width, (y1 - cy1) / crop.height, (x2 - cx1) / crop.width, (y2 - cy1) / crop.height]
        saved = (self.parts, self.prompt)
        try:
            self.parts, self.prompt = list(names), build_prompt(names)
            inputs, outputs, _ = self._forward(crop)
            probs, xyxy = self._box_scores(inputs, outputs)
            pos, distinct = self._token_maps()
        finally:
            self.parts, self.prompt = saved
        best, best_key = None, -1.0
        for q in range(len(xyxy)):
            ov = part_labels.iou(list(xyxy[q]), target)
            if ov < 0.3:
                continue
            k = float(probs[q].max()) * ov
            if k > best_key:
                best, best_key = q, k
        if best is None:
            return None
        return {l: s for l, s in part_labels.score_labels(probs[best], pos, distinct)}

    def _postprocess_phrase(self, inputs, outputs, image: Image.Image) -> List[Dict]:
        target_sizes = [image.size[::-1]]
        try:  # newer transformers: `threshold`; older: `box_threshold`
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=target_sizes)[0]
        except TypeError:
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=target_sizes)[0]

        boxes = res["boxes"].detach().float().cpu()
        scores = res["scores"].detach().float().cpu()
        raw = [str(l) for l in (res.get("text_labels") or res.get("labels") or [])]

        w, h = image.size
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

        canon = [canonicalize(l, self.parts) for l in raw]
        valid = [i for i, c in enumerate(canon) if c and box_area(boxes[i].tolist()) > 0]

        # 1) per-label NMS: remove duplicate boxes of the same part
        keep: List[int] = []
        for lbl in {canon[i] for i in valid}:
            idx = [i for i in valid if canon[i] == lbl]
            keep += [idx[k] for k in nms(boxes[idx], scores[idx], self.nms_iou).tolist()]

        # 2) cross-label NMS: same region labelled twice (front vs rear bumper)
        if keep:
            k2 = nms(boxes[keep], scores[keep], self.cross_label_iou).tolist()
            keep = [keep[k] for k in k2]

        dets = []
        for i in sorted(keep, key=lambda i: -float(scores[i])):
            b = [round(float(v), 1) for v in boxes[i].tolist()]
            dets.append({
                "label": canon[i],
                "raw_phrase": raw[i],
                "confidence": round(float(scores[i]), 4),
                "bbox": b,  # absolute pixels [x_min, y_min, x_max, y_max]
                "bbox_norm": [round(b[0] / w, 4), round(b[1] / h, 4),
                              round(b[2] / w, 4), round(b[3] / h, 4)],
                "whole_frame": box_area(b) / (w * h) > WHOLE_FRAME,
            })
        return dets

    @staticmethod
    def draw(image: Image.Image, dets: List[Dict], color=GREEN) -> Image.Image:
        canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = canvas.shape[:2]
        thick = max(2, round(min(w, h) / 300))
        scale = max(0.5, min(w, h) / 1000)
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
            text = f'{d["label"]} {d["confidence"]:.2f}'
            (tw, th), base = cv2.getTextSize(text, FONT, scale, max(1, thick // 2))
            ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 10
            cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 6, ty + base), color, -1)
            cv2.putText(canvas, text, (x1 + 3, ty - 2), FONT, scale, WHITE,
                        max(1, thick // 2), cv2.LINE_AA)
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    def detect_vehicle_type(self, image: Image.Image, min_conf: float = 0.35) -> Dict:
        """Zero-shot: is the main vehicle in the photo a car, motorcycle or scooter?
        Picks the biggest confident box (the claim vehicle fills the frame)."""
        dets = self.find(image, ["car", "motorcycle", "scooter"])
        W, H = image.size
        best, best_score = None, 0.0
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            score = d["confidence"] * ((x2 - x1) * (y2 - y1)) / (W * H)
            if d["confidence"] >= min_conf and score > best_score:
                best, best_score = d, score
        return {"vehicle_type": best["label"] if best else None,
                "confidence": round(best["confidence"], 3) if best else None,
                "bbox": [round(v, 1) for v in best["bbox"]] if best else None,
                "candidates": [{"label": d["label"], "confidence": d["confidence"],
                                "bbox": [round(v, 1) for v in d["bbox"]]} for d in dets]}

    def detect(self, image_path: str, vehicle_type: Optional[str] = None) -> Tuple[Image.Image, Dict]:
        """vehicle_type: car / motorcycle / scooter / auto / None (= the detector's default)."""
        t_total = time.perf_counter()
        image = load_image(image_path)
        type_info = None
        vt = self.vehicle_type
        if vehicle_type == "auto":
            type_info = self.detect_vehicle_type(image)
            vt = type_info["vehicle_type"] or self.vehicle_type
        elif norm_type(vehicle_type):
            vt = norm_type(vehicle_type)
        saved = (self.parts, self.prompt)
        if vt != self.vehicle_type and not self.custom_parts:   # per-photo vocabulary switch
            self.parts = VEHICLE_PARTS[vt]
            self.prompt = build_prompt(self.parts)
        try:
            inputs, outputs, infer_s = self._forward(image)
            t_post = time.perf_counter()
            dets = self._postprocess(inputs, outputs, image)
            prompt_used = self.prompt
        finally:
            self.parts, self.prompt = saved
        refine_notes: List[str] = []
        if self.refine and vt == "car":   # door-seam / side-skirt fixes are car geometry
            dets, refine_notes = refine_parts(np.array(image), dets)
        recheck_s = 0.0
        dropped_vehicle: List[Dict] = []
        if self.label_mode == "tokens" and vt == "car":
            dets, dropped_vehicle, notes = part_labels.drop_whole_vehicle_boxes(dets)
            refine_notes += notes
            refine_notes += part_labels.context_flags(dets)
            t_rc = time.perf_counter()
            for d, names in part_labels.ambiguous_pairs(dets)[:part_labels.MAX_RECHECKS]:
                if len(names) < 2:
                    continue
                scores = self.recheck(image, d, names)
                if scores is None:
                    refine_notes.append(f"recheck of '{d['label']}' found no matching box - stays ambiguous")
                    continue
                before = d["label"]
                won = part_labels.apply_recheck(d, scores)
                refine_notes.append(f"recheck {' vs '.join(names)} at {[int(v) for v in d['bbox']]}: "
                                    + (f"'{won}'" + (f" (was '{before}')" if won != before else " confirmed")
                                       if won else "still too close - stays ambiguous"))
            recheck_s = time.perf_counter() - t_rc
        post_s = time.perf_counter() - t_post

        result = {
            "image": str(image_path),
            "image_size": list(image.size),
            "model": self.model_id,
            "device": str(self.device),
            "prompt": prompt_used,
            "vehicle_type": vt,
            "vehicle_type_detection": type_info,
            "thresholds": {"box": self.box_threshold, "text": self.text_threshold,
                           "nms_iou": self.nms_iou, "cross_label_iou": self.cross_label_iou},
            "label_mode": self.label_mode,
            "ambiguous_parts": [d["label"] for d in dets if d.get("label_ambiguous")],
            "dropped_whole_vehicle_boxes": dropped_vehicle,
            "timing_s": {
                "inference": round(infer_s, 4),
                "postprocess": round(post_s, 4),
                "label_rechecks": round(recheck_s, 4),
                "end_to_end": round(time.perf_counter() - t_total, 4),
                "model_load_one_time": round(self.load_time_s, 2),
            },
            "parts": dets,
            "refine_notes": refine_notes,
            "no_parts_found": len(dets) == 0,
        }
        if self.device.type == "cuda":
            result["gpu_peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
        return self.draw(image, dets), result

    def find(self, image: Image.Image, labels: List[str]) -> List[Dict]:
        """Detect arbitrary objects (e.g. 'license plate') with a one-off prompt.
        Same loaded model, no extra GPU memory; no part-label fixes applied."""
        saved = (self.parts, self.prompt)
        try:
            self.parts = [l.strip().lower() for l in labels]
            self.prompt = build_prompt(self.parts)
            inputs, outputs, _ = self._forward(image)
            return self._postprocess(inputs, outputs, image)
        finally:
            self.parts, self.prompt = saved

    def benchmark(self, image_path: str, runs: int = 20) -> Dict:
        image = load_image(image_path)
        times = [self._forward(image)[2] for _ in range(runs)]
        return {
            "runs": runs,
            "mean_s": round(statistics.mean(times), 4),
            "median_s": round(statistics.median(times), 4),
            "min_s": round(min(times), 4),
            "fps": round(1.0 / statistics.mean(times), 2),
        }

    def close(self) -> None:
        """Free GPU memory (used by --compare before loading the next model)."""
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Saving + simple function API
# --------------------------------------------------------------------------- #
def save_outputs(annotated: Image.Image, result: Dict, out_dir: str, suffix: str = "") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_path = out / f"annotated_parts{suffix}.jpg"
    json_path = out / f"parts{suffix}.json"
    annotated.save(img_path, quality=95)
    json_path.write_text(json.dumps(result, indent=2))
    return {"annotated": str(img_path), "json": str(json_path)}


_DEFAULT: Optional[PartDetector] = None


def detect_parts(image_path: str) -> Tuple[Image.Image, Dict]:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PartDetector()
    return _DEFAULT.detect(image_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def print_result(result: Dict, bench: Optional[Dict]) -> None:
    t = result["timing_s"]
    dev = result["device"]
    if dev.startswith("cuda"):
        dev += f" ({torch.cuda.get_device_name(0)})"
    print(f"\n===== {result['model']} =====")
    print(f"Device           : {dev}")
    vtd = result.get("vehicle_type_detection") or {}
    print(f"Vehicle type     : {result.get('vehicle_type', 'car')}"
          + (f"  (auto-detected, conf {vtd.get('confidence')})" if vtd else ""))
    print(f"Model load       : {t['model_load_one_time']:.2f} s (one-time)")
    print(f"Inference        : {t['inference']:.4f} s")
    print(f"Post-process+NMS : {t['postprocess']:.4f} s")
    print(f"End-to-end       : {t['end_to_end']:.4f} s")
    if "gpu_peak_mem_mb" in result:
        print(f"GPU peak memory  : {result['gpu_peak_mem_mb']} MB")
    if bench:
        print(f"Benchmark        : {bench['runs']} runs | mean {bench['mean_s']} s | "
              f"median {bench['median_s']} s | {bench['fps']} FPS")
    print(f"Parts found ({len(result['parts'])}):")
    for d in result["parts"]:
        flag = "  [covers whole photo - not specific]" if d.get("whole_frame") else ""
        print(f"  {d['label']:<14} conf={d['confidence']:.3f}  bbox={[int(v) for v in d['bbox']]}{flag}")
    for n in result.get("refine_notes", []):
        print(f"  fix: {n}")
    if result["no_parts_found"]:
        print("  none - try a lower --box-threshold or a wider photo")


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-Settle AI - Grounding DINO part detection")
    ap.add_argument("image")
    ap.add_argument("--model", choices=list(MODEL_IDS), default="tiny")
    ap.add_argument("--compare", action="store_true", help="run tiny AND base, print a table")
    ap.add_argument("--parts", help="comma-separated part list (overrides default)")
    ap.add_argument("--box-threshold", type=float, default=0.35)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--runs", type=int, default=1, help=">1 runs a latency benchmark")
    ap.add_argument("--no-refine", action="store_true", help="skip the geometric label fixes")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--vehicle-type", choices=["car", "motorcycle", "scooter", "auto"], default="car")
    ap.add_argument("--labels", choices=["tokens", "phrase"], default=None,
                    help="part-label method: tokens (default, ambiguity kept) or phrase (original)")
    args = ap.parse_args()

    parts = args.parts.split(",") if args.parts else None
    sizes = ["tiny", "base"] if args.compare else [args.model]
    summary = []

    for size in sizes:
        try:
            det = PartDetector(model_size=size, parts=parts,
                               box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                               refine=not args.no_refine, label_mode=args.labels)
            annotated, result = det.detect(args.image, vehicle_type=args.vehicle_type)
            bench = det.benchmark(args.image, args.runs) if args.runs > 1 else None
            if bench:
                result["benchmark"] = bench
            paths = save_outputs(annotated, result, args.out_dir,
                                 suffix=f"_{size}" if args.compare else "")
            det.close()
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        except torch.cuda.OutOfMemoryError:
            print(f"[ERROR] GPU out of memory on {size} model.", file=sys.stderr)
            return 1

        print_result(result, bench)
        for name, p in paths.items():
            print(f"Saved {name:<9}: {p}")
        summary.append((size, result, bench))

    if len(summary) > 1:
        print("\n========== tiny vs base ==========")
        print(f"{'model':<6} {'latency(s)':>11} {'FPS':>7} {'GPU MB':>8} {'#parts':>7} {'mean conf':>10}")
        for size, r, b in summary:
            lat = b["median_s"] if b else r["timing_s"]["inference"]
            fps = b["fps"] if b else round(1 / max(lat, 1e-9), 2)
            confs = [d["confidence"] for d in r["parts"]]
            mc = f"{statistics.mean(confs):.3f}" if confs else "-"
            print(f"{size:<6} {lat:>11.4f} {fps:>7} {r.get('gpu_peak_mem_mb', '-'):>8} "
                  f"{len(r['parts']):>7} {mc:>10}")
        print("(latency = median of --runs if given, else single run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
